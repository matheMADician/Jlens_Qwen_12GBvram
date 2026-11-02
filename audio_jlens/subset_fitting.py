"""Fit a Jacobian lens using selected source and target token positions.

This module is a project-local alternative to
``jacobian_lens.jlens.fitting``. It keeps the external package unchanged while
allowing the source and target token subsets to be selected independently.

``torch.autograd.grad(outputs, inputs, grad_outputs=v)`` computes a
vector-Jacobian product:

    d(v dot outputs) / d inputs

Therefore, setting the cotangent to one at several target positions sums their
contributions. This determines which target tokens contribute to the estimate.

The resulting gradient is then averaged over ``source_tokens``.
For source subset ``S`` and target subset ``T``, each returned matrix is:

    J_l = (1 / |S|) * sum_s in S sum_t in T
          d target_activation[t] / d source_activation_l[s]

The ``valid_position_mask`` from the external implementation is still applied
to both subsets. Positions before ``skip_first`` and the final position are
removed because they are excluded by the original next-token estimator.

The returned ``J_l`` has shape ``[d_model, d_model]`` and is compatible with
the external ``JacobianLens`` class.

Call tree:
    subset_fit_experiment
    ├── shuffle prompts; optionally cap the input
    ├── for each prompt
    │   ├── Parser.parse_json_line
    │   ├── _compute_single_prompt_jacobian
    │   │   ├── mode=None: jacobian_for_prompt
    │   │   └── subset mode
    │   │       ├── _text_suffix_token_count
    │   │       ├── SubsetConfig: select source/target token positions
    │   │       └── subset_jacobian_for_prompt
    │   ├── accumulate combined and per-dataset Jacobian sums
    │   ├── optionally cache projected per-prompt features
    │   └── optionally save a cumulative milestone lens
    ├── save dataset lenses and consolidated projected features
    └── save combined lens, metadata, and final checkpoint

    subset_fit
    ├── mode=None: delegate the full fit to jacobian_lens.fitting.fit
    └── subset mode
        ├── for each prompt: _compute_single_prompt_jacobian
        ├── accumulate the running Jacobian sum
        └── save or resume the fit checkpoint
"""
import logging, math, os, time
from collections.abc import Sequence
from typing import Any
import torch
from audio_jlens.random_proj import GaussianRandomProjector as GP

logger = logging.getLogger(__name__)

from jacobian_lens.jlens.hooks import ActivationRecorder
from jacobian_lens.jlens.lens import JacobianLens
from jacobian_lens.jlens.protocol import LensModel

# Original fitting.py methods
from jacobian_lens.jlens.fitting import jacobian_for_prompt, _check_layer_indices, valid_position_mask, _atomic_save, fit
from json_parser import Parser


#: Positions before this index are excluded from the Jacobian average; early
#: positions act as attention sinks and have atypical residual statistics.
SKIP_FIRST_N_POSITIONS = 16

# define a configuration class for source and target token subsets

from dataclasses import dataclass, field
@dataclass
class SubsetConfig:
    text_start: int
    prompt_count: int
    mode: str
    source_tokens: Sequence[int] = field(init=False)
    target_tokens: Sequence[int] = field(init=False)
    
    def __post_init__(self):
        if self.mode == "S2T":
            self.source_tokens = range(self.text_start)
            self.target_tokens = range(self.text_start, self.prompt_count)
        elif self.mode == "S2S":
            self.source_tokens = self.target_tokens = range(self.prompt_count)
        elif self.mode == "T2T":
            self.source_tokens = self.target_tokens = range(self.text_start, self.prompt_count)
        elif self.mode == "S&T2T":
            self.source_tokens = range(self.prompt_count)
            self.target_tokens = range(self.text_start, self.prompt_count)
        else:
            raise ValueError(
                "Modes in subset_fit() must be S2T, S2S, T2T, or S&T2T."
            )

def _text_suffix_token_count(
    tokenizer: Any, processor: Any, prompt: str, data_root: str | None
) -> int:
    """Return the token count after the audio segment in a formatted prompt."""
    try:
        sample = Parser.parse_json_line(prompt, audio_root=data_root)
    except (IndexError, TypeError, AttributeError) as exc:
        raise ValueError("prompt is not a valid audio-chat JSONL record") from exc

    chat_prompt = Parser.build_chat_prompt(processor, sample.prompt, sample.audio_path)
    audio_eos = "<|audio_eos|>"
    if audio_eos not in chat_prompt:
        raise ValueError(
            f"formatted prompt is missing {audio_eos}; cannot locate text positions"
        )

    text_suffix = chat_prompt.split(audio_eos, 1)[1]
    text_tokens = tokenizer(
        text_suffix, add_special_tokens=False, return_tensors="pt"
    )
    return int(text_tokens["input_ids"].shape[1])


def subset_jacobian_for_prompt(
    model: LensModel,
    subset_config: SubsetConfig,
    source_layers: Sequence[int],
    prompt: str | None = None,
    input_ids: torch.Tensor | None = None,
    mode: str | None = None,
    target_layer: int | None = None,
    dim_batch: int = 8,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> tuple[dict[int, torch.Tensor], int, int]:
    """Compute the per-layer Jacobian estimator ``J_l`` for one prompt.

    ``source_tokens`` and ``target_tokens`` contain token positions in the
    encoded prompt. Target positions are summed by the vector-Jacobian
    product, while source positions are averaged after backpropagation.

    Runs one forward pass on the prompt replicated ``dim_batch`` times along
    the batch axis, retains the graph, then runs ``ceil(d_model / dim_batch)``
    backward passes against it. Each backward computes ``dim_batch`` rows of
    ``J_l`` at once: batch element ``b`` carries a one-hot cotangent at output
    dimension ``dim_start + b``, set at every selected target position. See
    the module docstring for the resulting estimator and how it relates to a
    strict per-position Jacobian.

    Args:
        model: The model to compute Jacobians for.
        input_ids: Already encoded ``[1, seq_len]`` token IDs. ``model.encode``
            must have been called immediately before this function so multimodal
            processor inputs remain cached by the model adapter.
        subset_config: Configuration object containing source and target token positions.
        source_layers: Layer indices ``l`` to compute ``J_l`` at.
        target_layer: Layer to take gradients with respect to. Defaults to the
            final layer; negative indices count from the end. In some cases,
            targeting the penultimate layer can give a better-conditioned
            ``J_l``.
        dim_batch: Output dimensions computed per backward pass. Higher uses
            more GPU memory (the prompt is replicated this many times); total
            backward FLOPs are unchanged.
        skip_first: Leading positions to exclude; see :func:`valid_position_mask`.

    Returns:
        ``(jacobians, seq_len, n_source_positions)``. ``jacobians`` maps each
        source layer to a ``[d_model, d_model]`` fp32 CPU tensor.
    """
    if mode is None:
        if prompt is None:
            raise ValueError("Missing prompt for subset_jacobian_for_prompt.")
        return jacobian_for_prompt(
            model=model,
            prompt=prompt,
            source_layers=source_layers,
            target_layer=target_layer,
            dim_batch=dim_batch,
            skip_first=skip_first,
        )
    elif input_ids is None:
        raise ValueError("Missing input_ids for subset_jacobian_for_prompt.")
    
    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = _check_layer_indices(
        source_layers, target_layer, n_layers
    )

    seq_len = input_ids.shape[1]
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)

    def position_tensor(positions: Sequence[int], name: str) -> torch.Tensor:
        if not positions:
            raise ValueError(f"{name} must not be empty")
        position_tensor = torch.tensor(positions, dtype=torch.long)
        if ((position_tensor < 0) | (position_tensor >= seq_len)).any():
            raise ValueError(
                f"{name} must contain positions in [0, {seq_len}); "
                f"got {list(positions)}"
            )
           
        position_tensor = position_tensor[position_mask[position_tensor]]
        if position_tensor.numel() == 0:
            raise ValueError(
                f"{name} has no positions after applying valid_position_mask"
            )
        return position_tensor

    source_positions = position_tensor(subset_config.source_tokens, "source_tokens")
    target_positions = position_tensor(subset_config.target_tokens, "target_tokens")
    n_source_positions = source_positions.numel()

    jacobians = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32)
        for layer in source_layers
    }
    n_passes = math.ceil(d_model / dim_batch)

    with (
        ActivationRecorder(
            model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=min(source_layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        # One forward on the prompt replicated dim_batch times. The retained
        # graph is reused for every backward pass below.
        replicated_ids = input_ids.expand(dim_batch, -1)
        model.forward(replicated_ids)
        target_activation = recorder.activations[
            target_layer
        ]  # [dim_batch, seq_len, d_model]
        source_activations = [recorder.activations[layer] for layer in source_layers]

        target_positions = target_positions.to(
            target_activation.device
        )
        source_positions = source_positions.to(target_activation.device)
        batch_indices = torch.arange(dim_batch, device=target_activation.device)
        cotangent = torch.zeros_like(target_activation)

        for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
            n_dims_this_pass = min(dim_batch, d_model - dim_start)
            # One-hot cotangent at dim (dim_start + b) for batch element b,
            # at every selected target position. Yields rows dim_start..+n.
            cotangent.zero_()
            cotangent[
                batch_indices[:n_dims_this_pass, None],
                target_positions[None, :],
                dim_start + batch_indices[:n_dims_this_pass, None],
            ] = 1.0
            grads = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_activations,
                grad_outputs=cotangent,
                retain_graph=(pass_idx < n_passes - 1),
            )
            for layer, grad in zip(source_layers, grads, strict=True):
                # grad: [dim_batch, seq_len, d_model] on whatever device this
                # layer lives on; mean over selected source positions -> rows.
                positions_on_device = source_positions.to(
                    grad.device, non_blocking=True
                )
                rows = (
                    grad[:n_dims_this_pass, positions_on_device, :].float().mean(dim=1)
                )
                jacobians[layer][dim_start : dim_start + n_dims_this_pass, :] = (
                    rows.cpu()
                )
            del grads
            if pass_idx % 100 == 0 or pass_idx == n_passes - 1:
                logger.debug(
                    "    pass %d/%d (dims %d-%d)",
                    pass_idx + 1,
                    n_passes,
                    dim_start,
                    dim_start + n_dims_this_pass,
                )

    return jacobians, seq_len, n_source_positions

def _compute_single_prompt_jacobian(
    model: LensModel,
    processor: Any,
    prompt: str,
    mode: str | None,
    source_layers: Sequence[int],
    target_layer: int,
    dim_batch: int,
    skip_first: int,
    effective_max_seq_len: int,
    data_root: str | None,
) -> tuple[dict[int, torch.Tensor], int, int]:
    """Helper to encode a single prompt and compute its per-layer Jacobian."""
    if mode is None:
        # Calls the unmodified function.
        return jacobian_for_prompt(
            model=model,
            prompt=prompt,
            source_layers=source_layers,
            target_layer=target_layer,
            dim_batch=dim_batch,
            max_seq_len=effective_max_seq_len,
            skip_first=skip_first,
        )

    input_ids = model.encode(prompt, max_length=effective_max_seq_len)
    prompt_count = input_ids.shape[1]
    prompt_text_count = _text_suffix_token_count(
        model.tokenizer, processor, prompt, data_root
    )
    text_start = prompt_count - prompt_text_count
    if text_start < 0:
        raise ValueError(
            f"text suffix has {prompt_text_count} tokens, exceeding "
            f"encoded prompt length {prompt_count}"
        )

    subset_config = SubsetConfig(
        mode=mode,
        text_start=text_start,
        prompt_count=prompt_count,
    )
    return subset_jacobian_for_prompt(
        model,
        subset_config=subset_config,
        source_layers=source_layers,
        input_ids=input_ids,
        mode=mode,
        target_layer=target_layer,
        dim_batch=dim_batch,
        skip_first=skip_first,
    )

def subset_fit_experiment(
    model: LensModel,
    processor: Any,
    prompts: Sequence[str],
    data_root: str,
    seed: int,
    output_dir: str,
    checkpoint_path: str,
    checkpoint_every: int | None = 1,
    mode: str | None = None,
    save_prompt_counts: Sequence[int] | None = None,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 3000,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    resume: bool = True,
    max_prompts: int | None = None,
    *,
    projectors: list[GP] | None = None,
    save_projected_features: bool = True,
    save_dataset_lenses: bool = True,
    save_milestone_lenses: bool = True,
) -> tuple[JacobianLens, dict[str, JacobianLens], dict[int, JacobianLens]]:
    """Fit combined, per-dataset, and cumulative Jacobian lenses in one pass.

    Prompts are shuffled deterministically with ``seed`` and then limited by
    ``max_prompts``, if supplied. Each valid prompt contributes its Jacobians
    to the combined and dataset-specific means. Milestone means are captured
    after the requested number of successfully processed prompts. Prompts that
    raise ``ValueError`` or ``OSError`` during parsing or fitting are logged
    and skipped.

    ``mode=None`` uses the ordinary Jacobian estimator. The subset modes
    ``"S2T"``, ``"S2S"``, ``"T2T"``, and ``"S&T2T"`` select source and target
    token positions relative to the text suffix of the formatted audio chat
    prompt; these modes require ``processor``.

    Fitting state is checkpointed atomically and can resume from
    ``checkpoint_path``. Resume validates source layers, target layer,
    ``skip_first``, shuffle seed, prompt cap, projector configuration,
    projected-feature setting, and milestone-save setting against the
    checkpoint. With projected-feature saving enabled, each prompt's
    projected Jacobians are flattened and concatenated across source layers,
    then written as durable shards for resuming and later per-dataset
    consolidation.

    Lens saving switches only control files: dataset and milestone lenses are
    still computed and returned when their corresponding saving option is
    false. The combined lens and fit metadata are always saved to
    ``output_dir``. If projectors are supplied, the projection matrix and
    projected combined lens are saved separately from full-size milestone
    lenses. Saving per-prompt projected features requires exactly one
    projector.

    Args:
        model: Model whose activations and Jacobians are used for fitting.
        processor: Multimodal processor used to format audio-chat prompts;
            required when ``mode`` selects token subsets.
        prompts: Input records, one JSONL string per prompt.
        data_root: Root directory used to resolve audio paths in each record.
        seed: Seed for the deterministic prompt shuffle; it must match the
            checkpoint when resuming.
        output_dir: Directory for the combined lens, metadata, optional dataset
            and milestone lenses, and optional projected artifacts.
        checkpoint_path: Path to the resumable fitting-state checkpoint.
        checkpoint_every: Save intermediate state after this many input
            positions. The final state is saved when fitting completes;
            ``None`` disables intermediate checkpoint writes.
        mode: Jacobian token-subset mode: ``"S2T"`` (source before the text
            suffix, target within it), ``"S2S"`` (all prompt positions for
            source and target), ``"T2T"`` (text suffix for both), ``"S&T2T"``
            (all source positions and text-suffix target positions), or
            ``None`` for ordinary fitting.
        save_prompt_counts: Successful-prompt counts at which to compute and
            return cumulative milestone lenses. Defaults to no milestones.
        source_layers: Model layers at which to compute Jacobians; defaults to
            the layers selected by the fitting layer validator.
        target_layer: Layer differentiated to compute Jacobians; defaults to
            the final layer as resolved by the fitting layer validator.
        dim_batch: Number of Jacobian output dimensions computed in each
            backward pass.
        max_seq_len: Maximum encoded prompt length passed to the model.
        skip_first: Number of leading token positions excluded by the Jacobian
            estimator.
        resume: If true and ``checkpoint_path`` exists, restore its fit state
            and continue from the next shuffled prompt.
        max_prompts: Optional maximum number of shuffled input records to fit.
            This cap is applied after shuffling.
        projectors: Optional unique random-projection configurations. Their
            matrices and projected aggregate lenses are saved under
            ``output_dir/projected``.
        save_projected_features: If true, persist a projected feature vector
            for each successfully fitted prompt for downstream t-SNE
            visualization. Requires exactly one projector.
        save_dataset_lenses: Save one full-dimensional lens file per dataset
            under ``output_dir/datasets`` (default true).
        save_milestone_lenses: Save full-size cumulative lens files at
            ``save_prompt_counts`` under ``output_dir/milestones`` (default
            true), regardless of whether projectors are supplied.

    Returns:
        A tuple containing the combined lens, a mapping from normalized
        dataset name to its lens, and a mapping from successful-prompt count
        to each computed milestone lens. Dataset and milestone mappings are
        populated independently of their file-saving switches.
    """
    import random

    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = _check_layer_indices(
        source_layers, target_layer, n_layers
    )

    if projectors is not None:
        if not projectors:
            raise ValueError("projectors must be non-empty when provided.")
        # Since the projected features still take up a substantial amount of memory,
            # we ensure that when projected features are saved, only one projector is used per run.
            # If you have a lot of RAM and Disc space, this check can be relaxed.
        if save_projected_features and len(projectors) != 1:
            raise ValueError(
                "Per-prompt feature saving requires exactly one projector per fit."
            )
        if any(projector.d_model != d_model for projector in projectors):
            raise ValueError("Every projector d_model must match the model d_model.")
        projector_names = [projector.name for projector in projectors]
        if len(set(projector_names)) != len(projector_names):
            raise ValueError(
                "Projector dimension/seed configurations must be unique."
            )

    # Deterministically shuffle prompts using a dedicated Random instance
    total_prompts_input = len(prompts)
    shuffled_prompts = list(prompts)
    rng = random.Random(seed)
    rng.shuffle(shuffled_prompts)
    if max_prompts is not None:
        shuffled_prompts = shuffled_prompts[:max_prompts]

    logger.info(
        "Fitting %d/%d prompts after deterministic shuffle (max_prompts=%s)",
        len(shuffled_prompts),
        total_prompts_input,
        max_prompts,
    )

    sorted_milestones = sorted(set(save_prompt_counts)) if save_prompt_counts else []

    # Directories for saving artifacts
    milestones_dir = (
        os.path.join(output_dir, "milestones")
        if output_dir and save_milestone_lenses
        else None
    )
    datasets_dir = (
        os.path.join(output_dir, "datasets")
        if output_dir and save_dataset_lenses
        else None
    )
    projected_root = os.path.join(output_dir, "projected") if output_dir and projectors else None

    if milestones_dir:
        os.makedirs(milestones_dir, exist_ok=True)
    if datasets_dir:
        os.makedirs(datasets_dir, exist_ok=True)
    if projected_root:
        os.makedirs(projected_root, exist_ok=True)

    # State containers
    total_jacobian_sum: dict[int, torch.Tensor] = {
        l: torch.zeros(d_model, d_model, dtype=torch.float32) for l in source_layers
    }
    dataset_jacobian_sums: dict[str, dict[int, torch.Tensor]] = {}
    dataset_counts: dict[str, int] = {}
    total_done: int = 0
    next_idx: int = 0
    saved_milestones: set[int] = set()
    milestone_lenses: dict[int, JacobianLens] = {}
    projection_cache_available = projectors is not None and save_projected_features
    fit_complete = False

    if resume and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for key, expected in (
            ("source_layers", source_layers),
            ("target_layer", target_layer),
            ("skip_first", skip_first),
            ("seed", seed),
        ):
            if key in state and state[key] != expected:
                raise ValueError(
                    f"checkpoint at {checkpoint_path} was fitted with {key}="
                    f"{state.get(key)!r}, not {expected!r}; pass resume=False to discard it"
                )
        if state.get("max_prompts") != max_prompts:
            raise ValueError(
                f"checkpoint at {checkpoint_path} was fitted with max_prompts="
                f"{state.get('max_prompts')!r}, not {max_prompts!r}; "
                "pass resume=False to discard it"
            )
        expected_projectors = (
            [projector.metadata.as_dict() for projector in projectors]
            if projectors is not None
            else []
        )
        if state.get("projectors", []) != expected_projectors:
            raise ValueError(
                f"checkpoint at {checkpoint_path} has different projector "
                "configurations; use a new run name or pass resume=False"
            )
        if state.get("save_projected_features", True) != save_projected_features:
            raise ValueError(
                f"checkpoint at {checkpoint_path} has a different "
                "save_projected_features setting; use a new run name or "
                "pass resume=False"
            )
        if state.get("save_milestone_lenses", True) != save_milestone_lenses:
            raise ValueError(
                f"checkpoint at {checkpoint_path} has a different "
                "save_milestone_lenses setting; use a new run name or "
                "pass resume=False"
            )
        total_jacobian_sum = state["total_jacobian_sum"]
        dataset_jacobian_sums = state["dataset_jacobian_sums"]
        dataset_counts = state["dataset_counts"]
        total_done = state["total_done"]
        next_idx = state["next_idx"]
        saved_milestones = set(state.get("saved_milestones", []))
        fit_complete = state.get("fit_complete", False)
        if projectors is not None and next_idx > 0:
            if fit_complete:
                projection_cache_available = False
            elif save_projected_features:
                assert projected_root
                for projector in projectors:
                    projector.prepare_feature_cache_for_resume(
                        projected_root,
                        next_prompt_index=next_idx,
                        expected_count=total_done,
                    )
            else:
                projection_cache_available = False
            if not fit_complete and not save_projected_features:
                projection_cache_available = False
            if fit_complete and save_projected_features:
                assert projected_root
                for projector in projectors:
                    projector.clear_disk_cache(projected_root)
            if not fit_complete and not projection_cache_available and save_projected_features:
                raise ValueError(
                    "Could not restore projected feature caches for this fit. "
                    "Use a new run name or pass resume=False to rebuild them."
                )
            for projector in projectors:
                assert projected_root
                projector_dir = projector.projected_directory(projected_root)
                if not (projector_dir / "projection_matrix.npz").is_file():
                    if fit_complete:
                        raise FileNotFoundError(
                            "Completed fit checkpoint is missing projected artifacts: "
                            f"{projector_dir}"
                        )
        logger.info(
            "Resuming experiment fitting from %d/%d (total_done=%d)",
            next_idx,
            len(shuffled_prompts),
            total_done,
        )

    if projected_root and projectors is not None:
        for projector in projectors:
            projector.save_projection_matrix(
                projected_root, source_layers=source_layers
            )

    def write_checkpoint() -> None:
        if checkpoint_path is not None:
            state = {
                "total_jacobian_sum": total_jacobian_sum,
                "dataset_jacobian_sums": dataset_jacobian_sums,
                "dataset_counts": dataset_counts,
                "total_done": total_done,
                "next_idx": next_idx,
                "saved_milestones": list(saved_milestones),
                "source_layers": source_layers,
                "target_layer": target_layer,
                "skip_first": skip_first,
                "seed": seed,
                "max_prompts": max_prompts,
                "projectors": (
                    [projector.metadata.as_dict() for projector in projectors]
                    if projectors is not None
                    else []
                ),
                "save_projected_features": save_projected_features,
                "save_milestone_lenses": save_milestone_lenses,
                "fit_complete": fit_complete,
            }
            _atomic_save(state, checkpoint_path)

    for prompt_idx, prompt in enumerate(shuffled_prompts):
        if prompt_idx < next_idx:
            continue

        start_time = time.perf_counter()
        try:
            # Parse record metadata (dataset_name)
            parsed_sample = Parser.parse_json_line(prompt, audio_root=data_root)
            raw_dataset_name = parsed_sample.dataset_name or "unknown"
            dataset_name = raw_dataset_name.lower()

            per_prompt_J, seq_len, n_valid = _compute_single_prompt_jacobian(
                model=model,
                prompt=prompt,
                mode=mode,
                source_layers=source_layers,
                target_layer=target_layer,
                dim_batch=dim_batch,
                skip_first=skip_first,
                effective_max_seq_len=max_seq_len,
                processor=processor,
                data_root=data_root,
            )
        except (ValueError, OSError) as exc:
            logger.warning("  skipping prompt %d: %s", prompt_idx, exc)
            next_idx = prompt_idx + 1
            continue

        # Accumulate to global running sum
        for l in source_layers:
            total_jacobian_sum[l] += per_prompt_J[l]
        total_done += 1
        next_idx = prompt_idx + 1

        # Accumulate to per-dataset running sum
        if dataset_name not in dataset_jacobian_sums:
            dataset_jacobian_sums[dataset_name] = {
                l: torch.zeros(d_model, d_model, dtype=torch.float32)
                for l in source_layers
            }
            dataset_counts[dataset_name] = 0
        for l in source_layers:
            dataset_jacobian_sums[dataset_name][l] += per_prompt_J[l]
        dataset_counts[dataset_name] += 1

        if projectors is not None and save_projected_features:
            sample_id = parsed_sample.sample_id
            prompt_id = (
                str(sample_id) if sample_id is not None else f"prompt_{prompt_idx}"
            )
            for projector in projectors:
                assert projected_root
                projector.cache_prompt(
                    per_prompt_J,
                    source_layers,
                    sample_id=prompt_id,
                    dataset_name=dataset_name,
                    prompt_index=prompt_idx,
                    cache_dir=projector.feature_cache_directory(projected_root),
                )

        prompt_norm = max(per_prompt_J[l].norm().item() for l in source_layers) / math.sqrt(d_model)
        logger.info(
            "  [exp] prompt %d/%d (total_done=%d, ds=%s) seq_len=%d n_valid=%d %.0fs "
            "max||J||/sqrt(d)=%.3f",
            prompt_idx + 1,
            len(shuffled_prompts),
            total_done,
            dataset_name,
            seq_len,
            n_valid,
            time.perf_counter() - start_time,
            prompt_norm,
        )

        # Save milestones
        if total_done in sorted_milestones and total_done not in saved_milestones:
            m_means = {l: total_jacobian_sum[l] / total_done for l in source_layers}
            m_lens = JacobianLens(jacobians=m_means, n_prompts=total_done, d_model=d_model)
            milestone_lenses[total_done] = m_lens
            if milestones_dir:
                save_file = os.path.join(milestones_dir, f"lens_count_{total_done}.pt")
                m_lens.save(save_file)
                logger.info("Saved milestone lens for count %d to %s", total_done, save_file)
            saved_milestones.add(total_done)

        if checkpoint_every is not None and next_idx % checkpoint_every == 0:
            write_checkpoint()

    if total_done == 0:
        raise ValueError("no prompts were successfully processed for experiment fitting")

    # Finalize per-dataset lenses
    dataset_lenses: dict[str, JacobianLens] = {}
    for ds_name, sums in dataset_jacobian_sums.items():
        cnt = dataset_counts[ds_name]
        ds_mean = {l: sums[l] / cnt for l in source_layers}
        ds_lens = JacobianLens(jacobians=ds_mean, n_prompts=cnt, d_model=d_model)
        dataset_lenses[ds_name] = ds_lens
        if datasets_dir:
            save_file = os.path.join(datasets_dir, f"lens_{ds_name}.pt")
            ds_lens.save(save_file)
            logger.info("Saved dataset lens for %s (n=%d) to %s", ds_name, cnt, save_file)

    if (
        projectors is not None
        and projected_root is not None
        and save_projected_features
        and projection_cache_available
    ):
        for projector in projectors:
            projector_dir = projector.save_features(
                projected_root,
                source_layers=list(source_layers),
                dataset_counts=dataset_counts,
                use_disk_cache=True,
            )
            logger.info(
                "Saved projected prompt features for %s to %s",
                projector.name,
                projector_dir,
            )

    # Final combined lens
    combined_mean = {l: total_jacobian_sum[l] / total_done for l in source_layers}
    combined_lens = JacobianLens(
        jacobians=combined_mean, n_prompts=total_done, d_model=d_model
    )

    if projectors is not None and projected_root is not None:
        for projector in projectors:
            projector.save_lens(
                combined_lens,
                projector.projected_directory(projected_root) / "lens_full.pt",
            )
    else:
        combined_save_file = os.path.join(output_dir, "lens_full.pt")
        combined_lens.save(combined_save_file)
    # Write metadata
    import json
    meta = {
        "seed": seed,
        "total_prompts_input": total_prompts_input,
        "total_prompts_selected": len(shuffled_prompts),
        "max_prompts": max_prompts,
        "total_done": total_done,
        "dataset_counts": dataset_counts,
        "saved_milestones": sorted(saved_milestones),
        "save_dataset_lenses": save_dataset_lenses,
        "save_milestone_lenses": save_milestone_lenses,
        "source_layers": list(source_layers),
        "target_layer": target_layer,
        "mode": mode,
    }
    if projectors is not None:
        meta["projectors"] = [
            projector.metadata.as_dict() for projector in projectors
        ]
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    fit_complete = True
    write_checkpoint()
    if projectors is not None and projected_root is not None and save_projected_features:
        for projector in projectors:
            projector.clear_disk_cache(projected_root)
    return combined_lens, dataset_lenses, milestone_lenses


def subset_fit(
    model: LensModel,
    prompts: Sequence[str],
    mode: str | None = None,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int | None = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    checkpoint_path: str | None = None,
    checkpoint_every: int | None = 1,
    resume: bool = True,
    *,
    processor: Any | None = None,
    data_root: str | None = None,
) -> JacobianLens:
    """Fit a subset Jacobian lens over a list of prompts.

    For each prompt, :func:`subset_jacobian_for_prompt` computes a matrix using
    the selected source and target token positions. These per-prompt matrices
    are accumulated as a running mean. The same token-position subsets are
    derived from each raw JSONL record after it is encoded once by ``model``.
    Text positions use the suffix following ``<|audio_eos|>`` in the formatted
    chat prompt.

    If ``checkpoint_path`` is set, the running sum is written every
    ``checkpoint_every`` prompts using an atomic replace and can be resumed
    after a restart. Prompts that are too short or have no valid selected
    positions are skipped, matching the behavior of the external fitter.

    Args:
        model: The model to fit on.
        prompts: Raw JSONL records to fit. See the README for guidance on
            corpus size and distribution.
        mode: Defines the source and target tokens for Jacobian matrix calculation.
            Can be one of the following:
            - "S2T"
            - "S2S"
            - "T2T"
            - "S&T2T"
            - ``None`` to delegate to the ordinary Jacobian fitter.
        source_layers: Layers to fit at. Defaults to every layer below
            ``target_layer``; negative indices count from the end.
        target_layer: See :func:`subset_jacobian_for_prompt`. Defaults to the
            final layer; negative indices count from the end.
        dim_batch: See :func:`subset_jacobian_for_prompt`.
        max_seq_len: Truncate each prompt to this many tokens. ``None`` uses 3000.
        skip_first: See :func:`valid_position_mask`. Positions before this
            index and the final position are excluded from both subsets.
        checkpoint_path: If set, write a resumable checkpoint here.
        checkpoint_every: Write the checkpoint every N prompts (default 1).
            ``None`` skips per-iteration writes and saves once at the end; the
            checkpoint can be large (``len(source_layers) * d_model**2 * 4``
            bytes), so raise this for large models.
        resume: If ``True`` and ``checkpoint_path`` exists, resume from it.

    Returns:
        The fitted :class:`JacobianLens`.
    """
    effective_max_seq_len = (
        max_seq_len if max_seq_len is not None else 3000
    )
    effective_resume = resume and checkpoint_path is not None

    if mode is None:
        return fit(
            model=model,
            prompts=prompts,
            source_layers=source_layers,
            target_layer=target_layer,
            dim_batch=dim_batch,
            max_seq_len=effective_max_seq_len,
            skip_first=skip_first,
            checkpoint_path=checkpoint_path,
            checkpoint_every=checkpoint_every,
            resume=effective_resume,
        )

    if processor is None:
        raise ValueError("processor is required when fitting a subset mode")

    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = _check_layer_indices(
        source_layers, target_layer, n_layers
    )

    logger.info(
        "fit: n_layers=%d d_model=%d, fitting %d source layers "
        "(target=L%d) on %d prompts",
        n_layers,
        d_model,
        len(source_layers),
        target_layer,
        len(prompts),
    )

    # Running state: sum of per-prompt Jacobians, success count, and the list
    # index to resume from. ``next_idx`` is tracked separately from ``n_done``
    # so a too-short prompt that was skipped is not re-processed on resume.
    jacobian_sum: dict[int, torch.Tensor]
    n_done: int
    next_idx: int
    if effective_resume:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path must be specified when resuming.")
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"checkpoint_path {checkpoint_path} does not exist.")
        
        # Load checkpoint
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        
        # Check if parameters match those used in the checkpoint.
        for key, expected in (
            ("source_layers", source_layers),
            ("target_layer", target_layer),
            ("skip_first", skip_first),
        ):
            if key in state and state[key] != expected:
                raise ValueError(
                    f"checkpoint at {checkpoint_path} was fitted with {key}="
                    f"{state[key]!r}, not {expected!r}; pass resume=False to discard it"
                )
        
        # Extract running state from the checkpoint.
        jacobian_sum, n_done, next_idx = (
            state["jacobian_sum"],
            state["n_done"],
            state["next_idx"],
        )
        logger.info(
            "  resuming from checkpoint: %d/%d prompts processed",
            next_idx,
            len(prompts),
        )
    else:
        jacobian_sum = {
            layer: torch.zeros(d_model, d_model, dtype=torch.float32)
            for layer in source_layers
        }
        n_done = 0
        next_idx = 0

    def write_checkpoint() -> None:
        if checkpoint_path is not None:
            _atomic_save(
                {
                    "jacobian_sum": jacobian_sum,
                    "n_done": n_done,
                    "next_idx": next_idx,
                    "source_layers": source_layers,
                    "target_layer": target_layer,
                    "skip_first": skip_first,
                },
                checkpoint_path,
            )

    sqrt_d = math.sqrt(d_model)
    for prompt_idx, prompt in enumerate(prompts):
        if prompt_idx < next_idx:
            continue
        start_time = time.perf_counter()
        try:
            per_prompt_J, seq_len, n_valid = _compute_single_prompt_jacobian(
                model=model,
                prompt=prompt,
                mode=mode,
                source_layers=source_layers,
                target_layer=target_layer,
                dim_batch=dim_batch,
                skip_first=skip_first,
                effective_max_seq_len=effective_max_seq_len,
                processor=processor,
                data_root=data_root,
            )
        except (ValueError, OSError) as exc:
            logger.warning("  skipping prompt %d: %s", prompt_idx, exc)
            next_idx = prompt_idx + 1
            continue

        # Per-prompt diagnostics, max over source layers: the prompt's own
        # Jacobian norm flags heavy-tailed outliers, and the relative shift
        # in the running mean tracks convergence (falls ~1/n once settled).
        prompt_norm = max(per_prompt_J[l].norm().item() for l in source_layers) / sqrt_d
        if n_done > 0:
            mean_rel_change = max(
                (
                    (per_prompt_J[l] - jacobian_sum[l] / n_done).norm()
                    / ((n_done + 1) * (jacobian_sum[l] / n_done).norm())
                ).item()
                for l in source_layers
            )
        else:
            mean_rel_change = float("nan")

        for layer in source_layers:
            jacobian_sum[layer] += per_prompt_J[layer]
        n_done += 1
        next_idx = prompt_idx + 1

        logger.info(
            "  prompt %d/%d  seq_len=%d n_valid=%d  %.0fs  "
            "max||J||/sqrt(d)=%.3f  max_d_mean=%.2e",
            prompt_idx + 1,
            len(prompts),
            seq_len,
            n_valid,
            time.perf_counter() - start_time,
            prompt_norm,
            mean_rel_change,
        )
        if checkpoint_every is not None and next_idx % checkpoint_every == 0:
            write_checkpoint()

    write_checkpoint()
    if n_done == 0:
        raise ValueError("no prompts were long enough to fit on")
    jacobian_mean = {layer: jacobian_sum[layer] / n_done for layer in source_layers}
    logger.info("fit: done, %d prompts", n_done)
    return JacobianLens(jacobians=jacobian_mean, n_prompts=n_done, d_model=d_model)
