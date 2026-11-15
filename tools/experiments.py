import os, math, json, random, statistics
from dataclasses import dataclass, field

import torch
from torch import Tensor, linalg

from audio_jlens.jlens import Jlens
from jacobian_lens.jlens.hooks import ActivationRecorder
from json_parser import Parser
from tools.math_tools import MathTools
from tools.matrix_tools import MatrixTools, LayerSVDMetrics
from model.model import Model, SWAP_DIRECTION_SOURCES

SWAP_POSITION_POLICIES = ("all", "audio", "after_audio", "last")

@dataclass
class passk_cache:
    """Per-prompt pass@k observations accumulated during evaluation."""

    prompt_cnt: int = 0
    best_token_pos: list[int] = field(default_factory=list)
    best_k_values: list[int] = field(default_factory=list)
    best_probability: list[float] = field(default_factory=list)
    best_layers: list[int] = field(default_factory=list)
    layer: int | None = None

#TODO write docstring for this class
@dataclass
class JlensComparison:
    """Holds comparison results between two Jlens instances."""
    cosine_similarity: float
    symmetric_relative_frobenius_dist: float
    frobenius_norm_ratio: float
    operator_norm_diff: float
    largest_sval_value_diff: float
    relative_largest_sval_diff: float
    sval_threshold: float
    left_subspace_rank1: int
    left_subspace_rank2: int
    left_principal_cosines: list[float]
    left_mean_principal_cosine: float
    right_subspace_rank1: int
    right_subspace_rank2: int
    right_principal_cosines: list[float]
    right_mean_principal_cosine: float
    
    
class Experiments:
    @staticmethod
    def ablation():
        return
    
    @staticmethod
    def comp_dataset_choice():
        pass

    @staticmethod
    def swap_token_positions(input_ids: Tensor, tokenizer, policy: str) -> list[int]:
        """Return the positions selected by a policy in SWAP_POSITION_POLICIES.

        "audio" covers <|audio_bos|> through <|audio_eos|>, and "after_audio"
        covers everything after <|audio_eos|>.
        """
        if policy not in SWAP_POSITION_POLICIES:
            raise ValueError(
                f"Unknown position policy {policy!r}; "
                f"expected one of {SWAP_POSITION_POLICIES}"
            )
        ids = input_ids.reshape(-1).tolist()
        if policy == "all":
            return list(range(len(ids)))
        if policy == "last":
            return [len(ids) - 1]

        audio_bos = tokenizer.convert_tokens_to_ids("<|audio_bos|>")
        audio_eos = tokenizer.convert_tokens_to_ids("<|audio_eos|>")
        if ids.count(audio_bos) != 1 or ids.count(audio_eos) != 1:
            raise ValueError(
                "Expected exactly one audio segment, found "
                f"{ids.count(audio_bos)} <|audio_bos|> and "
                f"{ids.count(audio_eos)} <|audio_eos|> tokens."
            )
        audio_start = ids.index(audio_bos)
        audio_end = ids.index(audio_eos)
        if audio_end < audio_start:
            raise ValueError("<|audio_eos|> appears before <|audio_bos|>.")
        if policy == "audio":
            return list(range(audio_start, audio_end + 1))
        positions = list(range(audio_end + 1, len(ids)))
        if not positions:
            raise ValueError("No tokens follow the audio segment.")
        return positions

    @staticmethod
    def swap_layer_bands(
        percent_bands: list[list[float]],
        n_layers: int,
        source_layers: list[int],
    ) -> dict[str, list[int]]:
        """Convert [start, end] depth percentages into lens layers.

        Depth runs from 0 at the first layer to 100 at the final layer, matching
        the paper's reindexing. Both ends of a band are inclusive.
        """
        if n_layers < 2:
            raise ValueError("At least two layers are required to define depth.")
        bands: dict[str, list[int]] = {}
        for band in percent_bands:
            if len(band) != 2 or not 0 <= band[0] <= band[1] <= 100:
                raise ValueError(
                    f"Each band must be [start, end] within 0-100, got {band!r}."
                )
            start, end = band
            layers = [
                layer
                for layer in sorted(source_layers)
                if start <= layer * 100 / (n_layers - 1) <= end
            ]
            if not layers:
                raise ValueError(f"Band {band!r} contains no lens layers.")
            bands[f"L{start:g}-{end:g}"] = layers
        return bands

    @staticmethod
    def swap_single_token(tokenizer, text: str) -> int | None:
        """Return the token ID when text is exactly one token, otherwise None."""
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return token_ids[0] if len(token_ids) == 1 else None

    @staticmethod
    def swap_candidate_tokens(
        tokenizer,
        source_text: str,
        words: list[str],
        count: int,
    ) -> dict[str, int]:
        """Pick the first single-token words, written in the same form as the source.

        A swap only compares like with like when the case and leading whitespace
        of the replacement match the token the model actually chose.
        """
        core = source_text.strip()
        prefix = source_text[: len(source_text) - len(source_text.lstrip())]
        candidates: dict[str, int] = {}
        for word in words:
            if word.lower() == core.lower():
                continue
            form = word.capitalize() if core[:1].isupper() else word.lower()
            token_id = Experiments.swap_single_token(tokenizer, prefix + form)
            if token_id is not None:
                candidates[word.lower()] = token_id
            if len(candidates) == count:
                break
        return candidates

    @staticmethod
    def swap_prompt_line(json_line: str, instruction: str) -> str:
        """Return the JSONL record with its user text replaced by instruction."""
        record = json.loads(json_line)
        record["messages"][0]["content"][1]["text"] = instruction
        return json.dumps(record, ensure_ascii=False)

    @staticmethod
    def swap_verbal_report(
        logger,
        model: Model,
        run_name: str,
        lens_path: str,
        prompts_path: str,
        data_root: str,
        instruction: str,
        candidate_words: list[str],
        layer_bands_percent: list[list[float]],
        candidates_per_prompt: int = 3,
        direction_sources: list[str] | None = None,
        position_policies: list[str] | None = None,
        alpha: float = 1.0,
        max_direction_cosine: float = 0.9,
        max_prompts: int | None = None,
        seed: int = 42,
        clamp_to_clean: bool = True,
    ) -> dict[str, object]:
        """Audio version of the paper's verbal-report swap experiment.

        Asks for a one-word answer, treats the model's first token as the source
        concept, and swaps it for other words of the same category. Each swap is
        scored on the next token. Pairs whose directions are too similar stay in
        ``rows`` but are excluded from ``summary``, because the pseudoinverse is
        unstable for nearly parallel directions.
        """
        direction_sources = direction_sources or list(SWAP_DIRECTION_SOURCES)
        position_policies = position_policies or ["all"]
        for source in direction_sources:
            if source not in SWAP_DIRECTION_SOURCES:
                raise ValueError(f"Unknown direction source {source!r}.")
        for policy in position_policies:
            if policy not in SWAP_POSITION_POLICIES:
                raise ValueError(f"Unknown position policy {policy!r}.")
        if candidates_per_prompt < 1:
            raise ValueError("candidates_per_prompt must be at least 1.")
        if max_prompts is not None and max_prompts < 1:
            raise ValueError("max_prompts must be at least 1 or None.")
        vocabulary = {word.lower() for word in candidate_words}

        json_lines = Parser.load_jsonl_lines(prompts_path)
        random.Random(seed).shuffle(json_lines)
        if max_prompts is not None:
            json_lines = json_lines[:max_prompts]

        # Swaps run on the same layers for every source so the controls stay comparable.
        model.load_Jlens(lens_path=lens_path)
        instance = model.load_model()
        tokenizer = instance.tokenizer
        bands = Experiments.swap_layer_bands(
            layer_bands_percent, instance.n_layers, model.lens.lens.source_layers
        )
        instance.load_data(data_root=data_root)

        rows: list[dict[str, object]] = []
        skipped: list[dict[str, object]] = []
        for prompt_index, raw_line in enumerate(json_lines):
            json_line = Experiments.swap_prompt_line(raw_line, instruction)
            sample = Parser.parse_json_line(json_line)
            input_ids = instance.encode(json_line)
            with torch.no_grad():
                baseline_residual = instance.forward(input_ids)
                answer_logits = instance.unembed(baseline_residual[0][[-1]])[0]
            source_token = int(answer_logits.argmax().item())
            source_text = tokenizer.decode([source_token])

            if source_text.strip().lower() not in vocabulary:
                skipped.append({
                    "sample_id": sample.sample_id,
                    "reason": "answer_not_in_candidates",
                    "answer": source_text,
                })
                continue
            candidates = Experiments.swap_candidate_tokens(
                tokenizer, source_text, candidate_words, candidates_per_prompt
            )
            if not candidates:
                skipped.append({
                    "sample_id": sample.sample_id,
                    "reason": "no_single_token_candidates",
                    "answer": source_text,
                })
                continue

            positions_by_policy = {
                policy: Experiments.swap_token_positions(input_ids, tokenizer, policy)
                for policy in position_policies
            }
            transcript = (sample.transcript or "").strip().lower()
            for word, target_token in candidates.items():
                for band_name, layers in bands.items():
                    for direction_source in direction_sources:
                        for policy, positions in positions_by_policy.items():
                            result = model.run_swap_inference(
                                jsonl_line=json_line,
                                data_root=data_root,
                                source_token=source_token,
                                target_token=target_token,
                                layers=layers,
                                token_pos=positions,
                                alpha=alpha,
                                direction_source=direction_source,
                                random_seed=seed + len(rows),
                                clamp_to_clean=clamp_to_clean,
                            )
                            rows.append(
                                Experiments._swap_row(
                                    tokenizer,
                                    result,
                                    sample_id=sample.sample_id,
                                    label_match=transcript == source_text.strip().lower(),
                                    source_token=source_token,
                                    target_token=target_token,
                                    target_word=word,
                                    band_name=band_name,
                                    direction_source=direction_source,
                                    position_policy=policy,
                                    max_direction_cosine=max_direction_cosine,
                                )
                            )
            logger.info(
                "%s swap progress: %d/%d prompts, %d swaps",
                run_name,
                prompt_index + 1,
                len(json_lines),
                len(rows),
            )

        return {
            "prompt_count": len(json_lines),
            "clamp_to_clean": clamp_to_clean,
            "alpha": alpha,
            "skipped": skipped,
            "bands": bands,
            "summary": Experiments._summarize_swap_rows(rows),
            "rows": rows,
        }

    @staticmethod
    def _swap_row(
        tokenizer,
        result,
        *,
        sample_id: str,
        label_match: bool,
        source_token: int,
        target_token: int,
        target_word: str,
        band_name: str,
        direction_source: str,
        position_policy: str,
        max_direction_cosine: float,
    ) -> dict[str, object]:
        """Score one swap from its next-token logits."""
        baseline = result.baseline_logits[0]
        swapped = result.swapped_logits[0]

        def rank(logits: Tensor, token: int) -> int:
            return int((logits > logits[token]).sum().item()) + 1

        baseline_gap = float((baseline[target_token] - baseline[source_token]).item())
        swapped_gap = float((swapped[target_token] - swapped[source_token]).item())
        direction_cosine = max(abs(v) for v in result.direction_cosines.values())
        return {
            "sample_id": sample_id,
            "label_match": label_match,
            "source_text": tokenizer.decode([source_token]),
            "target_word": target_word,
            "target_text": tokenizer.decode([target_token]),
            "band": band_name,
            "direction_source": direction_source,
            "position_policy": position_policy,
            "swapped_top_text": tokenizer.decode([int(swapped.argmax().item())]),
            "target_rank": rank(swapped, target_token),
            "source_rank": rank(swapped, source_token),
            "logit_gap_shift": swapped_gap - baseline_gap,
            "direction_cosine": direction_cosine,
            "excluded_high_cosine": direction_cosine > max_direction_cosine,
        }

    @staticmethod
    def _summarize_swap_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        """Aggregate swaps by direction source, layer band, and position policy."""
        groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
        for row in rows:
            key = (row["direction_source"], row["band"], row["position_policy"])
            groups.setdefault(key, []).append(row)

        summary = []
        for (direction_source, band, policy), group in sorted(groups.items()):
            kept = [row for row in group if not row["excluded_high_cosine"]]
            count = len(kept)
            summary.append({
                "direction_source": direction_source,
                "band": band,
                "position_policy": policy,
                "swap_count": count,
                "excluded_high_cosine": len(group) - count,
                "target_top1_rate": (
                    sum(row["target_rank"] == 1 for row in kept) / count if count else None
                ),
                "source_displaced_rate": (
                    sum(row["source_rank"] > 1 for row in kept) / count if count else None
                ),
                "median_target_rank": (
                    statistics.median(row["target_rank"] for row in kept) if count else None
                ),
                "mean_logit_gap_shift": (
                    statistics.fmean(row["logit_gap_shift"] for row in kept) if count else None
                ),
            })
        return summary


    def _passk_target_text(sample, target: str, logger) -> str | None:
        if target == "transcript":
            target_text = sample.transcript
            if target_text is None:
                logger.warning(
                    "Sample %s has no transcript to evaluate", sample.sample_id
                )
                return None
            target_text = target_text.strip()
            if target_text.startswith("(") and ")" in target_text:
                target_text = target_text.split(")", 1)[1].strip()
            return target_text

        if not target.startswith("label.") or not target[len("label."):].strip():
            raise ValueError(
                "Pass@k target must be 'transcript' or 'label.<field>'."
            )
        label_key = target[len("label."):]
        labels = sample.label
        if not isinstance(labels, dict) or label_key not in labels:
            raise ValueError(
                f"Sample {sample.sample_id} has no label field {label_key!r}."
            )
        target_value = labels[label_key]
        if not isinstance(target_value, str):
            raise ValueError(
                f"Pass@k label {label_key!r} for sample {sample.sample_id} "
                "must be a scalar string."
            )
        target_text = target_value.strip()
        if not target_text:
            raise ValueError(
                f"Pass@k label {label_key!r} for sample {sample.sample_id} "
                "must not be empty."
            )
        return target_text
    
    @staticmethod
    def draw_Jlens_cka_heatmap(
        logger,
        run_name: str,
        heatmap_path: str,
        lens1: Jlens,
        lens2: Jlens | None = None,
        device: str | torch.device | None = None,
        readout_weights: tuple[Tensor, Tensor | None] | None = None,
    ) -> torch.Tensor:
        """Draw a heatmap using Jacobian or output J-vector CKA between layers.
        The returned tensor is indexed as ``[lens1_layer, lens2_layer]``.
        
        The y axis represents the first lens, while the x axis does for the second.

        Args:
            lens1: First JLens instance, with its checkpoint already loaded.
            lens2: If not specified, compare lens1 to itself.
            device: Device for CKA calculation. Defaults to CUDA when available.
        """
        os.makedirs(heatmap_path, exist_ok=True)
        
        lens2 = lens1 if lens2 is None else lens2

        if lens1.lens is None or lens2.lens is None:
            raise ValueError("Both JLens instances must contain a loaded lens.")
        layers1 = lens1.lens.source_layers
        layers2 = lens2.lens.source_layers

        if readout_weights is None:
            heatmap = MathTools.cka_similarity(
                l1=lens1.lens.jacobians,
                l2=lens2.lens.jacobians,
                d1=layers1,
                d2=layers2,
                device=device,
            )
        else:
            _, heatmap = MathTools.j_vector_similarities(
                l1=lens1.lens.jacobians,
                l2=lens2.lens.jacobians,
                d1=layers1,
                d2=layers2,
                output_weights=readout_weights[0],
                norm_weight=readout_weights[1],
                device=device,
            )
        logger.info("Computed CKA heatmap on %s", device or ("cuda" if torch.cuda.is_available() else "cpu"))
        output_name = run_name or "heatmap"
        if readout_weights is not None:
            output_name += "_j_vectors"
            if readout_weights[1] is None:
                output_name += "_no_norm_scaling"
        MathTools.draw_heatmap(
            data=heatmap,
            png_path=os.path.join(heatmap_path, output_name),
            x_labels=layers2,
            y_labels=layers1,
            title=(
                "JLens J-vector CKA similarity"
                if readout_weights is not None
                else "JLens Jacobian cka similarity"
            ),
        )
        logger.info("Saved JLens heatmap to %s", heatmap_path)
        return heatmap

    @staticmethod
    def draw_Jlens_cosine_heatmap(
        logger,
        run_name: str,
        heatmap_path: str,
        lens1: Jlens,
        lens2: Jlens | None = None,
        device: str | torch.device | None = None,
        readout_weights: tuple[Tensor, Tensor | None] | None = None,
    ) -> torch.Tensor:
        """Draw pairwise cosine similarities between Jacobians or their J-vectors."""
        os.makedirs(heatmap_path, exist_ok=True)
        lens2 = lens1 if lens2 is None else lens2
        compute_device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if lens1.lens is None or lens2.lens is None:
            raise ValueError("Both JLens instances must contain a loaded lens.")
        layers1 = lens1.lens.source_layers
        layers2 = lens2.lens.source_layers
        if readout_weights is not None:
            heatmap, _ = MathTools.j_vector_similarities(
                l1=lens1.lens.jacobians,
                l2=lens2.lens.jacobians,
                d1=layers1,
                d2=layers2,
                output_weights=readout_weights[0],
                norm_weight=readout_weights[1],
                device=compute_device,
            )
        else:
            jacobians1 = {
                layer: lens1.lens.jacobians[layer].to(
                    device=compute_device, dtype=torch.float32
                )
                for layer in layers1
            }
            if lens1 is lens2:
                jacobians2 = jacobians1
            else:
                jacobians2 = {
                    layer: lens2.lens.jacobians[layer].to(
                        device=compute_device, dtype=torch.float32
                    )
                    for layer in layers2
                }
            heatmap = MathTools.sim_for_sets(
                l1=jacobians1,
                l2=jacobians2,
                d1=layers1,
                d2=layers2,
                func=MathTools.cos_similarity,
            )
        logger.info("Computed cosine heatmap on %s", compute_device)
        output_name = run_name or "heatmap"
        if readout_weights is not None:
            output_name += "_j_vectors"
            if readout_weights[1] is None:
                output_name += "_no_norm_scaling"
        MathTools.draw_heatmap(
            data=heatmap,
            png_path=os.path.join(heatmap_path, f"{output_name}_cosine"),
            x_labels=layers2,
            y_labels=layers1,
            title=(
                "JLens J-vector cosine similarity"
                if readout_weights is not None
                else "JLens Jacobian cosine similarity"
            ),
        )
        logger.info("Saved JLens cosine heatmap to %s", heatmap_path)
        return heatmap

    @staticmethod
    def draw_Jlens_svd_heatmaps(
        logger,
        run_name: str,
        output_dir: str,
        lens: Jlens,
        top_k: int,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[str]]:
        """Save per-layer sval spectra and adjacent-layer subspace heatmaps."""

        output_dir = os.path.join(output_dir, "SVD_analysis")
        if lens.lens is None:
            raise ValueError("JLens instance must contain a loaded lens.")

        spectra, similarities, layers, layer_pairs = MatrixTools.draw_svd_heatmaps(
            jacobians=lens.lens.jacobians,
            output_dir=output_dir,
            name=run_name or "jlens",
            top_k=top_k,
            device=device,
        )
        logger.info("Saved JLens SVD heatmaps to %s", output_dir)
        return spectra, similarities, layers, layer_pairs

    @staticmethod
    def compare_jacobians(
        jacobian1: Tensor,
        jacobian2: Tensor,
        *,
        sval_threshold: float = 0.5,
        device: torch.device | str = "cpu",
    ) -> JlensComparison:
        #TODO This is directly from Copilot. Consult 學長.
        """
        Compare matched Jacobians by similarity, dist, and sval subspaces.
        Jacobians must have the same shape.
        For Jacobians from different models, some comparison metrics may not be meaningful.
        If so, call the lower-level comparison functions.
        """
        # Validate inputs
        MatrixTools._validate_sval_threshold(sval_threshold)
        matrix1 = jacobian1.detach().to(device=device, dtype=torch.float32)
        matrix2 = jacobian2.detach().to(device=device, dtype=torch.float32)
        if matrix1.ndim != 2 or matrix2.ndim != 2:
            raise ValueError("Comparison matrices must be two-dimensional.")
        if not torch.isfinite(matrix1).all() or not torch.isfinite(matrix2).all():
            raise ValueError("Comparison matrices must contain only finite values.")
        if matrix1.shape != matrix2.shape:
            raise ValueError(
                "Jacobians must have matching shapes, got "
                f"{tuple(matrix1.shape)} and {tuple(matrix2.shape)}."
            )
        # Compute SVD
        svd_metrics1: LayerSVDMetrics = MatrixTools.SVD_decomp(matrix1)
        svd_metrics2: LayerSVDMetrics = MatrixTools.SVD_decomp(matrix2)
        if svd_metrics1.sval[0] == 0 or svd_metrics2.sval[0] == 0:
            raise ValueError(
                "Jacobian comparison metrics are undefined for an all-zero matrix."
            )
        # Compute frobenius dist
        matrix1_norm = linalg.vector_norm(matrix1)
        matrix2_norm = linalg.vector_norm(matrix2)
        mean_frobenius_norm = (matrix1_norm + matrix2_norm) / 2
        delta = matrix1 - matrix2
        relative_frobenius_dist = (
            linalg.vector_norm(delta) / mean_frobenius_norm
        )
        # Largest singular value of the difference
        operator_norm_diff = linalg.matrix_norm(delta, ord=2)
        # Normalized difference of largest singular
        largest_sval_diff = abs(svd_metrics1.sval[0] - svd_metrics2.sval[0])
        mean_operator_norm = (svd_metrics1.sval[0] + svd_metrics2.sval[0]) / 2
        relative_largest_sval_diff = largest_sval_diff / mean_operator_norm
        # Masks for strong singular values
        strong_mask1 = svd_metrics1.sval >= sval_threshold * svd_metrics1.sval[0]
        strong_mask2 = svd_metrics2.sval >= sval_threshold * svd_metrics2.sval[0]
        # Extract only the impactful left and right subspaces
        left_basis1 = svd_metrics1.left_svec[:, strong_mask1]
        left_basis2 = svd_metrics2.left_svec[:, strong_mask2]
        right_basis1 = svd_metrics1.right_svec[:, strong_mask1]
        right_basis2 = svd_metrics2.right_svec[:, strong_mask2]
        # Principal cosines of the two subspaces
        # Since SVD decomposition works as M -> U S Vh, the below effectively
        #   does MN^T -> U S Vh, that is, (U-1 M)(Vh-1^T N)^T -> S
        #   This maximizes the alignment between the subspace vectors after a linear transformation.
        # This is much better than directly comparing them. Copilot is the absolute goat.
        # See https://web.ma.utexas.edu/users/vandyke/notes/deep_learning_presentation/presentation.pdf
        left_principal_cosines = linalg.svdvals(
            left_basis1.T @ left_basis2
        ).clamp_(0, 1)
        right_principal_cosines = linalg.svdvals(
            right_basis1.T @ right_basis2
        ).clamp_(0, 1)

        def mean_or_zero(values: Tensor) -> float:
            return float(values.mean().item()) if values.numel() else 0.0

        return JlensComparison(
            cosine_similarity=MathTools.cos_similarity(matrix1, matrix2),
            symmetric_relative_frobenius_dist=float(
                relative_frobenius_dist.item()
            ),
            frobenius_norm_ratio=float((matrix2_norm / matrix1_norm).item()),
            operator_norm_diff=float(operator_norm_diff.item()),
            largest_sval_value_diff=float(
                largest_sval_diff.item()
            ),
            relative_largest_sval_diff=float(
                relative_largest_sval_diff.item()
            ),
            sval_threshold=sval_threshold,
            left_subspace_rank1=int(strong_mask1.sum().item()),
            left_subspace_rank2=int(strong_mask2.sum().item()),
            left_principal_cosines=left_principal_cosines.cpu().tolist(),
            left_mean_principal_cosine=mean_or_zero(left_principal_cosines),
            right_subspace_rank1=int(strong_mask1.sum().item()),
            right_subspace_rank2=int(strong_mask2.sum().item()),
            right_principal_cosines=right_principal_cosines.cpu().tolist(),
            right_mean_principal_cosine=mean_or_zero(right_principal_cosines),
        )

    @staticmethod
    def info_Jlens(
        logger,
        jacobians: dict[int, Tensor],
        layers: list[int],
        svd_top_k: int = 8,
        device: str | torch.device | None = None,
        svd_device: str | torch.device = "cpu",
        readout_weights: tuple[Tensor, Tensor | None] | None = None,
        compute_svd: bool = True,
    ) -> dict[str, object]:
        """Compute similarity analyses and optionally Jacobian SVD.

        SVD result keys are omitted when ``compute_svd`` is false.
        """
        if not layers:
            raise ValueError("Cannot analyze a JLens with no source layers.")
        if len(set(layers)) != len(layers) or any(
            layer not in jacobians for layer in layers
        ):
            raise ValueError("Source layers must be unique keys in jacobians.")

        compute_device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if readout_weights is None:
            cosine_jacobians = {
                layer: jacobians[layer].to(
                    device=compute_device, dtype=torch.float32
                )
                for layer in layers
            }
            cosine_heatmap = MathTools.sim_for_sets(
                l1=cosine_jacobians,
                l2=cosine_jacobians,
                d1=layers,
                d2=layers,
                func=MathTools.cos_similarity,
            )
            cka_heatmap = MathTools.cka_similarity(
                l1=jacobians,
                l2=jacobians,
                d1=layers,
                d2=layers,
                device=device,
            )
        else:
            cosine_heatmap, cka_heatmap = MathTools.j_vector_similarities(
                l1=jacobians,
                l2=jacobians,
                d1=layers,
                d2=layers,
                output_weights=readout_weights[0],
                norm_weight=readout_weights[1],
                device=compute_device,
            )
        logger.info(
            "Computed single-lens %s cosine and CKA analysis for layers %s",
            "J-vector" if readout_weights is not None else "Jacobian",
            layers,
        )
        result: dict[str, object] = {
            "layers": layers,
            "cosine_heatmap": cosine_heatmap,
            "cka_heatmap": cka_heatmap,
        }
        if compute_svd:
            spectra, adjacent_cosines, svd_layers, layer_pairs = (
                MatrixTools.analyze_layer_svd(
                    jacobians,
                    top_k=svd_top_k,
                    device=svd_device,
                )
            )
            result.update({
                "svd_spectra": spectra,
                "svd_adjacent_cosines": adjacent_cosines,
                "svd_layers": svd_layers,
                "svd_layer_pairs": layer_pairs,
            })
            logger.info("Computed Jacobian SVD analysis for layers %s", layers)
        return result

    @staticmethod
    def passk_Jlens(
        logger,
        model: Model,
        run_name: str,
        prompts_path: str,
        data_root: str,
        top_k: int,
        k_values: list[int],
        do_batch: bool = False,
        lens: list[Jlens] | None = None,
        layer: int | None = None,
        target: str = "transcript",
    ) -> dict[str, object]:
        """Run JLens and LogitLens pass@k evaluation for all prompts.

        In batch mode, evaluate each loaded JLens against the same prompts and
        return reports keyed by lens index.
        Applies both lenses once per prompt, then reports the first transcript
        token's best rank across positions/layers and independently at each layer.
        """
        if do_batch:
            lenses = lens or []
        else:
            if lens is not None:
                raise ValueError("Pass lens only when do_batch=True.")
            if not isinstance(model.lens, Jlens):
                raise ValueError("Load or bind a JLens on the model before evaluation.")
            lenses = [model.lens]

        result = Experiments._passk_Jlens_batch(
            logger=logger,
            model=model,
            run_name=run_name,
            prompts_path=prompts_path,
            data_root=data_root,
            top_k=top_k,
            k_values=k_values,
            lenses=lenses,
            layer=layer,
            target=target,
        )
        if do_batch:
            return result

        reports = result.get("lenses")
        logit_report = result.get("logit_lens")
        rows_by_lens = result.get("prediction_rows")
        if not isinstance(reports, dict) or not isinstance(rows_by_lens, dict):
            raise RuntimeError("Pass@k batch evaluation returned invalid results.")
        lens_report = reports.get("0")
        prediction_rows = rows_by_lens.get("0")
        if not isinstance(lens_report, dict) or not isinstance(logit_report, dict):
            raise RuntimeError("Pass@k batch evaluation returned invalid reports.")
        if not isinstance(prediction_rows, list):
            raise RuntimeError("Pass@k batch evaluation returned invalid prediction rows.")
        lens_by_layer = lens_report.pop("by_layer")
        logit_by_layer = logit_report.pop("by_layer")
        return {
            "k_values": result["k_values"],
            "auc_x_axis": result["auc_x_axis"],
            "layer": result["layer"],
            "target": result["target"],
            "position_policy": result["position_policy"],
            "layer_policy": result["layer_policy"],
            **lens_report,
            "logit_lens": logit_report,
            "by_layer": {
                "jlens": lens_by_layer,
                "logit_lens": logit_by_layer,
            },
            "prediction_rows": prediction_rows,
        }

    @staticmethod
    def _passk_Jlens_batch(
        *,
        logger,
        model: Model,
        run_name: str,
        prompts_path: str,
        data_root: str,
        top_k: int,
        k_values: list[int],
        lenses: list[Jlens],
        layer: int | None,
        target: str,
    ) -> dict[str, object]:
        """Evaluate several lenses using one model forward pass per prompt."""
        if not lenses:
            raise ValueError("Batch evaluation requires at least one loaded JLens.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not k_values or any(not isinstance(k, int) or k <= 0 for k in k_values):
            raise ValueError("k_values must contain positive integers")
        if target != "transcript" and not (
            target.startswith("label.") and target[len("label."):].strip()
        ):
            raise ValueError(
                "Pass@k target must be 'transcript' or 'label.<field>'."
            )
        jacobian_lenses = []
        for index, item in enumerate(lenses):
            if not isinstance(item, Jlens) or item.lens is None:
                raise ValueError(
                    f"Batch item {index} must be a loaded JLens instance."
                )
            jacobian_lenses.append(item.lens)

        sorted_k_values = sorted(set(k_values))
        json_lines = Parser.load_jsonl_lines(prompts_path)
        adapter = model.load_model()
        adapter.load_data(data_root)
        for index, jacobian_lens in enumerate(jacobian_lenses):
            if jacobian_lens.d_model != adapter.d_model:
                raise ValueError(
                    f"JLens at index {index} has d_model={jacobian_lens.d_model}, "
                    f"but the model has d_model={adapter.d_model}."
                )
            if not jacobian_lens.source_layers:
                raise ValueError(f"JLens at index {index} has no source layers.")
            if layer is not None and layer not in jacobian_lens.source_layers:
                raise ValueError(
                    f"Requested layer {layer} is not present in JLens at index {index}."
                )
            invalid_layers = sorted(
                set(jacobian_lens.source_layers)
                - set(range(adapter.n_layers - 1))
            )
            if invalid_layers:
                raise ValueError(
                    f"JLens at index {index} has invalid source layers "
                    f"{invalid_layers} for this model."
                )

        lens_layers = [
            jacobian_lens.source_layers if layer is None else [layer]
            for jacobian_lens in jacobian_lenses
        ]
        union_layers = sorted({item for layers in lens_layers for item in layers})
        final_layer = adapter.n_layers - 1
        logit_layers = list(range(adapter.n_layers)) if layer is None else [layer]
        logit_result_layers = (
            union_layers if layer is None else [layer]
        )
        lens_caches = [passk_cache() for _ in lenses]
        lens_summaries: list[
            dict[str, float | int | dict[int, float] | list[float]]
        ] = [{} for _ in lenses]
        lens_layer_caches = [
            {layer_idx: passk_cache() for layer_idx in current_layers}
            for current_layers in lens_layers
        ]
        lens_layer_summaries: list[
            dict[int, dict[str, float | int | dict[int, float] | list[float]]]
        ] = [{} for _ in lenses]
        logit_cache = passk_cache()
        logit_summary: dict[
            str, float | int | dict[int, float] | list[float]
        ] = {}
        logit_layer_caches = {
            layer_idx: passk_cache() for layer_idx in logit_result_layers
        }
        logit_layer_summaries: dict[
            int, dict[str, float | int | dict[int, float] | list[float]]
        ] = {}
        rows_by_lens: dict[str, list[dict[str, object]]] = {
            str(index): [] for index in range(len(lenses))
        }

        for json_line in json_lines:
            sample = Parser.parse_json_line(json_line)
            target_text = Experiments._passk_target_text(
                sample,
                target,
                logger,
            )
            if target_text is None:
                continue
            processor = model.processor
            if processor is None:
                raise RuntimeError("The configured model adapter has no processor.")
            target_ids = processor.tokenizer(
                target_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"]
            if target_ids.numel() == 0:
                raise ValueError(
                    f"Transcript for sample {sample.sample_id} produced no tokens"
                )
            ans_token = int(target_ids[0, 0].item())
            input_ids = adapter.encode(json_line, max_length=300)

            with torch.no_grad(), ActivationRecorder(
                adapter.layers, at=[*union_layers, final_layer]
            ) as recorder:
                adapter.forward(input_ids)
            activations = {
                layer_idx: recorder.activations[layer_idx].detach()
                for layer_idx in {*union_layers, final_layer}
            }

            def residual_at(layer_idx: int) -> Tensor:
                return activations[layer_idx][0].float()

            model_logits = adapter.unembed(residual_at(final_layer)).float().cpu()
            logit_data = (
                {
                    layer_idx: adapter.unembed(
                        residual_at(layer_idx)
                    ).float().cpu()
                    for layer_idx in logit_layers
                },
                model_logits,
                input_ids,
            )
            model.lens_cache.put("logit_lens", logit_data)
            logit_summary = Experiments.eval_passk(
                logger=logger,
                run_name=f"{run_name} logit_lens",
                lens_data=logit_data,
                passk_cache=logit_cache,
                ans_token=ans_token,
                k_values=sorted_k_values,
                layer=layer,
            )
            for layer_idx in logit_result_layers:
                logit_layer_summaries[layer_idx] = Experiments.eval_passk(
                    logger=logger,
                    run_name=f"{run_name} logit_lens layer {layer_idx}",
                    lens_data=logit_data,
                    passk_cache=logit_layer_caches[layer_idx],
                    ans_token=ans_token,
                    k_values=sorted_k_values,
                    layer=layer_idx,
                )

            for index, jacobian_lens in enumerate(jacobian_lenses):
                lens_data = (
                    {
                        layer_idx: adapter.unembed(
                            jacobian_lens.transport(
                                residual_at(layer_idx), layer_idx
                            )
                        ).float().cpu()
                        for layer_idx in lens_layers[index]
                    },
                    model_logits,
                    input_ids,
                )
                model.lens_cache.put("jlens", lens_data)
                rows_by_lens[str(index)].extend(
                    model.lens_cache.to_rows(top_k=top_k)
                )
                lens_summaries[index] = Experiments.eval_passk(
                    logger=logger,
                    run_name=f"{run_name} lens {index}",
                    lens_data=lens_data,
                    passk_cache=lens_caches[index],
                    ans_token=ans_token,
                    k_values=sorted_k_values,
                    layer=layer,
                )
                for layer_idx in lens_layers[index]:
                    lens_layer_summaries[index][layer_idx] = (
                        Experiments.eval_passk(
                            logger=logger,
                            run_name=f"{run_name} lens {index} layer {layer_idx}",
                            lens_data=lens_data,
                            passk_cache=lens_layer_caches[index][layer_idx],
                            ans_token=ans_token,
                            k_values=sorted_k_values,
                            layer=layer_idx,
                        )
                    )

        if not json_lines or not all(lens_summaries):
            raise RuntimeError("Pass@k evaluation did not process any prompts")

        def report(
            cache: passk_cache,
            summary: dict[str, float | int | dict[int, float] | list[float]],
        ) -> dict[str, object]:
            return {
                **summary,
                "best_answer_ranks": cache.best_k_values,
                "answer_probabilities": cache.best_probability,
                "target_positions": cache.best_token_pos,
                "best_layers": cache.best_layers,
            }

        return {
            "k_values": sorted_k_values,
            "auc_x_axis": "log(k)",
            "layer": layer,
            "target": target,
            "position_policy": "best rank over prompt positions",
            "layer_policy": (
                "selected layer only"
                if layer is not None
                else "all layers pooled for overall metrics; each layer scored independently in by_layer"
            ),
            "lenses": {
                str(index): {
                    **report(lens_caches[index], lens_summaries[index]),
                    "by_layer": {
                        str(layer_idx): report(
                            lens_layer_caches[index][layer_idx],
                            lens_layer_summaries[index][layer_idx],
                        )
                        for layer_idx in lens_layers[index]
                    },
                }
                for index in range(len(lenses))
            },
            "logit_lens": {
                **report(logit_cache, logit_summary),
                "by_layer": {
                    str(layer_idx): report(
                        logit_layer_caches[layer_idx],
                        logit_layer_summaries[layer_idx],
                    )
                    for layer_idx in logit_result_layers
                },
            },
            "prediction_rows": rows_by_lens,
        }

    @staticmethod
    def eval_passk(
        logger,
        run_name: str,
        lens_data: tuple[dict[int, Tensor], Tensor, Tensor],
        passk_cache: passk_cache,
        ans_token: int,
        k_values: list[int],
        layer: int | None = None,
    ) -> dict[str, float | int | dict[int, float] | list[float]]:
        """Accumulate one example and summarize pass@k over processed examples.
        Returns:
            Current aggregate prompt pass rates and normalized trapezoidal AUC
            integrated over log(k). Each prompt contributes its lowest answer-token
            rank across the selected layers and all prompt positions.
        """
        if not k_values or any(not isinstance(k, int) or k <= 0 for k in k_values):
            raise ValueError("k_values must contain positive integers")
        sorted_k_values = sorted(set(k_values))
        # Check if settings match
        if passk_cache.layer is not None and passk_cache.layer != layer:
            raise ValueError("Cannot mix layer settings in one passk_cache")
        passk_cache.layer = layer

        logits_by_layer, _, input_ids = lens_data
        if not logits_by_layer:
            raise ValueError("Cached JLens output has no layer logits")
        # If layer is not specified, consider all available layers
        #     pass@k then takes the best result over all layers
        if layer is not None:
            if layer not in logits_by_layer:
                raise ValueError(
                    f"Layer {layer} is not present in cached JLens predictions; "
                    f"available layers: {sorted(logits_by_layer)}"
                )
            selected_layers = [layer]
        else:
            selected_layers = sorted(logits_by_layer)

        prompt_length = int(input_ids.reshape(-1).numel())
        if prompt_length == 0:
            raise ValueError("Cached input_ids are empty")

        best_rank: int | None = None
        best_probability: float | None = None
        best_position: int | None = None
        best_layer: int | None = None
        # Iterate over each selected layer
        for layer_idx in selected_layers:
            layer_logits_tensor = logits_by_layer[layer_idx]
            if layer_logits_tensor.ndim != 2:
                raise ValueError(
                    f"Layer {layer_idx} logits must have shape [positions, vocab], "
                    f"got {tuple(layer_logits_tensor.shape)}"
                )
            if prompt_length > layer_logits_tensor.shape[0]:
                raise ValueError(
                    f"Prompt length {prompt_length} exceeds available positions in "
                    f"layer {layer_idx} logits"
                )
            if ans_token < 0 or ans_token >= layer_logits_tensor.shape[-1]:
                raise ValueError(
                    f"Answer token ID {ans_token} is outside the logits vocabulary"
                )
            # This counts how many logits are greater than the answer token's logits
            # instead of sorting the whole vocabulary
            position_logits = layer_logits_tensor[:prompt_length].float()
            answer_logits = position_logits[:, ans_token].unsqueeze(1)
            position_ranks = (position_logits > answer_logits).sum(dim=1) + 1
            # Determine the best rank and position for this layer
            # Yes, currently we are not considering the effect of token positions.
            # Future implementations can seperate the effect of different modalities
            layer_best_rank, layer_best_position = position_ranks.min(dim=0)
            rank = int(layer_best_rank.item())
            # Update the best rank
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_position = int(layer_best_position.item())
                best_layer = layer_idx
                best_probability = float(
                    torch.softmax(position_logits[best_position], dim=-1)[
                        ans_token
                    ].item()
                )

        if (
            best_rank is None
            or best_probability is None
            or best_position is None
            or best_layer is None
        ):
            raise RuntimeError("Could not determine answer-token rank")

        passk_cache.prompt_cnt += 1
        passk_cache.best_token_pos.append(best_position)
        passk_cache.best_k_values.append(best_rank)
        passk_cache.best_probability.append(best_probability)
        passk_cache.best_layers.append(best_layer)
        # Compute pass@k rates for the current prompt
        pass_rates = {
            k: sum(rank <= k for rank in passk_cache.best_k_values)
            / passk_cache.prompt_cnt
            for k in sorted_k_values
        }
        log_k_values = [math.log(k) for k in sorted_k_values]
        # Compute the Area Under the Curve (AUC) using the trapezoidal rule on log(k) values
        if len(sorted_k_values) == 1:
            normalized_auc = next(iter(pass_rates.values()))
        else:
            log_span = log_k_values[-1] - log_k_values[0]
            area = sum(
                (pass_rates[left] + pass_rates[right])
                * (log_right - log_left)
                / 2
                for left, right, log_left, log_right in zip(
                    sorted_k_values[:-1],
                    sorted_k_values[1:],
                    log_k_values[:-1],
                    log_k_values[1:],
                    strict=True,
                )
            )
            normalized_auc = area / log_span

        logger.info(
            "%s pass@k progress: %d prompts, normalized log(k) AUC=%.6f",
            run_name,
            passk_cache.prompt_cnt,
            normalized_auc,
        )
        return {
            "prompt_count": passk_cache.prompt_cnt,
            "normalized_auc": normalized_auc,
            "pass_rates": pass_rates,
            "log_k_values": log_k_values,
        }


        