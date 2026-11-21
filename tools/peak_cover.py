"""Peakn@k and Cover@k: audio versions of pass@k.

Copied from ``Experiments.eval_passk`` (tools/experiments.py) and changed so a
prompt is described by two numbers instead of one best rank:

* Peakn@k (strength): mean rank of the n best positions; the prompt passes
  when that mean is <= k. Peak1@k over all positions is the original pass@k.
* Cover@k (duration): fraction of positions whose rank is <= k, averaged
  over prompts.

When the answer occupies a single position (as in the text-only setting of
the original paper), both reduce to the paper's pass@k.

Other changes from ``eval_passk``:
* ``positions`` restricts scoring to chosen positions (e.g. audio tokens).
* ``ans_tokens`` accepts several token ids (surface forms such as " angry",
  "Angry"); each position uses the best-ranked form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class peak_cover_cache:
    """Per-prompt observations accumulated during evaluation."""

    prompt_cnt: int = 0
    peak_ranks: list[float] = field(default_factory=list)       # mean rank of top-n positions
    peak_layers: list[int] = field(default_factory=list)
    peak_positions: list[list[int]] = field(default_factory=list)
    cover_fractions: list[list[float]] = field(default_factory=list)  # per prompt, per k
    cover_layers: list[int] = field(default_factory=list)
    layer: int | None = None
    peak_n: int | None = None
    k_values: list[int] | None = None


def _normalized_log_auc(rates: dict[int, float], sorted_k_values: list[int]) -> float:
    """Trapezoidal AUC over log(k), normalized to [0, 1] (same as eval_passk)."""
    if len(sorted_k_values) == 1:
        return next(iter(rates.values()))
    log_k = [math.log(k) for k in sorted_k_values]
    area = sum(
        (rates[left] + rates[right]) * (log_right - log_left) / 2
        for left, right, log_left, log_right in zip(
            sorted_k_values[:-1], sorted_k_values[1:], log_k[:-1], log_k[1:], strict=True
        )
    )
    return area / (log_k[-1] - log_k[0])


def _check_k_values(k_values: list[int]) -> list[int]:
    if not k_values or any(not isinstance(k, int) or k <= 0 for k in k_values):
        raise ValueError("k_values must contain positive integers")
    return sorted(set(k_values))


def answer_ranks(position_logits: Tensor, token_ids: list[int]) -> Tensor:
    """Rank of each token at each position, shape [positions, len(token_ids)].

    rank = 1 + number of vocabulary logits strictly above the token's logit.
    """
    if any(t < 0 or t >= position_logits.shape[-1] for t in token_ids):
        raise ValueError(f"Answer token IDs {token_ids} are outside the logits vocabulary")
    return torch.stack(
        [(position_logits > position_logits[:, t : t + 1]).sum(dim=1) + 1 for t in token_ids],
        dim=1,
    )


def accumulate_ranks(
    cache: peak_cover_cache,
    ranks_by_layer: dict[int, Tensor],
    scored: list[int],
    k_values: list[int],
    peak_n: int = 5,
    layer: int | None = None,
) -> None:
    """Add one prompt from precomputed answer ranks (one value per scored position).

    ``ranks_by_layer`` maps layer -> 1-D rank tensor aligned with ``scored``.
    ``layer=None`` picks, per prompt, the layer with the lowest top-n mean for
    Peak and the layer with the highest mean coverage over ``k_values`` for Cover.
    """
    sorted_k_values = _check_k_values(k_values)
    if peak_n < 1:
        raise ValueError("peak_n must be at least 1")
    for name, value in (("layer", layer), ("peak_n", peak_n), ("k_values", sorted_k_values)):
        stored = getattr(cache, name)
        if stored is not None and stored != value:
            raise ValueError(f"Cannot mix {name} settings in one peak_cover_cache")
    cache.layer, cache.peak_n, cache.k_values = layer, peak_n, sorted_k_values

    if layer is not None:
        if layer not in ranks_by_layer:
            raise ValueError(f"Layer {layer} is not present; available: {sorted(ranks_by_layer)}")
        selected = [layer]
    else:
        selected = sorted(ranks_by_layer)
    if not selected:
        raise ValueError("No layers to score")

    k_tensor = torch.tensor(sorted_k_values)
    best_peak: tuple[float, int, list[int]] | None = None
    best_cover: tuple[float, int, list[float]] | None = None
    for layer_idx in selected:
        ranks = ranks_by_layer[layer_idx].reshape(-1).cpu()
        if ranks.numel() != len(scored):
            raise ValueError("Rank tensor does not match the scored positions")
        n = min(peak_n, ranks.numel())
        top_ranks, top_idx = torch.topk(ranks, n, largest=False)
        peak = float(top_ranks.double().mean().item())
        if best_peak is None or peak < best_peak[0]:
            best_peak = (peak, layer_idx, [scored[i] for i in top_idx.tolist()])
        fractions = (ranks.unsqueeze(0) <= k_tensor.unsqueeze(1)).double().mean(dim=1)
        mean_cover = float(fractions.mean().item())
        if best_cover is None or mean_cover > best_cover[0]:
            best_cover = (mean_cover, layer_idx, fractions.tolist())

    cache.prompt_cnt += 1
    cache.peak_ranks.append(best_peak[0])
    cache.peak_layers.append(best_peak[1])
    cache.peak_positions.append(best_peak[2])
    cache.cover_layers.append(best_cover[1])
    cache.cover_fractions.append(best_cover[2])


def summarize(cache: peak_cover_cache) -> dict[str, object]:
    """Aggregate Peakn@k / Cover@k rates and log(k) AUC over the cached prompts."""
    if cache.prompt_cnt == 0 or cache.k_values is None:
        raise RuntimeError("No prompts were accumulated")
    ks = cache.k_values
    peak_rates = {k: sum(r <= k for r in cache.peak_ranks) / cache.prompt_cnt for k in ks}
    cover_rates = {
        k: sum(fr[i] for fr in cache.cover_fractions) / cache.prompt_cnt
        for i, k in enumerate(ks)
    }
    return {
        "prompt_count": cache.prompt_cnt,
        "peak_n": cache.peak_n,
        "peak_pass_rates": peak_rates,
        "peak_normalized_auc": _normalized_log_auc(peak_rates, ks),
        "cover_rates": cover_rates,
        "cover_normalized_auc": _normalized_log_auc(cover_rates, ks),
        "log_k_values": [math.log(k) for k in ks],
    }


def eval_peak_cover(
    logger,
    run_name: str,
    lens_data: tuple[dict[int, Tensor], Tensor, Tensor],
    cache: peak_cover_cache,
    ans_tokens: int | list[int],
    k_values: list[int],
    peak_n: int = 5,
    positions: list[int] | None = None,
    layer: int | None = None,
) -> dict[str, object]:
    """Accumulate one prompt and summarize Peakn@k and Cover@k so far.

    Same inputs as ``Experiments.eval_passk`` plus ``peak_n``, ``positions`` and
    several answer tokens. ``positions=None`` scores every prompt position.
    For batch runs use ``accumulate_ranks`` + ``summarize`` (see run_peak_cover).
    """
    token_ids = [ans_tokens] if isinstance(ans_tokens, int) else list(ans_tokens)
    if not token_ids:
        raise ValueError("ans_tokens must not be empty")
    logits_by_layer, _, input_ids = lens_data
    if not logits_by_layer:
        raise ValueError("Cached JLens output has no layer logits")
    prompt_length = int(input_ids.reshape(-1).numel())
    if prompt_length == 0:
        raise ValueError("Cached input_ids are empty")
    scored = list(range(prompt_length)) if positions is None else sorted(set(positions))
    if not scored or scored[0] < 0 or scored[-1] >= prompt_length:
        raise ValueError(f"positions must be non-empty and inside [0, {prompt_length})")

    layers = [layer] if layer is not None else sorted(logits_by_layer)
    ranks_by_layer = {}
    for layer_idx in layers:
        if layer_idx not in logits_by_layer:
            raise ValueError(
                f"Layer {layer_idx} is not present in cached JLens predictions; "
                f"available layers: {sorted(logits_by_layer)}"
            )
        layer_logits = logits_by_layer[layer_idx]
        if layer_logits.ndim != 2 or prompt_length > layer_logits.shape[0]:
            raise ValueError(f"Layer {layer_idx} logits must have shape [>= {prompt_length}, vocab]")
        index = torch.tensor(scored, device=layer_logits.device)
        ranks_by_layer[layer_idx] = answer_ranks(layer_logits[index].float(), token_ids).min(dim=1).values

    accumulate_ranks(cache, ranks_by_layer, scored, k_values, peak_n, layer)
    summary = summarize(cache)
    logger.info(
        "%s Peak%d@k / Cover@k progress: %d prompts, AUC=%.6f / %.6f",
        run_name, peak_n, cache.prompt_cnt,
        summary["peak_normalized_auc"], summary["cover_normalized_auc"],
    )
    return summary


# ---------------------------------------------------------------------------
# Batch evaluation (modeled on Experiments._passk_Jlens_batch)
# ---------------------------------------------------------------------------
POSITION_POLICIES = ("audio", "all")
SURFACE_FORMS = (" {w}", "{w}", " {W}", "{W}")


def answer_token_ids(tokenizer, label_text: str) -> list[int]:
    """Token ids of the surface forms that are a single whole-word token.

    Forms that split into pieces (e.g. "angry" -> "ang" + "ry") are dropped, so a
    common fragment such as "ch" or "ger" never counts as the answer. If no form
    is a single token, fall back to the first token of " <label>".
    """
    single: list[int] = []
    for form in SURFACE_FORMS:
        text = form.format(w=label_text, W=label_text.capitalize())
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(tokens) == 1 and tokens[0] not in single:
            single.append(int(tokens[0]))
    if single:
        return single
    tokens = tokenizer(" " + label_text, add_special_tokens=False)["input_ids"]
    if not tokens:
        raise ValueError(f"Label {label_text!r} produced no tokens.")
    return [int(tokens[0])]


def scored_positions(input_ids: Tensor, tokenizer, policy: str) -> list[int]:
    """Positions to score: audio tokens between the markers, or every position."""
    if policy not in POSITION_POLICIES:
        raise ValueError(f"Unknown position policy {policy!r}; expected {POSITION_POLICIES}")
    from tools.experiments import Experiments

    if policy == "all":
        return Experiments.swap_token_positions(input_ids, tokenizer, "all")
    audio = Experiments.swap_token_positions(input_ids, tokenizer, "audio")
    inner = audio[1:-1]  # drop <|audio_bos|> and <|audio_eos|>
    if not inner:
        raise ValueError("The audio segment has no tokens between its markers.")
    return inner


def run_peak_cover(
    *,
    logger,
    model,
    run_name: str,
    prompts_path: str,
    data_root: str,
    k_values: list[int],
    lenses: list,
    label_keys: list[str],
    peak_n: int = 5,
    position_policy: str = "audio",
) -> dict[str, object]:
    """Evaluate Peakn@k and Cover@k for several lenses with one forward per prompt.

    ``lenses`` are loaded ``JacobianLens`` objects. LogitLens is scored on the
    same layers as the lenses (not the final layer) as a reference. Every lens
    is reported with layers pooled (best layer per prompt) and per layer.

    Per prompt and layer, only the scored positions are unembedded, and the
    ranks of every answer token of every label are computed once on the model's
    device; all labels and the pooled/per-layer caches reuse them.
    """
    from jacobian_lens.jlens.hooks import ActivationRecorder
    from json_parser import Parser
    from tools.experiments import Experiments

    if not lenses:
        raise ValueError("Peak/Cover evaluation requires at least one loaded JLens.")
    if not label_keys or len(set(label_keys)) != len(label_keys):
        raise ValueError("Peak/Cover evaluation requires distinct, non-empty label keys.")
    sorted_k_values = _check_k_values(k_values)

    json_lines = Parser.load_jsonl_lines(prompts_path)
    if not json_lines:
        raise RuntimeError("Peak/Cover evaluation did not find any prompts.")
    adapter = model.load_model()
    adapter.load_data(data_root)
    processor = model.processor
    if processor is None:
        raise RuntimeError("The configured model adapter has no processor.")
    tokenizer = processor.tokenizer
    for index, lens in enumerate(lenses):
        if lens.d_model != adapter.d_model:
            raise ValueError(
                f"JLens at index {index} has d_model={lens.d_model}, "
                f"but the model has d_model={adapter.d_model}."
            )

    lens_layers = [list(lens.source_layers) for lens in lenses]
    union_layers = sorted({layer for layers in lens_layers for layer in layers})
    lens_keys = [str(i) for i in range(len(lenses))] + ["logit_lens"]
    layers_of = {str(i): lens_layers[i] for i in range(len(lenses))}
    layers_of["logit_lens"] = union_layers
    caches = {
        label: {
            key: {"pooled": peak_cover_cache(), **{l: peak_cover_cache() for l in layers_of[key]}}
            for key in lens_keys
        }
        for label in label_keys
    }
    token_log: dict[str, dict[str, list[str]]] = {label: {} for label in label_keys}

    for prompt_index, json_line in enumerate(json_lines, start=1):
        sample = Parser.parse_json_line(json_line)
        answers: dict[str, list[int]] = {}
        for label in label_keys:
            text = Experiments._passk_target_text(sample, label)
            answers[label] = answer_token_ids(tokenizer, text)
            token_log[label].setdefault(text, [tokenizer.decode([t]) for t in answers[label]])
        all_ids = sorted({t for ids in answers.values() for t in ids})
        column = {t: i for i, t in enumerate(all_ids)}
        label_columns = {label: [column[t] for t in ids] for label, ids in answers.items()}

        input_ids = adapter.encode(json_line, max_length=300)
        scored = scored_positions(input_ids, tokenizer, position_policy)

        with torch.no_grad(), ActivationRecorder(adapter.layers, at=union_layers) as recorder:
            adapter.forward(input_ids)
        with torch.no_grad():
            # ranks[key][layer] -> [len(scored), len(all_ids)]
            ranks: dict[str, dict[int, Tensor]] = {key: {} for key in lens_keys}
            for layer in union_layers:
                residual = recorder.activations[layer][0].float()
                residual = residual[torch.tensor(scored, device=residual.device)]
                ranks["logit_lens"][layer] = answer_ranks(adapter.unembed(residual).float(), all_ids)
                for i, lens in enumerate(lenses):
                    if layer in lens_layers[i]:
                        logits = adapter.unembed(lens.transport(residual, layer)).float()
                        ranks[str(i)][layer] = answer_ranks(logits, all_ids)
            ranks = {
                key: {layer: value.cpu() for layer, value in by_layer.items()}
                for key, by_layer in ranks.items()
            }

        for label in label_keys:
            cols = label_columns[label]
            for key in lens_keys:
                label_ranks = {
                    layer: value[:, cols].min(dim=1).values for layer, value in ranks[key].items()
                }
                for layer_key, cache in caches[label][key].items():
                    accumulate_ranks(
                        cache, label_ranks, scored, sorted_k_values, peak_n,
                        layer=None if layer_key == "pooled" else layer_key,
                    )
        logger.info("%s Peak/Cover: processed prompt %d/%d (%d scored positions)",
                    run_name, prompt_index, len(json_lines), len(scored))

    def report(cache: peak_cover_cache) -> dict[str, object]:
        return {
            **summarize(cache),
            "peak_ranks": cache.peak_ranks,
            "peak_layers": cache.peak_layers,
            "peak_positions": cache.peak_positions,
            "cover_layers": cache.cover_layers,
        }

    return {
        label: {
            "k_values": sorted_k_values,
            "peak_n": peak_n,
            "position_policy": position_policy,
            "label": label,
            "answer_tokens": token_log[label],
            "lenses": {
                key: {
                    **report(caches[label][key]["pooled"]),
                    "by_layer": {
                        str(l): report(caches[label][key][l]) for l in layers_of[key]
                    },
                }
                for key in lens_keys
            },
        }
        for label in label_keys
    }
