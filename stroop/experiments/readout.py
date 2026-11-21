"""Shared lens-readout helpers for stroop experiments (pilot, conflict readout).

The idea: at every position of a region (audio or text), compare how highly a
lens ranks the *correct* gender's tokens against the *wrong* gender's tokens.

* ``rank`` is 1-based over the full vocabulary (1 = top prediction).
* A token *set* is scored by its best (lowest) rank among its members.
* ``win_rate`` = fraction of positions where the correct set outranks the wrong
  set. Chance is 0.5, and a lens with a fixed bias toward one gender gets ~0.5
  once averaged over balanced female/male clips, so it is robust to token-prior
  effects that raw ranks are not.

Torch is imported lazily so the aggregation code runs (and is tested) without it.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Iterable, Sequence

GENDER_WORD_SETS: dict[str, dict[str, tuple[str, ...]]] = {
    # The attribute words the gender question is about.
    "attr": {"female": ("female",), "male": ("male",)},
    # Broader words, including the ones that carry content gender in speech.
    "broad": {
        "female": ("female", "woman", "women", "girl", "lady", "she", "her"),
        "male": ("male", "man", "men", "boy", "gentleman", "he", "him", "his"),
    },
}
OTHER = {"female": "male", "male": "female"}
HIT_KS = (1, 10, 100)


def single_token_ids(tokenizer: Any, words: Iterable[str]) -> list[int]:
    """Ids of every surface form of ``words`` that is exactly one token.

    Forms: as-is, capitalised, each with and without a leading space, because
    Qwen's BPE gives "male" and " male" different ids.
    """
    ids: set[int] = set()
    for word in words:
        for form in {word, word.capitalize()}:
            for text in (form, " " + form):
                encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
                if len(encoded) == 1:
                    ids.add(int(encoded[0]))
    return sorted(ids)


def build_token_sets(tokenizer: Any) -> dict[str, dict[str, list[int]]]:
    """``{set_name: {gender: [token ids]}}``; raises if a set ends up empty."""
    sets = {
        name: {gender: single_token_ids(tokenizer, words) for gender, words in by_gender.items()}
        for name, by_gender in GENDER_WORD_SETS.items()
    }
    for name, by_gender in sets.items():
        for gender, ids in by_gender.items():
            if not ids:
                raise ValueError(f"No single-token form for set {name!r}, gender {gender!r}")
    return sets


def best_set_rank(logits: Any, token_ids: Sequence[int]) -> Any:
    """Per-position best rank of a token set: ``[positions, vocab] -> [positions]``.

    rank = 1 + number of vocabulary entries with a strictly higher logit.
    """
    import torch

    best = None
    for token_id in token_ids:
        target = logits[:, token_id : token_id + 1]
        rank = (logits > target).sum(dim=-1) + 1
        best = rank if best is None else torch.minimum(best, rank)
    return best


def region_stats(correct_ranks: Sequence[int], wrong_ranks: Sequence[int]) -> dict[str, float]:
    """Summarise per-position ranks of the correct and wrong sets over one region."""
    if len(correct_ranks) != len(wrong_ranks) or not correct_ranks:
        raise ValueError("Need the same, non-zero number of correct and wrong ranks")
    wins = sum(c < w for c, w in zip(correct_ranks, wrong_ranks))
    return {
        "n_positions": len(correct_ranks),
        "win_rate": wins / len(correct_ranks),
        "correct_best_rank": min(correct_ranks),
        "wrong_best_rank": min(wrong_ranks),
        "correct_median_rank": statistics.median(correct_ranks),
        "wrong_median_rank": statistics.median(wrong_ranks),
    }


GROUP_KEYS = ("lens", "region", "token_set", "layer")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-prompt rows by (lens, region, token_set, layer).

    ``balanced_win_rate`` averages the per-gender mean win rates, so a lens that
    always prefers one gender scores 0.5 rather than looking good on the
    majority gender.
    """
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in GROUP_KEYS)].append(row)

    summary = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0][:3])) + (item[0][3],)):
        by_gender: dict[str, list[float]] = defaultdict(list)
        for row in group:
            by_gender[row["gender"]].append(row["win_rate"])
        gender_means = {g: statistics.fmean(v) for g, v in sorted(by_gender.items())}
        best = [row["correct_best_rank"] for row in group]
        entry = dict(zip(GROUP_KEYS, key))
        entry.update({
            "n_prompts": len(group),
            "mean_win_rate": statistics.fmean(row["win_rate"] for row in group),
            "balanced_win_rate": statistics.fmean(gender_means.values()),
            "win_rate_by_gender": gender_means,
            "median_correct_best_rank": statistics.median(best),
            "median_wrong_best_rank": statistics.median(row["wrong_best_rank"] for row in group),
            **{f"hit@{k}": sum(r <= k for r in best) / len(best) for k in HIT_KS},
        })
        summary.append(entry)
    return summary
