#!/usr/bin/env python3
"""Compare lenses: Peakn@k and Cover@k AUC by layer, with bootstrap error bands.

Usage:
    python tools/plot_peak_cover_compare.py <run-dir> [<run-dir> ...] [--boot 1000]
where <run-dir> is an eval-peak-cover output folder such as
output/inference/Qwen2.5-Omni-7B/peak_cover_esd. For every label subfolder,
all ``*_peak_cover.json`` files are drawn together and saved as
``<label>/compare_lenses.png``. LogitLens (identical in every file) is drawn once.

Error bands: 95% bootstrap interval over prompts (resample prompts with
replacement, recompute the layer's AUC). Peak uses the per-prompt
``peak_ranks``; Cover uses the per-prompt ``cover_fractions`` when the JSON has
them (older reports do not, so their Cover is drawn without a band).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def log_auc(rates: np.ndarray, ks: list[int]) -> np.ndarray:
    """Normalized trapezoidal AUC over log(k); ``rates`` has k on the last axis."""
    if len(ks) == 1:
        return rates[..., 0]
    x = np.log(np.asarray(ks, dtype=float))
    area = ((rates[..., 1:] + rates[..., :-1]) / 2 * np.diff(x)).sum(axis=-1)
    return area / (x[-1] - x[0])


def peak_auc_samples(peak_ranks: list[float], ks: list[int], boot: int, rng) -> tuple[float, np.ndarray]:
    """Point AUC and bootstrap AUC samples from per-prompt Peak ranks."""
    hits = (np.asarray(peak_ranks)[:, None] <= np.asarray(ks)[None, :]).astype(float)  # [P, K]
    point = float(log_auc(hits.mean(0), ks))
    idx = rng.integers(0, len(hits), size=(boot, len(hits)))
    return point, log_auc(hits[idx].mean(1), ks)


def cover_auc_samples(fractions: list[list[float]], ks: list[int], boot: int, rng) -> tuple[float, np.ndarray]:
    """Point AUC and bootstrap AUC samples from per-prompt Cover fractions."""
    frac = np.asarray(fractions, dtype=float)  # [P, K]
    point = float(log_auc(frac.mean(0), ks))
    idx = rng.integers(0, len(frac), size=(boot, len(frac)))
    return point, log_auc(frac[idx].mean(1), ks)


def layer_series(by_layer: dict, ks: list[int], metric: str, boot: int, seed: int):
    """Per-layer AUC with 95% bootstrap bounds (bounds are None when unavailable)."""
    rng = np.random.default_rng(seed)
    layers = sorted(by_layer, key=int)
    point, low, high = [], [], []
    for layer in layers:
        item = by_layer[layer]
        if metric == "peak" and "peak_ranks" in item:
            value, samples = peak_auc_samples(item["peak_ranks"], ks, boot, rng)
        elif metric == "cover" and "cover_fractions" in item:
            value, samples = cover_auc_samples(item["cover_fractions"], ks, boot, rng)
        else:
            key = "peak_normalized_auc" if metric == "peak" else "cover_normalized_auc"
            point.append(float(item[key]))
            low.append(math.nan)
            high.append(math.nan)
            continue
        point.append(value)
        low.append(float(np.percentile(samples, 2.5)))
        high.append(float(np.percentile(samples, 97.5)))
    return [int(l) for l in layers], np.array(point), np.array(low), np.array(high)


def plot_label(label_dir: Path, boot: int) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(label_dir.glob("*_peak_cover.json"))
    if not files:
        return None
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    n, ks = runs[0]["peak_n"], runs[0]["k_values"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    for ax, (metric, title) in zip(axes, (("peak", f"Peak{n}@k AUC"), ("cover", "Cover@k AUC"))):
        curves = [(f"JLens {run.get('lens_name', '?')}", run["jlens"]["by_layer"], "-") for run in runs]
        curves.append(("LogitLens", runs[0]["logit_lens"]["by_layer"], "--"))
        for name, by_layer, style in curves:
            layers, point, low, high = layer_series(by_layer, ks, metric, boot, seed=0)
            color = "k" if name == "LogitLens" else None
            line, = ax.plot(layers, point, style, marker="o", ms=3, label=name, color=color)
            if not np.all(np.isnan(low)):
                ax.fill_between(layers, low, high, color=line.get_color(), alpha=0.18, lw=0)
        ax.set(title=title, xlabel="layer", ylabel="normalized AUC over log(k)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{label_dir.parent.name} / {runs[0]['label']} | "
                 f"{runs[0]['jlens']['prompt_count']} prompts | each layer scored independently | "
                 f"bands: 95% bootstrap over prompts ({boot}x)")
    fig.tight_layout()
    out = label_dir / "compare_lenses.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--boot", type=int, default=1000, help="Bootstrap resamples.")
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        for label_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            out = plot_label(label_dir, args.boot)
            if out:
                print(f"Saved {out}")


if __name__ == "__main__":
    main()
