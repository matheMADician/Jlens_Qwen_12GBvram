#!/usr/bin/env python3
"""Plot Peakn@k and Cover@k against k, one curve per layer.

Reads the ``*_peak_cover.json`` files written by ``main.py eval-peak-cover``
and draws, for JLens and LogitLens, the pass-rate curve of every layer
(colored by depth). The area under each curve is that layer's AUC.
The dashed line is a random-token reference (rank uniform over the vocabulary),
which is only a lower bound for a real baseline.

Usage:
    python tools/plot_peak_cover_curves.py <json> [<json> ...]
    python tools/plot_peak_cover_curves.py output/inference/Qwen2.5-Omni-7B/peak_cover_esd/*/*_peak_cover.json
Each figure is saved next to its JSON as ``<name>_curves.png``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VOCAB_SIZE = 152064      # Qwen2.5-Omni
TYPICAL_POSITIONS = 76   # median ESD audio length in tokens, for the Peak reference


def _rates(result: dict, key: str, ks: list[int]) -> list[float]:
    rates = result[key]
    return [float(rates[str(k)] if str(k) in rates else rates[k]) for k in ks]


def random_peak_reference(ks: list[int], peak_n: int, positions: int = TYPICAL_POSITIONS) -> list[float]:
    rng = np.random.default_rng(0)
    sims = np.sort(rng.integers(1, VOCAB_SIZE + 1, size=(20000, positions)), axis=1)[:, :peak_n].mean(1)
    return [float((sims <= k).mean()) for k in ks]


def plot_file(path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(path.read_text(encoding="utf-8"))
    ks, n = data["k_values"], data["peak_n"]
    lenses = [("JLens", data["jlens"]), ("LogitLens", data["logit_lens"])]
    metrics = [
        (f"Peak{n}@k", "peak_pass_rates", random_peak_reference(ks, n)),
        ("Cover@k", "cover_rates", [k / VOCAB_SIZE for k in ks]),
    ]

    layers = sorted(data["jlens"]["by_layer"], key=int)
    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(vmin=int(layers[0]), vmax=int(layers[-1]))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for col, (lens_name, result) in enumerate(lenses):
        for row, (title, key, reference) in enumerate(metrics):
            ax = axes[row][col]
            for layer in layers:
                ax.plot(ks, _rates(result["by_layer"][layer], key, ks),
                        color=cmap(norm(int(layer))), lw=1.2, alpha=0.9)
            ax.plot(ks, reference, "k--", lw=1.5, label="random token")
            ax.set_xscale("log", base=2)
            ax.set_ylim(0, 1 if key == "peak_pass_rates" else None)
            ax.grid(alpha=0.3)
            ax.set_title(f"{lens_name}: {title}", fontsize=9)
            ax.set_ylabel("pass rate" if key == "peak_pass_rates" else "covered fraction")
            ax.legend(loc="upper left", fontsize=8)
        axes[1][col].set_xlabel("k (log scale)")
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axes, label="layer", shrink=0.8)
    fig.suptitle(
        f"{data['label']} | lens {data.get('lens_name', '')} | {data['jlens']['prompt_count']} prompts "
        f"| each line = one layer (area under it = that layer's AUC)\n"
        f"Peak{n}@k: share of prompts whose best {n} positions have mean rank <= k   |   "
        f"Cover@k: share of audio positions with rank <= k",
        fontsize=10,
    )
    out = path.with_name(path.stem + "_curves.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("json_files", nargs="+", type=Path)
    for path in parser.parse_args().json_files:
        print(f"Saved {plot_file(path)}")


if __name__ == "__main__":
    main()
