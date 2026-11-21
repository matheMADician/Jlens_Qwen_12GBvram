#!/usr/bin/env python3
"""Compare lenses: Peakn@k and Cover@k AUC by layer, one line per lens.

Usage:
    python tools/plot_peak_cover_compare.py <run-dir> [<run-dir> ...]
where <run-dir> is an eval-peak-cover output folder such as
output/inference/Qwen2.5-Omni-7B/peak_cover_esd. For every label subfolder,
all ``*_peak_cover.json`` files are drawn together and saved as
``<label>/compare_lenses.png``. LogitLens (identical in every file) is drawn once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot_label(label_dir: Path) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(label_dir.glob("*_peak_cover.json"))
    if not files:
        return None
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    n = runs[0]["peak_n"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    for ax, (metric, title) in zip(
        axes, (("peak_normalized_auc", f"Peak{n}@k AUC"), ("cover_normalized_auc", "Cover@k AUC"))
    ):
        for run in runs:
            by_layer = run["jlens"]["by_layer"]
            layers = sorted(by_layer, key=int)
            ax.plot([int(l) for l in layers], [by_layer[l][metric] for l in layers],
                    marker="o", ms=3, label=f"JLens {run.get('lens_name', '?')}")
        logit = runs[0]["logit_lens"]["by_layer"]
        layers = sorted(logit, key=int)
        ax.plot([int(l) for l in layers], [logit[l][metric] for l in layers],
                "k--", label="LogitLens")
        ax.set(title=title, xlabel="layer", ylabel="normalized AUC over log(k)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{label_dir.parent.name} / {runs[0]['label']} | "
                 f"{runs[0]['jlens']['prompt_count']} prompts | each layer scored independently")
    fig.tight_layout()
    out = label_dir / "compare_lenses.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    for run_dir in parser.parse_args().run_dirs:
        for label_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            out = plot_label(label_dir)
            if out:
                print(f"Saved {out}")


if __name__ == "__main__":
    main()
