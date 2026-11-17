#!/usr/bin/env python3
"""Recolor saved t-SNE embeddings so every point from the same audio clip matches.

Built for the prompt-variant experiment, where each audio clip is paired with
many text prompts. Every clip gets a unique (color, marker) pair - 8 hues x 3
markers covers up to 24 clips - and thin spokes join each point to its clip's
centroid, which is labelled with the short clip id (e.g. ``a07``). Spoke length
therefore shows how far the text prompt moves a clip's J-lens.

Usage:
    python tools/plot_tsne_by_audio.py <embedding.npz | directory> [...]

A directory is searched recursively for ``tsne*_embedding.npz``. Each plot is
written next to its embedding as ``{normal,normalized}/<stem>_by_audio.png``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

# Validated categorical order (light mode). Kept in fixed order, never cycled.
HUES = (
	"#2a78d6",  # blue
	"#eb6834",  # orange
	"#1baf7a",  # aqua
	"#eda100",  # yellow
	"#e87ba4",  # magenta
	"#008300",  # green
	"#6250d6",  # violet
	"#e34948",  # red
)
MARKERS = ("o", "s", "^")
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def load_embedding(path: Path) -> tuple[np.ndarray, list[dict], str]:
	with np.load(path, allow_pickle=False) as data:
		if "sample_labels_json" not in data:
			raise ValueError(f"{path} has no saved sample labels.")
		embedding = np.asarray(data["embedding"], dtype=np.float64)
		labels = [json.loads(str(item)) for item in data["sample_labels_json"]]
		normalization = (
			str(data["feature_normalization"]) if "feature_normalization" in data else "none"
		)
	if embedding.ndim != 2 or embedding.shape[1] != 2 or len(labels) != len(embedding):
		raise ValueError(f"{path} has an unexpected embedding or label shape.")
	return embedding, labels, normalization


def plot_by_audio(
	embedding_path: str | Path,
	*,
	audio_key: str = "audio_id",
	legend_keys: Sequence[str] = (),
	spokes: bool = True,
) -> Path:
	"""Write one audio-colored plot for a saved embedding and return its path."""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	embedding_path = Path(embedding_path)
	embedding, labels, normalization = load_embedding(embedding_path)
	missing = [i for i, label in enumerate(labels) if audio_key not in label]
	if missing:
		raise ValueError(
			f"{embedding_path}: {len(missing)} rows have no {audio_key!r} label "
			f"(first row {missing[0]})."
		)
	audio_ids = sorted({str(label[audio_key]) for label in labels})
	capacity = len(HUES) * len(MARKERS)
	if len(audio_ids) > capacity:
		raise ValueError(
			f"{len(audio_ids)} audio clips exceed the {capacity} distinct "
			"color/marker pairs; plot a subset instead."
		)
	audio_values = np.asarray([str(label[audio_key]) for label in labels])

	figure, axis = plt.subplots(figsize=(11, 8))
	figure.patch.set_facecolor(SURFACE)
	axis.set_facecolor(SURFACE)
	for position, audio_id in enumerate(audio_ids):
		mask = audio_values == audio_id
		points = embedding[mask]
		centroid = points.mean(axis=0)
		# Hue changes fastest so neighbouring clips differ in color first.
		color = HUES[position % len(HUES)]
		marker = MARKERS[position // len(HUES)]
		if spokes:
			for x, y in points:
				axis.plot(
					[centroid[0], x], [centroid[1], y],
					color=color, linewidth=0.8, alpha=0.35, zorder=1,
				)
		first = labels[int(np.flatnonzero(mask)[0])]
		details = [str(first[key]) for key in legend_keys if key in first]
		short_id = audio_id.split("_", 1)[0]
		legend_label = short_id + (f" · {', '.join(details)}" if details else "")
		axis.scatter(
			points[:, 0], points[:, 1],
			s=42, color=color, marker=marker, edgecolors=SURFACE, linewidths=1.0,
			alpha=0.9, zorder=2, label=legend_label,
		)
		# Label above the cluster's top edge so it never hides the points.
		axis.annotate(
			short_id, (centroid[0], points[:, 1].max()), xytext=(0, 5),
			textcoords="offset points", fontsize=8, color=TEXT_PRIMARY,
			ha="center", va="bottom", zorder=3,
		)

	axis.set_title(
		f"t-SNE by audio clip ({len(audio_ids)} clips, {len(labels)} points, "
		f"normalization: {normalization})",
		color=TEXT_PRIMARY, fontsize=11, loc="left",
	)
	axis.set_xlabel("t-SNE component 1", color=TEXT_SECONDARY)
	axis.set_ylabel("t-SNE component 2", color=TEXT_SECONDARY)
	axis.tick_params(colors=TEXT_SECONDARY, labelsize=8)
	axis.grid(color=GRID, linewidth=0.6, zorder=0)
	for spine in axis.spines.values():
		spine.set_color(GRID)
	legend = axis.legend(
		title=audio_key, bbox_to_anchor=(1.02, 1), loc="upper left",
		fontsize=8, title_fontsize=9, frameon=False, labelcolor=TEXT_PRIMARY,
	)
	legend.get_title().set_color(TEXT_SECONDARY)
	figure.tight_layout()

	output_dir = embedding_path.parent / (
		"normalized" if normalization == "l2_per_prompt" else "normal"
	)
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / f"{embedding_path.stem}_by_audio.png"
	figure.savefig(output_path, dpi=200, facecolor=SURFACE)
	plt.close(figure)
	return output_path


def collect_embeddings(targets: Sequence[str]) -> list[Path]:
	paths: list[Path] = []
	for target in targets:
		path = Path(target)
		if path.is_dir():
			paths.extend(sorted(path.rglob("tsne*_embedding.npz")))
		elif path.is_file():
			paths.append(path)
		else:
			raise FileNotFoundError(f"Not found: {path}")
	if not paths:
		raise FileNotFoundError("No tsne*_embedding.npz files found.")
	return paths


def main(argv: Sequence[str] | None = None) -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("targets", nargs="+", help="Embedding .npz files or directories.")
	parser.add_argument("--audio-key", default="audio_id")
	parser.add_argument(
		"--legend-labels", nargs="*", default=None,
		help="Label fields shown next to each clip in the legend "
		"(default: emotion language gender, or focus_class, whichever exist).",
	)
	parser.add_argument("--no-spokes", action="store_true", help="Hide centroid spokes.")
	args = parser.parse_args(argv)
	legend_keys = (
		args.legend_labels
		if args.legend_labels is not None
		else ["emotion", "language", "gender", "focus_class"]
	)
	for path in collect_embeddings(args.targets):
		print(plot_by_audio(
			path, audio_key=args.audio_key, legend_keys=legend_keys,
			spokes=not args.no_spokes,
		))


if __name__ == "__main__":
	main()
