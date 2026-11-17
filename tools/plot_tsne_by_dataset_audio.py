#!/usr/bin/env python3
"""Plot a combined t-SNE: color = dataset, marker shape = audio clip within it.

Meant for embeddings built from several prompt-variant runs (e.g. ESD and
FSD50K on one map). Thin spokes join each point to its clip's centroid, which
is labelled ``<dataset>:<clip>`` (e.g. ``esd:a07``), so spoke length shows how
far the text prompt moves that clip.

Usage:
    python tools/plot_tsne_by_dataset_audio.py <embedding.npz | directory> [...]

A directory is searched recursively for ``tsne*_embedding.npz``. Each plot is
written next to its embedding as ``{normal,normalized}/<stem>_by_dataset_audio.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

try:
	from tools.plot_tsne_by_audio import (
		GRID, HUES, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, collect_embeddings,
		load_embedding,
	)
except ImportError:  # Executed as `python tools/plot_tsne_by_dataset_audio.py`.
	from plot_tsne_by_audio import (  # type: ignore[no-redef]
		GRID, HUES, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, collect_embeddings,
		load_embedding,
	)

# 15 filled shapes, then 5 hollow ones: 20 distinguishable clips per dataset.
FILLED_MARKERS = ("o", "s", "^", "v", "D", "<", ">", "p", "h", "*", "P", "X", "d", "8", "H")
HOLLOW_MARKERS = ("o", "s", "^", "D", "v")
MAX_DATASETS = 3  # Scatter plots stay legible with at most three hues.


def marker_style(index: int) -> tuple[str, bool]:
	"""Return (marker, hollow) for the index-th clip inside one dataset."""
	if index < len(FILLED_MARKERS):
		return FILLED_MARKERS[index], False
	hollow_index = index - len(FILLED_MARKERS)
	if hollow_index < len(HOLLOW_MARKERS):
		return HOLLOW_MARKERS[hollow_index], True
	raise ValueError(
		f"More than {len(FILLED_MARKERS) + len(HOLLOW_MARKERS)} clips in one "
		"dataset; plot a subset instead."
	)


def plot_by_dataset_audio(
	embedding_path: str | Path,
	*,
	audio_key: str = "audio_id",
	legend_keys: Sequence[str] = (),
	spokes: bool = True,
) -> Path:
	"""Write one dataset/audio plot for a saved embedding and return its path."""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib.lines import Line2D

	embedding_path = Path(embedding_path)
	embedding, labels, normalization = load_embedding(embedding_path)
	with np.load(embedding_path, allow_pickle=False) as data:
		if "dataset_labels" not in data:
			raise ValueError(f"{embedding_path} has no dataset labels.")
		datasets = np.asarray(data["dataset_labels"]).astype(str)
	if any(audio_key not in label for label in labels):
		raise ValueError(f"{embedding_path}: some rows have no {audio_key!r} label.")
	audio_values = np.asarray([str(label[audio_key]) for label in labels])
	dataset_names = sorted(set(datasets.tolist()))
	if len(dataset_names) > MAX_DATASETS:
		raise ValueError(f"{len(dataset_names)} datasets exceed the {MAX_DATASETS}-color limit.")

	figure, axis = plt.subplots(figsize=(15, 9.5))
	figure.patch.set_facecolor(SURFACE)
	axis.set_facecolor(SURFACE)
	audio_handles: list[Line2D] = []
	for dataset_index, dataset in enumerate(dataset_names):
		color = HUES[dataset_index]
		in_dataset = datasets == dataset
		for clip_index, audio_id in enumerate(sorted(set(audio_values[in_dataset].tolist()))):
			mask = in_dataset & (audio_values == audio_id)
			points = embedding[mask]
			centroid = points.mean(axis=0)
			marker, hollow = marker_style(clip_index)
			if spokes:
				for x, y in points:
					axis.plot(
						[centroid[0], x], [centroid[1], y],
						color=color, linewidth=0.7, alpha=0.3, zorder=1,
					)
			axis.scatter(
				points[:, 0], points[:, 1], s=40, marker=marker,
				facecolors="none" if hollow else color,
				edgecolors=color if hollow else SURFACE,
				linewidths=1.3 if hollow else 0.8, alpha=0.9, zorder=2,
			)
			short_id = audio_id.split("_", 1)[0]
			axis.annotate(
				f"{dataset}:{short_id}", (centroid[0], points[:, 1].max()),
				xytext=(0, 4), textcoords="offset points", fontsize=6.5,
				color=TEXT_PRIMARY, ha="center", va="bottom", zorder=3,
			)
			first = labels[int(np.flatnonzero(mask)[0])]
			details = [str(first[key]) for key in legend_keys if key in first]
			audio_handles.append(Line2D(
				[], [], linestyle="none", marker=marker, markersize=6,
				markerfacecolor="none" if hollow else color,
				markeredgecolor=color, markeredgewidth=1.2 if hollow else 0.5,
				label=f"{dataset}:{short_id}" + (f" · {', '.join(details)}" if details else ""),
			))

	axis.set_title(
		f"Combined t-SNE: {', '.join(dataset_names)} ({len(labels)} points, "
		f"normalization: {normalization}). Color = dataset, shape = audio clip",
		color=TEXT_PRIMARY, fontsize=11, loc="left",
	)
	axis.set_xlabel("t-SNE component 1", color=TEXT_SECONDARY)
	axis.set_ylabel("t-SNE component 2", color=TEXT_SECONDARY)
	axis.tick_params(colors=TEXT_SECONDARY, labelsize=8)
	axis.grid(color=GRID, linewidth=0.6, zorder=0)
	for spine in axis.spines.values():
		spine.set_color(GRID)

	dataset_handles = [
		Line2D([], [], linestyle="none", marker="s", markersize=9,
		       markerfacecolor=HUES[i], markeredgecolor=HUES[i], label=name)
		for i, name in enumerate(dataset_names)
	]
	dataset_legend = axis.legend(
		handles=dataset_handles, title="dataset", loc="upper left",
		bbox_to_anchor=(1.01, 1.0), fontsize=9, title_fontsize=9, frameon=False,
	)
	axis.add_artist(dataset_legend)
	audio_legend = axis.legend(
		handles=audio_handles, title=audio_key, loc="upper left",
		bbox_to_anchor=(1.01, 0.9), fontsize=6.5, title_fontsize=8, frameon=False,
		ncol=2 if len(audio_handles) > 24 else 1, columnspacing=1.0,
		handletextpad=0.3,
	)
	for legend in (dataset_legend, audio_legend):
		legend.get_title().set_color(TEXT_SECONDARY)
	figure.tight_layout()

	output_dir = embedding_path.parent / (
		"normalized" if normalization == "l2_per_prompt" else "normal"
	)
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / f"{embedding_path.stem}_by_dataset_audio.png"
	figure.savefig(output_path, dpi=200, facecolor=SURFACE)
	plt.close(figure)
	return output_path


def main(argv: Sequence[str] | None = None) -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("targets", nargs="+", help="Embedding .npz files or directories.")
	parser.add_argument("--audio-key", default="audio_id")
	parser.add_argument(
		"--legend-labels", nargs="*", default=None,
		help="Label fields shown next to each clip in the legend "
		"(default: emotion language gender focus_class, whichever exist).",
	)
	parser.add_argument("--no-spokes", action="store_true", help="Hide centroid spokes.")
	args = parser.parse_args(argv)
	legend_keys = (
		args.legend_labels
		if args.legend_labels is not None
		else ["emotion", "language", "gender", "focus_class"]
	)
	for path in collect_embeddings(args.targets):
		print(plot_by_dataset_audio(
			path, audio_key=args.audio_key, legend_keys=legend_keys,
			spokes=not args.no_spokes,
		))


if __name__ == "__main__":
	main()
