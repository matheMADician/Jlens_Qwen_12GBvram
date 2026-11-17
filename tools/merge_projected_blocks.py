#!/usr/bin/env python3
"""Merge per-prompt projected block features from several runs into one archive.

The output keeps only what ``projection_tsne.py create-blocks`` reads, so a
combined block-wise t-SNE can be built with the existing command. All inputs
must share one projection (dimension, seed, method, original width) and the
same block layer ranges; otherwise their points are not comparable.

Usage:
    python tools/merge_projected_blocks.py --features a/dataset_esd.npz \
        --features b/dataset_fsd50k.npz --output out/combined_block_features.npz
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

METADATA_KEYS = ("original_dim", "projection_dim", "projection_seed", "projection_method")
ROW_KEYS = ("block_features", "sample_ids", "sample_labels_json", "dataset_labels")


def merge_projected_blocks(feature_paths: Sequence[Path], output_path: Path) -> Path:
	if len(feature_paths) < 2:
		raise ValueError("Pass at least two --features archives to merge.")
	rows: dict[str, list[np.ndarray]] = {key: [] for key in ROW_KEYS}
	reference: dict[str, np.ndarray] | None = None
	for path in feature_paths:
		with np.load(path, allow_pickle=False) as data:
			missing = [k for k in (*ROW_KEYS, *METADATA_KEYS, "block_layer_ranges") if k not in data]
			if missing:
				raise ValueError(f"{path} is missing {missing}; was it fitted with --save-projected-blocks?")
			shared = {k: np.asarray(data[k]) for k in (*METADATA_KEYS, "block_layer_ranges")}
			if reference is None:
				reference = shared
			else:
				for key, value in shared.items():
					if not np.array_equal(value, reference[key]):
						raise ValueError(
							f"{path} has {key}={value.tolist()!r}, expected "
							f"{reference[key].tolist()!r}; runs are not comparable."
						)
			for key in ROW_KEYS:
				# Cast text to object first so differing string widths concatenate.
				value = np.asarray(data[key])
				rows[key].append(value.astype(object) if value.dtype.kind == "U" else value)
	assert reference is not None
	merged = {
		key: (
			np.concatenate(parts).astype(np.str_)
			if parts[0].dtype == object
			else np.concatenate(parts)
		)
		for key, parts in rows.items()
	}
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.NamedTemporaryFile(
		dir=output_path.parent, prefix=".merge-", suffix=".tmp", delete=False
	) as handle:
		temporary = Path(handle.name)
		try:
			np.savez_compressed(handle, **merged, **reference)
			handle.flush()
			os.fsync(handle.fileno())
			os.replace(temporary, output_path)
		finally:
			if temporary.exists():
				temporary.unlink()
	return output_path


def main(argv: Sequence[str] | None = None) -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--features", type=Path, action="append", required=True)
	parser.add_argument("--output", type=Path, required=True)
	args = parser.parse_args(argv)
	path = merge_projected_blocks(args.features, args.output)
	with np.load(path, allow_pickle=False) as data:
		labels, counts = np.unique(data["dataset_labels"], return_counts=True)
	print(f"Wrote {path}: " + ", ".join(f"{l}={c}" for l, c in zip(labels, counts)))


if __name__ == "__main__":
	main()
