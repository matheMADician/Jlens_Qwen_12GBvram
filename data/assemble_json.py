#!/usr/bin/env python3
"""Build a combined JLens dataset from multiple source dataset directories.

Examples:
    python data/assemble_json.py --seed 42 \
        --dataset data/esc50:10 \
        --dataset data/librispeech_test_clean:5 \
        --output data/combined

The script reads each dataset's prompts.jsonl, samples a deterministic subset,
checks that the selected audio files exist, and writes a merged JSONL manifest.
Audio files are not copied; audio_path values point to the original dataset
directories.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable


def parse_dataset_spec(spec: str) -> tuple[Path, int]:
    """Parse a dataset spec like 'data/esc50:10' or 'data/esc50=10'."""
    for sep in (":", "="):
        if sep in spec:
            path_str, count_str = spec.rsplit(sep, 1)
            try:
                count = int(count_str)
            except ValueError as exc:
                raise ValueError(f"Invalid count in dataset spec: {spec!r}") from exc
            path = Path(path_str).expanduser().resolve()
            return path, count
    raise ValueError(
        "Dataset spec must include a count, e.g. 'data/esc50:10' or 'data/esc50=10'"
    )


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_dataset_manifest(
    dataset_specs: Iterable[str],
    output_dir: Path,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_records: list[dict] = []

    for spec in dataset_specs:
        source_root, take_count = parse_dataset_spec(spec)
        manifest_path = source_root / "prompts.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Expected prompts.jsonl under dataset root: {manifest_path}"
            )

        records = load_jsonl(manifest_path)
        if take_count < 0:
            raise ValueError(f"Dataset count must be non-negative: {spec!r}")
        if take_count > len(records):
            raise ValueError(
                f"Requested {take_count} samples from {source_root}, but only {len(records)} exist"
            )

        dataset_name = source_root.name
        selected_indices = sorted(rng.sample(range(len(records)), take_count))
        for idx in selected_indices:
            record = dict(records[idx])
            audio_rel = record.get("audio_path")
            if not audio_rel:
                raise ValueError(f"Record missing 'audio_path' in {manifest_path}: {record!r}")

            source_file = (source_root / audio_rel).resolve()
            if not source_file.exists():
                raise FileNotFoundError(f"Audio not found: {source_file}")

            original_id = str(record.get("id", "sample"))
            record["id"] = str(len(combined_records))
            record["source_id"] = original_id
            record["audio_path"] = (Path(dataset_name) / audio_rel).as_posix()
            record["source_dataset"] = dataset_name
            combined_records.append(record)

    return combined_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample a fixed number of items from each dataset directory and "
            "write a combined JSONL manifest."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset spec in the form 'path:count' or 'path=count'. Can be used multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/combined_dataset"),
        help="Output directory for copied audio and merged prompts.jsonl",
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Optional positional dataset specs, also accepted as 'path:count'",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = list(args.dataset) + list(args.datasets)
    if not specs:
        raise SystemExit(
            "No dataset specs provided. Example: --dataset data/esc50:10 --dataset data/librispeech_test_clean:5"
        )

    records = build_dataset_manifest(specs, args.output, args.seed)
    output_jsonl = args.output / "prompts.jsonl"
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} samples to {output_jsonl}")
    print("audio_path values were rewritten to point to the original dataset files")


if __name__ == "__main__":
    main()
