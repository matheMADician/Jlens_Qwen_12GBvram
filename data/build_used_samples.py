#!/usr/bin/env python3
"""Build a source-aware exclusion manifest from assembled JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_used_samples(
    input_paths: list[Path],
    output_path: Path,
) -> int:
    """Write unique source dataset/index pairs, retaining source IDs when present."""
    used_samples: dict[tuple[str, int], str | None] = {}
    for input_path in input_paths:
        if not input_path.is_file():
            raise FileNotFoundError(f"Assembled JSONL not found: {input_path}")
        with input_path.open("r", encoding="utf-8") as input_file:
            for line_number, raw_line in enumerate(input_file, start=1):
                if not raw_line.strip():
                    continue
                try:
                    parsed_record: Any = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {input_path}, line {line_number}: "
                        f"{exc.msg}"
                    ) from exc
                if not isinstance(parsed_record, dict):
                    raise ValueError(
                        f"Expected a JSON object in {input_path}, line {line_number}."
                    )

                source_dataset = parsed_record.get("source_dataset")
                source_index = parsed_record.get("source_index")
                source_id = parsed_record.get("source_id")
                if not isinstance(source_dataset, str) or not source_dataset.strip():
                    raise ValueError(
                        f"Record in {input_path}, line {line_number} has no "
                        "non-empty source_dataset."
                    )
                if (
                    not isinstance(source_index, int)
                    or isinstance(source_index, bool)
                    or source_index < 0
                ):
                    raise ValueError(
                        f"Record in {input_path}, line {line_number} has no "
                        "non-negative integer source_index."
                    )
                if source_id is not None and (
                    not isinstance(source_id, (str, int))
                    or isinstance(source_id, bool)
                    or not str(source_id).strip()
                ):
                    raise ValueError(
                        f"Record in {input_path}, line {line_number} has an "
                        "invalid source_id."
                    )
                key = (source_dataset.strip(), source_index)
                normalized_source_id = (
                    None if source_id is None else str(source_id)
                )
                previous_source_id = used_samples.get(key)
                if key in used_samples and previous_source_id != normalized_source_id:
                    raise ValueError(
                        f"Conflicting source_id values for {key!r} in assembled inputs."
                    )
                used_samples[key] = normalized_source_id

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for (source_dataset, source_index), source_id in sorted(used_samples.items()):
            used_sample_record: dict[str, str | int] = {
                "source_dataset": source_dataset,
                "source_index": source_index,
            }
            if source_id is not None:
                used_sample_record["source_id"] = source_id
            output_file.write(
                json.dumps(used_sample_record, ensure_ascii=False) + "\n"
            )
    return len(used_samples)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan assembled JSONL manifests and write a unique source-index "
            "exclusion manifest."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        dest="input_paths",
        help="Assembled JSONL input; repeat for additional manifests.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file of used source dataset/index pairs.",
    )
    args = parser.parse_args()
    count = build_used_samples(args.input_paths, args.output)
    print(f"Wrote {count} unique source samples to {args.output}")


if __name__ == "__main__":
    main()
