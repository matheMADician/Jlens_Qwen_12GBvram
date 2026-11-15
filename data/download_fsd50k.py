"""Download a balanced, single-focus FSD50K pilot subset from Hugging Face."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from tqdm import tqdm

DATASET_ID = "Fhrozen/FSD50k"
LOGGER = logging.getLogger("download_fsd50k")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a balanced subset of selected FSD50K classes."
    )
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_vocabulary(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as vocabulary_file:
        return {
            row[1]
            for row in csv.reader(vocabulary_file)
            if len(row) >= 2
        }


def select_balanced_rows(
    rows: list[dict[str, str]],
    classes: list[str],
    samples_per_class: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select equal counts, excluding rows ambiguous across requested classes."""
    if samples_per_class < 1:
        raise ValueError("--samples-per-class must be a positive integer.")
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("--classes must contain distinct class names.")

    requested_classes = set(classes)
    rows_by_class = {class_name: [] for class_name in classes}
    for row_index, row in enumerate(rows):
        if row.get("split") != "train":
            continue
        labels = [value.strip() for value in row.get("labels", "").split(",")]
        matching_classes = requested_classes.intersection(labels)
        if len(matching_classes) != 1:
            continue
        focus_class = next(iter(matching_classes))
        rows_by_class[focus_class].append(
            {
                "source_index": row_index,
                "fname": row["fname"],
                "labels": labels,
                "focus_class": focus_class,
                "source_split": row["split"],
            }
        )

    insufficient = {
        class_name: len(class_rows)
        for class_name, class_rows in rows_by_class.items()
        if len(class_rows) < samples_per_class
    }
    if insufficient:
        details = ", ".join(
            f"{name}: {count}" for name, count in insufficient.items()
        )
        raise ValueError(
            f"Not enough unambiguous training clips for "
            f"{samples_per_class} samples per class ({details})."
        )

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for class_name in classes:
        class_rows = rows_by_class[class_name]
        selected.extend(rng.sample(class_rows, samples_per_class))
    rng.shuffle(selected)
    return selected


def download_selected_audio(
    rows: list[dict[str, Any]],
    audio_dir: Path,
    *,
    overwrite: bool,
) -> None:
    audio_dir.mkdir(parents=True, exist_ok=True)

    def download_one(row: dict[str, Any]) -> None:
        filename = f"clips/dev/{row['fname']}.wav"
        destination = audio_dir / f"{row['fname']}.wav"
        if destination.exists() and not overwrite:
            return
        cached_path = hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=filename,
        )
        shutil.copyfile(cached_path, destination)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            tqdm(
                executor.map(download_one, rows),
                total=len(rows),
                desc="Downloading selected clips",
                unit="clip",
            )
        )


def write_manifest(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    manifest_path = output_dir / "prompts.jsonl"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_dir,
        prefix=".prompts-",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            for row in rows:
                audio_path = f"audio/{row['fname']}.wav"
                record = {
                    "id": str(row["fname"]),
                    "source_index": row["source_index"],
                    "audio_path": audio_path,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "audio", "path": audio_path},
                                {"type": "text", "text": ""},
                            ],
                        }
                    ],
                    "transcript": "",
                    "label": {
                        "focus_class": row["focus_class"],
                        "labels": row["labels"],
                    },
                    "source_split": row["source_split"],
                    "dataset_name": "FSD50K",
                    "source_dataset": DATASET_ID,
                }
                temporary_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            os.replace(temporary_path, manifest_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
    return manifest_path


def download_fsd50k(args: argparse.Namespace) -> int:
    output_dir = args.output.resolve()
    manifest_path = output_dir / "prompts.jsonl"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{manifest_path} already exists; use --overwrite to replace it."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    vocabulary_path = Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename="labels/vocabulary.csv",
        )
    )
    vocabulary = read_vocabulary(vocabulary_path)
    unknown_classes = sorted(set(args.classes) - vocabulary)
    if unknown_classes:
        raise ValueError(f"Unknown FSD50K classes: {unknown_classes}")

    metadata_path = Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename="labels/dev.csv",
        )
    )
    with metadata_path.open(encoding="utf-8", newline="") as metadata_file:
        rows = list(csv.DictReader(metadata_file))
    selected_rows = select_balanced_rows(
        rows,
        args.classes,
        args.samples_per_class,
        args.seed,
    )
    download_selected_audio(
        selected_rows,
        output_dir / "audio",
        overwrite=args.overwrite,
    )
    manifest_path = write_manifest(selected_rows, output_dir)
    LOGGER.info(
        "Saved %d clips (%d per class, %d classes) to %s",
        len(selected_rows),
        args.samples_per_class,
        len(args.classes),
        manifest_path,
    )
    return len(selected_rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    download_fsd50k(parse_args())


if __name__ == "__main__":
    main()
