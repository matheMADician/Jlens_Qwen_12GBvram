"""Build stroop stimuli from LibriSpeech: speaker gender vs content gender.

Streams ``openslr/librispeech_asr``, keeps utterances whose transcript refers to
one gender, labels the speaker's gender from ``SPEAKERS.TXT``, and fills a
balanced 2x2 design: {female, male voice} x {female, male content}.

``SPEAKERS.TXT`` ships in every LibriSpeech split archive. One way to get it:

    curl -L https://openslr.trmal.net/resources/12/dev-clean.tar.gz \
        | tar -xz LibriSpeech/SPEAKERS.TXT

Example (run from the repo root):

    python -m stroop.data_prep.get_stroop_libri \
        --speakers-file LibriSpeech/SPEAKERS.TXT \
        --config clean --split train.100 \
        --per-cell 100 \
        --output data/datasets/stroop/libri

Output: ``<output>/audio/*.wav``, ``<output>/prompts.jsonl`` (trunk schema plus
a ``label`` object), and ``<output>/metadata.json`` (arguments and cell counts).
The text prompt is left empty; fill it later with ``data/assemble_json.py
--prompt-file stroop/prompts/<file>.txt``.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stroop.data_prep.gender_labels import (
    FEMALE,
    MALE,
    cell_name,
    content_gender,
    is_conflict,
    parse_speakers_file,
)

LOGGER = logging.getLogger("stroop.data_prep.get_stroop_libri")
CELLS = [(voice, text) for voice in (FEMALE, MALE) for text in (FEMALE, MALE)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--speakers-file", type=Path, required=True,
                        help="LibriSpeech SPEAKERS.TXT")
    parser.add_argument("--dataset", default="openslr/librispeech_asr")
    parser.add_argument("--config", default="clean", help="clean | other | all")
    parser.add_argument("--split", default="train.100")
    parser.add_argument("--per-cell", type=int, default=100,
                        help="Utterances to keep in each of the 4 cells")
    parser.add_argument("--first-person-only", action="store_true",
                        help="Keep only 'I am a man/woman' style self-references")
    parser.add_argument("--min-gender-words", type=int, default=1,
                        help="Minimum gendered words in a third-person transcript")
    parser.add_argument("--max-per-speaker", type=int, default=5,
                        help="Cap per speaker and cell so a few readers do not dominate")
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=10.0,
                        help="Seconds; long audio inflates sequence length")
    parser.add_argument("--max-scan", type=int, default=None,
                        help="Stop after scanning this many utterances")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--output", type=Path, default=Path("data/datasets/stroop/libri"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.per_cell < 1:
        parser.error("--per-cell must be at least 1")
    if args.min_gender_words < 1:
        parser.error("--min-gender-words must be at least 1")
    return args


def select(
    text: str,
    speaker_id: int,
    speakers: dict[int, str],
    *,
    first_person_only: bool,
    min_gender_words: int,
) -> dict[str, Any] | None:
    """Return the label for an utterance, or None if it does not fit the design."""
    speaker_gender = speakers.get(speaker_id)
    if speaker_gender is None:
        return None
    found = content_gender(text)
    if found.gender not in (MALE, FEMALE):
        return None
    if first_person_only and found.person != "first":
        return None
    gender_words = found.male_words if found.gender == MALE else found.female_words
    if found.person == "third" and len(gender_words) < min_gender_words:
        return None
    return {
        "speaker_gender": speaker_gender,
        "text_gender": found.gender,
        "conflict": is_conflict(speaker_gender, found.gender),
        "person": found.person,
        "cell": cell_name(speaker_gender, found.gender),
        "speaker_id": speaker_id,
        "gender_words": list(gender_words),
    }


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build(args: argparse.Namespace) -> Counter:
    # Heavy imports stay here so the selection logic can be tested without them.
    from datasets import Audio, load_dataset
    from tqdm import tqdm

    from data.get_data import build_messages, decode_audio, save_audio

    output_dir = args.output.resolve()
    metadata_path = output_dir / "prompts.jsonl"
    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metadata_path} already exists; use --overwrite to replace it")

    speakers = parse_speakers_file(args.speakers_file)
    LOGGER.info("Loaded %d speakers from %s", len(speakers), args.speakers_file)

    dataset = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    counts: Counter = Counter()
    per_speaker: Counter = Counter()
    skipped: Counter = Counter()
    records: list[dict[str, Any]] = []
    target_total = args.per_cell * len(CELLS)
    output_dir.mkdir(parents=True, exist_ok=True)

    scanned = 0
    progress = tqdm(desc="Scanning", unit="utt")
    for source_index, sample in enumerate(dataset):
        if args.max_scan is not None and scanned >= args.max_scan:
            break
        scanned += 1
        progress.update()
        label = select(
            sample["text"],
            int(sample["speaker_id"]),
            speakers,
            first_person_only=args.first_person_only,
            min_gender_words=args.min_gender_words,
        )
        if label is None:
            skipped["no_match"] += 1
            continue
        cell = label["cell"]
        if counts[cell] >= args.per_cell:
            skipped["cell_full"] += 1
            continue
        if per_speaker[(label["speaker_id"], cell)] >= args.max_per_speaker:
            skipped["speaker_cap"] += 1
            continue

        samples, source_rate = decode_audio(sample["audio"])
        duration = len(samples) / source_rate
        if not args.min_duration <= duration <= args.max_duration:
            skipped["duration"] += 1
            continue

        record_id = f"stroop-libri-{len(records):06d}"
        audio_rel_path = f"audio/{record_id}.wav"
        save_audio(sample["audio"], output_dir / audio_rel_path, args.sample_rate)
        records.append({
            # The model adapters require decimal ids (Instance.encode checks isdecimal()).
            "id": str(len(records)),
            "source_index": source_index,
            "audio_path": audio_rel_path,
            "messages": build_messages(audio_rel_path, None),
            "transcript": " ".join(str(sample["text"]).split()),
            "dataset_name": "stroop_libri",
            "source_dataset": f"{args.dataset}/{args.config}/{args.split}",
            "source_id": sample.get("id"),
            "duration": round(duration, 3),
            "label": label,
        })
        counts[cell] += 1
        per_speaker[(label["speaker_id"], cell)] += 1
        progress.set_postfix(dict(counts))
        if len(records) >= target_total:
            break
    progress.close()

    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    cell_counts = {cell_name(*c): counts[cell_name(*c)] for c in CELLS}
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "scanned": scanned,
        "kept": len(records),
        "cells": cell_counts,
        "skipped": dict(skipped),
        "speakers_per_cell": {
            cell_name(*c): len({sid for sid, cell in per_speaker if cell == cell_name(*c)})
            for c in CELLS
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("Wrote %d records to %s; cells: %s", len(records), metadata_path, cell_counts)
    short = [c for c, n in cell_counts.items() if n < args.per_cell]
    if short:
        LOGGER.warning("Cells below --per-cell %d: %s. Scan more data or relax filters.",
                       args.per_cell, short)
    return counts


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for name in ("httpx", "httpcore", "urllib3.connectionpool"):
        logging.getLogger(name).setLevel(logging.WARNING)
    build(parse_args(argv))


if __name__ == "__main__":
    main()
