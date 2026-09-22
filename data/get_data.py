"""Download and convert a Hugging Face audio dataset for JLens.

Example::

	python data/get_data.py \
		--dataset google/fleurs \
		--config en_us \
		--split train \
		--target 50 \
		--output data/jlens_dataset

The generated directory contains ``audio/*.wav`` and ``prompts.jsonl``. The
paths in ``prompts.jsonl`` are relative to the output directory, which is the
format expected by ``model/instance.py``.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from pathlib import Path
from typing import Any

import librosa
import soundfile as sf
from datasets import Audio, load_dataset


LOGGER = logging.getLogger("get_data")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Download an audio dataset from Hugging Face and create JLens JSONL."
	)
	parser.add_argument("--dataset", required=True, help="Hugging Face dataset id")
	parser.add_argument("--config", default=None, help="Optional dataset config")
	parser.add_argument("--split", default="train", help="Dataset split")
	parser.add_argument("--target", type=int, default=None, help="Maximum samples")
	parser.add_argument("--output", type=Path, default=Path("data/jlens_dataset"))
	parser.add_argument("--audio-column", default="audio")
	parser.add_argument(
		"--text-column",
		default=None,
		help="Text/label column; auto-detect when omitted",
	)
	parser.add_argument(
		"--prompt",
		default="<|audio_bos|><|AUDIO|><|audio_eos|>請寫出這段音訊的逐字稿：",
	)
	parser.add_argument("--min-duration", type=float, default=0.0)
	parser.add_argument("--max-duration", type=float, default=float("inf"))
	parser.add_argument("--sample-rate", type=int, default=16000)
	parser.add_argument(
		"--no-streaming",
		action="store_true",
		help="Load the complete split instead of streaming it",
	)
	parser.add_argument("--overwrite", action="store_true")
	return parser.parse_args()


def decode_audio(audio: dict[str, Any]) -> tuple[Any, int]:
	"""Decode an Audio(decode=False) value without relying on torchcodec."""
	if audio.get("bytes") is not None:
		return sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
	if audio.get("path") is not None:
		return sf.read(audio["path"], dtype="float32")
	raise ValueError("Audio sample has neither bytes nor path")


def save_audio(
	audio: dict[str, Any],
	destination: Path,
	sample_rate: int,
) -> float:
	samples, source_rate = decode_audio(audio)
	duration = len(samples) / source_rate
	if source_rate != sample_rate:
		samples = librosa.resample(
			y=samples, orig_sr=source_rate, target_sr=sample_rate
		)
	destination.parent.mkdir(parents=True, exist_ok=True)
	sf.write(destination, samples, sample_rate)
	return duration


def resolve_text_column(
	dataset: Any,
	requested: str | None,
	dataset_name: str,
) -> str | None:
	columns = list(dataset.column_names)
	if requested is not None:
		if requested not in columns:
			raise KeyError(
				f"Text column {requested!r} not found. Available columns: {columns}"
			)
		return requested

	for candidate in (
		"transcription",
		"transcript",
		"text",
		"sentence",
		"caption",
		"category",
		"label",
	):
		if candidate in columns:
			LOGGER.info("using %r as text column", candidate)
			return candidate

	LOGGER.warning(
		"No text or label column found in %s; transcript will be empty. "
		"Available columns: %s",
		dataset_name,
		columns,
	)
	return None


def download_dataset(args: argparse.Namespace) -> int:
	output_dir = args.output.resolve()
	audio_dir = output_dir / "audio"
	metadata_path = output_dir / "prompts.jsonl"

	if metadata_path.exists() and not args.overwrite:
		raise FileExistsError(
			f"{metadata_path} already exists; use --overwrite to replace it"
		)

	output_dir.mkdir(parents=True, exist_ok=True)
	dataset = load_dataset(
		args.dataset,
		args.config,
		split=args.split,
		streaming=not args.no_streaming,
	)
	dataset = dataset.cast_column(args.audio_column, Audio(decode=False))
	text_column = resolve_text_column(dataset, args.text_column, args.dataset)

	records: list[dict[str, Any]] = []
	for source_index, sample in enumerate(dataset):
		if args.target is not None and len(records) >= args.target:
			break

		audio = sample[args.audio_column]
		source_duration = None
		if args.min_duration > 0 or args.max_duration < float("inf"):
			_, source_rate = decode_audio(audio)
			source_duration = len(decode_audio(audio)[0]) / source_rate
			if not args.min_duration <= source_duration <= args.max_duration:
				continue

		record_id = f"sample-{len(records):06d}"
		audio_path = audio_dir / f"{record_id}.wav"
		duration = save_audio(audio, audio_path, args.sample_rate)
		text = "" if text_column is None else str(sample[text_column])
		text = text.replace("\n", " ").strip()
		records.append(
			{
				"id": record_id,
				"source_index": source_index,
				"audio_path": audio_path.relative_to(output_dir).as_posix(),
				"transcript": text,
				"prompt": args.prompt,
			}
		)
		LOGGER.info("saved %s/%s (%.2fs)", len(records), args.target or "-", duration)

	with metadata_path.open("w", encoding="utf-8") as metadata_file:
		for record in records:
			metadata_file.write(json.dumps(record, ensure_ascii=False) + "\n")

	LOGGER.info("created %s with %d samples", metadata_path, len(records))
	return len(records)


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
	download_dataset(parse_args())


if __name__ == "__main__":
	main()
