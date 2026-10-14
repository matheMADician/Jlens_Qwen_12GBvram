"""Download and convert a Hugging Face audio dataset for JLens.

Arguments:

* ``--dataset``: Hugging Face dataset ID (required).
* ``--config``: Dataset configuration, if required by the dataset.
* ``--split``: Dataset split to download (default: ``train``).
* ``--target``: Maximum number of samples to save; omit to process the full split.
* ``--output``: Output directory for audio and ``prompts.jsonl``.
* ``--audio-column``: Dataset column containing audio (default: ``audio``).
* ``--transcription``: Dataset column for the top-level transcript field;
  auto-detected from transcription-specific column names if omitted.
* ``--prompt-column``: Dataset column for user instructions; prompt is empty if omitted.
* ``--min-duration`` / ``--max-duration``: Optional audio-duration filters in seconds.
* ``--sample-rate``: Output audio sample rate in Hz (default: ``16000``).
* ``--no-streaming``: Load the complete split instead of streaming it.
* ``--overwrite``: Replace an existing ``prompts.jsonl`` file.

The generated directory contains ``audio/*.wav`` and ``prompts.jsonl``. Each
JSONL record includes a model-agnostic ``messages`` conversation; media paths
are relative to the output directory.

Example ``prompts.jsonl`` record::

	{
		"id": "sample-000000",
		"source_index": 0,
		"audio_path": "audio/sample-000000.wav",
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "audio", "path": "audio/sample-000000.wav"},
					{"type": "text", "text": "Dataset-provided instruction"}
				]
			}
		],
		"transcript": "音訊中的逐字稿"
	}

Set ``--prompt-column`` to populate the text part of the user message. If it is
omitted, the text part remains present with an empty string. Transcriptions are
stored in the top-level ``transcript`` field, not as assistant messages.
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
from tqdm import tqdm


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
		"--transcription",
		default=None,
		help="Transcript column; auto-detect when omitted",
	)
	parser.add_argument(
		"--prompt-column",
		default=None,
		help="Dataset column for user instructions; auto-detect when omitted",
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

def resolve_transcription_column(
	dataset: Any,
	requested: str | None,
	dataset_name: str,
) -> str | None:
	columns = list(dataset.column_names)
	if requested is not None:
		if requested not in columns:
			raise KeyError(
				f"Transcription column {requested!r} not found. "
				f"Available columns: {columns}"
			)
		return requested

	for candidate in (
		"transcription",
		"raw_transcription",
		"transcript",
	):
		if candidate in columns:
			LOGGER.info("using %r as transcription column", candidate)
			return candidate

	LOGGER.warning(
		"No transcription column found in %s; assistant messages will be omitted. "
		"Available columns: %s",
		dataset_name,
		columns,
	)
	return None


def resolve_prompt_column(
	dataset: Any,
	requested: str | None,
	dataset_name: str,
) -> str | None:
	columns = list(dataset.column_names)
	if requested is None:
		LOGGER.info(
			"No prompt column specified for %s; user prompts will be empty.",
			dataset_name,
		)
		return None
	if requested not in columns:
		raise KeyError(
			f"Prompt column {requested!r} not found. Available columns: {columns}"
		)
	return requested


def build_messages(
	audio_path: str,
	instruction: str | None,
) -> list[dict[str, Any]]:
	"""Build a model-agnostic conversation using the Hugging Face chat schema."""
	user_content: list[dict[str, str]] = [
		{"type": "audio", "path": audio_path},
		{"type": "text", "text": instruction or ""},
	]
	return [
		{
			"role": "user",
			"content": user_content,
		}
	]


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
	transcription_column = resolve_transcription_column(
		dataset, args.transcription, args.dataset
	)
	prompt_column = resolve_prompt_column(
		dataset, args.prompt_column, args.dataset
	)

	records: list[dict[str, Any]] = []
	progress_total = getattr(dataset, "num_rows", None)
	for source_index, sample in enumerate(
		tqdm(dataset, total=progress_total, desc="Processing dataset", unit="sample")
	):
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
		save_audio(audio, audio_path, args.sample_rate)
		raw_text = (
			None if transcription_column is None else sample[transcription_column]
		)
		text = "" if raw_text is None else " ".join(str(raw_text).split())
		raw_prompt = None if prompt_column is None else sample[prompt_column]
		prompt = (
			None if raw_prompt is None else " ".join(str(raw_prompt).split())
		)
		if not prompt:
			prompt = None
		audio_rel_path = audio_path.relative_to(output_dir).as_posix()
		records.append(
			{
				"id": record_id,
				"source_index": source_index,
				"audio_path": audio_rel_path,
				"messages": build_messages(
					audio_rel_path,
					prompt,
				),
				# Retain these legacy fields until the model-side loader is migrated.
				"transcript": text,
			}
		)

	with metadata_path.open("w", encoding="utf-8") as metadata_file:
		for record in records:
			metadata_file.write(json.dumps(record, ensure_ascii=False) + "\n")

	LOGGER.info("created %s with %d samples", metadata_path, len(records))
	return len(records)

def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
	for logger_name in ("httpx", "httpcore", "urllib3.connectionpool"):
		logging.getLogger(logger_name).setLevel(logging.WARNING)
	download_dataset(parse_args())

if __name__ == "__main__":
	main()
