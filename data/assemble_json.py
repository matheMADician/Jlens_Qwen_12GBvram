#!/usr/bin/env python3
"""Sample one dataset into an assembled JSONL manifest.
Jsonl example:
{
    "id": "sample-000000",
    "source_index": 0,
    "audio_path": "audio/sample-000000.wav",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": "audio/sample-000000.wav"},
                {"type": "text", "text": ""}
            ]
        }
    ],
    "transcript": "dog"
}

Examples:
    # Use prompts already present in the dataset's user messages.
    python data/assemble_json.py \
        --dataset data/datasets/SAKURA/AnimalQA \
        --count 5 \
        --seed 42 \
        --output data/test

    # Assign random prompts from a text file to the sampled records.
    python data/assemble_json.py \
        --dataset data/datasets/Clotho \
        --count 50 \
        --prompt-file data/llama/environment_sound_prompt.txt \
        --seed 42 \
        --output data/assembled

The source directory must contain ``prompts.jsonl`` and the audio referenced
by its records. Audio is not copied; paths in the output are relative to the
data root. Existing output is appended unless ``--override`` is specified.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

PROMPT_LINE = re.compile(r"^\s*\d+[.)]\s+(.+?)\s*$")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
	"""Load JSONL records with line-numbered errors."""
	if not path.is_file():
		raise FileNotFoundError(f"JSONL file not found: {path}")

	records: list[dict[str, Any]] = []
	with path.open("r", encoding="utf-8") as handle:
		for line_number, line in enumerate(handle, start=1):
			line = line.strip()
			if not line:
				continue
			try:
				record = json.loads(line)
			except json.JSONDecodeError as exc:
				raise ValueError(
					f"Invalid JSON in {path}, line {line_number}: {exc.msg}"
				) from exc
			if not isinstance(record, dict):
				raise ValueError(
					f"Expected a JSON object in {path}, line {line_number}"
				)
			records.append(record)
	return records


def load_prompts(path: Path) -> list[str]:
	"""
        Load prompts from numbered lists or one-prompt-per-line text files.
        Llama outputs tend to have a mix of numbered prompts and plain text lines,
        which this function can parse.
    """

	if not path.is_file():
		raise FileNotFoundError(f"Prompt file not found: {path}")

	lines = path.read_text(encoding="utf-8").splitlines()
	numbered_prompts = [
		match.group(1).strip()
		for line in lines
		if (match := PROMPT_LINE.match(line)) and match.group(1).strip()
	]
	if numbered_prompts:
		prompts = numbered_prompts
	else:
		prompts = [
			line.strip()
			for line in lines
			if line.strip() and not line.lstrip().startswith("#")
		]

	if not prompts:
		raise ValueError(f"No prompts found in {path}")
	return prompts


def write_dataset_counts(manifest_path: Path, counts_path: Path) -> None:
	"""Write per-dataset record totals for the complete assembled manifest."""
	counts: Counter[str] = Counter()
	for record in load_jsonl(manifest_path):
		dataset_name = record.get("dataset_name")
		label = (
			dataset_name.strip()
			if isinstance(dataset_name, str) and dataset_name.strip()
			else "<missing dataset_name>"
		)
		counts[label] += 1

	with counts_path.open("w", encoding="utf-8") as handle:
		for dataset_name, count in sorted(counts.items()):
			handle.write(f"{dataset_name}\t{count}\n")


def get_user_prompt(record: dict[str, Any]) -> str | None:
	"""Return the user text prompt, including a legacy top-level prompt."""
	messages = record.get("messages")
	if isinstance(messages, list):
		for message in messages:
			if not isinstance(message, dict) or message.get("role") != "user":
				continue
			content = message.get("content")
			if not isinstance(content, list):
				continue
			for part in content:
				if isinstance(part, dict) and part.get("type") == "text":
					text = part.get("text")
					if isinstance(text, str) and text.strip():
						return text.strip()

	legacy_prompt = record.get("prompt")
	if isinstance(legacy_prompt, str) and legacy_prompt.strip():
		return legacy_prompt.strip()
	return None


def set_user_prompt(record: dict[str, Any], prompt: str) -> None:
	"""Set the text part of the first user message, creating it if needed."""
	messages = record.get("messages")
	if not isinstance(messages, list):
		raise ValueError(
			f"Record {record.get('id', '<unknown>')!r} has no messages list"
		)

	for message in messages:
		if not isinstance(message, dict) or message.get("role") != "user":
			continue
		content = message.get("content")
		if not isinstance(content, list):
			raise ValueError(
				f"User message in record {record.get('id', '<unknown>')!r} "
				"must have list content"
			)
		for part in content:
			if isinstance(part, dict) and part.get("type") == "text":
				part["text"] = prompt
				record.pop("prompt", None)
				return
		content.append({"type": "text", "text": prompt})
		record.pop("prompt", None)
		return

	raise ValueError(
		f"Record {record.get('id', '<unknown>')!r} has no user message"
	)


def rebase_audio_paths(
	record: dict[str, Any],
	source_root: Path,
	data_root: Path,
	manifest_path: Path,
) -> None:
	"""Validate source audio and rewrite paths relative to the data root."""
	audio_rel = record.get("audio_path")
	if not isinstance(audio_rel, str) or not audio_rel:
		raise ValueError(f"Record missing 'audio_path' in {manifest_path}")

	source_file = (source_root / audio_rel).resolve()
	if not source_file.is_file():
		raise FileNotFoundError(f"Audio not found for {manifest_path}: {source_file}")
	try:
		combined_audio_path = source_file.relative_to(data_root).as_posix()
	except ValueError as exc:
		raise ValueError(
			f"Audio file {source_file} is outside data root {data_root}; "
			"pass a data root containing both the dataset and output paths"
		) from exc
	record["audio_path"] = combined_audio_path

	messages = record.get("messages")
	if not isinstance(messages, list):
		raise ValueError(
			f"Record {record.get('id', '<unknown>')!r} has no messages list"
		)
	for message in messages:
		if not isinstance(message, dict):
			continue
		content = message.get("content")
		if not isinstance(content, list):
			continue
		for part in content:
			if isinstance(part, dict) and part.get("type") == "audio":
				part["path"] = combined_audio_path


def normalize_existing_audio_path(
	record: dict[str, Any],
	output_dir: Path,
	data_root: Path,
	manifest_path: Path,
) -> None:
	"""Convert an existing output record's audio path to data-root-relative."""
	audio_rel = record.get("audio_path")
	if not isinstance(audio_rel, str) or not audio_rel:
		raise ValueError(f"Record missing 'audio_path' in {manifest_path}")

	candidates = (
		(data_root / audio_rel).resolve(),
		(output_dir / audio_rel).resolve(),
	)
	audio_file = next((path for path in candidates if path.is_file()), None)
	if audio_file is None:
		raise FileNotFoundError(
			f"Audio not found for {manifest_path}: tried "
			+ " and ".join(str(path) for path in candidates)
		)
	try:
		relative_audio_path = audio_file.relative_to(data_root).as_posix()
	except ValueError as exc:
		raise ValueError(
			f"Audio file {audio_file} is outside data root {data_root}"
		) from exc

	record["audio_path"] = relative_audio_path
	messages = record.get("messages")
	if not isinstance(messages, list):
		raise ValueError(
			f"Record {record.get('id', '<unknown>')!r} has no messages list"
		)
	for message in messages:
		if not isinstance(message, dict):
			continue
		content = message.get("content")
		if not isinstance(content, list):
			continue
		for part in content:
			if isinstance(part, dict) and part.get("type") == "audio":
				part["path"] = relative_audio_path


def dataset_label(source_root: Path) -> str:
	"""Keep nested dataset identities such as SAKURA/AnimalQA intact."""
	for parent in source_root.parents:
		if parent.name == "datasets":
			return source_root.relative_to(parent).as_posix()
	return source_root.name


def infer_data_root(source_root: Path) -> Path:
	"""Infer the data root as the parent of the dataset collection directory."""
	for parent in source_root.parents:
		if parent.name == "datasets":
			return parent.parent
	raise ValueError(
		f"Cannot infer data root from dataset path {source_root}; "
		"pass --data-root explicitly"
	)


def sample_dataset_records(
	dataset_dir: Path,
	dataset_name: str,
	count: int,
	prompt_file: Path | None,
	seed: int,
	existing_records: list[dict[str, Any]],
	data_root: Path | None = None,
) -> list[dict[str, Any]]:
	dataset_name = dataset_name.strip()
	if not dataset_name:
		raise ValueError("--dataset_name must not be empty")
	if count < 1:
		raise ValueError("--count must be at least 1")

	source_root = dataset_dir.expanduser().resolve()
	root = (data_root.expanduser().resolve() if data_root else infer_data_root(source_root))
	manifest_path = source_root / "prompts.jsonl"
	records = load_jsonl(manifest_path)
	if not records:
		raise ValueError(f"No records found in {manifest_path}")

	prompts = load_prompts(prompt_file.expanduser()) if prompt_file else None
	if prompts is None:
		candidate_indices = [
			index for index, record in enumerate(records)
			if get_user_prompt(record) is not None
		]
		if not candidate_indices:
			raise ValueError(
				f"No non-empty user prompts found in {manifest_path}; "
				"provide --prompt-file"
			)
	else:
		candidate_indices = list(range(len(records)))

	rng = random.Random(seed)
	selected_indices = rng.sample(candidate_indices, count)
	source_dataset = dataset_label(source_root)
	if prompts is not None:
		selected_prompts = [rng.choice(prompts) for _ in selected_indices]
	else:
		selected_prompts = [
			get_user_prompt(records[index]) or "" for index in selected_indices
		]

	next_id = max(
		(
			int(record["id"])
			for record in existing_records
			if str(record.get("id", "")).isdecimal()
		),
		default=-1,
	) + 1
	output_records: list[dict[str, Any]] = []
	for offset, (index, prompt) in enumerate(zip(selected_indices, selected_prompts)):
		record = copy.deepcopy(records[index])
		original_id = str(record.get("id", index))
		record["id"] = str(next_id + offset)
		record["source_id"] = original_id
		record["dataset_name"] = dataset_name
		record["source_dataset"] = source_dataset
		set_user_prompt(record, prompt)
		rebase_audio_paths(record, source_root, root, manifest_path)
		output_records.append(record)
	return output_records


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Sample one dataset and append records to an assembled JSONL."
	)
	parser.add_argument(
		"--dataset",
		type=Path,
		required=True,
		help="Dataset directory containing prompts.jsonl and its audio files",
	)
	parser.add_argument(
		"--dataset_name",
		required=True,
		help="Dataset label to write on every sampled JSONL record",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=0,
		help="Seed used to sample records and prompts (default: 0)",
	)
	parser.add_argument(
		"--count",
		type=int,
		required=True,
		help="Number of dataset records to sample",
	)
	parser.add_argument(
		"--prompt-file",
		type=Path,
		default=None,
		help=(
			"Optional text file of prompts. If omitted, records must already "
			"contain non-empty user prompts."
		),
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("data/assembled"),
		help="Output directory for prompts.jsonl (default: data/assembled)",
	)
	parser.add_argument(
		"--data-root",
		type=Path,
		default=None,
		help=(
			"Root that contains the source dataset and is used to make audio "
			"paths in the JSONL relative. Defaults to the parent of the nearest "
			"'datasets' directory in --dataset."
		),
	)
	parser.add_argument(
		"--override",
		action="store_true",
		help="Replace the output prompts.jsonl instead of appending",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	output_dir = args.output.expanduser().resolve()
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "prompts.jsonl"
	source_root = args.dataset.expanduser().resolve()
	data_root = (
		args.data_root.expanduser().resolve()
		if args.data_root is not None
		else infer_data_root(source_root)
	)

	existing_records = (
		[] if args.override or not output_path.exists() else load_jsonl(output_path)
	)
	for record in existing_records:
		normalize_existing_audio_path(record, output_dir, data_root, output_path)
	new_records = sample_dataset_records(
		dataset_dir=args.dataset,
		dataset_name=args.dataset_name,
		count=args.count,
		prompt_file=args.prompt_file,
		seed=args.seed,
		existing_records=existing_records,
		data_root=data_root,
	)

	all_records = existing_records + new_records
	with output_path.open("w", encoding="utf-8") as handle:
		for record in all_records:
			handle.write(json.dumps(record, ensure_ascii=False) + "\n")

	action = "Replaced" if args.override else "Appended"
	print(f"{action} {len(new_records)} records in {output_path}")
	counts_path = output_dir / "dataset_counts.txt"
	write_dataset_counts(output_path, counts_path)
	print(f"Updated per-dataset sample counts in {counts_path}")


if __name__ == "__main__":
	main()
