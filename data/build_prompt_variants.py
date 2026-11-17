#!/usr/bin/env python3
"""Pair a fixed set of audio clips with every prompt in a prompt-variant list.

The output manifest is a full cross product (audio x prompt), so per-prompt
J-lens features can be compared across text prompts while holding audio fixed.
Each record's ``label`` gains three scalar fields for t-SNE coloring:

* ``audio_id``     - identifies the shared audio clip
* ``prompt_id``    - identifies the text prompt variant
* ``prompt_group`` - coarse prompt category (e.g. emotion, off_domain, none)

Prompt-variant files are JSON lists of ``{"id", "group", "text"}`` objects.
An empty ``text`` produces an audio-only record.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

try:
	from data.assemble_json import (
		load_jsonl,
		set_user_prompt,
		stratified_sample_indices,
		write_dataset_counts,
	)
except ImportError:  # Executed as `python data/build_prompt_variants.py`.
	from assemble_json import (  # type: ignore[no-redef]
		load_jsonl,
		set_user_prompt,
		stratified_sample_indices,
		write_dataset_counts,
	)

RESERVED_LABELS = ("audio_id", "prompt_id", "prompt_group")


def load_prompt_variants(path: Path) -> list[dict[str, str]]:
	"""Load and validate a JSON list of prompt variants."""
	if not path.is_file():
		raise FileNotFoundError(f"Prompt-variant file not found: {path}")
	variants: Any = json.loads(path.read_text(encoding="utf-8"))
	if not isinstance(variants, list) or not variants:
		raise ValueError(f"{path} must contain a non-empty JSON list.")
	seen_ids: set[str] = set()
	seen_texts: set[str] = set()
	for position, variant in enumerate(variants):
		if not isinstance(variant, dict):
			raise ValueError(f"{path} entry {position} must be a JSON object.")
		for key in ("id", "group", "text"):
			if not isinstance(variant.get(key), str):
				raise ValueError(f"{path} entry {position} needs a string {key!r}.")
		if not variant["id"].strip() or not variant["group"].strip():
			raise ValueError(f"{path} entry {position} has an empty id or group.")
		if variant["id"] in seen_ids:
			raise ValueError(f"{path} has duplicate prompt id {variant['id']!r}.")
		text = variant["text"].strip()
		if text in seen_texts:
			raise ValueError(f"{path} has duplicate prompt text {text!r}.")
		seen_ids.add(variant["id"])
		seen_texts.add(text)
	return [
		{"id": v["id"], "group": v["group"], "text": v["text"].strip()}
		for v in variants
	]


def build_prompt_variant_records(
	records: list[dict[str, Any]],
	variants: list[dict[str, str]],
	num_audio: int,
	stratify_labels: list[str] | None,
	seed: int,
) -> list[dict[str, Any]]:
	"""Select audio clips and return one record per (audio, prompt) pair."""
	if num_audio <= 0:
		raise ValueError("--num-audio must be positive.")
	unique_indices: list[int] = []
	seen_audio: set[str] = set()
	for index, record in enumerate(records):
		audio_path = record.get("audio_path")
		if not isinstance(audio_path, str) or not audio_path:
			raise ValueError(f"Record {record.get('id', index)!r} has no audio_path.")
		if audio_path not in seen_audio:
			seen_audio.add(audio_path)
			unique_indices.append(index)
	if num_audio > len(unique_indices):
		raise ValueError(
			f"Requested {num_audio} audio clips, but only {len(unique_indices)} "
			"unique clips are available."
		)

	rng = random.Random(seed)
	if stratify_labels:
		selected = stratified_sample_indices(
			records, unique_indices, num_audio, stratify_labels, rng
		)
	else:
		selected = rng.sample(unique_indices, num_audio)
	# Order clips by original position so the manifest is stable and readable.
	selected.sort()

	output: list[dict[str, Any]] = []
	for audio_number, index in enumerate(selected):
		source = records[index]
		label = source.get("label", {})
		if not isinstance(label, dict):
			raise ValueError(f"Record {source.get('id', index)!r} has no label mapping.")
		clash = [key for key in RESERVED_LABELS if key in label]
		if clash:
			raise ValueError(
				f"Record {source.get('id', index)!r} already defines label fields {clash}."
			)
		audio_id = f"a{audio_number:02d}_{source.get('source_id', source.get('id', index))}"
		for variant in variants:
			record = copy.deepcopy(source)
			record["id"] = str(len(output))
			record["variant_source_id"] = str(source.get("id", index))
			record["prompt_text"] = variant["text"]
			record["label"] = {
				**label,
				"audio_id": audio_id,
				"prompt_id": variant["id"],
				"prompt_group": variant["group"],
			}
			set_user_prompt(record, variant["text"])
			output.append(record)
	return output


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--input", type=Path, required=True, help="Assembled JSONL to draw audio from.")
	parser.add_argument("--prompts", type=Path, required=True, help="Prompt-variant JSON list.")
	parser.add_argument("--num-audio", type=int, required=True)
	parser.add_argument("--stratify-labels", nargs="+", default=None)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--output", type=Path, required=True, help="Output directory.")
	parser.add_argument(
		"--override",
		action="store_true",
		help="Replace an existing prompts.jsonl in --output.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	manifest_path = args.output / "prompts.jsonl"
	if manifest_path.exists() and not args.override:
		raise FileExistsError(f"{manifest_path} exists; pass --override to replace it.")
	records = load_jsonl(args.input)
	variants = load_prompt_variants(args.prompts)
	output = build_prompt_variant_records(
		records, variants, args.num_audio, args.stratify_labels, args.seed
	)
	args.output.mkdir(parents=True, exist_ok=True)
	with manifest_path.open("w", encoding="utf-8") as handle:
		for record in output:
			handle.write(json.dumps(record, ensure_ascii=False) + "\n")
	write_dataset_counts(manifest_path, args.output / "dataset_counts.txt")
	print(
		f"Wrote {len(output)} records ({args.num_audio} audio x "
		f"{len(variants)} prompts) to {manifest_path}"
	)


if __name__ == "__main__":
	main()
