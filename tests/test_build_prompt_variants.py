import json
import tempfile
import unittest
from pathlib import Path

from data.build_prompt_variants import (
	build_prompt_variant_records,
	load_prompt_variants,
)


def make_record(index: int, emotion: str) -> dict:
	audio = f"datasets/esd/audio/sample-{index:06d}.wav"
	return {
		"id": str(index),
		"audio_path": audio,
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "audio", "path": audio},
					{"type": "text", "text": "original prompt"},
				],
			}
		],
		"label": {"emotion": emotion},
		"source_id": f"sample-{index:06d}",
		"dataset_name": "ESD",
	}


VARIANTS = [
	{"id": "p0", "group": "none", "text": ""},
	{"id": "p1", "group": "emotion", "text": "What is the speaker's tone?"},
	{"id": "p2", "group": "off_domain", "text": "What type of sound is that?"},
]


class BuildPromptVariantsTests(unittest.TestCase):
	def test_cross_product_shares_audio_and_labels_prompts(self) -> None:
		records = [make_record(i, ["anger", "neutral"][i % 2]) for i in range(10)]
		output = build_prompt_variant_records(records, VARIANTS, 4, ["emotion"], seed=1)

		self.assertEqual(len(output), 12)
		self.assertEqual([r["id"] for r in output], [str(i) for i in range(12)])
		audio_to_paths = {}
		for record in output:
			label = record["label"]
			audio_to_paths.setdefault(label["audio_id"], set()).add(record["audio_path"])
			text = record["messages"][0]["content"][1]["text"]
			variant = next(v for v in VARIANTS if v["id"] == label["prompt_id"])
			self.assertEqual(text, variant["text"])
			self.assertEqual(label["prompt_group"], variant["group"])
		self.assertEqual(len(audio_to_paths), 4)
		self.assertTrue(all(len(paths) == 1 for paths in audio_to_paths.values()))
		emotions = [r["label"]["emotion"] for r in output if r["label"]["prompt_id"] == "p0"]
		self.assertEqual(sorted(emotions), ["anger", "anger", "neutral", "neutral"])
		# Source records must not be mutated.
		self.assertEqual(records[0]["messages"][0]["content"][1]["text"], "original prompt")

	def test_rejects_too_many_audio_and_label_clash(self) -> None:
		records = [make_record(i, "anger") for i in range(3)]
		with self.assertRaises(ValueError):
			build_prompt_variant_records(records, VARIANTS, 4, None, seed=0)
		records[0]["label"]["prompt_id"] = "x"
		with self.assertRaises(ValueError):
			build_prompt_variant_records(records, VARIANTS, 3, None, seed=0)

	def test_load_prompt_variants_rejects_duplicates(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "variants.json"
			path.write_text(json.dumps(VARIANTS), encoding="utf-8")
			self.assertEqual(len(load_prompt_variants(path)), 3)
			path.write_text(json.dumps(VARIANTS + [VARIANTS[1]]), encoding="utf-8")
			with self.assertRaises(ValueError):
				load_prompt_variants(path)


if __name__ == "__main__":
	unittest.main()
