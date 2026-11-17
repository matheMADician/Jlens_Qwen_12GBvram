import json
import tempfile
import unittest
from pathlib import Path

from data.assemble_json import sample_dataset_records


class AssembleJsonLabelTests(unittest.TestCase):
    def _assemble(
        self,
        source_record: dict[str, object],
        dataset_name: str = "FSD50K",
        label_overrides: dict[str, str] | None = None,
    ) -> tuple[
        dict[str, object], dict[str, object]
    ]:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_root = Path(temporary_directory) / "data"
            source_root = data_root / "datasets" / "fsd50k_pilot"
            (source_root / "audio").mkdir(parents=True)
            (source_root / "audio" / "123.wav").write_bytes(b"audio")
            (source_root / "prompts.jsonl").write_text(
                json.dumps(source_record) + "\n",
                encoding="utf-8",
            )

            output_records = sample_dataset_records(
                dataset_dir=source_root,
                dataset_name=dataset_name,
                count=1,
                prompt_file=None,
                seed=42,
                existing_records=[],
                data_root=data_root,
                label_overrides=label_overrides,
            )
            return output_records[0], source_record

    def test_preserves_named_and_multi_value_labels(self) -> None:
        source_record: dict[str, object] = {
            "id": "123",
            "audio_path": "audio/123.wav",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "path": "audio/123.wav"},
                        {"type": "text", "text": "Describe this sound."},
                    ],
                }
            ],
            "transcript": "",
            "label": {
                "focus_class": "Dog",
                "labels": ["Dog", "Domestic_animals_and_pets"],
            },
        }

        assembled, original = self._assemble(source_record)

        self.assertEqual(assembled["label"], source_record["label"])
        self.assertEqual(assembled["source_id"], "123")
        self.assertEqual(
            original["label"],
            {
                "focus_class": "Dog",
                "labels": ["Dog", "Domestic_animals_and_pets"],
            },
        )

    def test_normalizes_esd_emotion_and_language_values(self) -> None:
        source_record: dict[str, object] = {
            "id": "123",
            "audio_path": "audio/123.wav",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "path": "audio/123.wav"},
                        {"type": "text", "text": "Describe this speech."},
                    ],
                }
            ],
            "label": {
                "emotion": "sadness",
                "language": "zh",
                "gender": "female",
                "speaker_id": "0001",
            },
        }

        assembled, original = self._assemble(source_record, dataset_name="ESD")

        self.assertEqual(
            assembled["label"],
            {
                "emotion": "sad",
                "language": "chinese",
                "gender": "female",
                "speaker_id": "0001",
            },
        )
        self.assertEqual(original["label"]["emotion"], "sadness")
        self.assertEqual(original["label"]["language"], "zh")

    def test_applies_label_overrides_without_mutating_source_record(self) -> None:
        source_record: dict[str, object] = {
            "id": "123",
            "audio_path": "audio/123.wav",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "path": "audio/123.wav"},
                        {"type": "text", "text": "Describe this speech."},
                    ],
                }
            ],
            "label": {"speaker_id": "speaker-1"},
        }

        assembled, original = self._assemble(
            source_record,
            dataset_name="de_de",
            label_overrides={"language": "german"},
        )

        self.assertEqual(
            assembled["label"],
            {"speaker_id": "speaker-1", "language": "german"},
        )
        self.assertEqual(original["label"], {"speaker_id": "speaker-1"})

    def test_legacy_record_gets_empty_label_mapping(self) -> None:
        source_record: dict[str, object] = {
            "id": "123",
            "audio_path": "audio/123.wav",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "path": "audio/123.wav"},
                        {"type": "text", "text": "Describe this sound."},
                    ],
                }
            ],
        }

        assembled, _ = self._assemble(source_record)

        self.assertEqual(assembled["label"], {})

    def test_used_samples_excludes_matching_source_dataset_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_root = Path(temporary_directory) / "data"
            source_root = data_root / "datasets" / "fsd50k_pilot"
            (source_root / "audio").mkdir(parents=True)
            records = []
            for sample_id, source_index in (("123", 10), ("456", 11)):
                (source_root / "audio" / f"{sample_id}.wav").write_bytes(b"audio")
                records.append(
                    {
                        "id": sample_id,
                        "source_index": source_index,
                        "audio_path": f"audio/{sample_id}.wav",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "audio",
                                        "path": f"audio/{sample_id}.wav",
                                    },
                                    {"type": "text", "text": "Describe this sound."},
                                ],
                            }
                        ],
                    }
                )
            (source_root / "prompts.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            assembled = sample_dataset_records(
                dataset_dir=source_root,
                dataset_name="FSD50K",
                count=1,
                prompt_file=None,
                seed=42,
                existing_records=[],
                data_root=data_root,
                used_samples={("fsd50k_pilot", 10)},
            )

        self.assertEqual(len(assembled), 1)
        self.assertEqual(assembled[0]["source_index"], 11)
        self.assertEqual(assembled[0]["source_id"], "456")

    def test_explicit_empty_prompt_allows_sources_without_text_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_root = Path(temporary_directory) / "data"
            source_root = data_root / "datasets" / "fsd50k_pilot"
            (source_root / "audio").mkdir(parents=True)
            (source_root / "audio" / "123.wav").write_bytes(b"audio")
            source_record: dict[str, object] = {
                "id": "123",
                "source_index": 10,
                "audio_path": "audio/123.wav",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "path": "audio/123.wav"},
                            {"type": "text", "text": ""},
                        ],
                    }
                ],
            }
            (source_root / "prompts.jsonl").write_text(
                json.dumps(source_record) + "\n",
                encoding="utf-8",
            )

            assembled = sample_dataset_records(
                dataset_dir=source_root,
                dataset_name="FSD50K",
                count=1,
                prompt_file=None,
                seed=42,
                existing_records=[],
                data_root=data_root,
                empty_prompt=True,
            )

        self.assertEqual(
            assembled[0]["messages"][0]["content"][1]["text"],
            "",
        )


if __name__ == "__main__":
    unittest.main()
