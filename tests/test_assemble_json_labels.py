import json
import tempfile
import unittest
from pathlib import Path

from data.assemble_json import sample_dataset_records


class AssembleJsonLabelTests(unittest.TestCase):
    def _assemble(self, source_record: dict[str, object]) -> tuple[
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
                dataset_name="FSD50K",
                count=1,
                prompt_file=None,
                seed=42,
                existing_records=[],
                data_root=data_root,
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


if __name__ == "__main__":
    unittest.main()
