import json
import tempfile
import unittest
from pathlib import Path

from data.download_fsd50k import select_balanced_rows, write_manifest
from json_parser import Parser


class Fsd50kDownloadTests(unittest.TestCase):
    def test_selects_equal_unambiguous_training_rows(self) -> None:
        rows = [
            {"fname": "1", "labels": "Dog,Animal", "split": "train"},
            {"fname": "2", "labels": "Dog,Animal", "split": "train"},
            {"fname": "3", "labels": "Cat,Animal", "split": "train"},
            {"fname": "4", "labels": "Dog,Cat,Animal", "split": "train"},
            {"fname": "5", "labels": "Cat,Animal", "split": "val"},
        ]

        selected = select_balanced_rows(
            rows,
            classes=["Dog", "Cat"],
            samples_per_class=1,
            seed=42,
        )

        self.assertEqual(
            {row["focus_class"] for row in selected},
            {"Dog", "Cat"},
        )
        self.assertNotIn("4", {row["fname"] for row in selected})
        self.assertNotIn("5", {row["fname"] for row in selected})

    def test_manifest_labels_parse_with_the_shared_json_parser(self) -> None:
        rows = [
            {
                "fname": "123",
                "labels": ["Dog", "Animal"],
                "focus_class": "Dog",
                "source_index": 0,
                "source_split": "train",
            }
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = write_manifest(rows, Path(temporary_directory))
            record = json.loads(manifest_path.read_text(encoding="utf-8"))

        sample = Parser.parse_json_line(json.dumps(record))
        self.assertEqual(sample.sample_id, "123")
        self.assertEqual(
            sample.label,
            {"focus_class": "Dog", "labels": ["Dog", "Animal"]},
        )

    def test_requires_enough_examples_for_each_class(self) -> None:
        with self.assertRaisesRegex(ValueError, "Not enough"):
            select_balanced_rows(
                [{"fname": "1", "labels": "Dog", "split": "train"}],
                classes=["Dog", "Cat"],
                samples_per_class=1,
                seed=42,
            )


if __name__ == "__main__":
    unittest.main()
