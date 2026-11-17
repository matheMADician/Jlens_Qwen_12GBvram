import json
import tempfile
import unittest
from pathlib import Path

from data.build_used_samples import build_used_samples


class BuildUsedSamplesTests(unittest.TestCase):
    def test_writes_unique_source_aware_ids_across_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "train.jsonl"
            second = root / "validation.jsonl"
            output = root / "metadata" / "used_samples.jsonl"
            first.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "0",
                                "source_dataset": "fsd50k_pilot",
                                "source_id": "64742",
                                "source_index": 98,
                            }
                        ),
                        json.dumps(
                            {
                                "id": "1",
                                "source_dataset": "esd",
                                "source_id": "sample-000407",
                                "source_index": 24509,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            second.write_text(
                json.dumps(
                    {
                        "id": "0",
                        "source_dataset": "fsd50k_pilot",
                        "source_id": "64742",
                        "source_index": 98,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            count = build_used_samples([first, second], output)
            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(count, 2)
        self.assertEqual(
            records,
            [
                {
                    "source_dataset": "esd",
                    "source_index": 24509,
                    "source_id": "sample-000407",
                },
                {
                    "source_dataset": "fsd50k_pilot",
                    "source_index": 98,
                    "source_id": "64742",
                },
            ],
        )

    def test_requires_source_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = root / "assembled.jsonl"
            manifest.write_text(
                json.dumps({"id": "0", "source_dataset": "esd"}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "source_index"):
                build_used_samples([manifest], root / "used.jsonl")


if __name__ == "__main__":
    unittest.main()
