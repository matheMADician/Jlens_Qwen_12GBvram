import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from data.get_data import download_dataset, stratified_sample_indices


class _ClassLabel:
    def __init__(self, names: list[str]) -> None:
        self.names = names

    def int2str(self, value: int) -> str:
        return self.names[value]


class _SequenceFeature:
    def __init__(self, feature: _ClassLabel) -> None:
        self.feature = feature


class _Dataset:
    column_names = ["audio", "transcript", "emotion", "tags"]
    num_rows = 1
    features = {
        "emotion": _ClassLabel(["neutral", "happy"]),
        "tags": _SequenceFeature(_ClassLabel(["bird", "animal"])),
    }

    def __init__(self) -> None:
        self.row = {
            "audio": {"path": "unused.wav", "bytes": None},
            "transcript": "A bird sings.",
            "emotion": 1,
            "tags": [0, 1],
        }

    def cast_column(self, column: str, feature: object) -> "_Dataset":
        return self

    def __iter__(self):
        return iter([self.row])

    def __len__(self) -> int:
        return self.num_rows


class GetDataLabelTests(unittest.TestCase):
    def test_writes_selected_label_columns_as_a_mapping(self) -> None:
        dataset = _Dataset()
        with tempfile.TemporaryDirectory() as temporary_directory:
            args = SimpleNamespace(
                output=Path(temporary_directory),
                stratify_column=None,
                stratify_columns=None,
                target=None,
                overwrite=False,
                no_streaming=True,
                dataset="test/dataset",
                config=None,
                split="train",
                audio_column="audio",
                transcription="transcript",
                prompt_column=None,
                label_columns=["emotion", "tags"],
                min_duration=0.0,
                max_duration=float("inf"),
                sample_rate=16000,
            )
            with (
                patch("data.get_data.load_dataset", return_value=dataset),
                patch("data.get_data.save_audio", return_value=1.5),
            ):
                count = download_dataset(args)

            output_path = Path(temporary_directory) / "prompts.jsonl"
            record = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(count, 1)
        self.assertEqual(record["transcript"], "A bird sings.")
        self.assertEqual(
            record["label"],
            {"emotion": "happy", "tags": ["bird", "animal"]},
        )

    def test_stratifies_equally_across_joint_label_combinations(self) -> None:
        strata = [
            (emotion, language)
            for emotion in ("calm", "angry")
            for language in ("en", "zh")
            for _ in range(3)
        ]

        selected = stratified_sample_indices(strata, target=8, seed=42)
        counts = Counter(strata[index] for index in selected)

        self.assertEqual(set(counts.values()), {2})

    def test_joint_stratification_rounds_target_up_to_multiple_of_strata(self) -> None:
        strata = [label for label in range(9) for _ in range(56)]

        selected = stratified_sample_indices(strata, target=500, seed=42)

        self.assertEqual(len(selected), 504)
        self.assertEqual(set(Counter(strata[index] for index in selected).values()), {56})


if __name__ == "__main__":
    unittest.main()
