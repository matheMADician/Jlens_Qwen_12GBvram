import random
import unittest
from collections import Counter

from data.assemble_json import stratified_sample_indices


class AssembleJsonStratificationTests(unittest.TestCase):
    def test_distributes_remainder_across_joint_label_combinations(self) -> None:
        records = []
        index = 0
        for emotion in range(5):
            for language in range(2):
                for gender in range(2):
                    for speaker_id in range(5):
                        for _ in range(10):
                            records.append(
                                {
                                    "id": str(index),
                                    "label": {
                                        "emotion": f"emotion_{emotion}",
                                        "language": f"language_{language}",
                                        "gender": f"gender_{gender}",
                                        "speaker_id": f"speaker_{speaker_id}",
                                    },
                                }
                            )
                            index += 1
        label_columns = ["emotion", "language", "gender", "speaker_id"]

        selected = stratified_sample_indices(
            records,
            list(range(len(records))),
            count=300,
            label_columns=label_columns,
            rng=random.Random(42),
        )
        counts = Counter(
            tuple(records[index]["label"][column] for column in label_columns)
            for index in selected
        )

        self.assertEqual(len(selected), 300)
        self.assertEqual(len(counts), 100)
        self.assertEqual(set(counts.values()), {3})


if __name__ == "__main__":
    unittest.main()
