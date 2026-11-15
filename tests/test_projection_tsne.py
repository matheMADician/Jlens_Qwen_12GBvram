import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from tools.projection_tsne import create_projection_tsne


class _FakeTSNE:
    inputs: list[np.ndarray] = []

    def __init__(self, **kwargs: object) -> None:
        pass

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        self.inputs.append(features.copy())
        return np.zeros((features.shape[0], 2), dtype=np.float32)


class ProjectionTsneTests(unittest.TestCase):
    def test_saves_raw_and_per_prompt_l2_normalized_embeddings(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            projected_root = Path(temporary_directory)
            projected_dir = projected_root / "dim_2_seed_42"
            projected_dir.mkdir()
            features = np.asarray(
                [
                    [3.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [2.0, 2.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
            np.savez_compressed(
                projected_dir / "dataset_test.npz",
                features=features,
                sample_ids=np.asarray(["1", "2", "3", "4"]),
                dataset_labels=np.asarray(["A", "A", "B", "B"]),
                sample_labels_json=np.asarray(
                    [
                        '{"emotion":"happy"}',
                        '{"emotion":"sad"}',
                        '{"focus_class":"Dog"}',
                        '{"focus_class":"Cat"}',
                    ]
                ),
                source_layers=np.asarray([1]),
                original_dim=np.asarray(8),
                projection_dim=np.asarray(2),
                projection_seed=np.asarray(42),
                projection_method=np.asarray("gaussian"),
            )
            _FakeTSNE.inputs = []

            with patch("sklearn.manifold.TSNE", _FakeTSNE):
                outputs = create_projection_tsne(projected_root, perplexity=2)

            paths = outputs[projected_dir]
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(path.is_file() for path in paths))
            np.testing.assert_array_equal(_FakeTSNE.inputs[0], features)
            np.testing.assert_allclose(
                np.linalg.norm(_FakeTSNE.inputs[1], axis=1),
                np.ones(features.shape[0]),
            )

            with np.load(paths[3], allow_pickle=False) as normalized:
                self.assertEqual(
                    str(normalized["feature_normalization"].item()),
                    "l2_per_prompt",
                )
                self.assertEqual(normalized["embedding"].shape, (4, 2))
                np.testing.assert_array_equal(
                    normalized["sample_labels_json"],
                    [
                        '{"emotion":"happy"}',
                        '{"emotion":"sad"}',
                        '{"focus_class":"Dog"}',
                        '{"focus_class":"Cat"}',
                    ],
                )


if __name__ == "__main__":
    unittest.main()
