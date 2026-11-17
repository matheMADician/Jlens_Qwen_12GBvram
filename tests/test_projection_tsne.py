import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from tools.projection_tsne import (
    create_block_projection_tsne,
    create_projection_tsne,
    plot_tsne_by_label,
)


class _FakeTSNE:
    inputs: list[np.ndarray] = []

    def __init__(self, **kwargs: object) -> None:
        pass

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        self.inputs.append(features.copy())
        return np.zeros((features.shape[0], 2), dtype=np.float32)


class ProjectionTsneTests(unittest.TestCase):
    def test_combines_projected_directories_and_keeps_dataset_colors(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            directories = [
                root / "esd" / "dim_2_seed_42",
                root / "fsd50k" / "dim_2_seed_42",
            ]
            for index, (directory, dataset_name) in enumerate(
                zip(directories, ("esd", "fsd50k"))
            ):
                directory.mkdir(parents=True)
                archive_data: dict[str, np.ndarray] = {
                    "features": np.asarray(
                        [
                            [1.0 + index, 0.0, 0.0, 0.0],
                            [0.0, 1.0 + index, 0.0, 0.0],
                        ],
                        dtype=np.float32,
                    ),
                    "sample_ids": np.asarray(
                        [f"{dataset_name}_1", f"{dataset_name}_2"]
                    ),
                    "dataset_labels": np.asarray([dataset_name, dataset_name]),
                    "source_layers": np.asarray([1]),
                    "original_dim": np.asarray(8),
                    "projection_dim": np.asarray(2),
                    "projection_seed": np.asarray(42),
                    "projection_method": np.asarray("gaussian"),
                }
                if dataset_name == "fsd50k":
                    archive_data["sample_labels_json"] = np.asarray(
                        ['{"class":"x"}', '{"class":"y"}']
                    )
                np.savez_compressed(
                    directory / f"dataset_{dataset_name}.npz",
                    **archive_data,
                )
            output_dir = root / "combined"
            _FakeTSNE.inputs = []

            with patch("sklearn.manifold.TSNE", _FakeTSNE):
                outputs = create_projection_tsne(
                    directories,
                    output_dir=output_dir,
                    perplexity=2,
                )

            self.assertEqual(set(outputs), {output_dir})
            paths = outputs[output_dir]
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertEqual(_FakeTSNE.inputs[0].shape, (4, 4))
            with np.load(paths[1], allow_pickle=False) as embedding:
                self.assertEqual(
                    set(embedding["dataset_labels"].tolist()),
                    {"esd", "fsd50k"},
                )
                self.assertEqual(
                    embedding["sample_labels_json"].tolist().count("{}"),
                    2,
                )

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

    def test_recolors_existing_embedding_by_scalar_label_without_reembedding(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            embedding_path = Path(temporary_directory) / "tsne_embedding.npz"
            np.savez_compressed(
                embedding_path,
                embedding=np.asarray(
                    [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
                    dtype=np.float32,
                ),
                sample_labels_json=np.asarray(
                    [
                        '{"emotion":"happy"}',
                        '{"emotion":"sad"}',
                        '{"emotion":"happy"}',
                    ]
                ),
                feature_normalization=np.asarray("none"),
            )

            output_path = plot_tsne_by_label(embedding_path, "emotion")

            self.assertTrue(output_path.is_file())
            self.assertEqual(
                output_path,
                Path(temporary_directory)
                / "normal"
                / "tsne_embedding_by_emotion.png",
            )

    def test_creates_per_block_embeddings_with_aligned_labels(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            feature_path = Path(temporary_directory) / "dataset_test.npz"
            np.savez_compressed(
                feature_path,
                block_features=np.asarray(
                    [
                        [
                            [[1.0, 0.0], [0.0, 0.0]],
                            [[0.0, 1.0], [0.0, 0.0]],
                        ],
                        [
                            [[0.0, 1.0], [1.0, 1.0]],
                            [[1.0, 0.0], [1.0, 1.0]],
                        ],
                        [
                            [[1.0, 1.0], [0.0, 1.0]],
                            [[1.0, 1.0], [1.0, 0.0]],
                        ],
                    ],
                    dtype=np.float32,
                ),
                block_layer_ranges=np.asarray([[0, 20], [21, 26]]),
                sample_ids=np.asarray(["1", "2", "3"]),
                sample_labels_json=np.asarray(
                    [
                        '{"class":"a"}',
                        '{"class":"b"}',
                        '{"class":"c"}',
                    ]
                ),
                dataset_labels=np.asarray(["test", "test", "test"]),
                original_dim=np.asarray(8),
                projection_dim=np.asarray(2),
                projection_seed=np.asarray(42),
                projection_method=np.asarray("scaled_gaussian"),
            )
            _FakeTSNE.inputs = []

            with patch("sklearn.manifold.TSNE", _FakeTSNE):
                outputs = create_block_projection_tsne(
                    feature_path,
                    perplexity=2,
                )

            self.assertEqual(set(outputs), {(0, 20), (21, 26)})
            self.assertEqual(len(_FakeTSNE.inputs), 4)
            for block_range, paths in outputs.items():
                for path in paths:
                    self.assertTrue(path.is_file())
                    with np.load(path, allow_pickle=False) as embedding:
                        self.assertEqual(
                            embedding["sample_ids"].tolist(),
                            ["1", "2", "3"],
                        )
                        self.assertEqual(
                            embedding["sample_labels_json"].tolist(),
                            ['{"class":"a"}', '{"class":"b"}', '{"class":"c"}'],
                        )
                        self.assertEqual(
                            embedding["source_layers"].tolist(),
                            list(block_range),
                        )


if __name__ == "__main__":
    unittest.main()
