import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from audio_jlens.random_proj import GaussianRandomProjector


class ProjectionLabelTests(unittest.TestCase):
    def test_saves_labels_aligned_with_projected_sample_ids(self) -> None:
        projector = GaussianRandomProjector(
            d_model=2,
            projection_dim=2,
            seed=42,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            projected_root = Path(temporary_directory)
            projector.cache_prompt(
                {0: torch.eye(2)},
                [0],
                sample_id="sample-1",
                dataset_name="pilot",
                label={"emotion": "happy", "tags": ["voice", "speech"]},
                prompt_index=0,
                cache_dir=projector.feature_cache_directory(projected_root),
            )
            output_dir = projector.save_features(
                projected_root,
                source_layers=[0],
                dataset_counts={"pilot": 1},
                use_disk_cache=True,
            )
            feature_path = output_dir / "dataset_pilot.npz"
            with np.load(feature_path, allow_pickle=False) as features:
                self.assertEqual(features["sample_ids"].tolist(), ["sample-1"])
                labels = json.loads(str(features["sample_labels_json"][0]))

        self.assertEqual(
            labels,
            {"emotion": "happy", "tags": ["voice", "speech"]},
        )

    def test_saves_averaged_block_matrices_and_layer_ranges(self) -> None:
        projector = GaussianRandomProjector(
            d_model=2,
            projection_dim=2,
            seed=42,
        )
        jacobians = {
            0: torch.eye(2),
            1: torch.eye(2) * 2,
            2: torch.eye(2) * 4,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            projected_root = Path(temporary_directory)
            projector.cache_prompt(
                jacobians,
                [0, 1, 2],
                sample_id="sample-1",
                dataset_name="pilot",
                label={"class": "one"},
                block_starts=[0, 2],
                prompt_index=0,
                cache_dir=projector.feature_cache_directory(projected_root),
            )
            output_dir = projector.save_features(
                projected_root,
                source_layers=[0, 1, 2],
                dataset_counts={"pilot": 1},
                use_disk_cache=True,
            )
            with np.load(output_dir / "dataset_pilot.npz", allow_pickle=False) as data:
                block_features = data["block_features"]
                ranges = data["block_layer_ranges"]
            with np.load(
                output_dir / "projection_matrix.npz",
                allow_pickle=False,
            ) as projection:
                projection_ranges = projection["block_layer_ranges"]

        self.assertEqual(block_features.shape, (1, 2, 2, 2))
        np.testing.assert_array_equal(ranges, [[0, 1], [2, 2]])
        np.testing.assert_array_equal(projection_ranges, ranges)
        np.testing.assert_allclose(
            block_features[0, 0],
            projector.project(torch.eye(2) * 1.5).numpy(),
        )
        np.testing.assert_allclose(
            block_features[0, 1],
            projector.project(torch.eye(2) * 4).numpy(),
        )


if __name__ == "__main__":
    unittest.main()
