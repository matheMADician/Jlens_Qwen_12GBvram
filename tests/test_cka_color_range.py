import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from Master import Master


class CkaColorRangeTests(unittest.TestCase):
    def setUp(self):
        self.jacobians = {
            0: torch.tensor([[1.0, 0.5], [0.2, 1.0]]),
            1: torch.tensor([[0.5, 1.0], [1.0, 0.3]]),
        }

    def master(self, output, matrix_settings):
        master = Master.__new__(Master)
        master.logger = logging.getLogger(__name__)
        master.run_name = "test"
        master.settings = {"experiment": {"matrix_analysis": matrix_settings}}
        master._experiment_analysis_dir = Mock(return_value=output)
        master.model = SimpleNamespace(
            load_model=Mock(return_value=SimpleNamespace(
                get_unembedding_weights=Mock(return_value=(
                    torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
                    torch.tensor([1.0, 2.0]),
                ))
            ))
        )
        return master

    def test_configured_and_default_ranges_for_both_modes(self):
        lens = SimpleNamespace(
            lens=SimpleNamespace(source_layers=[0, 1]),
            load_lens=Mock(),
            get_jacobian=Mock(return_value=self.jacobians),
        )
        for use_j_vectors in (False, True):
            with self.subTest(use_j_vectors=use_j_vectors), tempfile.TemporaryDirectory() as output:
                results = []
                for configured_range, expected in ((None, (0.0, 1.0)), ([0.5, 1], (0.5, 1))):
                    settings = {"SVD_top_k": 1}
                    if configured_range is not None:
                        settings["cka_color_range"] = configured_range
                    master = self.master(output, settings)
                    with (
                        patch("Master.Jlens", return_value=lens),
                        patch("Master.MathTools.draw_heatmap") as draw,
                    ):
                        result = master.info_Jlens(
                            "lens.pt", device="cpu", use_j_vectors=use_j_vectors,
                            compute_svd=False,
                        )
                    cosine, cka = draw.call_args_list
                    self.assertNotIn("vmin", cosine.kwargs)
                    self.assertEqual((cka.kwargs["vmin"], cka.kwargs["vmax"]), expected)
                    results.append(result["cka_heatmap"])
                torch.testing.assert_close(results[0], results[1])

    def test_invalid_ranges_fail_before_loading_model(self):
        for invalid in (None, [0.5], [1, 0.5], [-0.1, 1], [0, 1.1],
                        [True, 1], ["0.5", 1], [float("nan"), 1]):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as output:
                master = self.master(output, {"SVD_top_k": 1, "cka_color_range": invalid})
                with self.assertRaisesRegex(ValueError, "cka_color_range"):
                    master.info_Jlens("lens.pt", use_j_vectors=True, compute_svd=False)
                master.model.load_model.assert_not_called()
                self.assertFalse(any(Path(output).iterdir()))


if __name__ == "__main__":
    unittest.main()
