import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import main
from Master import Master
from tools.experiments import Experiments
from tools.matrix_tools import MatrixTools


class InfoJlensSvdTests(unittest.TestCase):
    def setUp(self):
        self.jacobians = {
            0: torch.tensor([[1.0, 0.5], [0.2, 1.0]]),
            1: torch.tensor([[0.5, 1.0], [1.0, 0.3]]),
        }
        self.readout = (
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            torch.tensor([1.0, 2.0]),
        )
        self.logger = logging.getLogger(__name__)

    def test_analysis_skips_svd_in_both_similarity_modes(self):
        for readout in (None, self.readout):
            with self.subTest(readout=readout is not None):
                with patch.object(MatrixTools, "analyze_layer_svd") as svd:
                    result = Experiments.info_Jlens(
                        self.logger,
                        self.jacobians,
                        [0, 1],
                        svd_top_k=1,
                        device="cpu",
                        readout_weights=readout,
                        compute_svd=False,
                    )
                svd.assert_not_called()
                self.assertEqual(
                    set(result), {"layers", "cosine_heatmap", "cka_heatmap"}
                )
                self.assertEqual(result["cosine_heatmap"].shape, (2, 2))
                self.assertEqual(result["cka_heatmap"].shape, (2, 2))

    def test_default_analysis_still_runs_svd(self):
        with patch.object(
            MatrixTools, "analyze_layer_svd", wraps=MatrixTools.analyze_layer_svd
        ) as svd:
            result = Experiments.info_Jlens(
                self.logger, self.jacobians, [0, 1], svd_top_k=1, device="cpu"
            )
        svd.assert_called_once()
        self.assertEqual(result["svd_spectra"].shape, (2, 2))
        self.assertEqual(result["svd_adjacent_cosines"].shape, (1, 1))

    def test_master_skips_svd_plots_in_both_modes(self):
        for use_j_vectors in (False, True):
            with self.subTest(use_j_vectors=use_j_vectors), tempfile.TemporaryDirectory() as output:
                master = Master.__new__(Master)
                master.logger = self.logger
                master.run_name = "test"
                master.settings = {"experiment": {"matrix_analysis": {"SVD_top_k": 1}}}
                instance = SimpleNamespace(
                    get_unembedding_weights=Mock(return_value=self.readout)
                )
                master.model = SimpleNamespace(load_model=Mock(return_value=instance))
                master._experiment_analysis_dir = Mock(return_value=output)
                lens = SimpleNamespace(
                    lens=SimpleNamespace(source_layers=[0, 1]),
                    load_lens=Mock(),
                    get_jacobian=Mock(return_value=self.jacobians),
                )
                with (
                    patch("Master.Jlens", return_value=lens),
                    patch("Master.MathTools.draw_heatmap") as draw,
                    patch.object(MatrixTools, "analyze_layer_svd") as svd,
                ):
                    result = master.info_Jlens(
                        "lens.pt",
                        device="cpu",
                        use_j_vectors=use_j_vectors,
                        compute_svd=False,
                    )
                svd.assert_not_called()
                self.assertEqual(draw.call_count, 2)
                self.assertFalse((Path(output) / "SVD_analysis").exists())
                self.assertNotIn("svd_spectra", result)
                self.assertEqual(master.model.load_model.call_count, int(use_j_vectors))

    def test_master_default_still_creates_svd_plots(self):
        with tempfile.TemporaryDirectory() as output:
            master = Master.__new__(Master)
            master.logger = self.logger
            master.run_name = "test"
            master.settings = {"experiment": {"matrix_analysis": {"SVD_top_k": 1}}}
            master._experiment_analysis_dir = Mock(return_value=output)
            lens = SimpleNamespace(
                lens=SimpleNamespace(source_layers=[0, 1]),
                load_lens=Mock(),
                get_jacobian=Mock(return_value=self.jacobians),
            )
            with (
                patch("Master.Jlens", return_value=lens),
                patch("Master.MathTools.draw_heatmap") as draw,
            ):
                result = master.info_Jlens("lens.pt", device="cpu")
            self.assertEqual(draw.call_count, 4)
            self.assertTrue((Path(output) / "SVD_analysis").is_dir())
            self.assertIn("svd_spectra", result)

    def test_cli_wires_flag_and_default(self):
        for flag, expected in (([], True), (["--no-svd"], False)):
            with self.subTest(flag=flag):
                master = Mock()
                with (
                    patch("main.Master", return_value=master),
                    patch("sys.argv", ["main.py", "info-jlens", "--lens-path", "lens.pt", *flag]),
                ):
                    main.main()
                self.assertEqual(
                    master.info_Jlens.call_args.kwargs["compute_svd"], expected
                )


if __name__ == "__main__":
    unittest.main()
