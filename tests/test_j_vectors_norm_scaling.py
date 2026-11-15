import contextlib
import io
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
from tools.math_tools import MathTools


class JVectorNormScalingTests(unittest.TestCase):
    def setUp(self):
        self.weights = torch.tensor(
            [[1.0, 0.1, 0.2], [0.2, 1.0, 0.0], [0.0, 0.3, 1.0],
             [2.0, -0.5, 0.5], [0.4, 0.7, -0.3]]
        )
        self.gamma = torch.tensor([4.0, 0.5, 2.0])
        self.jacobians = {
            0: torch.tensor([[1.0, 0.1, 0.0], [0.0, 0.5, 1.0], [0.3, 0.0, 1.0]]),
            1: torch.tensor([[0.2, 1.0, 0.0], [0.0, 1.0, 0.1], [1.0, 0.2, 0.7]]),
        }

    def test_scaled_and_unscaled_match_direct_products(self):
        results = []
        for gamma in (self.gamma, None):
            with self.subTest(scaled=gamma is not None):
                readout = self.weights if gamma is None else self.weights * gamma
                vectors = {
                    layer: readout @ jacobian
                    for layer, jacobian in self.jacobians.items()
                }
                expected_cosine = MathTools.sim_for_sets(
                    vectors, vectors, [0, 1], [0, 1], MathTools.cos_similarity
                )
                expected_cka = MathTools.cka_similarity(
                    vectors, vectors, [0, 1], [0, 1], device="cpu"
                )
                cosine, cka = MathTools.j_vector_similarities(
                    self.jacobians, self.jacobians, [0, 1], [0, 1],
                    self.weights, gamma, device="cpu", chunk_size=2,
                )
                torch.testing.assert_close(cosine, expected_cosine, atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(cka, expected_cka, atol=1e-6, rtol=1e-6)
                results.append((cosine, cka))
        self.assertGreater(abs((results[0][0] - results[1][0])[0, 1]).item(), 0.01)
        self.assertGreater(abs((results[0][1] - results[1][1])[0, 1]).item(), 0.01)

    def test_master_passes_scaling_and_uses_distinct_plot_names(self):
        with tempfile.TemporaryDirectory() as output:
            master = Master.__new__(Master)
            master.logger = logging.getLogger(__name__)
            master.run_name = "test"
            master.settings = {"experiment": {"matrix_analysis": {"SVD_top_k": 1}}}
            master._experiment_analysis_dir = Mock(return_value=output)
            instance = SimpleNamespace(
                get_unembedding_weights=Mock(return_value=(self.weights, self.gamma))
            )
            master.model = SimpleNamespace(load_model=Mock(return_value=instance))
            lens = SimpleNamespace(
                lens=SimpleNamespace(source_layers=[0, 1]),
                load_lens=Mock(),
                get_jacobian=Mock(return_value=self.jacobians),
            )
            with (
                patch("Master.Jlens", return_value=lens),
                patch("Master.MathTools.draw_heatmap") as draw,
                patch.object(Experiments, "info_Jlens", wraps=Experiments.info_Jlens) as analyze,
            ):
                for use_norm_scaling in (True, False):
                    master.info_Jlens(
                        "lens.pt", device="cpu", use_j_vectors=True,
                        compute_svd=False, use_norm_scaling=use_norm_scaling,
                    )
                    actual_norm = analyze.call_args.kwargs["readout_weights"][1]
                    self.assertIs(actual_norm, self.gamma if use_norm_scaling else None)
            names = [Path(call.kwargs["png_path"]).name for call in draw.call_args_list]
            self.assertEqual(names, [
                "test_j_vectors_cosine.png",
                "test_j_vectors_cka.png",
                "test_j_vectors_no_norm_scaling_cosine.png",
                "test_j_vectors_no_norm_scaling_cka.png",
            ])

    def test_cli_wiring_and_invalid_combination(self):
        master = Mock()
        with (
            patch("main.Master", return_value=master),
            patch("sys.argv", [
                "main.py", "info-jlens", "--lens-path", "lens.pt",
                "--use-j-vectors", "--no-norm-scaling", "--no-svd",
            ]),
        ):
            main.main()
        self.assertFalse(master.info_Jlens.call_args.kwargs["use_norm_scaling"])
        self.assertTrue(master.info_Jlens.call_args.kwargs["use_j_vectors"])
        with (
            patch("main.Master") as master_class,
            patch("sys.argv", [
                "main.py", "info-jlens", "--lens-path", "lens.pt",
                "--no-norm-scaling",
            ]),
            contextlib.redirect_stderr(io.StringIO()) as stderr,
            self.assertRaises(SystemExit) as exc,
        ):
            main.main()
        self.assertEqual(exc.exception.code, 2)
        self.assertIn("requires --use-j-vectors", stderr.getvalue())
        master_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
