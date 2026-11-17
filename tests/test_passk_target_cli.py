import unittest

import main


class PasskTargetCliTests(unittest.TestCase):
    def test_eval_target_accepts_a_label_field(self) -> None:
        args = main.build_parser().parse_args(
            ["eval-jlens", "--target", "label.emotion", "--target", "label.gender"]
        )

        self.assertEqual(args.target, ["emotion", "gender"])

    def test_eval_accepts_independent_lens_checkpoints(self) -> None:
        args = main.build_parser().parse_args(
            [
                "eval-jlens",
                "--lens-path",
                "esd.pt",
                "--lens-path",
                "fsd50k.pt",
                "--evaluate-lenses-independently",
                "--target",
                "label.emotion",
            ]
        )

        self.assertEqual(args.lens_paths, ["esd.pt", "fsd50k.pt"])
        self.assertTrue(args.evaluate_lenses_independently)

    def test_eval_target_rejects_empty_label_key(self) -> None:
        with self.assertRaises(SystemExit):
            main.build_parser().parse_args(
                ["eval-jlens", "--target", "label.", "--lens-path", "lens.pt"]
            )

    def test_eval_target_rejects_transcript(self) -> None:
        with self.assertRaises(SystemExit):
            main.build_parser().parse_args(
                ["eval-jlens", "--target", "transcript", "--lens-path", "lens.pt"]
            )


if __name__ == "__main__":
    unittest.main()
