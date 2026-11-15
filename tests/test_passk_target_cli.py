import unittest

import main


class PasskTargetCliTests(unittest.TestCase):
    def test_eval_target_defaults_to_transcript(self) -> None:
        args = main.build_parser().parse_args(["eval-jlens"])

        self.assertEqual(args.target, "transcript")

    def test_eval_target_accepts_a_label_field(self) -> None:
        args = main.build_parser().parse_args(
            ["eval-jlens", "--target", "label.emotion"]
        )

        self.assertEqual(args.target, "label.emotion")

    def test_eval_target_rejects_empty_label_key(self) -> None:
        with self.assertRaises(SystemExit):
            main.build_parser().parse_args(["eval-jlens", "--target", "label."])


if __name__ == "__main__":
    unittest.main()
