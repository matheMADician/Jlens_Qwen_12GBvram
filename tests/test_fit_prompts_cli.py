import unittest

import main


class FitPromptsCliTests(unittest.TestCase):
    def test_fit_prompts_path_is_optional(self) -> None:
        args = main.build_parser().parse_args(["calc-jlens-experiment"])

        self.assertIsNone(args.prompts_path)

    def test_fit_prompts_path_can_be_overridden(self) -> None:
        args = main.build_parser().parse_args(
            ["calc-jlens-experiment", "--prompts-path", "data/train/prompts.jsonl"]
        )

        self.assertEqual(args.prompts_path, "data/train/prompts.jsonl")

    def test_projected_blocks_can_be_enabled_from_cli(self) -> None:
        args = main.build_parser().parse_args(
            ["calc-jlens-experiment", "--save-projected-blocks"]
        )

        self.assertTrue(args.save_projected_blocks)


if __name__ == "__main__":
    unittest.main()
