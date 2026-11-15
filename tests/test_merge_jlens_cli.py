import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import main
from Master import Master


class MergeJlensCliTests(unittest.TestCase):
    def test_merge_cli_defaults_to_original_method(self):
        args = main.build_parser().parse_args(
            ["merge-jlens", "--lens-path", "one.pt", "--output-path", "merged.pt"]
        )

        self.assertEqual(args.merge_method, "original")

    def test_merge_cli_accepts_equal_method(self):
        args = main.build_parser().parse_args(
            [
                "merge-jlens",
                "--lens-path",
                "one.pt",
                "--output-path",
                "merged.pt",
                "--merge-method",
                "equal",
            ]
        )

        self.assertEqual(args.merge_method, "equal")

    def test_master_passes_selected_method_to_merge(self):
        merged_lens = Mock()
        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / "merged.pt"
            with patch.object(
                Master, "_merge_lenses", return_value=merged_lens
            ) as merge:
                result = Master.merge_Jlens(
                    ["one.pt", "two.pt"],
                    str(output_path),
                    merge_method="equal",
                )

        merge.assert_called_once_with(["one.pt", "two.pt"], "equal")
        merged_lens.save.assert_called_once_with(str(output_path))
        self.assertEqual(result, str(output_path))
