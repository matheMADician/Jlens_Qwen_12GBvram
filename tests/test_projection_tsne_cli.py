import unittest

from tools.projection_tsne import build_parser


class ProjectionTsneCliTests(unittest.TestCase):
    def test_calc_jlens_tsne_accepts_multiple_projected_directories(self) -> None:
        args = build_parser().parse_args(
            [
                "create",
                "--projected-dir",
                "run-a/projected",
                "--projected-dir",
                "run-b/projected",
                "--output-dir",
                "combined",
            ]
        )

        self.assertEqual(
            args.projected_dir,
            ["run-a/projected", "run-b/projected"],
        )
        self.assertEqual(args.output_dir, "combined")


if __name__ == "__main__":
    unittest.main()
