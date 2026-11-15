import argparse
import os

os.environ.setdefault("HF_HOME", "/work/c8763c8763/huggingface-cache")

from Master import Master


DEFAULT_LENS_PATH = (
    "output/checkpoints/Qwen2.5-Omni/"
    "train_1_no_SAKURA/experiment/lens_full.pt"
)
DEFAULT_PROMPTS_PATH = "data/assembled/test_SAKURA_AnimalQA/prompts.jsonl"


def parse_projector(value: str) -> list[int]:
    """Parse a projector argument formatted as DIMENSION:SEED."""
    try:
        dimension_text, seed_text = value.split(":", 1)
        dimension = int(dimension_text)
        seed = int(seed_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "projector must be formatted as DIMENSION:SEED, e.g. 128:42"
        ) from exc
    if dimension <= 0 or seed < 0:
        raise argparse.ArgumentTypeError(
            "projector dimension must be positive and seed must be non-negative"
        )
    return [dimension, seed]


def parse_layer_band(value: str) -> list[float]:
    """Parse a layer band argument formatted as START:END depth percentages."""
    try:
        start_text, end_text = value.split(":", 1)
        return [float(start_text), float(end_text)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "layer band must be formatted as START:END, e.g. 38:92"
        ) from exc
        
def parse_eval_target(value: str) -> str:
    """Parse a pass@k target as transcript or a scalar label field."""
    if value == "transcript":
        return value
    if value.startswith("label.") and value[len("label."):].strip():
        return value
    raise argparse.ArgumentTypeError(
        "target must be 'transcript' or 'label.<field>', e.g. label.emotion"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Master workflow.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--run-name",
        default=None,
        help="Name used for output files and checkpoints.",
    )
    common.add_argument(
        "--settings-file",
        default="settings.json",
        help="Settings JSON file to use for this command.",
    )
    commands = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="COMMAND",
    )

    create_model_settings = commands.add_parser(
        "create-model-settings",
        help="Create a separate settings file configured for a supported model.",
    )
    create_model_settings.add_argument(
        "--model-name",
        required=True,
        help="Supported model and adapter name to set in the copied settings.",
    )
    create_model_settings.add_argument(
        "--source-settings",
        default="settings.json",
        help="Settings file to copy.",
    )
    create_model_settings.add_argument(
        "--output-settings",
        required=True,
        help="Path for the separate settings file to create.",
    )

    merge = commands.add_parser(
        "merge-jlens",
        help="Merge multiple JLens checkpoints.",
    )
    merge.add_argument(
        "--lens-path",
        action="append",
        required=True,
        dest="lens_paths",
        help="Input lens checkpoint; pass this option once per checkpoint.",
    )
    merge.add_argument("--output-path", required=True)
    merge.add_argument(
        "--merge-method",
        choices=("original", "equal"),
        default="original",
        help="Use upstream prompt-count weights or an equal average across checkpoints.",
    )

    evaluate = commands.add_parser(
        "eval-jlens",
        help="Call Master.eval_Jlens to evaluate a trained JLens.",
        parents=[common],
    )
    evaluate.add_argument(
        "--lens-path",
        action="append",
        dest="lens_paths",
        help="Lens checkpoint; repeat to merge checkpoints in memory.",
    )
    evaluate.add_argument(
        "--merge-method",
        choices=("original", "equal"),
        default="original",
        help="Merge multiple lenses using upstream prompt weights or an equal average.",
    )
    evaluate.add_argument("--prompts-path", default=DEFAULT_PROMPTS_PATH)
    evaluate.add_argument(
        "--target",
        type=parse_eval_target,
        default="transcript",
        help=(
            "Pass@k reference: transcript (default) or a scalar label field, "
            "such as label.emotion."
        ),
    )
    evaluate.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Evaluate only this layer; default is all available layers.",
    )
    evaluate.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of top predictions to write to the inference CSV.",
    )

    swap = commands.add_parser(
        "eval-swap",
        help="Call Master.eval_swap to run the concept swap experiment.",
        parents=[common],
    )
    swap.add_argument(
        "--lens-path",
        required=True,
        help="Full-size JLens fitted on the configured model; projected lenses are rejected.",
    )
    swap.add_argument("--prompts-path", default=DEFAULT_PROMPTS_PATH)
    swap.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Number of prompts to run; defaults to experiment.swap.max_prompts.",
    )
    swap.add_argument(
        "--layer-band",
        action="append",
        type=parse_layer_band,
        default=None,
        metavar="START:END",
        help=(
            "Depth percentage band to swap at, e.g. 38:92; repeat for multiple "
            "bands. Defaults to experiment.swap.layer_bands_percent."
        ),
    )

    generate = commands.add_parser(
        "generate",
        help="Call Master.generate to generate text from audio prompts.",
        parents=[common],
    )
    generate.add_argument(
        "--prompts-path",
        default=None,
        help="Prompt JSONL path; defaults to inference.jsonl_path in settings.",
    )
    generate.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum generated tokens per prompt (default: 128).",
    )

    info = commands.add_parser(
        "info-jlens",
        help="Call Master.info_Jlens to analyze one JLens checkpoint.",
        parents=[common],
    )
    info.add_argument("--lens-path", required=True)
    info.add_argument("--device", default=None)
    info.add_argument("--svd-device", default="cpu")
    info.add_argument(
        "--no-svd",
        action="store_true",
        help="Skip Jacobian SVD analysis and its plots.",
    )
    info.add_argument(
        "--use-j-vectors",
        action="store_true",
        help=(
            "Load the configured model and compute cosine/CKA from all "
            "norm-scaled token unembedding vectors multiplied by each Jacobian."
        ),
    )
    info.add_argument(
        "--no-norm-scaling",
        action="store_true",
        help="With --use-j-vectors, omit the final norm weight and use WJ.",
    )

    compare = commands.add_parser(
        "comp-jlens",
        help="Call Master.comp_Jlens to compare two JLens checkpoints.",
        parents=[common],
    )
    compare.add_argument("--lens-path1", action="append", required=True, dest="lens_paths1")
    compare.add_argument("--lens-path2", action="append", required=True, dest="lens_paths2")
    compare.add_argument(
        "--merge-method",
        choices=("original", "equal"),
        default="original",
        help="Merge each lens group using upstream prompt weights or an equal average.",
    )
    compare.add_argument("--use-j-vectors", action="store_true")
    compare.add_argument(
        "--no-norm-scaling",
        action="store_true",
        help="With --use-j-vectors, omit the final norm weight and compare WJ.",
    )

    fit_experiment = commands.add_parser(
        "calc-jlens-experiment",
        help="Call Master.calc_Jlens_experiment to fit an experiment lens.",
        parents=[common],
    )
    fit_experiment.add_argument(
        "--prompts-path",
        default=None,
        help=(
            "Training JSONL manifest; defaults to fitting.jsonl_path "
            "from the settings file."
        ),
    )
    fit_experiment.add_argument(
        "--projector",
        action="append",
        type=parse_projector,
        default=None,
        metavar="DIM:SEED",
        help=(
            "Use this random projector for this run; repeat to select multiple "
            "projectors when per-prompt features are disabled."
        ),
    )
    fit_experiment.add_argument(
        "--save-projected-features",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Save per-prompt projected features for t-SNE; --no-save-projected-features "
            "keeps only projected aggregate lenses."
        ),
    )
    fit_experiment.add_argument(
        "--save-projected-blocks",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Save per-prompt projected block matrices using "
            "fitting.experiment.projection_block_starts."
        ),
    )
    fit_experiment.add_argument(
        "--save-dataset-lenses",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save per-dataset lenses; use --no-save-dataset-lenses to disable.",
    )
    fit_experiment.add_argument(
        "--save-milestone-lenses",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Save full-size cumulative milestone lenses; use "
            "--no-save-milestone-lenses to disable."
        ),
    )

    commands.add_parser(
        "test-inference",
        help="Call Master.test_inference using settings.json inputs.",
        parents=[common],
    )

    commands.add_parser(
        "test-training",
        help="Call Master.test_training using settings.json inputs.",
        parents=[common],
    )

    commands.add_parser(
        "test-lens",
        help="Call Master.test_lens using settings.json inputs.",
        parents=[common],
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "create-model-settings":
        settings_path = Master.create_model_settings_file(
            model_name=args.model_name,
            output_path=args.output_settings,
            source_path=args.source_settings,
        )
        print(settings_path)
        return
    if args.command == "merge-jlens":
        print(
            Master.merge_Jlens(
                args.lens_paths,
                args.output_path,
                merge_method=args.merge_method,
            )
        )
        return

    if args.command == "test-training" and not args.run_name:
        parser.error("test-training requires --run-name")
    if args.command == "info-jlens" and args.no_norm_scaling and not args.use_j_vectors:
        parser.error("--no-norm-scaling requires --use-j-vectors")
    if args.command == "comp-jlens" and args.no_norm_scaling and not args.use_j_vectors:
        parser.error("--no-norm-scaling requires --use-j-vectors")
    if args.command == "eval-jlens" and not args.lens_paths:
        args.lens_paths = [DEFAULT_LENS_PATH]
    master = Master(run_name=args.run_name, settings_path=args.settings_file)

    if args.command == "eval-jlens":
        master.eval_Jlens(
            lens_paths=args.lens_paths,
            prompts_path=args.prompts_path,
            top_k=args.top_k,
            layer=args.layer,
            merge_method=args.merge_method,
            target=args.target,
        )
    elif args.command == "eval-swap":
        master.eval_swap(
            lens_path=args.lens_path,
            prompts_path=args.prompts_path,
            max_prompts=args.max_prompts,
            layer_bands_percent=args.layer_band,
        )
    elif args.command == "generate":
        master.generate(
            prompts_path=args.prompts_path,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.command == "info-jlens":
        master.info_Jlens(
            lens_path=args.lens_path,
            device=args.device,
            svd_device=args.svd_device,
            use_j_vectors=args.use_j_vectors,
            compute_svd=not args.no_svd,
            use_norm_scaling=not args.no_norm_scaling,
        )
    elif args.command == "comp-jlens":
        master.comp_Jlens(
            lens_paths1=args.lens_paths1,
            lens_paths2=args.lens_paths2,
            use_j_vectors=args.use_j_vectors,
            use_norm_scaling=not args.no_norm_scaling,
            merge_method=args.merge_method,
        )
    elif args.command == "calc-jlens-experiment":
        master.calc_Jlens_experiment(
            prompts_path=args.prompts_path,
            projector_pairs=args.projector,
            save_projected_features=args.save_projected_features,
            save_projected_blocks=args.save_projected_blocks,
            save_dataset_lenses=args.save_dataset_lenses,
            save_milestone_lenses=args.save_milestone_lenses,
        )
    elif args.command == "test-inference":
        master.test_inference()
    elif args.command == "test-training":
        master.test_training()
    elif args.command == "test-lens":
        master.test_lens()
    else:
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
