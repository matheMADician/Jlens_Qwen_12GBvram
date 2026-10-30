import argparse

from Master import Master


DEFAULT_LENS_PATH = (
    "output/checkpoints/Qwen2.5-Omni/"
    "train_1_no_SAKURA/experiment/lens_full.pt"
)
DEFAULT_PROMPTS_PATH = "data/assembled/test_SAKURA_AnimalQA/prompts.jsonl"


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

    evaluate = commands.add_parser(
        "eval-jlens",
        help="Call Master.eval_Jlens to evaluate a trained JLens.",
        parents=[common],
    )
    evaluate.add_argument("--lens-path", default=DEFAULT_LENS_PATH)
    evaluate.add_argument("--prompts-path", default=DEFAULT_PROMPTS_PATH)
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

    compare = commands.add_parser(
        "comp-jlens",
        help="Call Master.comp_Jlens to compare two JLens checkpoints.",
        parents=[common],
    )
    compare.add_argument("--lens-path1", required=True)
    compare.add_argument("--lens-path2", required=True)

    commands.add_parser(
        "calc-jlens-experiment",
        help="Call Master.calc_Jlens_experiment to fit an experiment lens.",
        parents=[common],
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

    if args.command == "test-training" and not args.run_name:
        parser.error("test-training requires --run-name")
    master = Master(run_name=args.run_name, settings_path=args.settings_file)

    if args.command == "eval-jlens":
        master.eval_Jlens(
            lens_path=args.lens_path,
            prompts_path=args.prompts_path,
            top_k=args.top_k,
            layer=args.layer,
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
        )
    elif args.command == "comp-jlens":
        master.comp_Jlens(
            lens_path1=args.lens_path1,
            lens_path2=args.lens_path2,
        )
    elif args.command == "calc-jlens-experiment":
        master.calc_Jlens_experiment()
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
