"""Step 0 pilot: can the trunk's lens read the *speaker's* gender at audio positions?

For each SAKURA GenderQA clip, one forward pass records the residual at every
lens layer. Each residual is read with the J lens and with the logit lens
(control), and the correct vs wrong gender token sets are ranked at

* ``audio`` positions (between <|audio_bos|> and <|audio_eos|>, markers excluded)
* ``text`` positions (everything after <|audio_eos|>)

Decision rule (stroop/docs/TODO.md, step 0): a ``balanced_win_rate`` clearly
above 0.5 at **audio** positions means speaker gender is readable → continue.
Readable only at text positions → likely answer leakage → rethink.

Run both pilot files: the gender question asks for the answer; the
transcription prompt never mentions gender, so it is the stricter test.

Example (run from the repo root on the server):

    python -m stroop.experiments.pilot_readout \
        --lens-path output/checkpoints/<run>/lens.pt \
        --prompts data/assembled/stroop/sakura_gender/pilot_single.jsonl \
        --prompts data/assembled/stroop/sakura_gender/pilot_transcribe.jsonl \
        --run-name pilot_main_lens

Outputs in ``output/stroop/pilot/<run-name>/``: ``rows.csv`` (prompt x layer x
lens x region x token set), ``summary.json``, ``metadata.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stroop.experiments.readout import (
    OTHER,
    build_token_sets,
    best_set_rank,
    region_stats,
    summarize,
)

LOGGER = logging.getLogger("stroop.experiments.pilot_readout")
ROW_FIELDS = (
    "prompts_file", "sample_id", "gender", "lens", "region", "token_set", "layer",
    "n_positions", "win_rate", "correct_best_rank", "wrong_best_rank",
    "correct_median_rank", "wrong_median_rank",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--settings", type=Path, default=Path("settings.json"),
                        help="Trunk settings; only the model section is read")
    parser.add_argument("--lens-path", required=True, help="Full-size J lens (.pt)")
    parser.add_argument("--prompts", type=Path, action="append", required=True,
                        help="Pilot JSONL; repeat for several files")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--max-prompts", type=int, default=None, help="Per prompts file")
    parser.add_argument("--output-root", type=Path, default=Path("output/stroop/pilot"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def region_positions(input_ids: Any, tokenizer: Any) -> dict[str, list[int]]:
    """Audio positions without the two markers, and the text positions after them."""
    from tools.experiments import Experiments

    audio = Experiments.swap_token_positions(input_ids, tokenizer, "audio")[1:-1]
    text = Experiments.swap_token_positions(input_ids, tokenizer, "after_audio")
    if not audio:
        raise ValueError("No audio positions between the audio markers")
    return {"audio": audio, "text": text}


def read_prompt(
    instance: Any,
    jacobian_lens: Any,
    json_line: str,
    token_sets: dict[str, dict[str, list[int]]],
) -> list[dict[str, Any]]:
    """Score one prompt; returns one row per (layer, lens, region, token set)."""
    import torch
    from jacobian_lens.jlens.hooks import ActivationRecorder

    record = json.loads(json_line)
    gender = record["label"]["speaker_gender"]
    layers = list(jacobian_lens.source_layers)

    input_ids = instance.encode(json_line)
    regions = region_positions(input_ids, instance.tokenizer)
    with torch.no_grad():
        with ActivationRecorder(instance.layers, at=layers) as recorder:
            instance.forward(input_ids)
            activations = {layer: recorder.activations[layer][0].detach() for layer in layers}

        rows = []
        for layer in layers:
            residual = activations[layer].float()
            for lens_name in ("jlens", "logit_lens"):
                transported = jacobian_lens.transport(residual, layer) if lens_name == "jlens" else residual
                logits = instance.unembed(transported).float()  # [seq_len, vocab]
                for set_name, by_gender in token_sets.items():
                    correct = best_set_rank(logits, by_gender[gender])
                    wrong = best_set_rank(logits, by_gender[OTHER[gender]])
                    for region, positions in regions.items():
                        if not positions:
                            continue
                        stats = region_stats(correct[positions].tolist(), wrong[positions].tolist())
                        rows.append({
                            "sample_id": record["id"], "gender": gender, "lens": lens_name,
                            "region": region, "token_set": set_name, "layer": layer, **stats,
                        })
                del logits
    return rows


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    from model.model import Model

    output_dir = args.output_root / args.run_name
    if (output_dir / "summary.json").exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} already has results; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = json.loads(args.settings.read_text(encoding="utf-8"))["model"]
    model = Model(logger=LOGGER, MODEL_ID=settings["id"], adapter=settings["adapter"],
                  do_4bit=settings.get("load_in_4bit", False))
    model.load_Jlens(lens_path=args.lens_path)
    instance = model.load_model()
    instance.load_data(data_root=args.data_root)
    jacobian_lens = model.lens.lens
    token_sets = build_token_sets(instance.tokenizer)
    LOGGER.info("Token sets: %s", token_sets)

    rows: list[dict[str, Any]] = []
    for prompts_path in args.prompts:
        lines = [l for l in prompts_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if args.max_prompts is not None:
            lines = lines[: args.max_prompts]
        for index, line in enumerate(lines, 1):
            for row in read_prompt(instance, jacobian_lens, line, token_sets):
                rows.append({"prompts_file": prompts_path.stem, **row})
            if index % 25 == 0 or index == len(lines):
                LOGGER.info("%s: %d/%d prompts", prompts_path.stem, index, len(lines))

    with (output_dir / "rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        stem: summarize([r for r in rows if r["prompts_file"] == stem])
        for stem in dict.fromkeys(r["prompts_file"] for r in rows)
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model": settings,
        "args": {k: [str(p) for p in v] if isinstance(v, list) else str(v) if isinstance(v, Path) else v
                 for k, v in vars(args).items()},
        "lens_layers": list(jacobian_lens.source_layers),
        "token_sets": token_sets,
    }, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %d rows to %s", len(rows), output_dir)
    return rows


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run(parse_args(argv))


if __name__ == "__main__":
    main()
