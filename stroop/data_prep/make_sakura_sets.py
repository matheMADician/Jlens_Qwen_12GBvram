"""Build the stroop JSONL sets from the downloaded SAKURA GenderQA copy.

Input: ``data/datasets/stroop/sakura_gender/prompts.jsonl`` from
``stroop/data_prep/get_sakura_gender.sh`` (prompt = single-hop question,
transcript = speaker gender).

Output, in ``data/assembled/stroop/sakura_gender/`` (audio paths relative to
the data root, like ``data/assemble_json.py``):

    pilot_single.jsonl       Step 0: SAKURA's own single-hop question
    pilot_transcribe.jsonl   Step 0: same audio, transcription prompt
    fit_gender_{all,A,B}.jsonl   Step 1: gender question without gender words
    fit_trans_{all,A,B}.jsonl    Step 1: transcription prompt
    split.json, metadata.json

A and B are disjoint halves, balanced by speaker gender, and identical for both
tasks. Every record keeps its id and audio, so the two tasks differ only in the
text prompt.

Example (run from the repo root):

    python -m stroop.data_prep.make_sakura_sets
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stroop.data_prep.gender_labels import FEMALE_WORDS, MALE_WORDS, words

LOGGER = logging.getLogger("stroop.data_prep.make_sakura_sets")
PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Words that must not appear in fitting prompts: S2T targets are the prompt's
# text positions, so a gender word there would let the lens learn the prompt
# instead of the audio. See stroop/docs/AGENTS.md, section 2.
BANNED_PROMPT_WORDS = MALE_WORDS | FEMALE_WORDS | {
    "male", "female", "males", "females", "masculine", "feminine",
}


def load_prompts(path: Path) -> list[str]:
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    prompts = [p for p in prompts if p and not p.startswith("#")]
    if not prompts:
        raise ValueError(f"No prompts in {path}")
    return prompts


def check_prompts(prompts: list[str], name: str) -> None:
    """Raise if any fitting prompt contains a gender word."""
    for prompt in prompts:
        found = sorted(set(words(prompt)) & BANNED_PROMPT_WORDS)
        if found:
            raise ValueError(f"{name} prompt contains gender words {found}: {prompt!r}")


def with_prompt(record: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Copy a trunk-schema record with its user text replaced."""
    new = copy.deepcopy(record)
    new["messages"][0]["content"][1]["text"] = prompt
    return new


def rebase_audio(record: dict[str, Any], dataset_dir: Path, data_root: Path) -> dict[str, Any]:
    """Make audio paths relative to the data root, as the trunk's fitting code expects."""
    new = copy.deepcopy(record)
    rel = (dataset_dir / record["audio_path"]).resolve().relative_to(data_root.resolve()).as_posix()
    new["audio_path"] = rel
    new["messages"][0]["content"][0]["path"] = rel
    return new


def split_halves(records: list[dict[str, Any]], seed: int) -> tuple[list[str], list[str]]:
    """Disjoint A/B ids, balanced by speaker gender (``label.speaker_gender``)."""
    by_gender: dict[str, list[str]] = {}
    for record in records:
        by_gender.setdefault(record["label"]["speaker_gender"], []).append(record["id"])
    rng = random.Random(seed)
    half_a: list[str] = []
    half_b: list[str] = []
    for gender in sorted(by_gender):
        ids = sorted(by_gender[gender])
        rng.shuffle(ids)
        # Alternate which half gets the odd one out so totals stay balanced.
        cut = (len(ids) + (len(half_a) <= len(half_b))) // 2
        half_a += ids[:cut]
        half_b += ids[cut:]
    return sorted(half_a), sorted(half_b)


def assign(records: list[dict[str, Any]], prompts: list[str], seed: int) -> list[dict[str, Any]]:
    """Give each record a prompt drawn with a fixed seed."""
    rng = random.Random(seed)
    return [with_prompt(record, rng.choice(prompts)) for record in records]


def prepare_records(raw: list[dict[str, Any]], dataset_dir: Path, data_root: Path) -> list[dict[str, Any]]:
    """Rebase audio, renumber ids as decimals, and add ``label.speaker_gender``."""
    records = []
    for index, record in enumerate(raw):
        gender = str(record.get("transcript", "")).strip().lower()
        if gender not in ("male", "female"):
            raise ValueError(f"Record {record.get('id')} has transcript {gender!r}, expected male/female")
        new = rebase_audio(record, dataset_dir, data_root)
        # The model adapters require decimal ids (Instance.encode checks isdecimal()).
        new["id"] = str(index)
        new["source_id"] = record["id"]
        new["dataset_name"] = "stroop_sakura_gender"
        new["label"] = {"speaker_gender": gender}
        records.append(new)
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ids in input")
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/datasets/stroop/sakura_gender"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/assembled/stroop/sakura_gender"))
    parser.add_argument("--gender-prompts", type=Path, default=PROMPT_DIR / "ask_gender.txt")
    parser.add_argument("--trans-prompts", type=Path, default=PROMPT_DIR / "transcribe.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def build(args: argparse.Namespace) -> dict[str, int]:
    source = args.dataset_dir / "prompts.jsonl"
    raw = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = prepare_records(raw, args.dataset_dir, args.data_root)

    gender_prompts = load_prompts(args.gender_prompts)
    trans_prompts = load_prompts(args.trans_prompts)
    check_prompts(gender_prompts, "gender")
    check_prompts(trans_prompts, "transcription")

    half_a, half_b = split_halves(records, args.seed)
    in_a, in_b = set(half_a), set(half_b)

    fit_gender = assign(records, gender_prompts, args.seed)
    fit_trans = assign(records, trans_prompts, args.seed + 1)
    sets = {
        "pilot_single": records,  # keeps SAKURA's own single-hop question
        "pilot_transcribe": fit_trans,
        "fit_gender_all": fit_gender,
        "fit_gender_A": [r for r in fit_gender if r["id"] in in_a],
        "fit_gender_B": [r for r in fit_gender if r["id"] in in_b],
        "fit_trans_all": fit_trans,
        "fit_trans_A": [r for r in fit_trans if r["id"] in in_a],
        "fit_trans_B": [r for r in fit_trans if r["id"] in in_b],
    }

    args.output.mkdir(parents=True, exist_ok=True)
    existing = [name for name in sets if (args.output / f"{name}.jsonl").exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"{args.output} already has {existing}; use --overwrite")
    for name, subset in sets.items():
        write_jsonl(args.output / f"{name}.jsonl", subset)

    (args.output / "split.json").write_text(
        json.dumps({"seed": args.seed, "A": half_a, "B": half_b}, indent=2), encoding="utf-8"
    )
    counts = {name: len(subset) for name, subset in sets.items()}
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "counts": counts,
        "speaker_gender": {
            half: {g: sum(1 for r in records if r["id"] in ids and r["label"]["speaker_gender"] == g)
                   for g in ("female", "male")}
            for half, ids in (("A", in_a), ("B", in_b))
        },
        "gender_prompts": gender_prompts,
        "trans_prompts": trans_prompts,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s to %s", counts, args.output)
    return counts


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    build(parse_args(argv))


if __name__ == "__main__":
    main()
