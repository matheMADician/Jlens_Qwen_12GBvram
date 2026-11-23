"""S1 swap experiment: inspect SAKURA labels, run a baseline pilot, then swap.

Standalone on purpose: it only imports shared modules and never edits
main.py, Master.py or tools/experiments.py.

Run from the repository root:

    # Step 1: label distribution and data checks (no GPU, no model)
    python -m tools.swap_s1 labels --task emotion --task gender

    # Step 2: baseline pilot (no swap): what does the model answer to an
    # open question, and how often is it a label word and correct?
    python -m tools.swap_s1 pilot --settings-file <settings.json> \
        --task emotion --task gender --count 100

    # Small test first: 10 prompts with the full greedy reply
    python -m tools.swap_s1 pilot --settings-file <settings.json> \
        --task emotion --task gender --count 10 --generate-tokens 20 --run-name pilot_small

    # Step 3: swap on the prompts the pilot answered correctly, with sliding
    # windows of 8 layers (stride 2, aligned so S1 is one of them)
    python -m tools.swap_s1 swap --settings-file <settings.json>         --pilot-path output/swap_s1/pilot/pilot_gender_q1.json         --lens-path <lens_fsd50k.pt> --s1-start 5

Terms:
* baseline: one forward pass with no intervention. Its top-1 next token at the
  last position is the model's answer (the source token of a later swap).
* golden: SAKURA's ``label.attribute_label``.
* in_vocab: the answer token is a single-token form of one of the task's labels.
* correct: in_vocab and that label equals the golden label.
* swap: at the layers of one window and the positions of one policy, exchange
  the coordinates of the source and target J-lens directions (clamped to the
  baseline values), then read the last position's next-token logits.
* position policies: "audio" = audio tokens without <|audio_bos|>/<|audio_eos|>;
  "text" = every token after <|audio_eos|> except the last one, so the swap
  never edits the position that produces the answer.
  "last" = only the last token: the positive control, where a deep swap should
  flip the answer if the swap machinery works.

The SAKURA instruction is replaced by an open question that lists no options,
so single-hop and multi-hop records give the same input; only audio and the
golden label are used.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import csv
import time
from collections import Counter, defaultdict

DEFAULT_DATA_ROOT = "data"
DEFAULT_OUTPUT_ROOT = "output/swap_s1"
LABEL_KEY = "attribute_label"

# Open questions only: no answer options in the prompt.
TASKS: dict[str, dict[str, object]] = {
    "emotion": {
        "prompts_path": "data/datasets/SAKURA/EmotionQA/prompts.jsonl",
        "instructions": {
            "q1": "What emotion does the speaker express? Answer in one word.",
            "q2": "Can you identify the speaker's emotional state in this audio? Answer in one word.",
        },
    },
    "gender": {
        "prompts_path": "data/datasets/SAKURA/GenderQA/prompts.jsonl",
        "instructions": {
            "q1": "What is the speaker's gender? Answer in one word.",
            "q2": "Describe the speaker's gender in one word.",
        },
    },
}


# Data helpers (no model needed)
# ---------------------------------------------------------------------------
def load_records(prompts_path: str, data_root: str) -> list[dict]:
    """Read a manifest and make every record usable by the model adapter.

    Raw SAKURA downloads have ids such as "sample-000000" (the adapter needs
    decimal ids) and audio paths relative to the dataset folder (the adapter
    resolves them against ``data_root``). Each record gets ``id = str(index)``
    with the original id kept in ``original_id``, and an audio path that is
    missing under ``data_root`` but present next to the manifest is rewritten
    relative to ``data_root``. Assembled manifests pass through unchanged
    apart from the id.
    """
    manifest_dir = os.path.dirname(os.path.abspath(prompts_path))
    root = os.path.abspath(data_root)
    with open(prompts_path, "r", encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    for index, record in enumerate(records):
        record["original_id"] = record.get("id")
        record["id"] = str(index)
        path = audio_path(record)
        if path is None or os.path.isfile(os.path.join(root, path)):
            continue
        local = os.path.join(manifest_dir, path)
        if os.path.isfile(local):
            rebased = os.path.relpath(local, root).replace(os.sep, "/")
            record["messages"][0]["content"][0]["path"] = rebased
            if "audio_path" in record:
                record["audio_path"] = rebased
    return records


def golden_label(record: dict) -> str | None:
    value = (record.get("label") or {}).get(LABEL_KEY)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower()


def audio_path(record: dict) -> str | None:
    try:
        return record["messages"][0]["content"][0]["path"]
    except (KeyError, IndexError, TypeError):
        return None


def stratified_sample(records: list[dict], count: int, seed: int) -> list[dict]:
    """Take ``count`` records, as balanced across golden labels as possible."""
    by_label: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        label = golden_label(record)
        if label is not None:
            by_label[label].append(record)
    rng = random.Random(seed)
    for label in sorted(by_label):
        rng.shuffle(by_label[label])
    picked: list[dict] = []
    queues = {label: list(items) for label, items in sorted(by_label.items())}
    while len(picked) < count and any(queues.values()):
        for label in sorted(queues):
            if queues[label] and len(picked) < count:
                picked.append(queues[label].pop())
    return picked


def with_instruction(record: dict, instruction: str) -> str:
    """JSONL line with the user text replaced (same as Experiments.swap_prompt_line)."""
    record = json.loads(json.dumps(record))
    record["messages"][0]["content"][1]["text"] = instruction
    return json.dumps(record, ensure_ascii=False)


def inspect_task(task: str, prompts_path: str, data_root: str) -> dict[str, object]:
    """Step 1: label distribution plus the checks the pilot relies on."""
    records = load_records(prompts_path, data_root)
    labels = Counter(golden_label(r) for r in records)
    problems: Counter = Counter()
    for record in records:
        if golden_label(record) is None:
            problems["missing_attribute_label"] += 1
        if not str(record.get("id", "")).isdecimal():
            problems["non_decimal_id"] += 1
        content = (record.get("messages") or [{}])[0].get("content") or []
        if len(content) < 2 or content[1].get("type") != "text":
            problems["no_text_slot"] += 1
        path = audio_path(record)
        if path is None:
            problems["no_audio_path"] += 1
        elif not os.path.isfile(os.path.join(data_root, path)):
            problems["audio_file_missing"] += 1
    example = records[0] if records else None
    return {
        "task": task,
        "prompts_path": prompts_path,
        "record_count": len(records),
        "label_counts": {str(k): v for k, v in labels.most_common()},
        "problems": dict(problems),
        "example_instruction": (
            example["messages"][0]["content"][1].get("text") if example else None
        ),
    }


# Step 2: baseline pilot
# ---------------------------------------------------------------------------
def label_token_ids(tokenizer, labels: list[str]) -> dict[str, list[int]]:
    """Single-token surface forms of every label (" angry", "Angry", ...)."""
    from tools.peak_cover import answer_token_ids

    token_ids = {label: answer_token_ids(tokenizer, label) for label in labels}
    # Distinct lower-case labels give distinct single-token strings, so a shared
    # id can only come from answer_token_ids' fallback (first token of a split
    # word). Fail loudly instead of letting one label overwrite another.
    owner: dict[int, str] = {}
    for label, ids in token_ids.items():
        for token in ids:
            if token in owner:
                raise ValueError(
                    f"Labels {owner[token]!r} and {label!r} share token "
                    f"{tokenizer.decode([token])!r}; the answer would be ambiguous."
                )
            owner[token] = label
    return token_ids


def load_model(settings_path: str, data_root: str):
    """Build the model the same way Master.__init__ does, without Master."""
    import logging

    import model.model as M

    with open(settings_path, "r", encoding="utf-8") as file:
        settings = json.load(file)
    model = M.Model(
        logger=logging.getLogger(),
        do_4bit=settings["model"]["load_in_4bit"],
        MODEL_ID=settings["model"]["id"],
        adapter=settings["model"]["adapter"],
    )
    adapter = model.load_model()
    adapter.load_data(data_root)
    return model, adapter, model.processor.tokenizer


def run_pilot(
    *,
    settings_path: str,
    tasks: list[str],
    count: int,
    seed: int,
    data_root: str,
    output_root: str,
    run_name: str,
    top_n: int = 5,
    generate_tokens: int = 0,
) -> None:
    import torch

    model, adapter, tokenizer = load_model(settings_path, data_root)
    out_dir = os.path.join(output_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    for task in tasks:
        spec = TASKS[task]
        records = load_records(spec["prompts_path"], data_root)
        all_labels = sorted({l for l in map(golden_label, records) if l is not None})
        token_ids = label_token_ids(tokenizer, all_labels)
        id_to_label = {t: label for label, ids in token_ids.items() for t in ids}
        sample = stratified_sample(records, count, seed)

        for qid, instruction in spec["instructions"].items():
            rows = []
            seconds = []
            for index, record in enumerate(sample, start=1):
                line = with_instruction(record, instruction)
                start = time.perf_counter()
                with torch.no_grad():
                    input_ids = adapter.encode(line)
                    logits = adapter.unembed(adapter.forward(input_ids)[0][[-1]])[0].float()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                seconds.append(time.perf_counter() - start)
                # Full greedy reply, to see what the model says when the first
                # token is not a label (e.g. "The speaker sounds ...").
                generated = (
                    adapter.generate(input_ids, max_new_tokens=generate_tokens)
                    if generate_tokens > 0 else None
                )

                golden = golden_label(record)
                top = torch.topk(logits, top_n).indices.tolist()
                answer_id = top[0]
                answer_label = id_to_label.get(answer_id)
                golden_ids = token_ids.get(golden, [])
                golden_rank = min(
                    int((logits > logits[t]).sum().item()) + 1 for t in golden_ids
                ) if golden_ids else None
                rows.append({
                    "sample_id": record.get("original_id"),
                    "golden": golden,
                    "answer_text": tokenizer.decode([answer_id]),
                    "answer_label": answer_label,
                    "in_vocab": answer_label is not None,
                    "correct": answer_label is not None and answer_label == golden,
                    "golden_rank": golden_rank,
                    "top_texts": [tokenizer.decode([t]) for t in top],
                    "seq_len": int(input_ids.shape[1]),
                    "generated_text": generated,
                })
                print(f"[{task}/{qid}] {index}/{len(sample)} golden={golden} "
                      f"answer={rows[-1]['answer_text']!r}"
                      + (f" reply={generated!r}" if generated is not None else ""), flush=True)

            report = {
                "task": task,
                "instruction_id": qid,
                "instruction": instruction,
                "prompts_path": spec["prompts_path"],
                "seed": seed,
                "label_token_forms": {
                    label: [tokenizer.decode([t]) for t in ids]
                    for label, ids in token_ids.items()
                },
                "summary": summarize(rows, seconds),
                "kept_sample_ids": [r["sample_id"] for r in rows if r["correct"]],
                "rows": rows,
            }
            path = os.path.join(out_dir, f"pilot_{task}_{qid}.json")
            with open(path, "w", encoding="utf-8") as file:
                json.dump(report, file, indent=2, ensure_ascii=False)
            s = report["summary"]
            print(f"[{task}/{qid}] n={s['n']} in_vocab={s['in_vocab_rate']:.2f} "
                  f"accuracy={s['accuracy']:.2f} kept={s['kept']} "
                  f"sec/forward={s['mean_seconds_per_forward']:.3f} -> {path}", flush=True)


def summarize(rows: list[dict], seconds: list[float]) -> dict[str, object]:
    n = len(rows)
    per_label: dict[str, dict[str, object]] = {}
    for label in sorted({r["golden"] for r in rows}):
        group = [r for r in rows if r["golden"] == label]
        per_label[label] = {
            "n": len(group),
            "accuracy": sum(r["correct"] for r in group) / len(group),
            "answers": dict(Counter(r["answer_text"] for r in group).most_common()),
        }
    ranks = [r["golden_rank"] for r in rows if r["golden_rank"] is not None]
    return {
        "n": n,
        "in_vocab_rate": sum(r["in_vocab"] for r in rows) / n if n else 0.0,
        "accuracy": sum(r["correct"] for r in rows) / n if n else 0.0,
        "kept": sum(r["correct"] for r in rows),
        "median_golden_rank": statistics.median(ranks) if ranks else None,
        "answer_counts": dict(Counter(r["answer_text"] for r in rows).most_common()),
        "per_label": per_label,
        # The first forward includes warm-up; the median is the better estimate.
        "mean_seconds_per_forward": statistics.fmean(seconds) if seconds else 0.0,
        "median_seconds_per_forward": statistics.median(seconds) if seconds else 0.0,
    }


# Step 3: swap with sliding windows
# ---------------------------------------------------------------------------
POSITION_POLICIES = ("audio", "text", "last")
DEFAULT_POSITION_POLICIES = ["audio", "text"]  # "last" is the positive control only
DIRECTION_SOURCES = ("jlens", "logit", "random")


def sliding_windows(layers: list[int], width: int, stride: int, anchor: int) -> list[list[int]]:
    """Windows of ``width`` consecutive layers, ``stride`` apart, one starting at ``anchor``."""
    available = set(layers)
    low, high = min(layers), max(layers)
    first = low + (anchor - low) % stride
    windows = [
        list(range(start, start + width))
        for start in range(first, high - width + 2, stride)
        if all(layer in available for layer in range(start, start + width))
    ]
    if not any(window[0] == anchor for window in windows):
        raise ValueError(
            f"S1 window L{anchor}-{anchor + width - 1} is not inside the lens layers "
            f"{low}-{high}."
        )
    return windows


def window_name(window: list[int]) -> str:
    return f"L{window[0]}-{window[-1]}"


def swap_positions(input_ids, tokenizer, policy: str) -> list[int]:
    """Positions a policy edits (see the module docstring)."""
    from tools.experiments import Experiments

    audio = Experiments.swap_token_positions(input_ids, tokenizer, "audio")
    if policy == "audio":
        positions = audio[1:-1]
    elif policy == "text":
        positions = list(range(audio[-1] + 1, int(input_ids.shape[1]) - 1))
    elif policy == "last":
        positions = [int(input_ids.shape[1]) - 1]
    else:
        raise ValueError(f"Unknown position policy {policy!r}; expected {POSITION_POLICIES}")
    if not positions:
        raise ValueError(f"Position policy {policy!r} selected no tokens.")
    return positions


def same_form_token(tokenizer, source_text: str, word: str) -> int | None:
    """Token of ``word`` written like the source (leading space, capital), if single."""
    core = source_text.strip()
    prefix = source_text[: len(source_text) - len(source_text.lstrip())]
    form = word.capitalize() if core[:1].isupper() else word.lower()
    ids = tokenizer(prefix + form, add_special_tokens=False)["input_ids"]
    return int(ids[0]) if len(ids) == 1 else None


def bootstrap_ci(values: list[float], draws: int, seed: int) -> tuple[float, float] | None:
    """95% percentile interval of the mean, resampling ``values``."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choices(values, k=len(values))) for _ in range(draws)
    )
    return means[int(0.025 * draws)], means[min(draws - 1, int(0.975 * draws))]


def run_swap(
    *,
    settings_path: str,
    pilot_path: str,
    lens_path: str,
    s1_start: int,
    width: int,
    stride: int,
    direction_sources: list[str],
    position_policies: list[str],
    data_root: str,
    output_root: str,
    run_name: str,
    alpha: float = 1.0,
    max_direction_cosine: float = 0.9,
    max_prompts: int | None = None,
    seed: int = 42,
    bootstrap_draws: int = 1000,
) -> None:
    import torch

    with open(pilot_path, "r", encoding="utf-8") as file:
        pilot = json.load(file)
    task, instruction = pilot["task"], pilot["instruction"]
    kept = list(pilot["kept_sample_ids"])
    records = load_records(pilot["prompts_path"], data_root)
    by_id = {record["original_id"]: record for record in records}
    missing = [sample_id for sample_id in kept if sample_id not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} kept ids are not in {pilot['prompts_path']}: {missing[:5]}")
    selected = [by_id[sample_id] for sample_id in kept][:max_prompts]
    if not selected:
        raise ValueError("The pilot kept no prompts; nothing to swap.")

    model, adapter, tokenizer = load_model(settings_path, data_root)
    model.load_Jlens(lens_path=lens_path)
    jacobian_lens = getattr(model.lens, "lens", None)
    if jacobian_lens is None:
        raise RuntimeError(f"The JLens checkpoint did not load: {lens_path}")
    windows = sliding_windows(list(jacobian_lens.source_layers), width, stride, s1_start)
    union_layers = sorted({layer for window in windows for layer in window})
    labels = sorted({label for label in map(golden_label, records) if label is not None})
    token_ids = label_token_ids(tokenizer, labels)
    id_to_label = {t: label for label, ids in token_ids.items() for t in ids}
    print(f"[swap/{task}] {len(selected)} prompts, windows "
          f"{[window_name(w) for w in windows]}, policies {position_policies}, "
          f"directions {direction_sources}", flush=True)

    # Directions depend only on the token pair, so compute them once per pair.
    direction_cache: dict[tuple[int, int, str], dict] = {}

    def directions_for(source: int, target: int, direction_source: str) -> dict:
        key = (source, target, direction_source)
        if key not in direction_cache:
            direction_cache[key] = model._swap_directions(
                source, target, union_layers, direction_source, seed
            )
        return direction_cache[key]

    def logits_last(residual):
        return adapter.unembed(residual[0][[-1]])[0].float().cpu()

    def rank(logits, token: int) -> int:
        return int((logits > logits[token]).sum().item()) + 1

    rows: list[dict] = []
    skipped: list[dict] = []
    for index, record in enumerate(selected, start=1):
        start = time.perf_counter()
        line = with_instruction(record, instruction)
        golden = golden_label(record)
        with torch.no_grad():
            input_ids = adapter.encode(line)
            # One baseline per prompt; every swap reuses its residuals.
            baseline_final, clean = adapter.capture_forward(input_ids, union_layers)
            baseline = logits_last(baseline_final)
        source = int(baseline.argmax().item())
        source_text = tokenizer.decode([source])
        if id_to_label.get(source) != golden:
            skipped.append({"sample_id": record["original_id"], "reason": "answer_changed",
                            "answer": source_text})
            continue
        positions = {p: swap_positions(input_ids, tokenizer, p) for p in position_policies}

        for target_label in [label for label in labels if label != golden]:
            target = same_form_token(tokenizer, source_text, target_label)
            if target is None:
                skipped.append({"sample_id": record["original_id"], "reason": "target_not_single_token",
                                "answer": source_text, "target": target_label})
                continue
            baseline_gap = float(baseline[target] - baseline[source])
            for direction_source in direction_sources:
                directions = directions_for(source, target, direction_source)
                for window in windows:
                    window_dirs = {layer: directions[layer] for layer in window}
                    cosine = max(abs(float(s @ t)) for s, t in window_dirs.values())
                    for policy, policy_positions in positions.items():
                        with torch.no_grad():
                            swapped = logits_last(adapter.swap_forward(
                                input_ids, window_dirs, policy_positions, alpha=alpha,
                                clean_residuals={layer: clean[layer] for layer in window},
                            ))
                        swapped_gap = float(swapped[target] - swapped[source])
                        top = int(swapped.argmax().item())
                        rows.append({
                            "sample_id": record["original_id"],
                            "golden": golden,
                            "source_text": source_text,
                            "target_text": tokenizer.decode([target]),
                            "direction_source": direction_source,
                            "window": window_name(window),
                            "window_start": window[0],
                            "is_s1": window[0] == s1_start,
                            "position_policy": policy,
                            "n_positions": len(policy_positions),
                            "baseline_gap": baseline_gap,
                            "swapped_gap": swapped_gap,
                            "logit_gap_shift": swapped_gap - baseline_gap,
                            "flipped": top == target,
                            "swapped_top_text": tokenizer.decode([top]),
                            "target_rank": rank(swapped, target),
                            "source_rank": rank(swapped, source),
                            "direction_cosine": cosine,
                            "excluded_high_cosine": cosine > max_direction_cosine,
                        })
        print(f"[swap/{task}] {index}/{len(selected)} {record['original_id']} "
              f"answer={source_text!r} {time.perf_counter() - start:.1f}s", flush=True)

    summary = summarize_swaps(rows, bootstrap_draws, seed)
    out_dir = os.path.join(output_root, run_name)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"swap_{task}")
    with open(stem + ".json", "w", encoding="utf-8") as file:
        json.dump({
            "task": task, "instruction": instruction, "pilot_path": pilot_path,
            "lens_path": lens_path, "s1": window_name(list(range(s1_start, s1_start + width))),
            "windows": [window_name(w) for w in windows], "alpha": alpha,
            "clamp_to_clean": True, "max_direction_cosine": max_direction_cosine,
            "prompt_count": len(selected), "skipped": skipped, "summary": summary,
        }, file, indent=2, ensure_ascii=False)
    with open(stem + "_rows.csv", "w", newline="", encoding="utf-8-sig") as file:
        if rows:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plot_swap_summary(summary, stem + ".png", task=task, lens_path=lens_path,
                      s1_start=s1_start, prompt_count=len(selected) - len(
                          {s["sample_id"] for s in skipped if s["reason"] == "answer_changed"}))
    print(f"[swap/{task}] {len(rows)} swaps, {len(skipped)} skipped -> {stem}.json/.png", flush=True)


def summarize_swaps(rows: list[dict], draws: int, seed: int) -> list[dict]:
    """Per (direction, policy, window): average each prompt first, then over prompts.

    Error bars resample prompts, as in the Peak/Cover plots. High-cosine swaps
    are counted but left out, because the pseudoinverse is unstable for nearly
    parallel directions.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["direction_source"], row["position_policy"], row["window_start"])].append(row)
    summary = []
    for (direction_source, policy, start), group in sorted(groups.items()):
        kept = [row for row in group if not row["excluded_high_cosine"]]
        per_prompt: dict[str, list[dict]] = defaultdict(list)
        for row in kept:
            per_prompt[row["sample_id"]].append(row)
        shifts = [statistics.fmean(r["logit_gap_shift"] for r in rs) for rs in per_prompt.values()]
        flips = [statistics.fmean(float(r["flipped"]) for r in rs) for rs in per_prompt.values()]
        summary.append({
            "direction_source": direction_source,
            "position_policy": policy,
            "window": group[0]["window"],
            "window_start": start,
            "is_s1": group[0]["is_s1"],
            "prompt_count": len(per_prompt),
            "excluded_high_cosine": len(group) - len(kept),
            "mean_logit_gap_shift": statistics.fmean(shifts) if shifts else None,
            "logit_gap_shift_ci95": bootstrap_ci(shifts, draws, seed),
            "flip_rate": statistics.fmean(flips) if flips else None,
            "flip_rate_ci95": bootstrap_ci(flips, draws, seed),
            "median_target_rank": (
                statistics.median(r["target_rank"] for r in kept) if kept else None
            ),
        })
    return summary


def plot_swap_summary(summary: list[dict], png_path: str, *, task: str, lens_path: str,
                      s1_start: int, prompt_count: int) -> None:
    """Swap effect against window, one figure row per direction source.

    Two panels, each with its own y-axis: logit gap shift and flip rate. Lines
    are the position policies; the shaded column marks S1.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {  # validated categorical slots 1-3, plus a marker per policy
        "audio": ("#2a78d6", "o"), "text": ("#eb6834", "s"), "last": ("#1baf7a", "^"),
    }
    sources = sorted({s["direction_source"] for s in summary})
    fig, axes = plt.subplots(len(sources), 2, figsize=(13, 4.2 * len(sources)),
                             squeeze=False, sharey="col")  # same scale for jlens vs controls
    for row_axes, source in zip(axes, sources):
        entries = [s for s in summary if s["direction_source"] == source]
        windows = sorted({(s["window_start"], s["window"]) for s in entries})
        for ax, key, ci_key, ylabel in (
            (row_axes[0], "mean_logit_gap_shift", "logit_gap_shift_ci95",
             "logit gap shift (target - source)"),
            (row_axes[1], "flip_rate", "flip_rate_ci95", "flip rate (target becomes top-1)"),
        ):
            ax.axvspan(s1_start - 0.8, s1_start + 0.8, color="#d9d9d9", zorder=0)
            ax.text(s1_start, 1.0, " S1", transform=ax.get_xaxis_transform(),
                    va="top", ha="center", color="#555555", fontsize=9)
            for policy, (color, marker) in style.items():
                points = sorted((s["window_start"], s) for s in entries
                                if s["position_policy"] == policy and s[key] is not None)
                if not points:
                    continue
                xs = [x for x, _ in points]
                ax.plot(xs, [s[key] for _, s in points], color=color, marker=marker,
                        linewidth=2, markersize=6, label=f"swap {policy}")
                bands = [(x, s[ci_key]) for x, s in points if s[ci_key] is not None]
                if bands:
                    ax.fill_between([x for x, _ in bands], [c[0] for _, c in bands],
                                    [c[1] for _, c in bands], color=color, alpha=0.15,
                                    linewidth=0)
            ax.axhline(0, color="#999999", linewidth=0.8)
            if key == "flip_rate":
                ax.set_ylim(-0.05, 1.05)  # a rate; all-zero data would otherwise get a 1e-17 axis
            ax.set_xticks([start for start, _ in windows])
            ax.set_xticklabels([name for _, name in windows], rotation=45, ha="right")
            ax.set_xlabel("swapped window (8 layers)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"direction: {source}")
            ax.grid(alpha=0.3)
            ax.legend(loc="best", fontsize=9)
    lens_name = os.path.splitext(os.path.basename(lens_path))[0]
    fig.suptitle(f"S1 swap / {task} | {lens_name} | {prompt_count} prompts | "
                 "bands: 95% bootstrap over prompts")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    labels = commands.add_parser("labels", help="Step 1: label distribution and data checks.")
    labels.add_argument("--task", action="append", choices=sorted(TASKS), required=True)
    labels.add_argument("--data-root", default=DEFAULT_DATA_ROOT)

    pilot = commands.add_parser("pilot", help="Step 2: baseline answers, no swap.")
    pilot.add_argument("--settings-file", default="settings.json")
    pilot.add_argument("--task", action="append", choices=sorted(TASKS), required=True)
    pilot.add_argument("--count", type=int, default=100)
    pilot.add_argument("--seed", type=int, default=42)
    pilot.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    pilot.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    pilot.add_argument("--run-name", default="pilot")
    pilot.add_argument(
        "--generate-tokens", type=int, default=0,
        help="Also generate a greedy reply of this many tokens (0 = off; slower).",
    )

    swap = commands.add_parser("swap", help="Step 3: sliding-window swap on the pilot's kept prompts.")
    swap.add_argument("--settings-file", default="settings.json")
    swap.add_argument("--pilot-path", required=True, help="pilot_<task>_<q>.json from step 2.")
    swap.add_argument("--lens-path", required=True)
    swap.add_argument("--s1-start", type=int, required=True, help="First layer of S1.")
    swap.add_argument("--width", type=int, default=8)
    swap.add_argument("--stride", type=int, default=2)
    swap.add_argument("--direction-source", action="append", choices=DIRECTION_SOURCES,
                      help="Repeatable; defaults to jlens only.")
    swap.add_argument("--position", action="append", choices=POSITION_POLICIES,
                      help="Repeatable; defaults to audio and text (last = positive control).")
    swap.add_argument("--alpha", type=float, default=1.0)
    swap.add_argument("--max-prompts", type=int, default=None)
    swap.add_argument("--seed", type=int, default=42)
    swap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    swap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    swap.add_argument("--run-name", default="swap")

    args = parser.parse_args(argv)
    if args.command == "labels":
        for task in args.task:
            result = inspect_task(task, TASKS[task]["prompts_path"], args.data_root)
            print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "swap":
        if args.width < 1 or args.stride < 1:
            parser.error("--width and --stride must be at least 1")
        run_swap(
            settings_path=args.settings_file,
            pilot_path=args.pilot_path,
            lens_path=args.lens_path,
            s1_start=args.s1_start,
            width=args.width,
            stride=args.stride,
            direction_sources=args.direction_source or ["jlens"],
            position_policies=args.position or list(DEFAULT_POSITION_POLICIES),
            data_root=os.path.abspath(args.data_root),
            output_root=args.output_root,
            run_name=args.run_name,
            alpha=args.alpha,
            max_prompts=args.max_prompts,
            seed=args.seed,
        )
    else:
        if args.count < 1:
            parser.error("--count must be at least 1")
        run_pilot(
            settings_path=args.settings_file,
            tasks=args.task,
            count=args.count,
            seed=args.seed,
            data_root=os.path.abspath(args.data_root),
            output_root=args.output_root,
            run_name=args.run_name,
            generate_tokens=args.generate_tokens,
        )


if __name__ == "__main__":
    sys.exit(main())
