#!/usr/bin/env python3
"""Dump and plot where a label token ranks at every layer and prompt position.

Diagnostic for pass@k: instead of keeping only the best (layer, position),
store the full rank / log-probability grid so the distribution can be studied.
Every label value seen in the dataset is a candidate, so wrong labels act as a
noise-floor control for the true label. Each candidate is scored in four
surface forms ("angry", " angry", "Angry", " Angry", first token of each).

Usage (server, GPU):
    python tools/dump_label_ranks.py dump \
        --lens-path output/checkpoints/.../lens_full.pt \
        --prompts-path data/assembled/test/esd/prompts.jsonl \
        --target label.emotion --max-prompts 30 \
        --output output/diag/esd_emotion_ranks.npz

Usage (anywhere with numpy + matplotlib):
    python tools/dump_label_ranks.py plot \
        --input output/diag/esd_emotion_ranks.npz --output-dir output/diag/esd_emotion
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

FORMS = ("bare", "space", "Cap", "space_Cap")
LENSES = ("jlens", "logit_lens")
N_AUDIO_BINS = 10


def _form_text(word: str, form: str) -> str:
    text = word.capitalize() if "Cap" in form else word
    return " " + text if form.startswith("space") else text


# ---------------------------------------------------------------------------
# dump (needs the model)
# ---------------------------------------------------------------------------
def dump(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import torch
    from Master import Master
    from json_parser import Parser
    from jacobian_lens.jlens.hooks import ActivationRecorder
    from jacobian_lens.jlens.lens import JacobianLens

    if not args.target.startswith("label."):
        raise SystemExit("--target must look like label.emotion")
    label_key = args.target[len("label."):]

    master = Master(run_name=None, settings_path=args.settings_file)
    lens = (
        JacobianLens.load(args.lens_paths[0])
        if len(args.lens_paths) == 1
        else Master._merge_lenses(args.lens_paths, "original")
    )
    json_lines = Parser.load_jsonl_lines(args.prompts_path)
    all_samples = [Parser.parse_json_line(line) for line in json_lines]
    # Candidates come from the whole file so a small subset still sees every label;
    # --extra-candidates adds wrong-label controls that never occur in this file.
    present = sorted({str(s.label[label_key]).strip() for s in all_samples})
    candidates = sorted(set(present) | {c.strip() for c in args.extra_candidates})
    # Pick a small subset with every present label represented (round-robin by label).
    order = list(range(len(json_lines)))
    random.Random(args.seed).shuffle(order)
    by_label: dict[str, list[int]] = {c: [] for c in present}
    for i in order:
        by_label[str(all_samples[i].label[label_key]).strip()].append(i)
    limit = args.per_label * len(present) if args.per_label else args.max_prompts
    limit = min(limit, len(order))
    picked: list[int] = []
    while len(picked) < limit:
        for c in present:
            if by_label[c] and len(picked) < limit:
                picked.append(by_label[c].pop())
    json_lines = [json_lines[i] for i in picked]
    samples = [all_samples[i] for i in picked]
    true_words = [str(s.label[label_key]).strip() for s in samples]

    adapter = master.model.load_model()
    adapter.load_data(master._configured_data_root())
    tokenizer = master.model.processor.tokenizer

    # token ids: [candidate, form] -> first token id; also keep token counts
    token_ids = np.zeros((len(candidates), len(FORMS)), dtype=np.int64)
    token_counts = np.zeros_like(token_ids)
    for ci, word in enumerate(candidates):
        for fi, form in enumerate(FORMS):
            ids = tokenizer(_form_text(word, form), add_special_tokens=False)["input_ids"]
            token_ids[ci, fi] = ids[0]
            token_counts[ci, fi] = len(ids)
    unique_ids = sorted(set(token_ids.reshape(-1).tolist()))
    id_col = {tid: i for i, tid in enumerate(unique_ids)}
    col_index = np.vectorize(id_col.__getitem__)(token_ids)  # [C, F] -> column
    tid_tensor = torch.tensor(unique_ids)

    layers = list(lens.source_layers)
    final_layer = adapter.n_layers - 1
    audio_bos = tokenizer.convert_tokens_to_ids("<|audio_bos|>")
    audio_eos = tokenizer.convert_tokens_to_ids("<|audio_eos|>")

    arrays: dict[str, np.ndarray] = {}
    meta_samples = []
    for si, (line, sample) in enumerate(zip(json_lines, samples)):
        input_ids = adapter.encode(line, max_length=300)
        with torch.no_grad(), ActivationRecorder(
            adapter.layers, at=[*layers, final_layer]
        ) as recorder:
            adapter.forward(input_ids)
        ids = input_ids.reshape(-1).tolist()
        prompt_len = len(ids)

        for lens_name in LENSES:
            ranks = np.zeros((len(layers), prompt_len, len(unique_ids)), dtype=np.int32)
            logprobs = np.zeros_like(ranks, dtype=np.float16)
            for li, layer in enumerate(layers):
                with torch.no_grad():
                    resid = recorder.activations[layer][0].float()
                    if lens_name == "jlens":
                        resid = lens.transport(resid, layer)
                    logits = adapter.unembed(resid).float()[:prompt_len]
                    tids = tid_tensor.to(logits.device)
                    target = logits[:, tids]  # [P, T]
                    rank = torch.stack(
                        [(logits > target[:, t : t + 1]).sum(dim=1) + 1 for t in range(len(tids))],
                        dim=1,
                    )
                    lp = torch.log_softmax(logits, dim=-1)[:, tids]
                ranks[li] = rank.cpu().numpy()
                logprobs[li] = lp.cpu().numpy().astype(np.float16)
            arrays[f"s{si}_{lens_name}_rank"] = ranks
            arrays[f"s{si}_{lens_name}_logprob"] = logprobs

        audio_start = ids.index(audio_bos) if audio_bos in ids else -1
        audio_end = ids.index(audio_eos) if audio_eos in ids else -1
        arrays[f"s{si}_input_ids"] = np.asarray(ids, dtype=np.int64)
        meta_samples.append({
            "sample_id": sample.sample_id,
            "true": true_words[si],
            "true_index": candidates.index(true_words[si]),
            "audio_start": audio_start,
            "audio_end": audio_end,
            "prompt_len": prompt_len,
            "text_prompt": sample.prompt,
            "tokens": [tokenizer.decode([t]) for t in ids],
        })
        print(f"[{si + 1}/{len(samples)}] sample {sample.sample_id} "
              f"({true_words[si]}), {prompt_len} positions", flush=True)

    meta = {
        "label_key": label_key,
        "candidates": candidates,
        "forms": list(FORMS),
        "token_ids": token_ids.tolist(),
        "token_counts": token_counts.tolist(),
        "token_strings": [[tokenizer.decode([int(t)]) for t in row] for row in token_ids],
        "unique_ids": unique_ids,
        "col_index": col_index.tolist(),
        "layers": layers,
        "lens_paths": args.lens_paths,
        "prompts_path": args.prompts_path,
        "samples": meta_samples,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, meta=np.asarray(json.dumps(meta)), **arrays)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# plot (numpy + matplotlib only)
# ---------------------------------------------------------------------------
def _segments(s: dict) -> list[tuple[str, np.ndarray]]:
    """Split positions into: before audio, N audio bins, after audio."""
    p = s["prompt_len"]
    a0, a1 = s["audio_start"], s["audio_end"]
    if a0 < 0 or a1 < 0:
        return [("all", np.arange(p))]
    segs = [("pre", np.arange(0, a0))]
    audio = np.arange(a0 + 1, a1)  # exclude bos/eos markers
    for b, chunk in enumerate(np.array_split(audio, N_AUDIO_BINS)):
        segs.append((f"a{b}", chunk))
    segs.append(("post", np.arange(a1 + 1, p)))
    return segs


def plot(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(args.input)
    meta = json.loads(str(data["meta"]))
    cands, layers, samples = meta["candidates"], meta["layers"], meta["samples"]
    col = np.asarray(meta["col_index"])  # [C, F]
    C = len(cands)
    has_wrong = C >= 2   # with one candidate there is no wrong-label control
    if not has_wrong:
        print("Only one candidate label: skipping win rate and true-vs-wrong plots. "
              "Use --extra-candidates in dump to add controls.")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seg_names = [n for n, _ in _segments(samples[0])]
    report: dict[str, object] = {"candidates": cands, "token_strings": meta["token_strings"]}

    for lens in LENSES:
        # concept score = best over surface forms
        heat = np.full((len(samples), len(layers), len(seg_names)), np.nan)
        win = np.full_like(heat, np.nan)
        min_true = np.zeros((len(samples), len(layers)))
        min_wrong = np.zeros((len(samples), len(layers)))
        form_min = np.zeros((len(samples), len(layers), len(FORMS)))
        best_seg = []
        for si, s in enumerate(samples):
            rank = data[f"s{si}_{lens}_rank"]            # [L, P, T]
            lp = data[f"s{si}_{lens}_logprob"].astype(np.float32)
            cand_rank = rank[:, :, col].min(axis=-1)     # [L, P, C]
            cand_lp = lp[:, :, col].max(axis=-1)         # [L, P, C]
            t = s["true_index"]
            wrong = [c for c in range(C) if c != t]
            true_rank = cand_rank[:, :, t]               # [L, P]
            is_win = cand_lp.argmax(axis=-1) == t        # [L, P]
            for gi, (_, pos) in enumerate(_segments(s)):
                if len(pos):
                    heat[si, :, gi] = np.median(np.log10(true_rank[:, pos]), axis=1)
                    win[si, :, gi] = is_win[:, pos].mean(axis=1)
            min_true[si] = true_rank.min(axis=1)
            min_wrong[si] = (np.median(cand_rank[:, :, wrong].min(axis=1), axis=-1)
                             if has_wrong else np.nan)
            form_min[si] = rank[:, :, col[t]].min(axis=1)
            flat = np.argmin(true_rank)
            _, best_p = np.unravel_index(flat, true_rank.shape)
            for name, pos in _segments(s):
                if best_p in pos:
                    best_seg.append(name)
                    break
            else:
                best_seg.append("audio_marker")

        # 1. median log10 rank of the true label over layer x position segment
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        im = axes[0].imshow(np.nanmedian(heat, axis=0), aspect="auto", cmap="viridis_r")
        axes[0].set(title=f"{lens}: median log10 rank of true label",
                    xlabel="position segment", ylabel="layer")
        fig.colorbar(im, ax=axes[0])
        if has_wrong:
            diff = np.nanmean(win, axis=0) - 1 / C
            lim = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)), 1e-6)
            im = axes[1].imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
            axes[1].set(title=f"{lens}: win rate among {C} candidates minus chance ({1 / C:.2f})",
                        xlabel="position segment", ylabel="layer")
            fig.colorbar(im, ax=axes[1])
        else:
            axes[1].set_title("win rate: needs >= 2 candidates")
        for ax in axes:
            ax.set_xticks(range(len(seg_names)), seg_names, rotation=45)
            ax.set_yticks(range(len(layers)), layers, fontsize=6)
        fig.tight_layout()
        fig.savefig(out / f"{lens}_heatmaps.png", dpi=150)
        plt.close(fig)

        # 2. current pass@k rule (min over positions) for true vs wrong labels
        fig, ax = plt.subplots(figsize=(8, 5))
        curves = [(min_true, "true label")]
        if has_wrong:
            curves.append((min_wrong, "wrong labels (median)"))
        for arr, name in curves:
            med = np.median(arr, axis=0)
            q1, q3 = np.percentile(arr, [25, 75], axis=0)
            ax.plot(layers, med, label=name)
            ax.fill_between(layers, q1, q3, alpha=0.2)
        ax.set(yscale="log", xlabel="layer", ylabel="min rank over positions",
               title=f"{lens}: min-over-positions rank, true vs wrong")
        ax.invert_yaxis()
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{lens}_min_true_vs_wrong.png", dpi=150)
        plt.close(fig)

        # 3. surface-form sensitivity
        fig, ax = plt.subplots(figsize=(8, 5))
        for fi, form in enumerate(FORMS):
            ax.plot(layers, np.median(form_min[:, :, fi], axis=0), label=form)
        ax.set(yscale="log", xlabel="layer", ylabel="median min rank",
               title=f"{lens}: true label by surface form")
        ax.invert_yaxis()
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{lens}_forms.png", dpi=150)
        plt.close(fig)

        names, counts = np.unique(best_seg, return_counts=True)
        report[lens] = {
            "best_position_segment_counts": dict(zip(names.tolist(), counts.tolist())),
            "median_min_rank_true_by_layer": dict(zip(layers, np.median(min_true, 0).tolist())),
            "median_min_rank_wrong_by_layer": (
                dict(zip(layers, np.median(min_wrong, 0).tolist())) if has_wrong else None
            ),
        }

    (out / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    for si in args.samples:
        if 0 <= si < len(samples):
            _plot_one_sample(plt, data, meta, si, out / "per_sample")
    print(f"Saved plots and summary.json to {out}")


def _plot_one_sample(plt, data, meta: dict, si: int, out: Path) -> None:
    """Layer x position heatmaps of the true label's rank and probability."""
    from matplotlib.colors import LogNorm

    out.mkdir(parents=True, exist_ok=True)
    s = meta["samples"][si]
    t, layers = s["true_index"], meta["layers"]
    col = np.asarray(meta["col_index"])[t]          # this label's 4 forms
    tokens = s.get("tokens")
    a0, a1, P = s["audio_start"], s["audio_end"], s["prompt_len"]

    for lens in LENSES:
        rank = data[f"s{si}_{lens}_rank"][:, :, col].min(axis=-1)                 # [L, P]
        prob = np.exp(data[f"s{si}_{lens}_logprob"][:, :, col].astype(np.float32)).max(axis=-1) * 100

        fig, axes = plt.subplots(2, 1, figsize=(max(12, P / 12), 10), sharex=True)
        panels = (
            (rank, "rank (1 = top, log color)", "viridis_r", LogNorm(vmin=1, vmax=max(rank.max(), 2))),
            (prob, "probability % (log color)", "magma",
             LogNorm(vmin=max(prob.min(), 1e-6), vmax=max(prob.max(), 1e-5))),
        )
        for ax, (grid, name, cmap, norm) in zip(axes, panels):
            im = ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm,
                           origin="lower", interpolation="nearest")
            fig.colorbar(im, ax=ax, label=name)
            ax.set_ylabel("layer")
            ax.set_yticks(range(len(layers))[::2], layers[::2])
            for x in (a0, a1):
                if x >= 0:
                    ax.axvline(x, color="cyan", lw=1, ls="--")
            best_l, best_p = np.unravel_index(np.argmin(rank), rank.shape)
            ax.plot(best_p, best_l, marker="x", color="red", ms=10, mew=2)

        # x ticks: every token outside the audio, sparse ticks inside it
        if tokens:
            ticks = [p for p in range(P) if not (a0 < p < a1)]
            ticks += list(range(a0 + 1, a1, 10)) if a0 >= 0 and a1 >= 0 else []
            ticks = sorted(set(ticks))
            labels = [repr(tokens[p])[1:-1][:12] if not (a0 < p < a1) else str(p) for p in ticks]
            axes[1].set_xticks(ticks, labels, rotation=90, fontsize=6)
        axes[1].set_xlabel("position (cyan dashes = audio start / end, red x = best rank)")
        axes[0].set_title(
            f"{lens} | sample {s['sample_id']} | true = {s['true']} "
            f"(best of {meta['forms']}) | min rank = {rank.min()}"
        )
        fig.tight_layout()
        fig.savefig(out / f"sample{si}_{s['sample_id']}_{lens}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dump", help="Run the model and save rank grids (GPU).")
    d.add_argument("--lens-path", action="append", dest="lens_paths", required=True)
    d.add_argument("--prompts-path", required=True)
    d.add_argument("--target", required=True, help="e.g. label.emotion")
    d.add_argument("--max-prompts", type=int, default=10,
                   help="Subset size; labels are balanced round-robin.")
    d.add_argument("--per-label", type=int, default=0,
                   help="If set, take this many samples per label value (overrides --max-prompts).")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--extra-candidates", nargs="*", default=[],
                   help="Extra label values scored as wrong-label controls, e.g. english chinese.")
    d.add_argument("--settings-file", default="settings.json")
    d.add_argument("--output", required=True)
    p = sub.add_parser("plot", help="Plot a dump file.")
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--samples", type=int, nargs="*", default=[0, 1, 2],
                   help="Sample indices (in dump order) to draw per-prompt heatmaps for.")
    args = parser.parse_args()
    if args.command == "dump":
        dump(args)
    else:
        plot(args)


if __name__ == "__main__":
    main()
