# stroop — context for AI assistants

This file is the detailed brief for any AI assistant (Claude Code, Copilot, Codex, ...) working in `stroop/`.
Human-facing docs are `stroop/README.md` (what/why) and `stroop/docs/TODO.md` (status). Keep those short; put detail here.
Last revised: 2026-11-16.

---

## 1. Ownership and working rules

- Owner: RaTa0314. Branch: `stroop` (created from `main` @ `7f8b17e`).
- The rest of the repo ("trunk") is shared with teammates. A teammate (matheMADician) has unpushed work that touches
  `main.py` and `tools/experiments.py` (commit `2224c67`, adds pass@k targets `transcript | label.<field>`).
  Those two files are merge-conflict hot spots.
- **Do not edit trunk files** (`main.py`, `Master.py`, `tools/*`, `model/*`, `audio_jlens/*`, `data/get_data.py`,
  `json_parser.py`, `settings.json`) from stroop work. Import from them. If a trunk function does not fit, wrap it in
  `stroop/` instead of changing it. If a trunk change is truly needed, stop and ask the owner.
- **Do not copy trunk code** into `stroop/`; copies drift from fixes made upstream.
- Run scripts from the repo root as modules: `python -m stroop.<script>` so `from model.model import Model` works.
- Do not commit or push unless the owner asks.
- Human docs are written in Traditional Chinese and must stay short. Code, docstrings, and this file may be English.

### File layout

```
Jlens_Qwen_12GBvram/
├── stroop/                 code, prompt files, small configs (committed)
│   ├── README.md           human: purpose, experiments, how to run (short)
│   ├── CLAUDE.md           one line, imports docs/AGENTS.md for Claude Code (must stay here to auto-load)
│   ├── docs/
│   │   ├── TODO.md         human: status checklist (short)
│   │   └── AGENTS.md       this file
│   ├── data_prep/          code that builds stimuli/JSONL (code only; data goes to data/datasets/stroop/)
│   │   ├── gender_labels.py      pure gender-labelling logic
│   │   └── get_stroop_libri.py   LibriSpeech stimulus builder
│   ├── experiments/        experiment code
│   │   ├── readout.py            shared: gender token sets, per-position ranks, win rate, summaries
│   │   └── pilot_readout.py      Step 0
│   ├── tests/              unittest tests (torch tests skip when torch is missing)
│   └── prompts/            plain-text prompt lists, one prompt per line
├── data/datasets/stroop/   downloaded audio + prompts.jsonl (NOT committed)
└── output/stroop/          lenses, CSV/JSON results, figures (NOT committed)
```

`.gitignore` already ignores `/output/`, `*.pt`, `/data/**/*.wav|mp3|flac`, and **`/data/**/*.jsonl`**.
Consequence: any JSONL that must be versioned (e.g. a tiny fixture for tests) must live outside `data/`,
for example `stroop/fixtures/`. Prompt lists in `stroop/prompts/` are `.txt` and are committed.

---

## 2. Research question

**Stroop analogy.** In the Stroop task the word "RED" is printed in blue ink; word meaning and ink colour conflict.
The LALM analogue: a female voice says a male-referring word ("he", "the man", "I am a man").
Two kinds of gender information coexist: **speaker gender** (acoustic) and **content gender** (semantic).

**Core hypothesis.** A J lens is an averaged Jacobian, `J_l = E_{prompts, t, t'≥t}[∂h_final,t' / ∂h_l,t]`.
Because the average is taken over a data distribution, the lens is (approximately) the best linear readout of the
model's expected future output *under that distribution* (see Yan et al., §3, Stein-lemma argument). Different
tasks engage different downstream circuits (attention patterns, MLP gating), so the local linearisation differs.
Therefore:

**Design (decided 2026-11-16).** Two lenses fit on the **same audio** (SAKURA GenderQA, 500 clips) with different
prompts, so the only difference is the task:
- `L_gender`: audio + a gender question from `stroop/prompts/ask_gender.txt`. **Not** SAKURA's
  `single_instruction`: its options "(a) Male (b) Female" put gender words in the prompt text, and S2T targets are
  exactly those text positions, so the lens could learn the option tokens instead of the audio. The transcription
  prompt has no gender words, so prompts must be gender-word-free on both sides.
- `L_trans`: audio + a transcription prompt (`stroop/prompts/transcribe.txt`).
Fitting a J lens needs no labels, so SAKURA's lack of transcripts does not matter for `L_trans`.

- H1: `L_gender` yields token directions `d_l(tok) = J_lᵀ u_tok` for {male, female, ...} that differ from those
  of `L_trans` by more than fitting noise.
- H2: on LibriSpeech conflict stimuli under a *transcription* prompt (gender never asked), `L_gender` reads speaker
  gender at audio positions while `L_trans` reads content gender.
- H3 (causal): swapping along the `L_gender` direction flips the speaker-gender answer but not the transcript;
  swapping along the `L_trans` direction does the opposite (double dissociation).

**Why same audio is clean.** The chat format puts audio before the text prompt and the LM is causal, so hidden
states at audio positions are *identical* under both prompts. Only the downstream computation (what the text
positions do with those states) differs, so any difference between `L_gender` and `L_trans` at audio positions is
purely a difference in readout — exactly the claim being tested.

**Both outcomes are reportable.**
- Directions separate → task-dependent lenses can disentangle acoustic vs semantic information; possibly a
  label-free attribute readout tool.
- Directions converge in the workspace band → the model maps acoustic gender into the same verbalisable concept as
  text ("global workspace" across modalities). Still a finding.
- The bad outcome is "not found": speaker gender is not in the J-space at audio positions at all. Step 0 tests this
  cheaply before anything expensive is built.

**Why this is novel.** The original J-lens paper fits one generic text corpus and only tested method variants and
restricting the text distribution (no meaningful gain). It never compared lenses fit on different tasks or
modalities. AudioLens (the owner's mentor's paper) used a fixed readout (logit lens) at the final text token only.

---

## 3. Background papers (verified 2026-11-16)

1. **Anthropic, "Verbalizable Representations Form a Global Workspace in Language Models"** (2026),
   https://transformer-circuits.pub/2026/workspace/index.html, code: https://github.com/anthropics/jacobian-lens
   (vendored in `jacobian_lens/`).
   - Lens: `lens_l(h) = unembed(norm(J_l h))`, `J_l = E[∂h_final/∂h_l]`, averaged over prompts, source positions,
     and current+future target positions. Default corpus: 1000 sequences × 128 tokens, pretraining-like.
   - Beats logit/tuned lens with as few as ~10 prompts; small gains beyond. So ~100–200 prompts per lens is a
     reasonable budget.
   - Workspace band ≈ depth 38–92 (on a 0–100 scale) **for Claude**; may not transfer to Qwen.
   - First third of layers: readouts noisy / uninterpretable.
   - J-space carries a median of only 6–7% of a concept vector's variance → much information is not verbalisable.
   - Only single-token concepts are readable.
   - Swap experiments: clamped lens-coordinate swap; text continuation and anomaly detection are "automatic"
     (largely unmoved by swaps).
2. **Yan et al., "Short Horizons and Sparse Concepts: a Mathematical View of the Readout in the J-lens"**,
   arXiv:2608.25347. Theory: J ≈ least-squares slope of future output on h (exact under Gaussian h). Jacobian energy
   splits into diagonal (t'≈t, "short horizon", next-token-like) and sparse critical positions ("sparse concepts",
   intermediate concepts). Their diagonal-mask decoupling failed. No intervention experiments. The owner considers
   the experiments weak but the theory useful.
3. **Chih-Kai Yang, Neo Ho, Yi-Jyun Lee, Hung-yi Lee, "AudioLens"**, arXiv:2506.05140 (ASRU 2025). Logit lens on
   LALMs (incl. Qwen2-Audio-Instruct), SAKURA data, reads at the final text token "is". Findings: for wrong
   predictions attribute info peaks then fades with depth; earlier resolution ↔ higher accuracy; masking audio at the
   last token hurts → models query audio directly rather than aggregating at attribute-mentioning positions.
   **Chih-Kai Yang is the owner's mentor.**
4. **Chih-Kai Yang, Neo Ho, Yen-Ting Piao, Hung-yi Lee, "SAKURA"**, arXiv:2505.13237 (Interspeech 2025 oral).
   Multi-hop reasoning benchmark over speech/audio: gender, language, emotion, animal; single-hop vs multi-hop
   questions on the same audio. LALMs often perceive correctly but fail to use the info in multi-hop reasoning.
5. Disentanglement references (encoder-side, trained; not directly our method): AutoVC (Qian 2019), AdaIN-VC
   (Chou, Yeh, Lee 2019), SpeechSplit (Qian 2020). Concept-erasure tools (LEACE, INLP) are a possible extra causal
   test: erase a gender subspace from audio-token residuals and check lens readout + transcript.

---

## 4. Trunk APIs used (verified against `main` @ 7f8b17e)

- `data/get_data.py`: `save_audio(audio, destination, sample_rate) -> duration`,
  `build_messages(audio_path, instruction) -> messages`, `decode_audio`, `stratified_sample_indices`.
  `HF_HOME` defaults to `/work/c8763c8763/huggingface-cache` (server path).
  It keeps only one column as `transcript`; it drops `speaker_id` and anything else.
- `data/assemble_json.py`: samples a downloaded dataset into an assembled JSONL; can assign random prompts from a
  text file (`--prompt-file`); output paths are relative to the data root.
- `json_parser.Parser`: `load_jsonl_lines(path)`, `parse_json_line(line, audio_root=...) -> ParsedSample`,
  `build_chat_prompt(processor, text, audio_path)`.
  `ParsedSample` fields on main: `sample_id, audio_path, prompt, transcript, dataset_name`.
  **No `label` field on main yet**; the teammate's commit 2224c67 reads `sample.label` for `label.<field>` targets.
  Until that lands, stroop code should read extra fields from the raw JSON record itself.
- `model.model.Model`:
  - `load_Jlens(lens_path=..., lens_name=...)` loads one lens into `self.lens` (single slot). To use two lenses,
    load sequentially, or read the saved `.pt` jacobians directly for pure linear-algebra work.
  - `_token_directions(token_id, layers) -> {layer: unit vector}` computes `d_l = u J_l`, where `u` is the
    unembedding row scaled by the final-norm weight (`Instance.get_unembedding_direction`). Private-named but fine
    to call; or reimplement the 3-line formula on raw jacobians in stroop.
  - `apply_lens(json_line, data_root, run_name, ..., use_Jlens=True)` fills `LensOutputCache` with per-layer,
    per-position logits; `LensOutputCache.find_token(lens, token_id)`.
  - `run_swap_inference(jsonl_line, data_root, source_token, target_token, layers, token_pos, read_positions,
    alpha, direction_source in {"jlens","logit","random"}, random_seed, clamp_to_clean) -> SwapInferenceResult`
    (written by the owner, commit 7f8b17e). Reads next-token logits at `read_positions` (default last).
  - `fit_Jlens_for_experiment(jsonl_path, seed, mode=None|"S2T"|"S2S"|"T2T"|"S&T2T", max_prompts, ...)`; CLI:
    `python main.py calc-jlens-experiment` with `settings.json -> fitting.experiment.mode`.
- `audio_jlens/subset_fitting.py`: `SubsetConfig` modes. With `text_start` = number of tokens before the text
  suffix (split at `<|audio_eos|>`):
  - `S2T`: source = positions `[0, text_start)` (audio + preamble), target = text positions.
  - `S2S`: all → all. `T2T`: text → text. `S&T2T`: all → text.
  - `mode=None` delegates to the vendored `jacobian_lens.fitting.fit` (standard estimator).
  - Skips the first 16 positions (attention sinks) and the final position.
- `tools/experiments.py` (main): `passk_Jlens` reports the first transcript token's **best rank over all prompt
  positions** and per layer. Not position-restricted. `swap_token_positions(input_ids, tokenizer, policy)` with
  policies `all | last | audio | after_audio` locates `<|audio_bos|> … <|audio_eos|>`; reuse it for audio-position
  restriction.
- `tools/experiments.py`: `draw_Jlens_cosine_heatmap` / `compare_jacobians` / `comp-jlens` compare **whole J
  matrices**, not per-token directions. Not a substitute for step 2.
- `settings.json` (main) currently: model `Qwen/Qwen2.5-Omni-7B`; fitting JSONL
  `data/assembled/train_upscale_1/prompts.jsonl`; `mode: null`; random projection on (128, seed 42).
  The general lens was fit on a mix of Clotho, ESC-50, Fleurs, CREMA-D (per the trunk's TODO.md); no speaker-gender questions.

Unverified, check before relying on it:
- Whether Qwen2.5-Omni uses the same `<|audio_bos|>/<|audio_eos|>` markers as Qwen2-Audio.
- What lenses / pass@k outputs already exist on the server (`output/` is not in the local checkout).
- (Resolved) SAKURA GenderQA has `single_instruction` / `single_answer`.

---

## 5. Design decisions and confounds

1. **One factor at a time.** `L_gender` and `L_trans` share audio, prompt count, model, and fitting mode; only the
   prompt differs. The trunk's general lens (`L_main`, mixed audio, mode=None) is used **only in the Step 0 pilot**,
   never in the main comparison, because it differs in data and estimator as well.
   **Use `mode="S2T"` for both lenses.** With `mode=None`, the average includes audio→audio Jacobian pairs. Those
   are identical under both prompts (causal LM, audio first), so they dilute any difference and bias the cosine
   toward 1, i.e. toward "cannot separate". S2T (source = audio, target = text positions) keeps exactly the part
   that depends on the task. If time allows, also fit both with mode=None to show the dilution effect.
2. **Noise baseline.** Two lenses always differ. Split the 500 clips into **disjoint** halves A/B (250 each) and fit
   `L_gender_A`, `L_gender_B`, `L_trans_A`, `L_trans_B`. Report `cos(L_gender, L_trans)` against
   `cos(L_gender_A, L_gender_B)` and `cos(L_trans_A, L_trans_B)` per layer. Also report the cross-task cosine on
   halves (`L_gender_A` vs `L_trans_B`) so every compared pair has the same prompt count.
   `settings.experiment.seed` only shuffles, so different seeds can overlap — split the JSONL into two files.
3. **Match prompt counts** across compared lenses (lens quality depends weakly on count; 250 per half is plenty).
   **SAKURA is used for fitting here**, so stroop results are not comparable with the trunk's SAKURA pass@k.
4. **Answer-leakage trap.** A lens fit where the model is about to say "male" may just read the upcoming answer.
   Evaluation of H2 must use a prompt that never asks about gender (transcription), and read at **audio
   positions**.
5. **Content gender: third person vs first person.** LibriSpeech gives "female voice reads *he said*". That makes
   content gender ≠ speaker gender (enough for H1/H2 lens readout), but it is not a self-contradiction. A true
   Stroop conflict about the speaker ("I am a man" in a female voice) needs first-person sentences (rare in
   audiobooks; may need TTS or targeted search). Label which kind each sample is.
6. **Token sets, not single tokens.** Semantic content uses man/woman/he/she; attribute answers use male/female.
   Compare token *sets*. Qwen BPE: leading-space variants (" male" vs "male") are different ids. Use the trunk
   helper `swap_single_token` logic (only keep words that are exactly one token in the needed form) and decide the
   form (with/without leading space, capitalised) from what the model actually emits in the answer.
7. **Layers.** Report all layers, but interpret mainly in the middle band; the first third is expected to be noise.
   Do not hard-code Claude's 38–92 band as truth for Qwen.
8. **Model choice** is open (Qwen2-Audio-7B-Instruct vs Qwen2.5-Omni-7B). Use whatever the existing general lens
   was fit for; every lens in one comparison must come from the same model.

---

## 6. Steps (spec)

### Step 0 — Pilot: is speaker gender readable at audio positions?

**Status (2026-11-16): `stroop/experiments/pilot_readout.py` written; aggregation unit-tested locally; the
torch/model path is untested (no torch locally) — first server run is the real test.**

How it works: one forward per clip with `jacobian_lens.jlens.hooks.ActivationRecorder` on all lens source
layers; for each layer, read the residual with the J lens (`JacobianLens.transport`) and the logit lens, then
`Instance.unembed` → full-vocab logits `[seq, vocab]` on the model device. Ranks are computed per position for
the correct and wrong gender token sets (`readout.GENDER_WORD_SETS`: `attr` = female/male, `broad` adds
woman/she/... and man/he/...). Regions: `audio` (strictly between the audio markers) and `text` (after
`<|audio_eos|>`), via trunk `Experiments.swap_token_positions`. Only small per-region stats are kept, so memory
stays at one `[seq, vocab]` tensor at a time (full-sequence lens outputs via `Model.apply_lens` would be ~GBs).

Run:
```
python -m stroop.experiments.pilot_readout --lens-path <full-size lens.pt>     --prompts data/assembled/stroop/sakura_gender/pilot_single.jsonl     --prompts data/assembled/stroop/sakura_gender/pilot_transcribe.jsonl     --run-name pilot_main_lens [--max-prompts 20 for a smoke test]
```
Model id/adapter come from `--settings` (default trunk `settings.json`); the lens must be fit for that model.

Read `summary.json` → per prompts file → entries keyed by lens/region/token_set/layer. Main number:
`balanced_win_rate` (mean of per-gender win rates; 0.5 = chance, robust to a fixed gender bias). Also
`hit@k` of the correct set's best rank. Decision: jlens/audio/attr clearly above 0.5 (and above logit_lens) in
middle layers on `pilot_transcribe` → continue. Only text region above chance → answer leakage.

Things to check on the first server run (assumptions not verifiable locally):
- `Instance.unembed` applies the final norm before `W_U` (LensModel protocol says it does).
- `recorder.activations[layer]` is `[batch, seq, d_model]` (used as `[0]`).
- Qwen2.5-Omni uses `<|audio_bos|>` / `<|audio_eos|>`; if not, `region_positions` raises.
- The trunk lens is full-size (projected lenses lack usable `transport`).
- `Instance.encode` ignores `max_length` (no truncation), so long clips use more memory, not fewer tokens.

Original spec:
- Data: SAKURA GenderQA with `single_instruction` (verified to exist); plus the same audio with a transcription
  prompt.
- Lens: existing trunk lens `L_main` (+ logit lens as control). Note the answer format is multiple choice
  ("(b) Female"), so read the rank of the gender word token set, not the first answer token.
- Readout: per layer, rank of the speaker-gender token (set) at audio positions only (min over audio positions, and
  also mean rank, to avoid search-space inflation). Compare with the trunk's all-position pass@k.
- Decision: readable at audio positions → continue. Only readable at text positions → likely answer leakage;
  rethink before Steps 1–4.
- Depends on teammate push (label targets) if reusing `passk_Jlens`; otherwise write a small standalone readout
  using `apply_lens` + `swap_token_positions(policy="audio")`.

### Data — `stroop/data_prep/get_stroop_libri.py` (do first; independent, CPU only)

**Status (2026-11-16): written, unit-tested locally, not yet run on real data.**
- Pure labelling logic is in `stroop/data_prep/gender_labels.py` (no heavy imports); the downloader imports `datasets`,
  `tqdm`, and `data.get_data` lazily inside `build()` so tests run without them.
- Tests: `python -m unittest discover -s stroop/tests -t .` (stdlib `unittest`; pytest is not installed locally,
  but pytest also collects these).
- Importing `data.get_data` sets `HF_HOME` to the server path `/work/c8763c8763/huggingface-cache` unless already
  set. When running locally, export `HF_HOME` first.
- `data/` has no `__init__.py`; `from data.get_data import ...` works as a namespace package when run from the
  repo root.
- LibriSpeech streams in speaker order, so `--max-per-speaker` (default 5 per speaker per cell) keeps cells from
  being dominated by the first few readers. Expect to scan many utterances for the male-voice/female-text cell and
  far more with `--first-person-only`.

Original spec:
- Source: HF `openslr/librispeech_asr` (columns include `speaker_id`, `chapter_id`, `text`; verify).
  Speaker gender from LibriSpeech `SPEAKERS.TXT` (OpenSLR resource 12; format `ID | SEX | SUBSET | MINUTES | NAME`,
  `;` comment lines). Verify location/format before coding against it.
- Filter: transcript contains gendered words; compute `text_gender` ∈ {male, female, mixed, none} from word lists
  (male: he, him, his, himself, man, men, boy, father, ...; female: she, her, hers, herself, woman, women, girl,
  mother, ...). Keep `male`/`female` only; drop `mixed`.
- Also record `person`: `first` if the sentence has "I am a man"-type self-reference, else `third`.
- Balance the 2×2 cells {speaker F/M} × {content F/M}; record counts per cell.
- Output: `data/datasets/stroop/libri/audio/*.wav` + `prompts.jsonl`, reusing `save_audio` and `build_messages`.
  Record schema (superset of trunk schema, so trunk loaders still work):
  ```json
  {"id": "...", "source_index": 0, "audio_path": "audio/x.wav",
   "messages": [{"role": "user", "content": [{"type": "audio", "path": "audio/x.wav"},
                                             {"type": "text", "text": ""}]}],
   "transcript": "<verbatim text>",
   "dataset_name": "stroop_libri",
   "label": {"speaker_gender": "female", "text_gender": "male", "person": "third",
             "speaker_id": 1234, "gender_words": ["he", "his"]}}
  ```
- Prompts are filled later (via `assemble_json.py --prompt-file stroop/prompts/<x>.txt` or a stroop helper):
  `prompts/ask_gender.txt` for fitting/asking, `prompts/transcribe.txt` for H2 readout.
- CLI args mirror `get_data.py` where sensible (`--split`, `--target`, `--seed`, `--output`, `--overwrite`).
  Streaming + filtering means scanning until the per-cell quota is met.

### Step 1 — Fit lenses

**Status (2026-11-16): data prep written + unit-tested, not run on real data.**
`bash stroop/data_prep/get_sakura_gender.sh` → `data/datasets/stroop/sakura_gender/`;
`python -m stroop.data_prep.make_sakura_sets` → `data/assembled/stroop/sakura_gender/` with
`pilot_single`, `pilot_transcribe`, `fit_{gender,trans}_{all,A,B}.jsonl`, `split.json`, `metadata.json`.
Audio paths are relative to `data/` (use `data_root=data`). `check_prompts` refuses fitting prompts that
contain gender words.

- Data: SAKURA GenderQA (HF `SLLM-multi-hop/GenderQA`, split `test`, 500 rows; columns `audio`,
  `attribute_label`, `single_instruction`, `single_answer`, `multi_instruction`, `multi_answer`; audio from
  Common Voice, 1.5–20.8 s). Download once with trunk `get_data.py --transcription attribute_label` (no prompt column).
  Build two JSONLs with the same ids and audio paths: prompts from `prompts/ask_gender.txt` and from
  `prompts/transcribe.txt`. Neither prompt set may contain gender words (see §2).
- Split ids into disjoint halves A/B (fixed seed), identical for both tasks.
- Fit with `mode="S2T"`: `L_gender`, `L_trans` (all 500), and `L_gender_A/B`, `L_trans_A/B`. Optionally mode=None.
- LibriSpeech is **test only** (Steps 3–4); never fit on it. If more fitting data is ever needed, use another
  dataset or speaker-disjoint LibriSpeech speakers.
- Use trunk CLI with a stroop-specific settings file (e.g. `stroop/settings_stroop.json`, passed via the trunk's
  settings flag) so `settings.json` is not edited. Turn off projection/t-SNE outputs to save time if not needed;
  swap/direction work needs **full-size** lenses (projected lenses are rejected by swap).
- Outputs under `output/stroop/lenses/`.

### Step 2 — `stroop/compare_directions.py` (the go/no-go test)
- Inputs: two or more lens `.pt` paths, a token word list, model id (for unembedding + final norm only).
- For each layer l and token w: `d_l^X(w) = normalize(u_w J_l^X)`. Metrics:
  - per-token cosine `cos(d^gender, d^trans)` vs noise `cos(d^gender_A, d^gender_B)`, `cos(d^trans_A, d^trans_B)`
  - subspace overlap of the token-set span (principal angles) — robust to single-token noise
  - also `cos(d^X(male), d^X(female))` within a lens (are the two genders separated at all?)
- Output: CSV (layer × token × pair) + one line plot per pair with the noise band.
- Pure linear algebra; can run on CPU after loading jacobians and the unembedding/norm weights.

### Step 3 — `stroop/conflict_readout.py`
- Data: LibriSpeech conflict cells (speaker ≠ content) and congruent cells as control; transcription prompt.
- For each lens: per layer, at audio positions, rank/logit of speaker-gender token set vs content-gender token set.
- Expected under H2: `L_gender` favours speaker gender, `L_trans` favours content gender; congruent cells show both.
- Note: content words occur at specific audio frames; speaker gender is global. Report position profiles too.

### Step 4 — Causal swap
- Reuse `run_swap_inference`, loading `L_gender` then `L_trans`.
- Two prompts per sample: ask-gender (read answer token) and transcription (read the first content-word position or
  generate). Swap female↔male along each lens's directions at audio positions, mid layers.
- Controls: `logit` and `random` directions (already supported), high-cosine exclusion (`max_direction_cosine`).
- Success = double dissociation (H3).

---

## 7. Conventions for code in stroop/

- Python ≥3.10 style matching trunk (type hints, dataclasses, `logging` not `print`).
- Each script: module docstring with purpose + example command; `argparse`; deterministic seeds; refuse to
  overwrite outputs without `--overwrite`.
- Save a JSON metadata file next to every result (inputs, lens paths, model id, seed, git commit, date).
- Tests in `stroop/tests/` mirroring trunk style (`tests/test_swap_tiny_model.py` shows how to test with a tiny
  model, no real weights). Pure functions (word filtering, gender labelling, cosine metrics) must have unit tests.
- When a step finishes, add one line to `docs/TODO.md` → "結果紀錄" and keep README in sync if the plan changes.
