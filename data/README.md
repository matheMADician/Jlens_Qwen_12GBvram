## Downloading

Use `get_data.sh` to download datasets into model-agnostic chat conversations.
Each dataset directory contains `audio/` and a `prompts.jsonl` manifest.

The generated rows use this structure. The top-level `transcript` is retained
as dataset metadata; the user prompt is part of `messages`.

```json
{
  "id": "sample-000000",
  "audio_path": "audio/sample-000000.wav",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "audio", "path": "audio/sample-000000.wav"},
        {"type": "text", "text": ""}
      ]
    }
  ],
  "transcript": "reference transcript"
}
```

## Assembling a sampled manifest

`assemble_json.py` samples from one dataset directory per run. The output
manifest is `prompts.jsonl` inside the directory passed to `--output`.
Audio is not copied; audio paths are rewritten relative to that output
directory. The assembler also writes `dataset_counts.txt` beside the manifest,
listing the total number of records for each `dataset_name` in the assembled
manifest.

To assign prompts from a numbered or one-prompt-per-line text file:

```bash
python data/assemble_json.py \
  --dataset data/datasets/Clotho \
  --dataset_name Clotho \
  --count 50 \
  --prompt-file data/llama/environment_sound_prompt.txt \
  --seed 42 \
  --output data/assembled
```

When `--prompt-file` is omitted, only source records with a non-empty prompt
already in a user message are eligible; the command fails if none are present.
Each selected record receives a randomly selected prompt from the file, with
replacement, so prompt files may contain fewer prompts than the requested
record count.

`--dataset_name` is required and is stored on each record as a top-level field,
separate from `source_dataset` (the source folder path/name) and `source_id`
(the original record ID).

By default, records append to an existing output manifest. Pass `--override`
to replace it. Use the same seed and `--override` to reproduce a result
without accumulating duplicate rows. The source dataset name and source record
ID are retained in `source_dataset` and `source_id`.

`assemble_json.sh` demonstrates sampling SAKURA's pre-prompted AnimalQA data
for testing; SAKURA is not included in training data by default.

## Downloading CREMA-D

Run `bash data/get_cremad.sh` from the repository root to sample 1,000 clips
from the `mteb/crema-d` `train` split into `data/datasets/crema_d`. Sampling is
random within each emotion class and approximately balanced across the six
labels (166 or 167 clips per class); the seed is fixed at 42. The label is
written as a human-readable emotion name to each record's top-level
`transcript` field.

The script uses `--stratify-column` in `get_data.py`. Stratified sampling needs
to inspect all labels, so it loads the full split locally before saving only
the selected audio files. Use `--stratify-column` with `--target` for other
categorical datasets; `--seed` controls the reproducible random selection.

The data download scripts and `main.py` default Hugging Face's cache location
to `/work/c8763c8763/huggingface-cache` when `HF_HOME` is not already set.
An explicitly exported `HF_HOME` takes precedence. Existing cache directories
are not moved or deleted.
