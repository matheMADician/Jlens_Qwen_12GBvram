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
