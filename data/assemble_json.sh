#!/usr/bin/env bash

# Sample 300 records, balanced across FSD50K focus classes.
# Keep the seed fixed for reproducibility; --override prevents duplicate
# records from accumulating when this script is rerun.
set -euo pipefail

PYTHON="${PYTHON:-python}"
dataset=data/datasets/fsd50k_pilot
output=data/assembled/train/fsd50k

"$PYTHON" data/assemble_json.py \
    --dataset "$dataset" \
    --dataset_name FSD50K \
    --seed 42 \
    --count 300 \
    --stratify-labels focus_class \
    --output "$output" \
    --override \
    --prompt-file data/llama/environment_sound_prompt.txt

cp "$0" "$output/assemble_json.sh"