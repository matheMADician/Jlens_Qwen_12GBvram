#!/usr/bin/env bash

# Dev log: assemble the ESD validation set while excluding source indices used
# for ESD training. A count of 100 selects one record from each observed joint
# emotion/language/gender/speaker_id combination.
#
# To refresh the ESD training exclusion list after changing its training set:
#      python data/build_used_samples.py \
#        --input data/assembled/train/esd/prompts.jsonl \
#        --output data/assembled/train/esd/used_samples.jsonl
# To assemble validation data from the same ESD pool:
#      python data/assemble_json.py \
#        --dataset data/datasets/esd \
#        --dataset_name ESD \
#        --seed 43 \
#        --count 100 \
#        --stratify-labels emotion language gender speaker_id \
#        --used-samples data/assembled/train/esd/used_samples.jsonl \
#        --output data/assembled/test/esd \
#        --empty-prompt \
#        --override
#
# --override replaces the destination prompts.jsonl. Remove it only when you
# intentionally want to append to an existing manifest.
# --empty-prompt explicitly permits audio-only evaluation records when the
# source manifest has no non-empty text prompt. Do not pass --prompt-file too.
set -euo pipefail

PYTHON="${PYTHON:-python}"
datasets=(
    "de_de"
)
dataset_root=data/datasets/fleurs
output_root=data/assembled/test/fleurs_de

for dataset in "${datasets[@]}"; do
    output="$output_root/$dataset"
    "$PYTHON" data/assemble_json.py \
        --dataset "$dataset_root/$dataset" \
        --dataset_name "$dataset" \
        --seed 42 \
        --count 50 \
        --label-overrides language=german \
        --output "$output" \
        --override \
        --empty-prompt
done

"$PYTHON" data/build_used_samples.py \
    --input "$output_root/de_de/prompts.jsonl" \
    --output "$output_root/de_de/used_samples.jsonl"

# Prompt file is omitted for validation set for now
# Since text prompt might disturb the readout
#TODO The current method has to be changed, but I haven't figured out the best approach yet.

cp "$0" "$output_root/assemble_json.sh"