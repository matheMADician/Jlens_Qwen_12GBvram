#!/usr/bin/env bash

# SAKURA is reserved for testing and already contains user prompts.
# Keep the seed fixed for reproducibility; --override prevents duplicate
# records from accumulating when this script is rerun.
seed=42
splits=(
    "AnimalQA"
    "EmotionQA"
    "GenderQA"
    "LanguageQA"
)
output=data/assembled/test_SAKURA
data_root=data/datasets/SAKURA

for split in "${splits[@]}"; do
    python data/assemble_json.py \
        --dataset $data_root/$split \
        --dataset_name SAKURA_$split \
        --seed $seed \
        --count 100 \
        --output $output/$split/ \
        --override
done