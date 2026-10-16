#!/usr/bin/env bash

# SAKURA is reserved for testing and already contains user prompts.
# Keep the seed fixed for reproducibility; --override prevents duplicate
# records from accumulating when this script is rerun.
seed=1145141919810
output=data/assembled/train_2_no_SAKURA
data_root=data/datasets
prompt_root=data/llama

python data/assemble_json.py \
        --dataset $data_root/Clotho \
        --dataset_name Clotho \
        --count 15 \
        --seed $seed \
        --prompt-file $prompt_root/environment_single.txt \
        --output $output

python data/assemble_json.py \
        --dataset $data_root/esc50 \
        --dataset_name esc50 \
        --count 15 \
        --seed $seed \
        --prompt-file $prompt_root/environment_single.txt \
        --output $output

python data/assemble_json.py \
        --dataset $data_root/fleurs_en_us \
        --dataset_name fleurs \
        --count 30 \
        --seed $seed \
        --prompt-file $prompt_root/speech_single.txt \
        --output $output