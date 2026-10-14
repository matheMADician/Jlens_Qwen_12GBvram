#!/usr/bin/env bash

# SAKURA is reserved for testing and already contains user prompts.
# Keep the seed fixed for reproducibility; --override prevents duplicate
# records from accumulating when this script is rerun.
seed=1145141919810
output=data/assembled/train_1_no_SAKURA
data_root=data/datasets
prompt_root=data/llama

python data/assemble_json.py \
        --dataset $data_root/Clotho \
        --dataset_name Clotho \
		--prompt-file $prompt_root/environment_sound_prompt.txt \
        --count 150 \
        --seed $seed \
        --output $output

python data/assemble_json.py \
        --dataset $data_root/esc50 \
        --dataset_name ESC50 \
		--prompt-file $prompt_root/environment_sound_prompt.txt \
        --count 150 \
        --seed $seed \
        --output $output

python data/assemble_json.py \
        --dataset $data_root/fleurs_en_us \
        --dataset_name Fleurs_en_us \
		--prompt-file $prompt_root/speech_general_prompt.txt \
        --count 300 \
        --seed $seed \
        --output $output

