#!/usr/bin/env bash
set -euo pipefail

python data/get_data.py \
	--dataset mteb/crema-d \
	--config default \
	--split train \
	--target 1000 \
	--stratify-column label \
	--seed 42 \
	--output data/datasets/crema_d \
	--audio-column audio \
	--transcription label \
	--overwrite
