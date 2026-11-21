#!/usr/bin/env bash
# Download SAKURA GenderQA with the single-hop question as the prompt.
# Run from the repo root. Next step: python -m stroop.data_prep.make_sakura_sets
#
# The trunk keeps its own copy (data/datasets/SAKURA/GenderQA, multi-hop prompts);
# this copy lives under data/datasets/stroop/ so the two never mix.
set -euo pipefail

python data/get_data.py \
	--dataset SLLM-multi-hop/GenderQA \
	--split test \
	--output data/datasets/stroop/sakura_gender \
	--audio-column audio \
	--transcription attribute_label \
	--prompt-column single_instruction \
	--overwrite
