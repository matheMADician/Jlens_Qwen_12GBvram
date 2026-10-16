#!/usr/bin/env bash

# This may take some time. If you see the progress bar stuck, it is likely still downloading data in the background.

# ``--dataset``: Hugging Face dataset ID (required).
# ``--config``: Dataset configuration, if required by the dataset.
# ``--split``: Dataset split to download (default: ``train``).
# ``--target``: Maximum number of samples to save; omit to process the full split.
# ``--output``: Output directory for audio and ``prompts.jsonl``.
# ``--audio-column``: Dataset column containing audio (default: ``audio``).
# ``--transcription``: Dataset column for the top-level transcript field;
#   auto-detected from transcription-specific column names if omitted.
# ``--prompt-column``: Dataset column for user instructions; prompt is empty if omitted.
# ``--min-duration`` / ``--max-duration``: Optional audio-duration filters in seconds.
# ``--sample-rate``: Output audio sample rate in Hz (default: ``16000``).
# ``--no-streaming``: Load the complete split instead of streaming it.
# ``--overwrite``: Replace an existing ``prompts.jsonl`` file.


python data/get_data.py \
	--dataset SLLM-multi-hop/AnimalQA \
	--config default \
	--split test \
	--target 500 \
	--output data/datasets/SAKURA/AnimalQA \
	--audio-column audio \
	--transcription single_answer \
	--prompt-column multi_instruction \
	--overwrite

python data/get_data.py \
	--dataset SLLM-multi-hop/GenderQA \
	--config default \
	--split test \
	--target 500 \
	--output data/datasets/SAKURA/GenderQA \
	--audio-column audio \
	--transcription single_answer \
	--prompt-column multi_instruction \
	--overwrite

python data/get_data.py \
	--dataset SLLM-multi-hop/EmotionQA \
	--config default \
	--split test \
	--target 500 \
	--output data/datasets/SAKURA/EmotionQA \
	--audio-column audio \
	--transcription single_answer \
	--prompt-column multi_instruction \
	--overwrite

python data/get_data.py \
	--dataset SLLM-multi-hop/LanguageQA \
	--config default \
	--split test \
	--target 500 \
	--output data/datasets/SAKURA/LanguageQA \
	--audio-column audio \
	--transcription single_answer \
	--prompt-column multi_instruction \
	--overwrite