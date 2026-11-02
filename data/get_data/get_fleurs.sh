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
	--dataset google/fleurs \
	--config cmn_hans_cn \
	--split train \
	--target 250 \
	--seed 42 \
	--output data/datasets/fleurs/cmn_hans_cn \
	--audio-column audio \
	--transcription transcription \
	--overwrite

python data/get_data.py \
	--dataset google/fleurs \
	--config en_us \
	--split train \
	--target 250 \
	--seed 42 \
	--output data/datasets/fleurs/en_us \
	--audio-column audio \
	--transcription transcription \
	--overwrite

python data/get_data.py \
	--dataset google/fleurs \
	--config de_de \
	--split train \
	--target 250 \
	--seed 42 \
	--output data/datasets/fleurs/de_de \
	--audio-column audio \
	--transcription transcription \
	--overwrite