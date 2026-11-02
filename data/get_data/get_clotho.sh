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
	--dataset mteb/Clotho \
	--split test \
	--target 500 \
	--seed 42 \
	--output data/datasets/clotho \
	--audio-column audio \
	--transcription text \
	--overwrite