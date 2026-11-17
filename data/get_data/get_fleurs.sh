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

languages=(
	"fi_fi"
	"ka_ge"
	"tr_tr"
	"ar_eg"
	"hi_in"
	"ta_in"
	"th_th"
	"vi_vn"
	"id_id"
	"ja_jp"
	"cmn_hans_cn"
	"cmn_hant_tw"
	"cmn_hant_hk"
	"en_us"
	"de_de"
)

for lang in "${languages[@]}"; do
	python data/get_data.py \
		--dataset google/fleurs \
		--config "$lang" \
		--split train \
		--target 100 \
		--seed 42 \
		--output "data/datasets/fleurs/$lang" \
		--audio-column audio \
		--transcription transcription \
		--overwrite
done

cp "$0" "data/datasets/fleurs/get_fleurs.sh"