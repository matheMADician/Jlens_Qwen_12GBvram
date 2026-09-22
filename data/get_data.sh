#!bin/bash

# target: Maximum samples



python data/get_data.py \
	--dataset google/fleurs \
	--config en_us \
	--split train \
	--target 100 \
	--output data/fleurs_en_us \
	--audio-column audio \
	--text-column transcription \
	--prompt "<|audio_bos|><|AUDIO|><|audio_eos|>請寫出這段音訊的逐字稿：" \
	--overwrite