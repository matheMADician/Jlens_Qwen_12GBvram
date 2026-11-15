#!/usr/bin/env bash

set -euo pipefail

# These are exact FSD50K vocabulary names. Clips matching multiple selected
# classes are excluded so each clip has one focus_class for balanced coloring.
classes=(
	"Dog"
	"Cat"
	"Bird_vocalization_and_bird_call_and_bird_song"
	"Insect"
	"Chicken_and_rooster"
	"Rain"
	"Waves_and_surf"
	"Wind"
	"Fire"
	"Door"
	"Train"
	"Aircraft"
	"Electric_guitar"
	"Piano"
	"Trumpet"
)

python data/download_fsd50k.py \
	--classes "${classes[@]}" \
	--samples-per-class 50 \
	--seed 42 \
	--output data/datasets/fsd50k_pilot \
	--overwrite
