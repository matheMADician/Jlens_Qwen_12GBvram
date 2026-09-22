#!/bin/bash

# WARNING: FOR REPRODUCIBILITY, DO NOT CHANGE THE SEED (這麼臭的種子有更改的必要嗎)
python data/assemble_json.py --seed 1145141919810 \
        --dataset data/SAKURA_animal:5 \
        --output data/