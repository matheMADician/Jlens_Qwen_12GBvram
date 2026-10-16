#!/bin/bash

# Usage: sbatch job.sh
# To see the queue: squeue --me
# To see the progress: scontrol show job <JOBID>
# To see logs: cat slurm-<JOBID>.out

#SBATCH --account=mst113234
#SBATCH --partition=dev
#SBATCH --gpus-per-node=1
#SBATCH --output=output/logs/%x-%j.out

export HF_HOME="/work/c8763c8763/huggingface-cache"
mkdir -p "$HF_HOME"

/home/c8763c8763/.conda/envs/jlens/bin/python /home/c8763c8763/Jlens/Jlens/main.py --run-name train_1_no_SAKURA_On_Animal