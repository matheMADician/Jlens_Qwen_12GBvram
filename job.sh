#!/bin/bash

# Usage: sbatch job.sh
# To see the queue: squeue --me
# To see the progress: scontrol show job <JOBID>
# To see logs: cat slurm-<JOBID>.out

#SBATCH --account=mst113234
#SBATCH --partition=dev
#SBATCH --gpus-per-node=1

/home/c8763c8763/.conda/envs/jlens/bin/python /home/c8763c8763/Jlens/Jlens/main.py --run-name esc50-50_fleurs_en_us-50_X_esc50-50_libri-50