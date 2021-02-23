#!/bin/bash
#SBATCH -A atlas
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -c 10
#SBATCH -t 60

conda activate missing-mass-env
wandb agent vladov3000/MissingMassRegression-hwwlnulnu/dsa468hu
