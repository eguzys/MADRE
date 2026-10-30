#!/usr/bin/env bash
#SBATCH --job-name=eval_all
#SBATCH --account=projects
#SBATCH --gpus=5060ti:1
#SBATCH --time=08:00:00
#SBATCH --output=eval_all_%j.out

source /work/projects/eguzys/miniconda3/etc/profile.d/conda.sh
conda activate /work/scratch/eguzys/padre_env
cd /work/projects/eguzys/code/src
srun bash eval_all.sh
