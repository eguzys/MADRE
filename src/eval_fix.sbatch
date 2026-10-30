#!/usr/bin/env bash
#SBATCH --job-name=eval_fix
#SBATCH --account=projects
#SBATCH --gpus=5060ti:1
#SBATCH --time=01:00:00
#SBATCH --output=eval_fix_%j.out

source /work/projects/eguzys/miniconda3/etc/profile.d/conda.sh
conda activate /work/scratch/eguzys/padre_env
cd /work/projects/eguzys/code/src
srun bash eval_fix.sh
