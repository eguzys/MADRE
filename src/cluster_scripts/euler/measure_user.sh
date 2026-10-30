#!/bin/bash
#SBATCH --job-name=user_latent
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/cluster/scratch/eguzys/user_latent_%j.out
source /cluster/scratch/eguzys/pisa_env/bin/activate
python /cluster/scratch/eguzys/measure_user_latent.py 1500000
