#!/bin/bash
#SBATCH --job-name=enc_final
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=/cluster/scratch/eguzys/pisa_data/enc_final_%j.log

module load stack/2024-06 python/3.11.6
source /cluster/scratch/eguzys/pisa_env/bin/activate
cd /cluster/scratch/eguzys
python train_final_encoder.py
