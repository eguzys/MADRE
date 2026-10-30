#!/bin/bash
#SBATCH --job-name=noise_floor
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/eguzys/pisa_data/noise_floor_%j.log

set -euo pipefail

module load stack/2024-06 python/3.11.6
source /cluster/scratch/eguzys/pisa_env/bin/activate

export SWEEP_SCRIPT_DIR=/cluster/scratch/eguzys

python /cluster/scratch/eguzys/noise_floor.py \
  --npz /cluster/scratch/eguzys/pisa_data/user_sets.npz \
  --encoder /cluster/scratch/eguzys/pisa_data/scaling_out/deepsets_h64_o8.pt \
  --hidden 64 \
  --output 8 \
  --sigma-dB 4.0 \
  --cap 200000 \
  --out /cluster/scratch/eguzys/pisa_data/noise_floor.json

echo "DONE $(date)"
