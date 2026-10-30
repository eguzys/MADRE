#!/bin/bash
#SBATCH --job-name=per_feat
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/eguzys/pisa_data/per_feature_recon_%j.log

set -euo pipefail

module load stack/2024-06 python/3.11.6
source /cluster/scratch/eguzys/pisa_env/bin/activate

export SWEEP_SCRIPT_DIR=/cluster/scratch/eguzys
export NPZ_X_KEY=tokens
export NPZ_M_KEY=masks

python /cluster/scratch/eguzys/per_feature_recon.py \
  --npz     /cluster/scratch/eguzys/pisa_data/user_sets.npz \
  --encoder /cluster/scratch/eguzys/pisa_data/scaling_out/deepsets_h64_o8.pt \
  --hidden  64 \
  --output  8 \
  --out     /cluster/scratch/eguzys/pisa_data/per_feature_recon.json

echo "DONE $(date)"
