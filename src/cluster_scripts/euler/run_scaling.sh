#!/bin/bash
#SBATCH --job-name=scaling
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/eguzys/pisa_data/scaling_%j.log

# --- Euler runbook: guest account is CPU-only; this is a CPU job by design.
# --- DeepSets AE on 200k tokens / 150 epochs / 6 cells finishes well under 6h.

set -euo pipefail

module load stack/2024-06 python/3.11.6
source /cluster/scratch/eguzys/pisa_env/bin/activate

# so the script can import DeepSetsAE from the loose sweep file
export SWEEP_SCRIPT_DIR=/cluster/scratch/eguzys

# out-dir parent must pre-exist
mkdir -p /cluster/scratch/eguzys/pisa_data/scaling_out

python /cluster/scratch/eguzys/scaling_by_setsize.py \
  --npz       /cluster/scratch/eguzys/pisa_data/user_sets.npz \
  --out-dir   /cluster/scratch/eguzys/pisa_data/scaling_out \
  --cap       200000 \
  --epochs    150 \
  --hidden    32 64 128 \
  --output    8 16

echo "DONE $(date)"
