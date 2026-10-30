#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/eguzys/event_sweep/base_%A_%a.log
#SBATCH --array=0-2
set -e
source /cluster/scratch/eguzys/pisa_env/bin/activate
cd /cluster/scratch/eguzys/resilient-backbone/src/encoder_experiments

MODES=(real shuffle noise)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}

echo "=== $MODE ==="
python distributional_recon_events.py \
  --npz /cluster/scratch/eguzys/data/events_cut.npz \
  --encoder /cluster/scratch/eguzys/event_sweep/ev_h64_o8.pt \
  --hidden 64 --output 8 --latent-mode $MODE \
  --out /cluster/scratch/eguzys/event_sweep/base_${MODE}.json
