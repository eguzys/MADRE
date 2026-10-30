#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/eguzys/event_sweep/evsw_%A_%a.log
#SBATCH --array=0-7
set -e
source /cluster/scratch/eguzys/pisa_env/bin/activate
cd /cluster/scratch/eguzys/resilient-backbone/src/encoder_experiments

H=(64 64 64 64 128 128 128 128)
O=(4 8 12 16 4 8 12 16)
h=${H[$SLURM_ARRAY_TASK_ID]}
o=${O[$SLURM_ARRAY_TASK_ID]}
TAG="ev_h${h}_o${o}"
OUT=/cluster/scratch/eguzys/event_sweep

echo "=== $TAG ==="
python train_final_encoder.py \
  --npz /cluster/scratch/eguzys/data/events_cut.npz \
  --hidden $h --output $o --epochs 150 \
  --out $OUT/${TAG}.pt

python /cluster/scratch/eguzys/distributional_recon_events.py \
  --npz /cluster/scratch/eguzys/data/events_cut.npz \
  --encoder $OUT/${TAG}.pt \
  --hidden $h --output $o \
  --out $OUT/${TAG}_recon.json
