#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/eguzys/event_sweep/gap_%A_%a.log
#SBATCH --array=0-15
set -e
source /cluster/scratch/eguzys/pisa_env/bin/activate
cd /cluster/scratch/eguzys/resilient-backbone/src/encoder_experiments

H=(64 64 64 64 128 128 128 128); O=(4 8 12 16 4 8 12 16)
CELL=$((SLURM_ARRAY_TASK_ID % 8))
MODES=(real shuffle); MODE=${MODES[$((SLURM_ARRAY_TASK_ID / 8))]}
h=${H[$CELL]}; o=${O[$CELL]}; TAG="ev_h${h}_o${o}"

python distributional_recon_events.py \
  --npz /cluster/scratch/eguzys/data/events_cut.npz \
  --encoder /cluster/scratch/eguzys/event_sweep/${TAG}.pt \
  --hidden $h --output $o --latent-mode $MODE \
  --out /cluster/scratch/eguzys/event_sweep/gap_${TAG}_${MODE}.json
