#!/bin/bash
#SBATCH --job-name=ev_sweep
#SBATCH --array=0-7
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/eguzys/logs/%x_%A_%a.out
#SBATCH --error=/cluster/scratch/eguzys/logs/%x_%A_%a.out

set -e
HIDDENS=(64 64 64 64 128 128 128 128)
OUTPUTS=(4 8 12 16 4 8 12 16)
H=${HIDDENS[$SLURM_ARRAY_TASK_ID]}
O=${OUTPUTS[$SLURM_ARRAY_TASK_ID]}
echo "=== EVENT CELL h=${H} o=${O} ==="
echo "NOTE: event sets are SPARSE (p50 ~0). If the diag says most cols are"
echo "      constant-zero, these numbers describe a token that should change."

cd /cluster/scratch/eguzys
apptainer exec --bind /cluster/scratch/eguzys/code:/work,/cluster/scratch/eguzys/data:/data omnet.sif bash -c "
  cd /work/src/encoder_experiments
  python -u train_final_encoder.py \
    --npz /data/omnet_events_merged.npz \
    --hidden ${H} --output ${O} --epochs 150 --batch-size 256 \
    --out /data/events_h${H}_o${O}.pt"
