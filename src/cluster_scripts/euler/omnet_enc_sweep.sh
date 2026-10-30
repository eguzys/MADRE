#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=oenc
#SBATCH --output=/cluster/scratch/eguzys/oenc_%a_%j.log
set -e
module load stack/2024-06 python/3.11.6
source /cluster/scratch/eguzys/pisa_env/bin/activate
cd /cluster/scratch/eguzys/resilient-backbone/src/encoder_experiments
HS=(64 64 64 64 128 128 128 128)
OS=(4 8 12 16 4 8 12 16)
H=${HS[$SLURM_ARRAY_TASK_ID]}
O=${OS[$SLURM_ARRAY_TASK_ID]}
echo "=== hidden=$H output=$O ==="
python train_final_encoder.py \
  --npz /cluster/scratch/eguzys/omnet_sets_merged.npz \
  --hidden $H --output $O --epochs 400 --batch-size 256 --cap 200000 \
  --out /cluster/scratch/eguzys/omnet_enc/omnet_h${H}_o${O}.pt
echo DONE
