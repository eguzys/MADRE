#!/bin/bash
#SBATCH --job-name=enc_sweep
#SBATCH --array=0-7
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/eguzys/logs/%x_%A_%a.out
#SBATCH --error=/cluster/scratch/eguzys/logs/%x_%A_%a.out
set -e
module load stack/2024-06 python/3.11.6
source /cluster/scratch/eguzys/pisa_env/bin/activate
python -c "import torch" || { echo "torch missing after activate"; exit 1; }

NPZ=/cluster/scratch/eguzys/data/omnet_sets_merged.npz
SRC=/cluster/scratch/eguzys/resilient-backbone/src/encoder_experiments
mkdir -p /cluster/scratch/eguzys/omnet_enc

HIDDENS=(64 64 64 64 128 128 128 128)
OUTPUTS=(4 8 12 16 4 8 12 16)
H=${HIDDENS[$SLURM_ARRAY_TASK_ID]}
O=${OUTPUTS[$SLURM_ARRAY_TASK_ID]}
CKPT="/cluster/scratch/eguzys/omnet_enc/deepsets_h${H}_o${O}.pt"
JSON="/cluster/scratch/eguzys/omnet_enc/recon_h${H}_o${O}.json"
echo "=== CELL h=${H} o=${O} ==="

cd "$SRC"
python -u train_final_encoder.py \
  --npz "$NPZ" --hidden ${H} --output ${O} --epochs 150 --batch-size 256 \
  --out "$CKPT"
python -u distributional_recon_omnet.py \
  --npz "$NPZ" --encoder "$CKPT" --hidden ${H} --output ${O} \
  --out "$JSON"
echo DONE
