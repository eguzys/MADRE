#!/bin/bash
#SBATCH --job-name=recon_only
#SBATCH --array=0-7
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/eguzys/logs/%x_%A_%a.out
#SBATCH --error=/cluster/scratch/eguzys/logs/%x_%A_%a.out
set -e
module load stack/2024-06 python/3.11.6
source /cluster/scratch/eguzys/pisa_env/bin/activate
python -c "import torch" || { echo "torch missing"; exit 1; }

NPZ=/cluster/scratch/eguzys/data/omnet_sets_merged.npz
SRC=/cluster/scratch/eguzys/resilient-backbone/src/encoder_experiments
cd "$SRC"

HIDDENS=(64 64 64 64 128 128 128 128)
OUTPUTS=(4 8 12 16 4 8 12 16)
H=${HIDDENS[$SLURM_ARRAY_TASK_ID]}
O=${OUTPUTS[$SLURM_ARRAY_TASK_ID]}
CKPT="/cluster/scratch/eguzys/omnet_enc/deepsets_h${H}_o${O}.pt"
JSON="/cluster/scratch/eguzys/omnet_enc/recon_h${H}_o${O}.json"
echo "=== RECON h=${H} o=${O} ==="
python -u distributional_recon_omnet.py \
  --npz "$NPZ" --encoder "$CKPT" --hidden ${H} --output ${O} --out "$JSON"
echo DONE
