#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=oencev
#SBATCH --output=/cluster/scratch/eguzys/oencev_%j.log
set -e
module load stack/2024-06 python/3.11.6
source /cluster/scratch/eguzys/pisa_env/bin/activate
cd /cluster/scratch/eguzys/resilient-backbone/src/encoder_experiments
for H in 64 128; do
  for O in 4 8 12 16; do
    CK=/cluster/scratch/eguzys/omnet_enc/omnet_h${H}_o${O}.pt
    [ -f "$CK" ] || { echo "MISSING $CK"; continue; }
    echo "===== hidden=$H output=$O ====="
    python distributional_recon_omnet.py \
      --npz /cluster/scratch/eguzys/omnet_sets_merged.npz \
      --encoder $CK --hidden $H --output $O --cap 120000 \
      --out /cluster/scratch/eguzys/omnet_enc/setw_h${H}_o${O}.json
  done
done
echo DONE
