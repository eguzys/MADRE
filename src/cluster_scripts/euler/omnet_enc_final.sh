#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=oencf
#SBATCH --output=/cluster/scratch/eguzys/oencf_%j.log
set -e
module load stack/2024-06 python/3.11.6
source /cluster/scratch/eguzys/pisa_env/bin/activate
cd /cluster/scratch/eguzys/resilient-backbone/src/encoder_experiments
python train_final_encoder.py \
  --npz /cluster/scratch/eguzys/omnet_sets_merged.npz \
  --hidden 64 --output 12 --epochs 150 --batch-size 256 \
  --out /cluster/scratch/eguzys/omnet_enc/omnet_final_h64_o12.pt
python distributional_recon_omnet.py \
  --npz /cluster/scratch/eguzys/omnet_sets_merged.npz \
  --encoder /cluster/scratch/eguzys/omnet_enc/omnet_final_h64_o12.pt \
  --hidden 64 --output 12 --cap 120000 \
  --out /cluster/scratch/eguzys/omnet_enc/setw_final_h64_o12.json
echo DONE
