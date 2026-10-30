#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/eguzys/recon_%x_%j.log
set -e
source /cluster/scratch/eguzys/pisa_env/bin/activate
cd /cluster/scratch/eguzys/resilient-backbone/src
python encoder_experiments/distributional_recon.py \
  --npz /cluster/scratch/eguzys/shift_${F}.npz \
  --encoder MANET/encoders/deepsets_h64_o8.pt \
  --out /cluster/scratch/eguzys/recon_${F}.json
echo DONE
