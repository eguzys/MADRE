#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=6G
#SBATCH --output=/cluster/scratch/eguzys/efn3_%x_%j.log
set -e
source /cluster/scratch/eguzys/pisa_env/bin/activate
cd /cluster/scratch/eguzys/resilient-backbone/src
export PADRE_EVAL_USERS=$U
python evaluation/evaluate.py \
  /cluster/scratch/eguzys/ffn_raw_n3_99e0dd65/checkpoints/checkpoint_3000000.pt \
  --policy trained --condition sinr --numb-of-nodes 3 -n 100 --no-render
echo DONE
