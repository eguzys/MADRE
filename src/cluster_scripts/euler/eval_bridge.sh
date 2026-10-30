#!/bin/bash
#SBATCH --job-name=bridge_u15
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/cluster/scratch/eguzys/bridge_u15_%j.out
. /cluster/scratch/eguzys/pisa_env/bin/activate
cd /cluster/scratch/eguzys/resilient-backbone/src
export PADRE_EVAL_USERS=15
python -u evaluation/evaluate.py \
  /cluster/scratch/eguzys/ffn_frozen_bridge/checkpoints/checkpoint_3000000.pt \
  --no-render --policy trained --condition sinr --numb-of-nodes 4 -n 100
echo "EXIT=$?"
