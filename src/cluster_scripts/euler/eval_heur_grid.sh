#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/eguzys/ehg_%x_%j.log
set -e
source /cluster/scratch/eguzys/pisa_env/bin/activate
cd /cluster/scratch/eguzys/resilient-backbone/src
export PADRE_EVAL_USERS=$U
python evaluation/evaluate.py --policy $POL --condition sinr \
  --numb-of-nodes $N -n 100 --no-render
echo DONE
