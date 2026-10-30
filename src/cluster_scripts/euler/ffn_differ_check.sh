#!/bin/bash
#SBATCH --job-name=ffn_diff
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/cluster/scratch/eguzys/ffn_diff_%j.log
set -uo pipefail
module load stack/2024-06 python/3.11.6
source /cluster/scratch/eguzys/pisa_env/bin/activate
cd /cluster/scratch/eguzys/resilient-backbone/src
DIR=/cluster/scratch/eguzys/ffn_raw_e553ef82

echo "########## U=5 ##########"
PADRE_EVAL_USERS=5  python -m evaluation.evaluate $DIR/checkpoints/checkpoint_3000000.pt \
  --policy trained --condition sinr --numb-of-nodes 4 -n 3 --no-render
echo "########## U=20 ##########"
PADRE_EVAL_USERS=20 python -m evaluation.evaluate $DIR/checkpoints/checkpoint_3000000.pt \
  --policy trained --condition sinr --numb-of-nodes 4 -n 3 --no-render
echo "########## DONE ##########"
