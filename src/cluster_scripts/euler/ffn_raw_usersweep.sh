#!/bin/bash
#SBATCH --job-name=ffnr_u
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/cluster/scratch/eguzys/ffnr_u_%j.log
set -uo pipefail
module load stack/2024-06 python/3.11.6
source /cluster/scratch/eguzys/pisa_env/bin/activate
cd /cluster/scratch/eguzys/resilient-backbone/src

DIR=/cluster/scratch/eguzys/ffn_raw_e553ef82
U=${PADRE_EVAL_USERS:?must set PADRE_EVAL_USERS}
echo "=== FFN-RAW @ prob=1.0, ${U} users, N=4, 100 eps ==="
python -m evaluation.evaluate $DIR/checkpoints/checkpoint_3000000.pt \
  --policy trained --condition sinr --numb-of-nodes 4 -n 100 --no-render
echo "DONE u=${U} $(date)"
