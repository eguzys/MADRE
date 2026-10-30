#!/bin/bash
#SBATCH --job-name=probe_norender
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/cluster/scratch/eguzys/probe_norender_%j.out
. /cluster/scratch/eguzys/pisa_env/bin/activate
cd /cluster/scratch/eguzys/resilient-backbone/src
export PADRE_EVAL_USERS=5
python -u evaluation/evaluate.py \
  /cluster/scratch/eguzys/normalizedmappo_static_stmodel__af33a1cd_26_07_21-00_01_34/checkpoints/checkpoint_3000000.pt \
  --no-render --policy trained --condition sinr --numb-of-nodes 4 -n 3
echo "PROBE_EXIT=$?"
