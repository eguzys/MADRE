#!/bin/bash
#SBATCH --job-name=eval_ds_raw
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/eguzys/eval_ds_raw_%j.log
set -uo pipefail
module load stack/2024-06 python/3.11.6
source /cluster/scratch/eguzys/pisa_env/bin/activate
cd /cluster/scratch/eguzys/resilient-backbone/src

CKPT=/cluster/scratch/eguzys/ds_raw_61c39ef7/checkpoints/checkpoint_3000000.pt
YAML=conf/task/padre/static.yaml
cp "$YAML" "${YAML}.dsbak"
trap 'mv -f "${YAML}.dsbak" "$YAML"' EXIT

for N in 3 4 5; do
  sed -i -E "s/^([[:space:]]*numbOfNodes:[[:space:]]*)[0-9]+/\1${N}/" "$YAML"
  echo "=================================================="
  echo "=== DeepSets-raw ZERO-SHOT @ N=${N}, 15 users, 100 eps ==="
  grep -E '^\s*numbOfNodes:' "$YAML"
  echo "=================================================="
  python -m evaluation.evaluate "$CKPT" \
    --policy trained --condition sinr --numb-of-nodes ${N} -n 100 --no-render
done
echo "DONE DeepSets-raw N=3/4/5 $(date)"
