#!/bin/bash
#SBATCH --job-name=eval_raw
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/eguzys/eval_raw_%x_%j.log
set -uo pipefail
module load stack/2024-06 python/3.11.6
source /cluster/scratch/eguzys/pisa_env/bin/activate
cd /cluster/scratch/eguzys/resilient-backbone/src

: "${ARCH:?set ARCH=st|ds}"
: "${N:?set N=3|4|5}"

case "$ARCH" in
  st) CKPT=/cluster/scratch/eguzys/st_raw_0a58342b/checkpoints/checkpoint_3000000.pt ;;
  ds) CKPT=/cluster/scratch/eguzys/ds_raw_61c39ef7/checkpoints/checkpoint_3000000.pt ;;
  *)  echo "bad ARCH=$ARCH"; exit 1 ;;
esac

echo "=== ${ARCH^^}-raw ZERO-SHOT @ N=${N}, 15 users, 100 eps ==="
python -m evaluation.evaluate "$CKPT" \
  --policy trained --condition sinr --numb-of-nodes ${N} -n 100 --no-render
echo "DONE ${ARCH} N=${N} $(date)"
