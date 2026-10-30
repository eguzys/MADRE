#!/bin/bash
#SBATCH --job-name=evgrid
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/eguzys/evg_%x_%j.log
set -uo pipefail
module load stack/2024-06 python/3.11.6
source /cluster/scratch/eguzys/pisa_env/bin/activate
cd /cluster/scratch/eguzys/resilient-backbone/src

: "${ARCH:?set ARCH=st|ds}"
: "${N:?set N=3|4|5}"
: "${PADRE_EVAL_USERS:?set PADRE_EVAL_USERS}"
export PADRE_EVAL_USERS

case "$ARCH" in
  st) CKPT=/cluster/scratch/eguzys/st_raw_0a58342b/checkpoints/checkpoint_3000000.pt ;;
  ds) CKPT=/cluster/scratch/eguzys/ds_raw_61c39ef7/checkpoints/checkpoint_3000000.pt ;;
  *)  echo "bad ARCH=$ARCH"; exit 1 ;;
esac

echo "=== ${ARCH^^}-raw @ N=${N}, ${PADRE_EVAL_USERS} users, 100 eps ==="
python -m evaluation.evaluate "$CKPT" \
  --policy trained --condition sinr --numb-of-nodes ${N} -n 100 --no-render
echo "DONE ${ARCH} N=${N} u=${PADRE_EVAL_USERS} $(date)"
