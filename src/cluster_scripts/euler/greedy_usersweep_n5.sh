#!/bin/bash
#SBATCH --job-name=greedy_n5
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --array=5-22%1
#SBATCH --output=/cluster/scratch/eguzys/greedy_n5_u%a_%A.log
set -uo pipefail
module load stack/2024-06 python/3.11.6
source /cluster/scratch/eguzys/pisa_env/bin/activate
cd /cluster/scratch/eguzys/resilient-backbone/src

U=${SLURM_ARRAY_TASK_ID}
export PADRE_EVAL_USERS=$U
YAML=conf/task/padre/static.yaml

# gotcha 58: sed-patch numbOfNodes to 5 BEFORE model/env build; restore on exit.
# per-task backup name so parallel array tasks don't clobber each other's backup.
BAK="${YAML}.n5bak.${SLURM_ARRAY_TASK_ID}"
cp "$YAML" "$BAK"
trap 'mv -f "$BAK" "$YAML"' EXIT
sed -i -E 's/^([[:space:]]*numbOfNodes:[[:space:]]*)[0-9]+/\15/' "$YAML"
echo "--- static.yaml numbOfNodes after patch ---"
grep -E '^\s*numbOfNodes:' "$YAML"

echo "=== GREEDY @ prob=1.0, ${U} users, N=5, 100 eps ==="
python -m evaluation.evaluate dummy_greedy_ignores_this.pt \
  --policy greedy --condition sinr --numb-of-nodes 5 -n 100 --no-render
echo "DONE u=${U} N=5 $(date)"
