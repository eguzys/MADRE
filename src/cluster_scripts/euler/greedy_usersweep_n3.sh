#!/bin/bash
#SBATCH --job-name=greedy_n3
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/cluster/scratch/eguzys/greedy_n3_u_%j.log
set -uo pipefail
module load stack/2024-06 python/3.11.6
source /cluster/scratch/eguzys/pisa_env/bin/activate
cd /cluster/scratch/eguzys/resilient-backbone/src

U=${PADRE_EVAL_USERS:?must set PADRE_EVAL_USERS}
YAML=conf/task/padre/static.yaml

# gotcha 58: sed-patch numbOfNodes to 3 BEFORE model/env build; restore on exit
cp "$YAML" "${YAML}.n3bak"
trap 'mv -f "${YAML}.n3bak" "$YAML"' EXIT
sed -i -E 's/^([[:space:]]*numbOfNodes:[[:space:]]*)[0-9]+/\13/' "$YAML"
echo "--- static.yaml numbOfNodes after patch ---"
grep -E '^\s*numbOfNodes:' "$YAML"

echo "=== GREEDY @ prob=1.0, ${U} users, N=3, 100 eps ==="
python -m evaluation.evaluate dummy_greedy_ignores_this.pt \
  --policy greedy --condition sinr --numb-of-nodes 3 -n 100 --no-render
echo "DONE u=${U} N=3 $(date)"
