#!/bin/bash
#SBATCH --job-name=greedy_u
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/cluster/scratch/eguzys/greedy_u_%j.log
set -uo pipefail
module load stack/2024-06 python/3.11.6
source /cluster/scratch/eguzys/pisa_env/bin/activate
cd /cluster/scratch/eguzys/resilient-backbone/src

U=${PADRE_EVAL_USERS:?must set PADRE_EVAL_USERS}
echo "=== GREEDY @ prob=1.0, ${U} users, N=4, 100 eps ==="
python -m evaluation.evaluate dummy_greedy_ignores_this.pt --policy greedy --condition sinr --numb-of-nodes 4 -n 100 --no-render
echo "=== RANDOM @ prob=1.0, ${U} users, N=4, 100 eps ==="
python -m evaluation.evaluate dummy_greedy_ignores_this.pt --policy random --condition sinr --numb-of-nodes 4 -n 100 --no-render || echo "random failed (known) — greedy is what matters"
echo "DONE u=${U} $(date)"
