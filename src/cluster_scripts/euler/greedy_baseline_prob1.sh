#!/bin/bash
#SBATCH --job-name=greedy_base
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/eguzys/greedy_baseline_%j.log
set -uo pipefail
module load stack/2024-06 python/3.11.6
source /cluster/scratch/eguzys/pisa_env/bin/activate
cd /cluster/scratch/eguzys/resilient-backbone/src

CKPT=dummy_greedy_ignores_this.pt

echo "=== GREEDY @ prob=1.0, 15 users, N=4 ==="
python -m evaluation.evaluate "$CKPT" --policy greedy --condition sinr --numb-of-nodes 4 -n 100 --no-render

echo ""
echo "=== RANDOM @ prob=1.0, 15 users, N=4 ==="
python -m evaluation.evaluate "$CKPT" --policy random --condition sinr --numb-of-nodes 4 -n 100 --no-render || echo "RANDOM FAILED (known customMovementOperator issue) — greedy is the baseline that matters"

echo ""
echo "ALL DONE $(date)"
