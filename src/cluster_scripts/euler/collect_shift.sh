#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/eguzys/shift_%x_%j.log
set -e
source /cluster/scratch/eguzys/pisa_env/bin/activate
cd /cluster/scratch/eguzys/resilient-backbone/src
if [ "$POL" = "greedy" ]; then
  python collect_user_sets.py --policy greedy --episodes 60 --steps 1000 \
    --nodes 4 --out /cluster/scratch/eguzys/shift_greedy.npz
else
  python collect_user_sets.py --policy trained --device cpu \
    --checkpoint "$CKPT" --episodes 60 --steps 1000 \
    --nodes 4 --out /cluster/scratch/eguzys/shift_${ARCH}.npz
fi
echo DONE
