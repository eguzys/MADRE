#!/bin/bash
#SBATCH --job-name=omnet_collect
#SBATCH --array=0-299%50
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/cluster/scratch/eguzys/logs/%x_%A_%a.out
#SBATCH --error=/cluster/scratch/eguzys/logs/%x_%A_%a.out

set -e
SEED=$((1000 + SLURM_ARRAY_TASK_ID))
OUT="/data/omnet_sets_seed${SEED}.npz"
echo "TASK ${SLURM_ARRAY_TASK_ID}  SEED ${SEED}  ->  ${OUT}"
cd /cluster/scratch/eguzys
apptainer exec --bind /cluster/scratch/eguzys/code:/work,/cluster/scratch/eguzys/data:/data omnet.sif bash -c "
  source /opt/omnetpp-6.4.0/setenv -f
  cd /opt/inet4.6 && source setenv -f
  cd /work/omnet_module && python -u collect_omnet_sets.py \
    --seed-start ${SEED} --n-episodes 1 --steps 1000 --out ${OUT}"
