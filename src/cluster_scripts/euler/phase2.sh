#!/bin/bash
#SBATCH --job-name=phase2
#SBATCH --array=0-19
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/cluster/scratch/eguzys/logs/%x_%A_%a.out
#SBATCH --error=/cluster/scratch/eguzys/logs/%x_%A_%a.out

set -e
SEED=${SLURM_ARRAY_TASK_ID}
cd /cluster/scratch/eguzys
apptainer exec --bind /cluster/scratch/eguzys/code:/work,/cluster/scratch/eguzys/data:/data omnet.sif bash -c "
  source /opt/omnetpp-6.4.0/setenv -f
  cd /opt/inet4.6 && source setenv -f
  cd /work/omnet_module && python -u phase2_delivered.py \
    --seed ${SEED} --steps 1000 --out /data/phase2_seed${SEED}.json"
