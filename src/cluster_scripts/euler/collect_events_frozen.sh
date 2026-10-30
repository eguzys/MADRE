#!/bin/bash
#SBATCH --job-name=collect_events_frozen
#SBATCH --array=0-9
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/eguzys/logs/cef_%A_%a.log
set -e

# host venv drives (has torch, numpy, pyzmq, highspy) — Option B: only the
# OMNeT BINARY runs in-container, Python stays outside. Same pattern as the
# revalidation job. The OLD container-runs-Python pattern fails because
# omnet_env.py now imports torch (user encoder wired, commit 4c834b6).
source /cluster/scratch/eguzys/pisa_env/bin/activate

export OMNET_APPTAINER_SIF=/cluster/scratch/eguzys/omnet.sif
export OMNET_HOST_CODE=/cluster/scratch/eguzys/resilient-backbone
export OMNET_CONTAINER_CODE=/work
export OMNET_INET_DIR=/opt/inet4.6
export OMNETPP_DIR=/opt/omnetpp-6.4.0
export OMNET_SMOKE_DIR=/cluster/scratch/eguzys/resilient-backbone/omnet_smoke

cd /cluster/scratch/eguzys/resilient-backbone/omnet_module

# 10 array tasks x 30 episodes = 300, seeds 1000-1299 (reset seeded seed+ep)
BASE_SEED=$((1000 + SLURM_ARRAY_TASK_ID * 30))
python3 collect_omnet_user_sets.py \
  --episodes 30 --seed $BASE_SEED \
  --out /cluster/scratch/eguzys/data/events_frozen_${SLURM_ARRAY_TASK_ID}.npz

echo "DONE task $SLURM_ARRAY_TASK_ID (seeds ${BASE_SEED}..$((BASE_SEED+29)))"
