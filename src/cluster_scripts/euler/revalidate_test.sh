#!/bin/bash
#SBATCH --job-name=revalidate_frozen
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/eguzys/revalidate_%j.log
set -e

# host venv (numpy, benchmarl, pyzmq, highspy) — drives from OUTSIDE the container
source /cluster/scratch/eguzys/pisa_env/bin/activate

# Option-B wrapper: host Python drives, OMNeT binary runs in-container
export OMNET_APPTAINER_SIF=/cluster/scratch/eguzys/omnet.sif
export OMNET_HOST_CODE=/cluster/scratch/eguzys/resilient-backbone
export OMNET_CONTAINER_CODE=/work
export OMNET_INET_DIR=/opt/inet4.6
export OMNETPP_DIR=/opt/omnetpp-6.4.0
export OMNET_SMOKE_DIR=/cluster/scratch/eguzys/resilient-backbone/omnet_smoke

cd /cluster/scratch/eguzys/resilient-backbone/omnet_module

python3 task_validation.py \
  --arenas 1700 \
  --seeds 1 \
  --max-clusters 2 \
  --json /cluster/scratch/eguzys/revalidate_frozen_n100.json \
  --verbose

echo DONE
