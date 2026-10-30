#!/bin/bash
#SBATCH --job-name=eventcount_probe
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/eguzys/logs/ecprobe_%j.log
set -e
source /cluster/scratch/eguzys/pisa_env/bin/activate
export OMNET_LOG_STDOUT=/cluster/scratch/eguzys/logs/eventcount_stdout.log
export OMNET_APPTAINER_SIF=/cluster/scratch/eguzys/omnet.sif
export OMNET_HOST_CODE=/cluster/scratch/eguzys/resilient-backbone
export OMNET_CONTAINER_CODE=/work
export OMNET_INET_DIR=/opt/inet4.6
export OMNETPP_DIR=/opt/omnetpp-6.4.0
export OMNET_SMOKE_DIR=/cluster/scratch/eguzys/resilient-backbone/omnet_smoke
cd /cluster/scratch/eguzys/resilient-backbone/omnet_module
python3 /cluster/scratch/eguzys/eventcount_drive.py
echo "=== EVENT_COUNT parse ==="
python3 /cluster/scratch/eguzys/eventcount_parse.py
