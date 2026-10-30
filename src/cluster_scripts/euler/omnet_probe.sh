#!/bin/bash
#SBATCH --job-name=omnet_probe
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/eguzys/logs/%x_%j.out
#SBATCH --error=/cluster/scratch/eguzys/logs/%x_%j.out

set -e
cd /cluster/scratch/eguzys
apptainer exec --bind /cluster/scratch/eguzys/code:/work omnet.sif bash -c '
  source /opt/omnetpp-6.4.0/setenv -f
  cd /opt/inet4.6 && source setenv -f
  cd /work/omnet_module && python -u sensed_set_probe.py'
