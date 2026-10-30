#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/eguzys/oc_%x_%j.log
set -e
R=/cluster/scratch/eguzys/resilient-backbone
apptainer exec --bind $R:/work --bind /cluster/scratch/eguzys:/out /cluster/scratch/eguzys/omnet.sif \
  bash -c "export OMNET_INET_DIR=/opt/inet4.6 && \
           export OMNET_SMOKE_DIR=/work/omnet_smoke && \
           cd /work/omnet_module && \
           python collect_omnet_user_sets.py --episodes 50 --seed $S \
             --out /out/omnet_sets_${S}.npz"
echo DONE
