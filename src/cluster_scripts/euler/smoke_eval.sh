#!/bin/bash
#SBATCH --job-name=eval_smoke
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/eguzys/eval_smoke_%j.out
source /cluster/scratch/eguzys/pisa_env/bin/activate
export OMNET_APPTAINER_SIF=/cluster/scratch/eguzys/omnet.sif
export OMNET_HOST_CODE=/cluster/scratch/eguzys/resilient-backbone
export OMNET_INET_DIR=/opt/inet4.6
export OMNETPP_DIR=/opt/omnetpp-6.4.0
cd /cluster/scratch/eguzys/resilient-backbone/omnet_module
python -c "import highspy; print('highspy ok')"
python evaluate_omnet.py --policy random --num-drones 5 -n 1 \
  --out /cluster/scratch/eguzys/smoke_n5.csv
RC=$?
echo "=== exit $RC ==="
