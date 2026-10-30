#!/bin/bash
#SBATCH --job-name=omnet_g4
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/eguzys/g4_%j.out
#
# G4: the trained path of evaluate_omnet.py, end-to-end on a real checkpoint
# (FFN N=5 from job 2122402, 26-wide drone token, current normalisation).
set -uo pipefail

source /cluster/scratch/eguzys/pisa_env/bin/activate
export OMNET_APPTAINER_SIF=/cluster/scratch/eguzys/omnet.sif
export OMNET_HOST_CODE=/cluster/scratch/eguzys/resilient-backbone
export OMNET_INET_DIR=/opt/inet4.6
export OMNETPP_DIR=/opt/omnetpp-6.4.0

CKPT=/cluster/scratch/eguzys/normalizedmappo_static_mlp__4b8bcef4_26_07_20-21_52_24/checkpoints/checkpoint_840000.pt

echo "=== G4 trained path, FFN N=5, 2 episodes ==="
echo "omnet_smoke procs BEFORE: $(pgrep -u $USER -c omnet_smoke || echo 0)"

cd /cluster/scratch/eguzys/resilient-backbone/omnet_module
python evaluate_omnet.py "$CKPT" --policy trained --num-drones 5 \
  --seed-start 0 -n 2 --out /cluster/scratch/eguzys/g4_trained.csv
RC=$?

echo "omnet_smoke procs AFTER: $(pgrep -u $USER -c omnet_smoke || echo 0)"
if [ $RC -ne 0 ]; then echo "=== FAILED rc=$RC ==="; exit $RC; fi
echo "=== OK $(wc -l < /cluster/scratch/eguzys/g4_trained.csv) lines ==="
cat /cluster/scratch/eguzys/g4_trained.csv
