#!/bin/bash
#SBATCH --job-name=ds_smoke
#SBATCH --array=0-9
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/eguzys/baselines/logs/ds_smoke_%A_%a.out
#
# 8 cells (greedy|random x N in {3,4,5,6}), each sharded 10 ways over
# seeds 0-99. Task id -> (cell, shard):  cell = id / 10, shard = id % 10.
# Every row carries its own seed, so shards concatenate with a cat.
set -uo pipefail

source /cluster/scratch/eguzys/pisa_env/bin/activate
export OMNET_APPTAINER_SIF=/cluster/scratch/eguzys/omnet.sif
export OMNET_HOST_CODE=/cluster/scratch/eguzys/resilient-backbone
export OMNET_INET_DIR=/opt/inet4.6
export OMNETPP_DIR=/opt/omnetpp-6.4.0

CKPT=/cluster/scratch/eguzys/normalizedmappo_static_deepsetsmodel__eb0b0eff_26_07_20-22_50_48/checkpoints/checkpoint_1680000.pt
SEED_START=$(( SLURM_ARRAY_TASK_ID * 10 ))
OUT=/cluster/scratch/eguzys/baselines/ds_trained_n5_s${SEED_START}.csv
echo "=== SMOKE (mid-training ckpt 1680000, NOT a quotable result) shard=$SLURM_ARRAY_TASK_ID seeds=${SEED_START}..$(( SEED_START + 9 )) ==="

cd /cluster/scratch/eguzys/resilient-backbone/omnet_module
python evaluate_omnet.py "$CKPT" --policy trained --num-drones 5 \
  --seed-start "$SEED_START" -n 10 --out "$OUT"
RC=$?
if [ $RC -ne 0 ]; then echo "=== FAILED rc=$RC ==="; exit $RC; fi
echo "=== OK $(wc -l < "$OUT") lines ==="
