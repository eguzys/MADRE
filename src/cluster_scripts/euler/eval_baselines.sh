#!/bin/bash
#SBATCH --job-name=omnet_base
#SBATCH --array=0-79
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/eguzys/baselines/logs/base_%A_%a.out
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

POLICIES=(greedy random)
NS=(3 4 5 6)

CELL=$(( SLURM_ARRAY_TASK_ID / 10 ))
SHARD=$(( SLURM_ARRAY_TASK_ID % 10 ))
POLICY=${POLICIES[$(( CELL / 4 ))]}
N=${NS[$(( CELL % 4 ))]}
SEED_START=$(( SHARD * 10 ))

OUT=/cluster/scratch/eguzys/baselines/${POLICY}_n${N}_s${SEED_START}.csv
echo "=== task=$SLURM_ARRAY_TASK_ID policy=$POLICY N=$N seeds=${SEED_START}..$(( SEED_START + 9 )) ==="

cd /cluster/scratch/eguzys/resilient-backbone/omnet_module
python evaluate_omnet.py --policy "$POLICY" --num-drones "$N" \
  --seed-start "$SEED_START" -n 10 --out "$OUT"
RC=$?
if [ $RC -ne 0 ]; then echo "=== FAILED rc=$RC ==="; exit $RC; fi
echo "=== OK $(wc -l < "$OUT") lines ==="
