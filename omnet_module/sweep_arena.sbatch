#!/bin/bash
#SBATCH --job-name=arena_sweep
#SBATCH --array=0-23
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/eguzys/arena_sweep/logs/sweep_%A_%a.out
#
# 24-cell grid: N in {3,4,5} x L in {1000,1200,1400,1700} x max_clusters in
# {2,3}. ONE array task per cell (sweep_arena.py is one-cell-per-invocation
# by design), each writing its own CSV + JSON -- results harvest incrementally
# even if some tasks are still running or a few fail. 8h/task is generous:
# P1 (probe_greedy) is the LP-heavy probe, up to greedy-steps x n_drones x 9
# max-flow solves per seed, x100 seeds.
#
# Submit:
#   mkdir -p /cluster/scratch/eguzys/arena_sweep/logs
#   sbatch omnet_module/sweep_arena.sbatch
#
# Harvest (after some/all of the array finishes):
#   python omnet_module/harvest_sweep.py /cluster/scratch/eguzys/arena_sweep
#
# Re-run just the cells that failed/are missing: resubmit with
#   sbatch --array=<comma-separated task ids> omnet_module/sweep_arena.sbatch
set -uo pipefail

# ── House pattern for driving OMNeT++ on Euler, copied from
# omnet_module/eval_baselines.sh / g4.sh (the existing OMNeT-array
# precedents -- src/eval_fix2.sbatch and src/eval_all.sbatch are GPU
# BenchMARL-evaluation jobs that never launch OMNeT directly and do not set
# any of these vars, so they are not the right template for this job). ──
source /cluster/scratch/eguzys/pisa_env/bin/activate
export OMNET_APPTAINER_SIF=/cluster/scratch/eguzys/omnet.sif
export OMNET_HOST_CODE=/cluster/scratch/eguzys/code_git
export OMNET_INET_DIR=/opt/inet4.6
export OMNETPP_DIR=/opt/omnetpp-6.4.0

NS=(3 4 5)
LS=(1000 1200 1400 1700)
MCS=(2 3)

# idx -> (N, L, max_clusters): 4 L-values x 2 mc-values = 8 cells per N.
IDX=$SLURM_ARRAY_TASK_ID
N_IDX=$(( IDX / 8 ))
REM=$(( IDX % 8 ))
L_IDX=$(( REM / 2 ))
MC_IDX=$(( REM % 2 ))

N=${NS[$N_IDX]}
L=${LS[$L_IDX]}
MC=${MCS[$MC_IDX]}

OUT_DIR=/cluster/scratch/eguzys/arena_sweep
mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/sweep_N${N}_L${L}_mc${MC}.csv"
JSON="$OUT_DIR/sweep_N${N}_L${L}_mc${MC}.json"

echo "=== task=$IDX  N=$N  L=$L  max_clusters=$MC ==="
echo "=== csv=$CSV  json=$JSON ==="

cd /cluster/scratch/eguzys/code_git/omnet_module
python sweep_arena.py --n-drones "$N" --arena "$L" --max-clusters "$MC" \
  --seeds 100 --seed-start 0 --out "$CSV" --json "$JSON" --verbose
RC=$?
if [ $RC -ne 0 ]; then echo "=== FAILED rc=$RC ==="; exit $RC; fi
echo "=== OK $(wc -l < "$CSV") csv lines ==="
