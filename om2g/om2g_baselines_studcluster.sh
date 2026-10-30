#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  om2g_baselines_studcluster.sh — hover / random / greedy / aggregate at the
#  OM2G radio, on the D-INFK student cluster instead of arton.
# ═══════════════════════════════════════════════════════════════════════════
#
#  Same policies, same math, same om2g_track.py source of truth as
#  om2g_baselines_arton.sh -- only the CLUSTER-SPECIFIC bits differ:
#    - no $SLURM_CONF override (arton needs one, this cluster does not)
#    - paths: /work/projects/eguzys/code, not arton's $SCRATCH
#    - conda env lives at /work/scratch/eguzys/padre_env directly
#    - Slurm resources: --account=projects --partition=jobs --gpus=N
#      (this cluster rejects --gres=gpu:type:N: "TRES per node not allowed")
#
#  Usage: sbatch --account=projects --partition=jobs ... \
#         om2g/om2g_baselines_studcluster.sh <policy> [episodes]
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail

POLICY="${1:-hover}"
EPISODES="${2:-40}"

RB=/work/projects/eguzys/code
cd "$RB"

source /work/projects/eguzys/miniconda3/etc/profile.d/conda.sh
conda activate /work/scratch/eguzys/padre_env

export OM2G_TRACK=1
eval "$(python om2g/om2g_config.py --arm compressed --shell)"

# Baselines exchange no state -- see om2g_baselines_arton.sh for why this is
# unset (a deliberately harder bar for the trained arms, not a shortcut).
unset OM2G_CONTROL_ON_AIR_BYTES

export OMNET_APPTAINER_SIF=$RB/omnet.sif
export OMNET_HOST_CODE=$RB
export OMNET_CONTAINER_CODE=/work
export OMNET_INET_DIR=/opt/inet4.6
export OMNETPP_DIR=/opt/omnetpp-6.4.0
export CUDA_VISIBLE_DEVICES=""

mkdir -p "$RB/om2g_out"
OUT="$RB/om2g_out/OM2G_baseline_${POLICY}_N5_n${EPISODES}${OM2G_RUN_TAG:+_${OM2G_RUN_TAG}}_studcluster.csv"

echo "═══════════════════════════════════════════════════════════════════"
echo "  OM2G BASELINE (student-cluster) — policy=$POLICY  N=5  episodes=$EPISODES"
echo "  reward unit: kbps (NOT the OMNET track's Mbps)"
echo "  out=$OUT"
echo "═══════════════════════════════════════════════════════════════════"

cd omnet_module
exec python evaluate_omnet.py \
  --policy "$POLICY" \
  --num-drones 5 \
  --episodes "$EPISODES" \
  --seed-start 0 \
  --out "$OUT"
