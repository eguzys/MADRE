#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  om2g_task_gate.sh — re-run the six C1–C6 task-validation criteria at the
#  OM2G radio (IEEE 802.15.4, 250 kbps fixed-rate).
# ═══════════════════════════════════════════════════════════════════════════
#
#  ⚠️  OM2G TRACK.  NOT the OMNET track.  See code/om2g/OM2G_README.md.
#
#  The GEOMETRY is unchanged from the OMNET v2 config, so the layout half of
#  the criteria should carry over.  The REWARD is not: capacity is now
#  min(Shannon, 250 kbps) * gate instead of uncapped Shannon, which is much
#  flatter near a drone.  C3 (headroom, P2 >= 2 x P1) and C6 (uplink coverage)
#  are the ones that could move.  This is why the gate is re-run rather than
#  assumed.
#
#  Usage:  sbatch --cpus-per-task=8 --mem=16G --time=8:00:00 \
#                 --job-name=OM2G_task_gate ... om2g/om2g_task_gate.sh [seeds]
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail

SEEDS="${1:-30}"

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
SCRATCH="${OM2G_SCRATCH:-/usr/itetnas04/data-scratch-01/eguzys/data}"
RB=$SCRATCH/resilient-backbone
export TMPDIR=$SCRATCH/tmp
source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate $SCRATCH/padre_env
cd "$RB"

export OM2G_TRACK=1
eval "$(python om2g/om2g_config.py --arm compressed --shell)"

# The simulator runs in-container; this python process stays outside.
export OMNET_APPTAINER_SIF=$SCRATCH/omnet.sif
export OMNET_HOST_CODE=$RB
export OMNET_CONTAINER_CODE=/work
export OMNET_INET_DIR=/opt/inet4.6
export OMNETPP_DIR=/opt/omnetpp-6.4.0
export CUDA_VISIBLE_DEVICES=""

mkdir -p "$RB/om2g_out"
OUT="$RB/om2g_out/OM2G_task_gate_L2000_N5.json"

echo "═══════════════════════════════════════════════════════════════════"
echo "  OM2G TASK GATE — C1..C6 at IEEE 802.15.4 (250 kbps fixed rate)"
echo "  L=2000  N=5  users=17  max_clusters=2  seeds=$SEEDS"
echo "  anchor=5 so the SIX criteria (Tier 1) apply, not Tier 2"
echo "═══════════════════════════════════════════════════════════════════"

cd omnet_module
exec python task_validation.py \
  --arenas 2000 \
  --drones 5 --anchor 5 \
  --users 17 \
  --max-clusters 2 \
  --seeds "$SEEDS" \
  --expect-geometry \
  --greedy-steps 40 \
  --verbose \
  --json "$OUT"
