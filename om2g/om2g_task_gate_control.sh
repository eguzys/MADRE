#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  om2g_task_gate_control.sh — THE CONTROL for om2g_task_gate.sh
# ═══════════════════════════════════════════════════════════════════════════
#
#  Same geometry (L=2000, N=5, 17 users, max_clusters=2, spread_eff=350) and
#  the same six criteria, but on the **OMNET track's 802.11g radio**.
#
#  Purpose: separate the effect of the radio from the effect of the geometry
#  on the criteria.  Either
#
#    (a) the 250 kbps fixed-rate capacity model moves the criteria (an OM2G
#        property), or
#    (b) the v2 config (L=2000, spread 350) already sits outside C1-C6 on
#        802.11g (a property of the geometry, independent of the radio).
#
#  Running both and comparing tells them apart.  This script is (b).
#
#  ⚠️ This is the ONE place an OMNET-radio number is produced from the om2g/
#     directory.  Its output is named ..._CONTROL_omnet_radio... so it can
#     never be mistaken for an OM2G result.  Its reward is in Mbps.
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

# Deliberately NOT the OM2G environment: no OM2G_TRACK, no OM2G_* radio
# overrides, the OMNET ini, the OMNET TX powers.  Geometry copied from the v2
# launcher so the ONLY difference from om2g_task_gate.sh is the PHY.
unset OM2G_TRACK OM2G_ARM OMNET_INI OMNET_CONFIG
export OMNET_DRONE_TX_mW=200.0
export OMNET_USER_TX_mW=10.0
export OMNET_STEP_M=60.0
export OMNET_NUM_DRONES=5
export OMNET_ENCODER_MODE=learned
export OMNET_USER_ENC_OUT_DIM=8
export OMNET_EXTRA_ARGS="--*.rlBridge.arenaWidth=2000.0 \
--*.rlBridge.arenaHeight=2000.0 \
--**.mobility.constraintAreaMaxX=2000.0m \
--**.mobility.constraintAreaMaxY=2000.0m \
--*.rlBridge.maxClusters=2 \
--*.rlBridge.pCross=0.5 \
--*.rlBridge.userSpread=350.0 \
--*.host[*].wlan[*].radio.transmitter.power=200.0mW \
--*.user[*].wlan[*].radio.transmitter.power=10.0mW \
--**.host[*].mobility.maxSpeed=60.0mps \
--*.numUsers=17 \
--*.jammer.mobility.activate=999999 \
--*.rlBridge.droneSpawnMode=\\\"staging\\\" \
--*.rlBridge.maxSteps=250"

export OMNET_APPTAINER_SIF=$SCRATCH/omnet.sif
export OMNET_HOST_CODE=$RB
export OMNET_CONTAINER_CODE=/work
export OMNET_INET_DIR=/opt/inet4.6
export OMNETPP_DIR=/opt/omnetpp-6.4.0
export CUDA_VISIBLE_DEVICES=""

mkdir -p "$RB/om2g_out"
OUT="$RB/om2g_out/CONTROL_omnet_radio_task_gate_L2000_N5.json"

echo "═══════════════════════════════════════════════════════════════════"
echo "  CONTROL — C1..C6 at the OMNET 802.11g radio, OM2G geometry"
echo "  ⚠️ These are OMNET-RADIO numbers (Mbps). NOT OM2G results."
echo "  L=2000  N=5  users=17  max_clusters=2  spread=350  seeds=$SEEDS"
echo "═══════════════════════════════════════════════════════════════════"

cd omnet_module
exec python task_validation.py \
  --arenas 2000 --drones 5 --anchor 5 --users 17 --max-clusters 2 \
  --seeds "$SEEDS" --greedy-steps 40 --expect-geometry --verbose \
  --json "$OUT"
