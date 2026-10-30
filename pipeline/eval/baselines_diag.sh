#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  baselines_diag.sh / baselines_diag_ub.sh — scripted baselines at the OM2G
#  radio (shard-path copy of om2g/om2g_baselines_arton.sh).
# ═══════════════════════════════════════════════════════════════════════════
#
#  OM2G track (IEEE 802.15.4). The reward unit is kbps (the OMNET track uses Mbps).
#
#  `oracle` reads exact global state and is an upper reference; `aggregate`
#  exchanges a pooled moment summary and is charged for it.
#
#  Usage: sbatch ... om2g/om2g_baselines_arton.sh <policy> [episodes] [seed_start]
#
#  Default episodes is 100. seed_start lets a smaller run be topped up to
#  n=100: pass the old count as seed_start and (100 - old count) as episodes;
#  the two CSVs' rows concatenate into the full sample.
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail

POLICY="${1:-hover}"
EPISODES="${2:-100}"
SEED_START="${3:-0}"

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
SCRATCH=/usr/itetnas04/data-scratch-01/eguzys/data
RB=$SCRATCH/resilient-backbone
export TMPDIR=$SCRATCH/tmp
source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate $SCRATCH/padre_env
cd "$RB"

export OM2G_TRACK=1
eval "$(python om2g/om2g_config.py --arm compressed --shell)"; [ -n "${DIAG_SENDERPROB:-}" ] && export OMNET_EXTRA_ARGS="$OMNET_EXTRA_ARGS --*.rlBridge.senderProb=$DIAG_SENDERPROB"

# ── Each policy pays for the state exchange it requires ─────────────────────
# `oracle` reads every peer's and every user's exact position, which a drone
# could only obtain by transmission, so it is charged the uncompressed user
# block. `aggregate` needs the user cloud's global moments, pooled across
# drones, so it pays one compressed-class message. Charges from om2g_track:
#
#   hover / random / spsa : no charge  (send nothing)
#   greedy   (realistic)  : one-hop position beacon,  51 B  -> 0.82%
#   aggregate             : pooled moment summary,   131 B  -> 2.10%
#   oracle                : uncompressed user block, 2949 B -> 47.18%
#                           (reported as an upper bound)
case "${POLICY:-}" in
  hover|random|spsa|scripted_relay|relay_informed)
    unset OM2G_CONTROL_ON_AIR_BYTES ;;
  relay_blind)
    # A deployable relay heuristic: it broadcasts its own position like greedy
    # does, so it pays the position beacon. relay_informed reads exact global
    # state and is left uncharged as an upper bound.
    export OM2G_CONTROL_ON_AIR_BYTES=$(python3 -c \
      "import sys;sys.path.insert(0,'$RB/om2g');import om2g_track as T;print(T.om2g_beacon_on_air_bytes())") ;;
  greedy|greedy_pooled)
    export OM2G_CONTROL_ON_AIR_BYTES=$(python3 -c \
      "import sys;sys.path.insert(0,'$RB/om2g');import om2g_track as T;print(T.om2g_beacon_on_air_bytes())") ;;
  aggregate)
    export OM2G_CONTROL_ON_AIR_BYTES=$(python3 -c \
      "import sys;sys.path.insert(0,'$RB/om2g');import om2g_track as T;print(T.om2g_message_on_air_bytes('compressed',8))") ;;
  engineered|engineered_exact|dispatch|relay_explore)
    # PAYLOAD-MATCHED charge. The moment message is 5 floats + a count = 24 B
    # of payload; framed the same way as greedy's beacon (om2g_beacon_on_air_bytes:
    # payload + 8 UDP + 20 IP + 15 MAC/PHY) that is 24+43 = 67 B on air, one frame.
    #
    # OM2G_FORCE_CHARGE_BYTES=131 charges the L=8 learned message size instead
    # (sensitivity row).
    export OM2G_CONTROL_ON_AIR_BYTES=${OM2G_FORCE_CHARGE_BYTES:-67} ;;
  oracle)
    export OM2G_CONTROL_ON_AIR_BYTES=$(python3 -c \
      "import sys;sys.path.insert(0,'$RB/om2g');import om2g_track as T;print(T.om2g_message_on_air_bytes('uncompressed'))") ;;
  *)
    echo "baselines: unknown POLICY='${POLICY:-}' -- refusing to guess a charge" >&2
    exit 2 ;;
esac
echo "[charge] POLICY=${POLICY} OM2G_CONTROL_ON_AIR_BYTES=${OM2G_CONTROL_ON_AIR_BYTES:-UNSET}"

export OMNET_APPTAINER_SIF=$SCRATCH/omnet.sif
export OMNET_HOST_CODE=$RB
export OMNET_CONTAINER_CODE=/work
export OMNET_INET_DIR=/opt/inet4.6
export OMNETPP_DIR=/opt/omnetpp-6.4.0
export CUDA_VISIBLE_DEVICES=""

mkdir -p "$RB/om2g_out"
# $OM2G_RUN_TAG keeps a re-measurement from overwriting earlier CSVs. The
# realistic greedy runs are tagged `greedy1hop`/`greedyNhop`; files named
# OM2G_baseline_greedy_* without a hop tag hold the oracle variant.
FILETAG="$POLICY"
[ "$POLICY" = "greedy" ] && FILETAG="greedy${OM2G_GREEDY_HOPS:-1}hop"
[ "$POLICY" = "greedy_pooled" ] && FILETAG="greedypooled${OM2G_GREEDY_HOPS:-1}hop"
# engineered honours OM2G_GREEDY_HOPS too, so the hop count is part of the
# filename (an h0 and an h1 run must not overwrite each other).
[ "$POLICY" = "engineered" ] && FILETAG="engineered${OM2G_GREEDY_HOPS:-1}hop"
[ "$POLICY" = "engineered_exact" ] && FILETAG="engineeredexact${OM2G_GREEDY_HOPS:-1}hop"
[ "$POLICY" = "dispatch" ] && FILETAG="dispatch${OM2G_GREEDY_HOPS:-1}hop"
[ "$POLICY" = "relay_explore" ] && FILETAG="relayexplore"
OUT="/usr/itetnas04/data-scratch-01/eguzys/data/u100diag/out/OM2G_baseline_${FILETAG}_N${OM2G_DRONES:-5}_n${EPISODES}${OM2G_RUN_TAG:+_${OM2G_RUN_TAG}}_seed${SEED_START}.csv"

echo "═══════════════════════════════════════════════════════════════════"
echo "  OM2G BASELINE — policy=$POLICY  N=${OM2G_DRONES:-5}  episodes=$EPISODES  spawn=${OM2G_SPAWN:-default}"
echo "  reward unit: kbps (NOT the OMNET track's Mbps)"
echo "  out=$OUT"
echo "═══════════════════════════════════════════════════════════════════"

cd omnet_module
exec python evaluate_omnet.py \
  --policy "$POLICY" \
  --num-drones ${OM2G_DRONES:-5} \
  --episodes "$EPISODES" \
  --seed-start "$SEED_START" \
  --out "$OUT"
