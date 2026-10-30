#!/bin/bash
# Relay-planner launcher v3: priced and information-matched relay planners via
# evaluate_relay2.py.
#
# relay_explore / relay_blind pool every drone's heard user positions and are
# charged 67 B or a 51 B position beacon. A position beacon carries no user state
# (om2g_track.om2g_beacon_on_air_bytes), so the planners below are charged for the
# user state they actually exchange.
set -uo pipefail

POLICY="${1:-relay_priced}"
EPISODES="${2:-100}"
SEED_START="${3:-0}"

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
SCRATCH=/usr/itetnas04/data-scratch-01/eguzys/data
RB=$SCRATCH/resilient-backbone
F=$SCRATCH/u100diag/suite
export TMPDIR=$SCRATCH/tmp
source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate $SCRATCH/padre_env
cd "$RB"

export OM2G_TRACK=1
eval "$(python om2g/om2g_config.py --arm compressed --shell)"; [ -n "${DIAG_SENDERPROB:-}" ] && export OMNET_EXTRA_ARGS="$OMNET_EXTRA_ARGS --*.rlBridge.senderProb=$DIAG_SENDERPROB"

REAL_POLICY=hover
RUNNER="$F/evaluate_relay2.py"
export SUITE_POLICY=$POLICY
export OM2G_RELAY_MODE=${OM2G_RELAY_MODE:-blind}

msg() { python3 -c "import sys;sys.path.insert(0,'$RB/om2g');import om2g_track as T;print(T.om2g_message_on_air_bytes('$1',${2:-8}))"; }
beacon() { python3 -c "import sys;sys.path.insert(0,'$RB/om2g');import om2g_track as T;print(T.om2g_beacon_on_air_bytes())"; }

case "${POLICY:-}" in
  relay_fair)
    # Positions-only user block: the planner reads x,y and nothing else, so it
    # is billed for two floats per user rather than the full 17-value token.
    export OM2G_CONTROL_ON_AIR_BYTES=$(python3 -c "
import sys;sys.path.insert(0,'$RB/om2g');import om2g_track as T,math
N=int('${OM2G_USERS:-40}')
fixed=T.om2g_message_floats('uncompressed')-T.OM2G_NUM_USERS*T.OM2G_OBS_FEATURES
fl=fixed+N*2
payload=T.OM2G_GEO_BYTES+(fl-2)*T.OM2G_FLOAT_BYTES
per=((T.OM2G_MTU_BYTES-20)//8)*8
n=max(1,math.ceil((payload+8)/per))
print(payload+8+(20+9+6)*n)
") ;;
  relay_priced)
    # Pools every drone's heard-user positions -> bill the uncompressed block.
    export OM2G_CONTROL_ON_AIR_BYTES=$(msg uncompressed) ;;
  relay_moments)
    # Sends only the two-moment summary -> bill one compressed message, as AGG.
    export OM2G_CONTROL_ON_AIR_BYTES=$(msg compressed 8) ;;
  relay_1hop)
    # Sends only what a one-hop neighbour round carries -> position beacon.
    export OM2G_CONTROL_ON_AIR_BYTES=$(beacon) ;;
  *)
    echo "relay2: unknown POLICY='${POLICY:-}' -- refusing to guess a charge" >&2
    exit 2 ;;
esac

echo "[charge] POLICY=${POLICY} REAL_POLICY=${REAL_POLICY} OM2G_CONTROL_ON_AIR_BYTES=${OM2G_CONTROL_ON_AIR_BYTES} RELAY_MODE=${OM2G_RELAY_MODE}"

export OMNET_APPTAINER_SIF=$SCRATCH/omnet.sif
export OMNET_HOST_CODE=${OMNET_HOST_CODE:-$RB}   # user-beacon build: respect a preset (ub_smoke binary); unset = default build
export OMNET_CONTAINER_CODE=/work
export OMNET_INET_DIR=/opt/inet4.6
export OMNETPP_DIR=/opt/omnetpp-6.4.0
export CUDA_VISIBLE_DEVICES=""

mkdir -p "$F/out"
OUT="$F/out/OM2G_baseline_${POLICY}_N${OM2G_DRONES:-5}_n${EPISODES}${OM2G_RUN_TAG:+_${OM2G_RUN_TAG}}_seed${SEED_START}.csv"
echo "  RELAY2 BASELINE - policy=$POLICY  N=${OM2G_DRONES:-5}  episodes=$EPISODES  spawn=${OM2G_SPAWN:-default}"
echo "  MINSEP=${OM2G_MINSEP:-} MAXNN=${OM2G_MAXNN:-} CLUSTERS=${OM2G_CLUSTERS:-} USERS=${OM2G_USERS:-40} SENDERPROB=${DIAG_SENDERPROB:-1.0}"
echo "  out=$OUT"

cd omnet_module
exec python "$RUNNER" \
  --policy "$REAL_POLICY" \
  --num-drones ${OM2G_DRONES:-5} \
  --episodes "$EPISODES" \
  --seed-start "$SEED_START" \
  --out "$OUT"
