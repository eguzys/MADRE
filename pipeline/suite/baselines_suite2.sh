#!/bin/bash
# Baseline-suite launcher v2: the scripted baselines plus kmeans / force / rsr /
# lattice / maxcov via evaluate_suite.py and GTM via evaluate_gtm.py.
set -uo pipefail

POLICY="${1:-hover}"
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

REAL_POLICY="$POLICY"
RUNNER="evaluate_omnet.py"
beacon() { python3 -c "import sys;sys.path.insert(0,'$RB/om2g');import om2g_track as T;print(T.om2g_beacon_on_air_bytes())"; }
case "${POLICY:-}" in
  hover|random|spsa|scripted_relay|relay_informed)
    unset OM2G_CONTROL_ON_AIR_BYTES ;;
  gtm|rsr)
    export SUITE_POLICY=$POLICY; REAL_POLICY=hover; RUNNER="$F/evaluate_suite.py"
    unset OM2G_CONTROL_ON_AIR_BYTES ;;
  lattice)
    export SUITE_POLICY=lattice; REAL_POLICY=hover; RUNNER="$F/evaluate_suite.py"
    unset OM2G_CONTROL_ON_AIR_BYTES ;;
  kmeans|maxcov)
    export SUITE_POLICY=$POLICY; REAL_POLICY=hover; RUNNER="$F/evaluate_suite.py"
    export OM2G_CONTROL_ON_AIR_BYTES=${OM2G_FORCE_CHARGE_BYTES:-67} ;;
  force)
    export SUITE_POLICY=force; REAL_POLICY=hover; RUNNER="$F/evaluate_suite.py"
    export OM2G_CONTROL_ON_AIR_BYTES=$(beacon) ;;
  relay_blind)
    export OM2G_CONTROL_ON_AIR_BYTES=$(beacon) ;;
  relay_blind_honest)
    export OM2G_RELAY_HONEST=1; REAL_POLICY=relay_blind
    export OM2G_CONTROL_ON_AIR_BYTES=${OM2G_FORCE_CHARGE_BYTES:-67} ;;
  greedy|greedy_pooled)
    export OM2G_CONTROL_ON_AIR_BYTES=$(beacon) ;;
  aggregate)
    export OM2G_CONTROL_ON_AIR_BYTES=$(python3 -c \
      "import sys;sys.path.insert(0,'$RB/om2g');import om2g_track as T;print(T.om2g_message_on_air_bytes('compressed',8))") ;;
  engineered|engineered_exact|dispatch|relay_explore)
    export OM2G_CONTROL_ON_AIR_BYTES=${OM2G_FORCE_CHARGE_BYTES:-67} ;;
  oracle)
    export OM2G_CONTROL_ON_AIR_BYTES=$(python3 -c \
      "import sys;sys.path.insert(0,'$RB/om2g');import om2g_track as T;print(T.om2g_message_on_air_bytes('uncompressed'))") ;;
  *)
    echo "baselines: unknown POLICY='${POLICY:-}' -- refusing to guess a charge" >&2
    exit 2 ;;
esac
echo "[charge] POLICY=${POLICY} REAL_POLICY=${REAL_POLICY} OM2G_CONTROL_ON_AIR_BYTES=${OM2G_CONTROL_ON_AIR_BYTES:-UNSET} RELAY_HONEST=${OM2G_RELAY_HONEST:-0} GTM_OBJ=${SUITE_GTM_OBJ:-}"

export OMNET_APPTAINER_SIF=$SCRATCH/omnet.sif
export OMNET_HOST_CODE=$RB
export OMNET_CONTAINER_CODE=/work
export OMNET_INET_DIR=/opt/inet4.6
export OMNETPP_DIR=/opt/omnetpp-6.4.0
export CUDA_VISIBLE_DEVICES=""

mkdir -p "$F/out"
OUT="$F/out/OM2G_baseline_${POLICY}_N${OM2G_DRONES:-5}_n${EPISODES}${OM2G_RUN_TAG:+_${OM2G_RUN_TAG}}_seed${SEED_START}.csv"
echo "  SUITE BASELINE v2 — policy=$POLICY  N=${OM2G_DRONES:-5}  episodes=$EPISODES  spawn=${OM2G_SPAWN:-default}"
echo "  MINSEP=${OM2G_MINSEP:-} MAXNN=${OM2G_MAXNN:-} CLUSTERS=${OM2G_CLUSTERS:-} USERS=${OM2G_USERS:-40} SENDERPROB=${DIAG_SENDERPROB:-1.0}"
echo "  out=$OUT"

cd omnet_module
exec python "$RUNNER" \
  --policy "$REAL_POLICY" \
  --num-drones ${OM2G_DRONES:-5} \
  --episodes "$EPISODES" \
  --seed-start "$SEED_START" \
  --out "$OUT"
