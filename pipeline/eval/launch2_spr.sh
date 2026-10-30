#!/bin/bash
# Population-generic OM2G eval launcher (scratch copy; works on tik42x/tik43x/arton nodes).
# Args: <cell-tag> <users> <seed_start> <episodes> <ckpt rel to src or "best"> <out-name>
# Env (optional): DIAG_SENDERPROB=<p>  injects --*.rlBridge.senderProb=<p> into OMNET_EXTRA_ARGS after om2g_config.
#                 DIAG_TELEM=0 disables telemetry/trace/hook dumps (they are on by default).
TAG=$1; USERS=$2; SS=$3; EP=$4; CK=$5; NAME=$6
S=/usr/itetnas04/data-scratch-01/eguzys/data/u100diag
mkdir -p $S/out $S/logs
export DIAG_HOME=$S
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# $DIAG_SPAWN overrides the drone spawn: "cluster" (default) places every drone on a
# user cluster, "uniform" spreads the drones over the area independently of the users.
export OM2G_SPAWN_FORCE=${DIAG_SPAWN:-cluster}
export OM2G_OUT_SUFFIX=_${NAME}
if [ "$USERS" != "40" ]; then export OM2G_USERS=$USERS; else unset OM2G_USERS; fi
POST=""
if [ -n "${DIAG_SENDERPROB:-}" ]; then
  POST="export OMNET_EXTRA_ARGS=\"\$OMNET_EXTRA_ARGS --*.rlBridge.senderProb=${DIAG_SENDERPROB}\""
fi
# $OM2G_DEMAND_ALPHA / $OM2G_DEMAND_FLOOR are read by RLBridge.cc through
# getenv, not through OMNET_EXTRA_ARGS, so they only need to reach the sim's
# environment -- ew.sh re-exports every OM2G_* line it is handed.
[ -n "${DIAG_ALPHA:-}" ] && POST="$POST
export OM2G_DEMAND_ALPHA=${DIAG_ALPHA}"
[ -n "${DIAG_FLOOR:-}" ] && POST="$POST
export OM2G_DEMAND_FLOOR=${DIAG_FLOOR}"
# MEASURED-THROUGHPUT MODE (opt-in via DIAG_RAW=1). omnet_env.py creates its
# stdout with tempfile.mkstemp() i.e. in $TMPDIR, and om2g_eval_arton.sh:36
# forces TMPDIR=$SCRATCH/tmp -- a SHARED dir with random filenames, so the
# DELIVERED: records cannot be attributed to a cell/shard. POST_OVERRIDES is
# eval'd at om2g_eval_arton.sh:64, AFTER line 36, so setting TMPDIR here wins.
# Gives one dir per (family, shard) holding that shard's OMNeT stdout.
if [ "${DIAG_RAW:-0}" = "1" ]; then
  RAWDIR=$S/omnet_raw/${NAME}_s${SS}
  rm -rf $RAWDIR; mkdir -p $RAWDIR
  POST="$POST
export TMPDIR=$RAWDIR"
fi
# Generic late-override passthrough (DIAG_POST). The control charge of a scripted
# baseline is set in om2g_baselines_arton.sh, which this shard path bypasses, so a
# baseline submitted here would otherwise be charged the cell's on-air bytes. Example
# for engineered: DIAG_POST="export OM2G_CONTROL_ON_AIR_BYTES=67".
[ -n "${DIAG_POST:-}" ] && POST="$POST
$DIAG_POST"
[ -n "$POST" ] && export OM2G_EXTRA_POST="$POST"
RUN=${NAME}_s${SS}
if [ "${DIAG_TELEM:-1}" = "1" ]; then
  export OM2G_DUMP_TRACE=$S/out/trace_${RUN}.json
  export OM2G_TELEMETRY_CSV=$S/out/telem_${RUN}.csv
fi
export HOOK_OUT=$S/out/hook_${RUN}.json
[ "$CK" != "best" ] && export DIAG_CKPT="$CK"
cd /usr/itetnas04/data-scratch-01/eguzys/data/resilient-backbone
exec bash $S/ew_spr.sh "$TAG" "$EP" "$SS" "${DIAG_MINSEP:-}" > $S/logs/${RUN}.log 2>&1
