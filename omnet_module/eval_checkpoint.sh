#!/bin/bash
# eval_checkpoint.sh — score ONE trained checkpoint against the baselines.
#
# Usage: bash omnet_module/eval_checkpoint.sh <checkpoint.pt> [N_SEEDS] [JOBS] [OUTDIR]
#
# Pulls a checkpoint off Arton first if given a bare job/run name? No -- it takes
# a LOCAL path. Fetch with:
#   scp eguzys@login.ee.ethz.ch:/usr/itetnas04/data-scratch-01/eguzys/data/\
# resilient-backbone/out/<run>/checkpoints/checkpoint_<N>.pt  <dest>
#
# WHAT THIS SCORES. infos["global_reward"] -- the TEAM reward, Sigma_k tau_k in
# Mbps summed over the episode. NOT the per-drone difference reward D_i the
# policy trains on. The two differ by ~4-5x, so a training curve must not be
# read as if it were this number. evaluate_omnet.py forces OMNET_DIFFERENCE_REWARD=0 for
# exactly this reason.
#
# PEER_LINK_MASK must match the training run (aa732995 and its successor both
# train with it ON). OMNeTEnv echoes the setting at construction; check both
# logs agree before believing a number.
set -uo pipefail

CKPT="${1:?usage: eval_checkpoint.sh <checkpoint.pt> [n_seeds] [jobs] [outdir]}"
N_SEEDS="${2:-100}"
JOBS="${3:-8}"
OUTDIR="${4:-out/omnet_v3_verdict}"
N_DRONES=5

cd "$(dirname "$0")/.."
[ -f "$CKPT" ] || { echo "FATAL: no such checkpoint: $CKPT"; exit 1; }
CKPT="$(readlink -f "$CKPT")"

# evaluation.policies.load_trained_policy reads config.pkl from
# ckpt.parent.parent and otherwise falls back to Hydra compose -- a PADRE-ONLY
# path, because OMNeT's observation shape is module-constant-driven rather than
# config-driven, so a bare .pt cannot be reconstructed from src/conf. Checking
# here turns a confusing downstream failure into one line.
CFG_PKL="$(dirname "$(dirname "$CKPT")")/config.pkl"
if [ ! -f "$CFG_PKL" ]; then
  echo "FATAL: no config.pkl at $CFG_PKL"
  echo "       load_trained_policy needs it two levels above the checkpoint:"
  echo "         <run>/config.pkl"
  echo "         <run>/checkpoints/checkpoint_<N>.pt"
  echo "       Fetch the pair with:  bash omnet_module/fetch_checkpoint.sh <frames>"
  exit 1
fi

mkdir -p "$OUTDIR/parts"

unset OMNET_APPTAINER_SIF OMNET_USER_TX_mW OMNET_DRONE_TX_mW OMNET_STEP_M
export OMNET_MAX_STEPS=250
export OMNET_ENCODER_MODE=learned
export OMNET_PEER_LINK_MASK="${OMNET_PEER_LINK_MASK:-1}"

if [ "$N_SEEDS" -gt 100 ]; then
  echo "FATAL: N_SEEDS=$N_SEEDS runs into the reserved 100-199 seed band."; exit 1
fi

echo "=============================================================="
echo " trained-policy eval"
echo "   ckpt=$CKPT"
echo "   n=$N_SEEDS  jobs=$JOBS  peer_link_mask=$OMNET_PEER_LINK_MASK"
echo "=============================================================="

CHUNK=$(( (N_SEEDS + JOBS - 1) / JOBS ))
JOBFILE="$OUTDIR/parts/.jobs"; : > "$JOBFILE"
s=0
while [ "$s" -lt "$N_SEEDS" ]; do
  n=$CHUNK; [ $(( s + n )) -gt "$N_SEEDS" ] && n=$(( N_SEEDS - s ))
  echo "$s $n" >> "$JOBFILE"; s=$(( s + CHUNK ))
done

run_chunk() {
  start="$1"; count="$2"
  out="$OUTDIR/parts/trained_${start}.csv"
  log="$OUTDIR/parts/trained_${start}.log"
  if [ -s "$out" ]; then echo "  skip  seeds $start+$count (exists)"; return 0; fi
  python3 omnet_module/evaluate_omnet.py "$CKPT" --policy trained \
      --num-drones "$N_DRONES" --seed-start "$start" -n "$count" \
      --out "$out" > "$log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$out" ]; then
    echo "  FAIL  seeds $start+$count (rc=$rc) -- see $log"; return 1
  fi
  echo "  done  seeds $start+$count"
}
export -f run_chunk
export OUTDIR N_DRONES CKPT

t0=$(date +%s)
xargs -a "$JOBFILE" -P "$JOBS" -L 1 bash -c 'run_chunk "$@"' _
echo "wall time: $(( $(date +%s) - t0 ))s"

DEST="$OUTDIR/trained_n${N_DRONES}.csv"
echo "seed,episode_reward,steps" > "$DEST"
cat "$OUTDIR"/parts/trained_*.csv 2>/dev/null | grep -v '^seed,' | sort -t, -k1,1n >> "$DEST"
got=$(( $(wc -l < "$DEST") - 1 ))
echo "$DEST: $got episodes"

BASE="${OMNET_BASELINE_DIR:-out/omnet_v3_baselines}"
if [ -f "$BASE/hover_n${N_DRONES}.csv" ]; then
  python3 omnet_module/compare_eval.py "$DEST" "$BASE"
else
  echo "(no baselines at $BASE -- run eval_baselines_parallel.sh, then:"
  echo "   python3 omnet_module/compare_eval.py $DEST $BASE )"
fi
