#!/bin/bash
# eval_baselines_parallel.sh — the three control baselines over N seeds, run
# JOBS-ways in parallel, then concatenated into one CSV per policy.
#
# Safe to parallelise: each OMNeTEnv binds its ZMQ socket with
# bind_to_random_port (omnet_env.py:1502) so the OS hands every process a free
# port, and evaluate_omnet.py passes --*.numDrones on the OMNeT command line
# rather than rewriting omnetpp.ini, so there is no shared mutable file to race
# on (see the comment above _patched_num_drones).
#
# Scores infos["global_reward"] -- the TEAM reward -- so these numbers are
# directly comparable to every historical CSV and to the trained policy.
# evaluate_omnet.py forces OMNET_DIFFERENCE_REWARD=0 for itself; D_i is a
# training signal only.
#
# Usage: bash omnet_module/eval_baselines_parallel.sh [N_SEEDS] [JOBS] [OUTDIR]
set -uo pipefail

N_SEEDS="${1:-100}"
JOBS="${2:-8}"
OUTDIR="${3:-out/omnet_v3_baselines}"
N_DRONES=5

cd "$(dirname "$0")/.."
mkdir -p "$OUTDIR/parts"

# The harness sets these itself; exporting them here as well would give the
# simulator and the harness two different configs. Only the two it reads go here.
unset OMNET_APPTAINER_SIF OMNET_USER_TX_mW OMNET_DRONE_TX_mW OMNET_STEP_M
export OMNET_MAX_STEPS=250
export OMNET_ENCODER_MODE=learned
# Must match training. Run aa732995 trained with the mask ON; measured effect on
# the score is <=15% either way, and the value is pinned rather than defaulted
# so training and evaluation always agree.
export OMNET_PEER_LINK_MASK="${OMNET_PEER_LINK_MASK:-1}"

# Chunk the seed range so each worker owns a contiguous block. Seeds 100-199 are
# a reserved evaluation band and must never appear here.
CHUNK=$(( (N_SEEDS + JOBS - 1) / JOBS ))
if [ "$N_SEEDS" -gt 100 ]; then
  echo "FATAL: N_SEEDS=$N_SEEDS would run into the reserved 100-199 seed band."
  exit 1
fi

echo "=============================================================="
echo " baselines: n=$N_SEEDS seeds x {greedy,hover,random}, ${JOBS}-way parallel"
echo "   chunk=$CHUNK  peer_link_mask=$OMNET_PEER_LINK_MASK  out=$OUTDIR"
echo "=============================================================="

JOBFILE="$OUTDIR/parts/.jobs"
: > "$JOBFILE"
for policy in greedy hover random; do
  s=0
  while [ "$s" -lt "$N_SEEDS" ]; do
    n=$CHUNK
    [ $(( s + n )) -gt "$N_SEEDS" ] && n=$(( N_SEEDS - s ))
    echo "$policy $s $n" >> "$JOBFILE"
    s=$(( s + CHUNK ))
  done
done
echo "queued $(wc -l < "$JOBFILE") chunks"

run_chunk() {
  policy="$1"; start="$2"; count="$3"
  out="$OUTDIR/parts/${policy}_${start}.csv"
  log="$OUTDIR/parts/${policy}_${start}.log"
  if [ -s "$out" ]; then echo "  skip  $policy seeds $start+$count (exists)"; return 0; fi
  python3 omnet_module/evaluate_omnet.py --policy "$policy" \
      --num-drones "$N_DRONES" --seed-start "$start" -n "$count" \
      --out "$out" > "$log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$out" ]; then
    echo "  FAIL  $policy seeds $start+$count (rc=$rc) -- see $log"
    return 1
  fi
  echo "  done  $policy seeds $start+$count"
}
export -f run_chunk
export OUTDIR N_DRONES

t0=$(date +%s)
# --halt never: one bad chunk must not discard the other 23; failures are
# reported and the concatenation below reports how many episodes actually landed.
xargs -a "$JOBFILE" -P "$JOBS" -L 1 bash -c 'run_chunk "$@"' _
t1=$(date +%s)
echo "wall time: $(( t1 - t0 ))s"

echo
for policy in greedy hover random; do
  dest="$OUTDIR/${policy}_n${N_DRONES}.csv"
  echo "seed,episode_reward,steps" > "$dest"
  # Sort by seed NUMERICALLY so the CSV is deterministic regardless of the
  # order chunks happened to finish in.
  cat "$OUTDIR"/parts/${policy}_*.csv 2>/dev/null \
    | grep -v '^seed,' | sort -t, -k1,1n >> "$dest"
  got=$(( $(wc -l < "$dest") - 1 ))
  uniq_seeds=$(tail -n +2 "$dest" | cut -d, -f1 | sort -un | wc -l)
  status="OK"
  [ "$got" -ne "$N_SEEDS" ] && status="INCOMPLETE (expected $N_SEEDS)"
  [ "$uniq_seeds" -ne "$got" ] && status="DUPLICATE SEEDS"
  echo "$dest: $got episodes, $uniq_seeds unique seeds -- $status"
done
