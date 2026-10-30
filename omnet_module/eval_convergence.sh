#!/bin/bash
# eval_convergence.sh — trace trained + greedy + hover per STEP, then report the
# value each converges to and how fast, via convergence.py.
#
# Usage: bash omnet_module/eval_convergence.sh <checkpoint.pt> [N_SEEDS] [OUTDIR] [SHARDS]
#
# Seeds shard across workers exactly like eval_checkpoint.sh. That is safe for
# traces because a shard owns a CONTIGUOUS seed range and writes whole episodes;
# the per-shard trace files concatenate into one (seed,step,reward) table with
# every episode intact. All three policies run concurrently, so keep
# 3 * SHARDS at or under the core count.
set -uo pipefail

CKPT="${1:?usage: eval_convergence.sh <checkpoint.pt> [n_seeds] [outdir] [shards]}"
N_SEEDS="${2:-20}"
OUTDIR="${3:-out/convergence}"
SHARDS="${4:-5}"
N_DRONES=5

cd "$(dirname "$0")/.."
[ -f "$CKPT" ] || { echo "FATAL: no such checkpoint: $CKPT"; exit 1; }
CKPT="$(readlink -f "$CKPT")"

# Same world binding as eval_checkpoint.sh -- these MUST agree or the trained
# policy and the baselines are scored in different simulators.
unset OMNET_APPTAINER_SIF OMNET_USER_TX_mW OMNET_DRONE_TX_mW OMNET_STEP_M
export OMNET_MAX_STEPS=250
export OMNET_ENCODER_MODE=learned
export OMNET_PEER_LINK_MASK="${OMNET_PEER_LINK_MASK:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"   # laptop GPU OOMs; CPU is fine

mkdir -p "$OUTDIR/parts"
CHUNK=$(( (N_SEEDS + SHARDS - 1) / SHARDS ))

run_shard() {
  pol="$1"; start="$2"; count="$3"
  extra=""
  [ "$pol" = "trained" ] && extra="$CKPT"
  python3 omnet_module/evaluate_omnet.py $extra --policy "$pol" \
      --num-drones "$N_DRONES" --seed-start "$start" -n "$count" \
      --out "$OUTDIR/parts/ep_${pol}_${start}.csv" \
      --trace "$OUTDIR/parts/trace_${pol}_${start}.csv" \
      > "$OUTDIR/parts/${pol}_${start}.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && echo "  FAIL $pol seeds $start+$count (rc=$rc)"
  return 0
}
export -f run_shard
export CKPT N_DRONES OUTDIR

echo "tracing trained/greedy/hover, n=$N_SEEDS seeds, $SHARDS shards each -> $OUTDIR"
JOBFILE="$OUTDIR/parts/.jobs"; : > "$JOBFILE"
for pol in trained greedy hover; do
  s=0
  while [ "$s" -lt "$N_SEEDS" ]; do
    n=$CHUNK; [ $(( s + n )) -gt "$N_SEEDS" ] && n=$(( N_SEEDS - s ))
    echo "$pol $s $n" >> "$JOBFILE"; s=$(( s + CHUNK ))
  done
done

t0=$(date +%s)
xargs -a "$JOBFILE" -P $(( SHARDS * 3 )) -L 1 bash -c 'run_shard "$@"' _
echo "wall time: $(( $(date +%s) - t0 ))s"

for pol in trained greedy hover; do
  dest="$OUTDIR/trace_${pol}.csv"
  echo "seed,step,reward" > "$dest"
  cat "$OUTDIR"/parts/trace_${pol}_*.csv 2>/dev/null | grep -v '^seed,' >> "$dest"
  got=$(( $(wc -l < "$dest") - 1 ))
  echo "$dest: $got steps"
done

echo
python3 omnet_module/convergence.py \
    "$OUTDIR/trace_hover.csv" "$OUTDIR/trace_trained.csv" "$OUTDIR/trace_greedy.csv"
