#!/bin/bash
# eval_aggregate_local.sh - run the INFORMATION-MATCHED baseline (--policy
# aggregate) at the cardinalities where it is missing.
#
# WHY: the cardinality-transfer matrix is normalised entirely against `greedy`,
# which reads all 25 user positions exactly while every learned architecture
# sees them compressed 40:1 into 8 numbers. `aggregate` is the same marginal
# hill-climb restricted to that compressed view, and it existed only at N=5, so
# no column but N=5 had a fair denominator.
#
#   ./omnet_module/eval_aggregate_local.sh          # N=4 and N=6
#   AGG_NS="4 6 3" ./omnet_module/eval_aggregate_local.sh
#
# No checkpoint and no GPU: aggregate is a numpy heuristic, the whole cost is
# the OMNeT children. Cells run SEQUENTIALLY at 8-wide (8 python + 8 OMNeT =
# 16 cores); two cells at once oversubscribes 2x and is measurably slower.
# Idempotent - existing chunks are skipped, safe to re-run after a crash.
set -uo pipefail
cd /home/emilis/Documents/thesis/code

PY=/home/emilis/pythonEnviroment/padre/bin/python
NS=${AGG_NS:-4 6}

# The laptop has a NATIVELY built OMNeT binary; the apptainer branch in
# omnet_env.py (~line 1628) falls back to a spawn that does not exist here.
unset OMNET_APPTAINER_SIF OMNET_USER_TX_mW OMNET_DRONE_TX_mW OMNET_STEP_M
export OMNET_MAX_STEPS=250 OMNET_ENCODER_MODE=learned OMNET_PEER_LINK_MASK=1
export CUDA_VISIBLE_DEVICES=""
# Root is at ~1.7G free; keep OMNeT's per-child stdout logs off it.
export TMPDIR="${TMPDIR:-/tmp}/aggtmp"
mkdir -p "$TMPDIR"

for N in $NS; do
  OUT="out/n100_N${N}"
  mkdir -p "$OUT/parts"
  FINAL="$OUT/aggregate_n100.csv"
  # A header-only CSV from a failed run is still "non-empty" -- require rows.
  if [ -f "$FINAL" ] && [ "$(grep -vc '^seed,' "$FINAL")" -gt 0 ]; then
    echo "== skip N=$N ($(grep -vc '^seed,' "$FINAL") episodes already)"; continue
  fi
  rm -f "$FINAL"
  echo "== aggregate N=$N  $(date '+%T')"

  JF="$OUT/parts/.jobs_aggregate"; : > "$JF"
  # 20 chunks of 5 seeds. NO trailing whitespace: `xargs -L 1` treats a line
  # ending in a blank as a continuation and silently joins it to the next.
  for s in $(seq 0 5 95); do echo "$s 5" >> "$JF"; done

  export OUT PY N
  run_one() {
    local o="$OUT/parts/aggregate_$1.csv"
    [ -s "$o" ] && return 0
    "$PY" omnet_module/evaluate_omnet.py --policy aggregate \
      --num-drones "$N" --train-num-drones "$N" \
      --seed-start "$1" -n "$2" --out "$o" > "$OUT/parts/aggregate_$1.log" 2>&1
  }
  export -f run_one
  xargs -a "$JF" -P 8 -L 1 bash -c 'run_one "$@"' _

  echo "seed,episode_reward,steps" > "$FINAL"
  cat "$OUT"/parts/aggregate_*.csv 2>/dev/null | grep -v '^seed,' | sort -t, -k1,1n >> "$FINAL"
  EP=$(( $(wc -l < "$FINAL") - 1 ))
  echo "   -> $EP episodes  $(date '+%T')"
  # Report loudly when no episodes were written: a job can exit 0 with an
  # empty CSV.
  if [ "$EP" -eq 0 ]; then
    echo "   !! ZERO episodes -- first chunk log:"; sed -n '1,40p' "$OUT/parts/aggregate_0.log"
  fi
done
echo "########## done $(date '+%F %T') ##########"
