#!/bin/bash
# eval_n100.sh - extend every evaluation from n=20 to n=100 (seeds 0-99).
#
# WHY: measured, the paired gap carries ~±0.15 SE at n=20 and the
# sigma effects being chased are 0.1-0.2, so almost every comparison lands
# "not significant". n=100 cuts SE ~2.2x to ~±0.07 and makes them decisive.
# Cost is only ~16 min/arm on 16 cores. Seeds 100-199 are a reserved band -
# 0-99 is exactly the usable range.
#
# Runs ONE cell at a time at 8-wide (8 python + 8 OMNeT children = 16 cores).
# Running two cells at once oversubscribes 2x and is SLOWER - measured, load hit 87.
#
# Idempotent: existing chunks are skipped, so it is safe to re-run after a crash.
set -uo pipefail
cd /home/emilis/Documents/thesis/code

PY=/home/emilis/pythonEnviroment/padre/bin/python
RREPO="${OM2G_SCRATCH:-/usr/itetnas04/data-scratch-01/eguzys/data}/resilient-backbone"
SSH="ssh -o ConnectTimeout=30 -o BatchMode=yes eguzys@login.ee.ethz.ch"

# N is configurable, so this harness can produce the cardinality-transfer
# matrix.
#   EVAL_N       - N to EVALUATE at (the env's drone count).
#   EVAL_TRAIN_N - N the checkpoint was TRAINED at. Defaults to EVAL_N.
#                  Must be set for off-diagonal cells: reload_from_file
#                  rebuilds the centralised critic (width 4N+102) from the
#                  CURRENT constants and force-loads the checkpoint's tensors,
#                  so a cross-N load dies on a critic eval never uses. See the
#                  CROSS-N note in evaluate_omnet.py's main().
#   EVAL_OUT     - output dir, so an N sweep does not overwrite out/n100.
# Defaults reproduce the original N=5 behaviour exactly.
N=${EVAL_N:-5}
TRAIN_N=${EVAL_TRAIN_N:-$N}
OUT=${EVAL_OUT:-out/n100}
mkdir -p "$OUT/parts"

# The user-encoder width must match what the checkpoint was TRAINED with or
# the model rebuild dies on a bare shape mismatch (see user_encoder.py).
# Exported unchanged when unset so the out_dim=8 arms behave as before.
if [ -n "${OMNET_USER_ENC_OUT_DIM:-}" ]; then
  export OMNET_USER_ENC_OUT_DIM
  echo "## user-encoder out_dim = $OMNET_USER_ENC_OUT_DIM"
fi
echo "## eval N=$N  (checkpoint trained at N=$TRAIN_N)  -> $OUT"

unset OMNET_APPTAINER_SIF OMNET_USER_TX_mW OMNET_DRONE_TX_mW OMNET_STEP_M
export OMNET_MAX_STEPS=250 OMNET_ENCODER_MODE=learned OMNET_PEER_LINK_MASK=1
export CUDA_VISIBLE_DEVICES=""
export OUT PY

# Keep at least this many GB free; each checkpoint is ~405 MB, so 1 GB leaves
# >2x headroom over one checkpoint. A skipped fetch leaves a hole in the
# matrix, so the threshold is kept low.
need_space() {
  local free; free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  [ "${free:-0}" -lt 1 ] && { echo "  ! only ${free}G free - skipping fetch"; return 1; }
  return 0
}

run_cell() {   # run_cell <name> <policy-args...>
  local name="$1"; shift
  local final="$OUT/${name}_n100.csv"
  [ -s "$final" ] && { echo "== skip $name (done)"; return 0; }
  echo "== $name  $(date '+%T')"
  local jf="$OUT/parts/.jobs_$name"; : > "$jf"
  # 20 chunks of 5 seeds. No trailing field: xargs -L 1 joins on trailing blanks.
  for s in $(seq 0 5 95); do echo "$s 5" >> "$jf"; done
  export name; export CELL_ARGS="$*"; export N TRAIN_N
  run_one() {
    local o="$OUT/parts/${name}_$1.csv"
    [ -s "$o" ] && return 0
    "$PY" omnet_module/evaluate_omnet.py $CELL_ARGS \
      --num-drones "$N" --train-num-drones "$TRAIN_N" \
      --seed-start "$1" -n "$2" --out "$o" > "$OUT/parts/${name}_$1.log" 2>&1
  }
  export -f run_one
  xargs -a "$jf" -P 8 -L 1 bash -c 'run_one "$@"' _
  echo "seed,episode_reward,steps" > "$final"
  cat "$OUT"/parts/${name}_*.csv 2>/dev/null | grep -v '^seed,' | sort -t, -k1,1n >> "$final"
  echo "   -> $(( $(wc -l < "$final") - 1 )) episodes"
}

fetch_arm() {   # fetch_arm <tag> -> echoes local checkpoint path
  local tag="$1"
  # Pick the run dir with the HIGHEST checkpoint, not the newest mtime. A tag
  # can have more than one run dir (an interrupted run, or a restarted
  # allocation whose newer dir has no checkpoints yet). Ranking by frame count
  # selects the complete run regardless of how many siblings exist.
  local best; best=$($SSH "for d in $RREPO/src/out/benchmarl_omnet/*${tag}*/*/; do \
      f=\$(ls \$d/checkpoints/ 2>/dev/null | grep -oE '^checkpoint_[0-9]+\.pt\$' \
           | sed -E 's/checkpoint_([0-9]+)\.pt/\1/' | sort -n | tail -1); \
      [ -n \"\$f\" ] && echo \"\$f \$d\"; done | sort -n | tail -1" | tr -d '\r')
  [ -z "$best" ] && return 1
  local fr="${best%% *}"
  local rd="${best#* }"
  [ -z "$fr" ] || [ "$fr" -lt 500000 ] && return 1
  echo "   [fetch_arm] $tag -> frame $fr in $(basename "${rd%/}")" >&2
  local dst="out/arton_ckpt/$(basename "${rd%/}")"
  mkdir -p "$dst/checkpoints"
  [ -f "$dst/checkpoints/checkpoint_${fr}.pt" ] || {
    need_space || return 1
    scp -q -o BatchMode=yes "eguzys@login.ee.ethz.ch:$rd/config.pkl" "$dst/config.pkl" || return 1
    scp -q -o BatchMode=yes "eguzys@login.ee.ethz.ch:$rd/checkpoints/checkpoint_${fr}.pt" "$dst/checkpoints/" || return 1
  }
  readlink -f "$dst/checkpoints/checkpoint_${fr}.pt"
}

# $EVAL_ARMS overrides the arm list (space-separated arton run tags). The
# special token "st_baseline" means the local 00fa5f10 ST checkpoint, which is
# already fetched and is the reference every other arm is paired against.
# Unset => the default arm list below.
ARMS=${EVAL_ARMS-st_baseline selftok_ffn selftok_ds ent003_st ent001_st}
ST_LOCAL=out/arton_ckpt/normalizedmappo_static_stmodel__00fa5f10_26_08_09-14_03_08/checkpoints/checkpoint_1002000.pt

echo "########## baselines (prerequisite for every paired gap) ##########"
# Baselines are N-specific: hover and greedy at N=4 are different numbers from
# N=5, so the gap normalisation needs its OWN pair per column of the matrix.
# Skipped only if $EVAL_SKIP_BASELINES is set (they are already done for that N).
if [ -z "${EVAL_SKIP_BASELINES:-}" ]; then
  run_cell greedy --policy greedy
  run_cell hover  --policy hover
fi

echo "########## arms ##########"
for tag in $ARMS; do
  if [ "$tag" = "st_baseline" ]; then
    ck=$(readlink -f "$ST_LOCAL")
    [ -f "$ck" ] || { echo "== st_baseline: local checkpoint missing, skipping"; continue; }
  else
    ck=$(fetch_arm "$tag") || { echo "== $tag: fetch failed, skipping"; continue; }
  fi
  run_cell "$tag" "$ck --policy trained"
done

echo "########## done $(date '+%F %T') ##########"
