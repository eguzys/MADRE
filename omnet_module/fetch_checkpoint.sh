#!/bin/bash
# fetch_checkpoint.sh — pull a checkpoint off Arton WITH the config.pkl it needs.
#
# WHY THIS EXISTS. evaluation.policies.load_trained_policy looks for config.pkl
# at ckpt.parent.parent/"config.pkl" and only falls back to Hydra compose when
# it is absent -- and that fallback is PADRE-only, because OMNeT's observation
# shape is module-constant-driven, not config-driven, so nothing in a bare .pt
# plus src/conf/task/padre/*.yaml can reconstruct it. A plain
# `scp .../checkpoint_N.pt /tmp/` therefore produces a file that loads down the
# wrong path and fails confusingly. This copies the pair, in the layout the
# loader expects:
#
#     <dest>/<run>/config.pkl
#     <dest>/<run>/checkpoints/checkpoint_<N>.pt
#
# Usage: bash omnet_module/fetch_checkpoint.sh <frames|latest> [dest] [run_glob]
#   frames   e.g. 48000, or "latest" for the highest-numbered checkpoint
set -uo pipefail

FRAMES="${1:-latest}"
DEST="${2:-$(cd "$(dirname "$0")/.." && pwd)/out/arton_ckpt}"
RUN_GLOB="${3:-omnet_st_n5_enclearned_v2_N5_L2000_sprd350_pX05_staging}"

HOST=eguzys@login.ee.ethz.ch
REMOTE_ROOT=/usr/itetnas04/data-scratch-01/eguzys/data/resilient-backbone/src/out/benchmarl_omnet

echo "locating run under $REMOTE_ROOT/$RUN_GLOB ..."
RUN_DIR=$(ssh -o BatchMode=yes "$HOST" \
  "ls -dt $REMOTE_ROOT/$RUN_GLOB/normalizedmappo_* 2>/dev/null | head -1")
if [ -z "$RUN_DIR" ]; then
  echo "FATAL: no experiment dir under $REMOTE_ROOT/$RUN_GLOB"
  echo "       (check the run name; the save suffix encodes the spread)"
  exit 1
fi
echo "  run: $RUN_DIR"

if [ "$FRAMES" = "latest" ]; then
  CKPT=$(ssh -o BatchMode=yes "$HOST" \
    "ls -1 $RUN_DIR/checkpoints/checkpoint_*.pt 2>/dev/null \
       | sed 's#.*/checkpoint_\([0-9]*\)\.pt#\1 &#' | sort -n -k1,1 | tail -1 | cut -d' ' -f2-")
else
  CKPT="$RUN_DIR/checkpoints/checkpoint_${FRAMES}.pt"
fi
if [ -z "$CKPT" ] || ! ssh -o BatchMode=yes "$HOST" "test -f '$CKPT'"; then
  echo "FATAL: no checkpoint '$CKPT'."
  echo "       available:"
  ssh -o BatchMode=yes "$HOST" "ls -1 $RUN_DIR/checkpoints/ 2>/dev/null | tail -12" \
    | sed 's/^/         /'
  exit 1
fi
echo "  ckpt: $CKPT"

NAME=$(basename "$RUN_DIR")
LOCAL="$DEST/$NAME"
mkdir -p "$LOCAL/checkpoints"
scp -q "$HOST:$CKPT" "$LOCAL/checkpoints/" || { echo "FATAL: scp of checkpoint failed"; exit 1; }
scp -q "$HOST:$RUN_DIR/config.pkl" "$LOCAL/" || { echo "FATAL: scp of config.pkl failed -- the checkpoint alone is not loadable"; exit 1; }

LOCAL_CKPT="$LOCAL/checkpoints/$(basename "$CKPT")"
echo
echo "fetched:"
ls -la "$LOCAL/config.pkl" "$LOCAL_CKPT"
echo
echo "next:"
echo "  bash omnet_module/eval_checkpoint.sh $LOCAL_CKPT 20 8 out/omnet_v3_verdict"
