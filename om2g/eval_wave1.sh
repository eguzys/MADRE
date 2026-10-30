#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  eval_wave1.sh — score a wave-1 cell at its best checkpoint.
#  Usage: eval_wave1.sh <cell-tag> [episodes] [seed_start] [minsep_override]
#
#  Reads the cell's OWN file (arch_out/cells/<tag>.sh) for every export --
#  the world AND the arm-scoped knobs (OM2G_LATENT_DIM, OM2G_CODEC_MODE,
#  OM2G_CODEC_PATH, partner tag): unlike world_from_cell.py, an eval of a
#  TRAINED checkpoint must reproduce the obs geometry it trained with, so
#  arm-scoped vars are KEPT. Dropped: run-scoped (RUN_TAG/TMPDIR/
#  EXTRA_OVERRIDES) and training-only shaping; OM2G_SPAWN forced to cluster
#  (the cell files' own convention). minsep_override enables the separation
#  transfer sweep: it moves OM2G_MINSEP/OM2G_MAXNN only.
#  Checkpoint: best-by-training-return (om2g/pick_best_ckpt.py).
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail
TAG="${1:?cell tag}"
EPISODES="${2:-100}"
SEED_START="${3:-0}"
MINSEP_OVR="${4:-}"
RB="${OM2G_SCRATCH:-/usr/itetnas04/data-scratch-01/eguzys/data}/resilient-backbone"
cd "$RB"
CELL=arch_out/cells/$TAG.sh
[ -f "$CELL" ] || { echo "FATAL: no cell $CELL"; exit 1; }
# every export except run-scoped + shaping; LAST wins, like bash
while IFS= read -r line; do
  case "$line" in
    *"export OM2G_RUN_TAG="*|*"export OM2G_TMPDIR="*|*"export OM2G_EXTRA_OVERRIDES="*) continue;;
    *"export OM2G_SHAPE_"*) continue;;
    *"export "OM2G_*|*"export "OMNET_*) eval "$line";;
  esac
done < <(grep -E '^\s*export (OM2G|OMNET)_' "$CELL")
# Spawn: INHERIT the cell's own value (the wave-1 cells set "explore"); spawn
# changes arm rankings, so eval spawn follows the cell. Fall back to cluster
# only if the cell names nothing.
export OM2G_SPAWN="${OM2G_SPAWN:-cluster}"
# A spawn override MUST be applied here, before om2g_config.py --shell bakes
# droneSpawnMode into OMNET_EXTRA_ARGS inside om2g_eval_arton.sh. Putting it
# in OM2G_EVAL_POST_OVERRIDES is TOO LATE -- the world string is already
# built, so the var would change and the simulator would not.
[ -n "${OM2G_SPAWN_FORCE:-}" ] && export OM2G_SPAWN="$OM2G_SPAWN_FORCE"
if [ -n "$MINSEP_OVR" ]; then
  export OM2G_MINSEP=$MINSEP_OVR
  # keep the band width proportional to the cell's own (nn - minsep)
  export OM2G_MAXNN=$((MINSEP_OVR + 300))
  echo "  separation override: MINSEP=$OM2G_MINSEP MAXNN=$OM2G_MAXNN"
fi
# Hand the cell's own vars down so they survive om2g_config.py --shell.
POST=$(grep -E '^[[:space:]]*export (OM2G|OMNET)_' "$CELL" \
  | grep -vE 'OM2G_RUN_TAG=|OM2G_TMPDIR=|OM2G_EXTRA_OVERRIDES=|OM2G_SHAPE_' \
  | sed 's/^[[:space:]]*//')
POST="$POST
export OM2G_SPAWN=$OM2G_SPAWN"
if [ -n "$MINSEP_OVR" ]; then
  POST="$POST
export OM2G_MINSEP=$OM2G_MINSEP
export OM2G_MAXNN=$OM2G_MAXNN"
fi
# Per-policy control charge (baselines): om2g_config exports the LEARNED
# arm's rate, which does not apply to a non-learned baseline. See the
# case block in om2g_baselines_arton.sh: engineered pays 67 B payload-
# matched, and a policy that exchanges no state pays 0.
[ -n "${OM2G_EXTRA_POST:-}" ] && POST="$POST
$OM2G_EXTRA_POST"
export OM2G_EVAL_POST_OVERRIDES="$POST"
ARM=$(grep -oE 'om2g_launch_arton.sh +[a-z]+' "$CELL" | awk '{print $2}')
RUN_DIR=$(ls -1dt src/out/OM2G/OM2G_${ARM}_*_${TAG}/ 2>/dev/null | head -1)
[ -n "$RUN_DIR" ] || { echo "FATAL: no run dir for $TAG"; exit 1; }
OUT_TAG="$TAG"
# Distinguish runs of the SAME cell under a different eval world (e.g. a
# spawn transfer test) so their CSVs never collide with the native ones.
[ -n "${OM2G_OUT_SUFFIX:-}" ] && OUT_TAG="${TAG}${OM2G_OUT_SUFFIX}"
[ -n "$MINSEP_OVR" ] && OUT_TAG="${OUT_TAG}_at${MINSEP_OVR}"  # append, do not clobber a suffix
# COMMON-FRAME SCORING. "best:" picks the peak-training-return checkpoint out of
# however many a cell happens to have, so a cell that trained longer gets more
# draws and the comparison is no longer matched on training budget.
# OM2G_EVAL_CKPT_FRAMES pins every cell to the SAME frame count instead.
# Unset => unchanged "best:" behaviour.
CKSPEC="best:$RUN_DIR"
if [ -n "${OM2G_EVAL_CKPT_FRAMES:-}" ]; then
  INNER=$(ls -1dt "$RUN_DIR"*/ 2>/dev/null | head -1)
  CKABS="${INNER}checkpoints/checkpoint_${OM2G_EVAL_CKPT_FRAMES}.pt"
  [ -s "$CKABS" ] || { echo "FATAL: no checkpoint_${OM2G_EVAL_CKPT_FRAMES}.pt under $RUN_DIR"; exit 1; }
  CKSPEC=$(realpath --relative-to="$RB/src" "$CKABS")
  echo "  common-frame checkpoint: $CKSPEC"
fi
exec bash om2g/om2g_eval_arton.sh "$ARM" "$CKSPEC" "$OUT_TAG" "$EPISODES" "$SEED_START"
