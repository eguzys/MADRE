#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  ew.sh (and the _ub / _spr / _spr_ub copies) — evaluate one trained cell.
#  Usage: <script> <cell-tag> [episodes] [seed_start] [minsep_override]
#
#  Reads the cell's own file (arch_out/cells/<tag>.sh) for every export:
#  the world and the arm-scoped knobs (OM2G_LATENT_DIM, OM2G_CODEC_MODE,
#  OM2G_CODEC_PATH, partner tag). An eval of a trained checkpoint must
#  reproduce the observation geometry it trained with, so arm-scoped vars are
#  kept. Dropped: run-scoped vars (RUN_TAG/TMPDIR/EXTRA_OVERRIDES) and the
#  training-only shaping weights. minsep_override runs the separation
#  transfer sweep: it moves OM2G_MINSEP/OM2G_MAXNN only.
#  Checkpoint: highest training return (om2g/pick_best_ckpt.py).
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail
TAG="${1:?cell tag}"
EPISODES="${2:-100}"
SEED_START="${3:-0}"
MINSEP_OVR="${4:-}"
RB=/usr/itetnas04/data-scratch-01/eguzys/data/resilient-backbone
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
# DIAG_SPREADING (spreading-factor override) replaces the cell's OM2G_CTRL_SPREADING
# before om2g_config builds the world (drone tx power) and again in POST (runtime airtime charge).
[ -n "${DIAG_SPREADING:-}" ] && { export OM2G_CTRL_SPREADING=$DIAG_SPREADING; echo "  spreading override: OM2G_CTRL_SPREADING=$OM2G_CTRL_SPREADING"; }
# Spawn: inherit the cell's own value; fall back to cluster if the cell names none.
export OM2G_SPAWN="${OM2G_SPAWN:-cluster}"
# A spawn override is applied here, before om2g_config.py --shell writes
# droneSpawnMode into OMNET_EXTRA_ARGS inside the eval launcher;
# OM2G_EVAL_POST_OVERRIDES is applied only after the world string is built.
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
# Per-policy control charge for baselines: om2g_config exports the learned arm's
# rate; the case block in om2g_baselines_arton.sh sets the baseline charge
# (engineered pays a payload-matched 67 B, a policy that exchanges no state pays 0).
[ -n "${OM2G_EXTRA_POST:-}" ] && POST="$POST
$OM2G_EXTRA_POST"
[ -n "${DIAG_SPREADING:-}" ] && POST="$POST
export OM2G_CTRL_SPREADING=$DIAG_SPREADING"
export OM2G_EVAL_POST_OVERRIDES="$POST"
ARM=$(grep -oE 'om2g_launch_arton.sh +[a-z]+' "$CELL" | awk '{print $2}')
RUN_DIR=$(ls -1dt src/out/OM2G/OM2G_${ARM}_*_${TAG}/ 2>/dev/null | head -1)
[ -n "$RUN_DIR" ] || { echo "FATAL: no run dir for $TAG"; exit 1; }
OUT_TAG="$TAG"
# Distinguish runs of the SAME cell under a different eval world (e.g. a
# spawn transfer test) so their CSVs never collide with the native ones.
[ -n "${OM2G_OUT_SUFFIX:-}" ] && OUT_TAG="${TAG}${OM2G_OUT_SUFFIX}"
[ -n "$MINSEP_OVR" ] && OUT_TAG="${OUT_TAG}_at${MINSEP_OVR}"  # append, do not clobber a suffix
exec bash $DIAG_HOME/oea.sh "$ARM" "${DIAG_CKPT:-best:$RUN_DIR}" "$OUT_TAG" "$EPISODES" "$SEED_START"
