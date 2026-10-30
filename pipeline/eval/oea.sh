#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  oea.sh / oea_ub.sh — score ONE trained OM2G checkpoint (shard-path copy of
#  om2g/om2g_eval_arton.sh)
# ═══════════════════════════════════════════════════════════════════════════
#
#  OM2G track (IEEE 802.15.4). Reward is in kbps, not the OMNET-track Mbps.
#
#  Usage:  oea.sh <arm> <checkpoint_path_rel_to_src> <tag> [episodes] [seed_start]
#
#  Default episodes is 100. seed_start lets a smaller run be topped up to
#  n=100 without re-scoring seeds already measured: pass the old episode count
#  as seed_start and (100 - old count) as episodes; the two CSVs concatenate.
#
#  The arm sets $OM2G_ENCODER and $OMNET_USER_ENC_OUT_DIM, and a checkpoint
#  trained with one encoder cannot be rebuilt with the other (the parameter
#  shapes differ), so pass it explicitly.
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail

ARM="${1:?arm required: compressed|uncompressed}"
CKPT="${2:?checkpoint path (relative to src/) required}"
TAG="${3:?tag required}"
EPISODES="${4:-100}"
SEED_START="${5:-0}"

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
SCRATCH=/usr/itetnas04/data-scratch-01/eguzys/data
RB=$SCRATCH/resilient-backbone
export TMPDIR=$SCRATCH/tmp
source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate $SCRATCH/padre_env
cd "$RB"

# ── best-checkpoint scoring ─────────────────────────────────────────────────
# CKPT may be "best:<run_dir>" (absolute or relative to the repo root): the
# checkpoint at peak training return is resolved by om2g/pick_best_ckpt.py.
# Training returns peak at ~150-190k frames; the same rule is used for every
# arm and codec mode and selects on the training metric only.
if [[ "$CKPT" == best:* ]]; then
  RUN_DIR="${CKPT#best:}"
  CKPT_ABS=$(python om2g/pick_best_ckpt.py "$RUN_DIR" --table) || {
    echo "FATAL: pick_best_ckpt failed for $RUN_DIR"; exit 1; }
  echo "  best-checkpoint rule: $CKPT_ABS"
  # evaluate_omnet.py takes the path relative to src/
  CKPT=$(realpath --relative-to="$RB/src" "$CKPT_ABS")
fi

export OM2G_TRACK=1
eval "$(python om2g/om2g_config.py --arm "$ARM" --shell)"
# ── cell-scoped vars take precedence over the canonical config ─────────────
#  om2g_config.py --shell emits the canonical world; the cell's own exports
#  (e.g. OMNET_PEER_LINK_MASK) are re-applied here so the env builds the
#  observation width the checkpoint was trained with.
[ -n "${OM2G_EVAL_POST_OVERRIDES:-}" ] && eval "$OM2G_EVAL_POST_OVERRIDES"
echo "[post] OM2G_CONTROL_ON_AIR_BYTES=${OM2G_CONTROL_ON_AIR_BYTES:-unset} (after POST overrides)"


export OMNET_APPTAINER_SIF=$SCRATCH/omnet.sif
export OMNET_HOST_CODE=$RB
export OMNET_CONTAINER_CODE=/work
export OMNET_INET_DIR=/opt/inet4.6
export OMNETPP_DIR=/opt/omnetpp-6.4.0

mkdir -p $DIAG_HOME/out
# Baselines are scored in the same OM2GEnv as the trained policies, so both are
# charged for control traffic the same way.
POLICY="${OM2G_EVAL_POLICY:-trained}"
OUT="$DIAG_HOME/out/OM2G_${POLICY}_${TAG}_n${EPISODES}_seed${SEED_START}.csv"

echo "═══════════════════════════════════════════════════════════════════"
echo "  OM2G EVAL — arm=$ARM  tag=$TAG  episodes=$EPISODES  seed_start=$SEED_START"
echo "  encoder=$OM2G_ENCODER  latent=$OMNET_USER_ENC_OUT_DIM"
echo "  ckpt=$CKPT"
echo "  out=$OUT"
echo "═══════════════════════════════════════════════════════════════════"

cd src
exec python $DIAG_HOME/hookeval.py "$CKPT" \
  ${OM2G_EVAL_TRACE:+--trace "$OM2G_EVAL_TRACE"} \
  --policy "$POLICY" --num-drones ${OM2G_DRONES:-5} --episodes "$EPISODES" --seed-start "$SEED_START" \
  --out "$OUT"
