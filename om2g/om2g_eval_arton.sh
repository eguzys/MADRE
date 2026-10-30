#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  om2g_eval_arton.sh — score ONE trained OM2G checkpoint
# ═══════════════════════════════════════════════════════════════════════════
#
#  ⚠️  OM2G TRACK (IEEE 802.15.4).  Reward is in kbps, NOT the OMNET Mbps.
#      See code/om2g/OM2G_README.md.
#
#  Usage:  om2g_eval_arton.sh <arm> <checkpoint_path_rel_to_src> <tag> [episodes] [seed_start]
#
#  Default episodes is 100, not 40: any number reported as a FINAL finding
#  needs n>=100 (n=20-40 is fine for a quick sanity check mid-campaign, but
#  is not sufficient to report). seed_start lets a prior n=40/n=20 run be
#  TOPPED UP to n=100 without re-scoring seeds already measured -- pass the
#  old episode count as seed_start and (100 - old count) as episodes, then
#  the two CSVs' rows concatenate into the full n=100 sample.
#
#  The arm matters for more than bookkeeping: it sets $OM2G_ENCODER and
#  $OMNET_USER_ENC_OUT_DIM, and a checkpoint trained with one encoder CANNOT be
#  rebuilt with the other -- the parameter shapes differ (8-wide latent vs the
#  max_users*13 passthrough, 520 at the 40-user population). Scoring an arm with the wrong value fails as a bare
#  tensor-shape mismatch, so pass it explicitly and never let it default.
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail

ARM="${1:?arm required: compressed|uncompressed}"
CKPT="${2:?checkpoint path (relative to src/) required}"
TAG="${3:?tag required}"
EPISODES="${4:-100}"
SEED_START="${5:-0}"

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
SCRATCH="${OM2G_SCRATCH:-/usr/itetnas04/data-scratch-01/eguzys/data}"
RB=$SCRATCH/resilient-backbone
export TMPDIR=$SCRATCH/tmp
source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate $SCRATCH/padre_env
cd "$RB"

# ── best-checkpoint scoring ─────────────────────────────────────────────────
# CKPT may be "best:<run_dir>" (absolute or relative to the repo root): the
# checkpoint at PEAK TRAINING RETURN is resolved by om2g/pick_best_ckpt.py.
# Measured: returns peak at ~150-190k frames then drift down, so scoring the
# LAST checkpoint both under-reports every arm and penalises slow-converging
# small-L rungs twice. The same rule applies to every arm of both codec
# modes; it selects on the TRAINING metric only.
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
# ── cell-scoped vars must WIN over the canonical config ──────────────────
#  om2g_config.py --shell emits a CANONICAL world and clobbers per-cell
#  choices (e.g. OMNET_PEER_LINK_MASK), which would give the env a different
#  observation width than the checkpoint expects.
[ -n "${OM2G_EVAL_POST_OVERRIDES:-}" ] && eval "$OM2G_EVAL_POST_OVERRIDES"
echo "[post] OM2G_CONTROL_ON_AIR_BYTES=${OM2G_CONTROL_ON_AIR_BYTES:-unset} (after POST overrides)"


export OMNET_APPTAINER_SIF=$SCRATCH/omnet.sif
export OMNET_HOST_CODE=${OMNET_HOST_CODE:-$RB}
export OMNET_CONTAINER_CODE=/work
export OMNET_INET_DIR=/opt/inet4.6
export OMNETPP_DIR=/opt/omnetpp-6.4.0

mkdir -p "$RB/om2g_out"
# Baselines are scored in the SAME env as the policies (OM2GEnv), so both
# are charged for control traffic the same way.
POLICY="${OM2G_EVAL_POLICY:-trained}"
OUT="$RB/om2g_out/OM2G_${POLICY}_${TAG}_n${EPISODES}_seed${SEED_START}.csv"

echo "═══════════════════════════════════════════════════════════════════"
echo "  OM2G EVAL — arm=$ARM  tag=$TAG  episodes=$EPISODES  seed_start=$SEED_START"
echo "  encoder=$OM2G_ENCODER  latent=$OMNET_USER_ENC_OUT_DIM"
echo "  ckpt=$CKPT"
echo "  out=$OUT"
echo "═══════════════════════════════════════════════════════════════════"

cd src
exec python ../omnet_module/evaluate_omnet.py "$CKPT" \
  ${OM2G_EVAL_TRACE:+--trace "$OM2G_EVAL_TRACE"} \
  --policy "$POLICY" --num-drones ${OM2G_DRONES:-5} --episodes "$EPISODES" --seed-start "$SEED_START" \
  --out "$OUT"
