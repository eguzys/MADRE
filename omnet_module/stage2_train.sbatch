#!/bin/bash
# stage2_train.sbatch — TEMPLATE. TIER C: short training run proving a
# shortlisted config is actually learnable, not just predicted to be.
#
# ============================================================================
# THIS IS A TEMPLATE. Do not sbatch it directly -- generate a filled copy
# with gen_stage2_job.py, which substitutes every __TOKEN__ below from a
# Tier-B survivor JSON (sweep_config.py's --json output) and writes a
# ready-to-submit file next to it. REQUIRED FILLS if you edit this by hand
# instead:
#   __ARCH__            st|deepsets|ffn|mlp
#   __NUM_DRONES__       from the shortlisted config's "N"
#   __EXTRA_ARGS__       $OMNET_EXTRA_ARGS string (config_knobs.build_omnet_extra_args)
#   __DRONE_TX_MW__ __USER_TX_MW__ __STEP_M__   the three dual-site knobs,
#                        exported so THIS training subprocess's own import of
#                        max_flow_reward.py/greedy_oracle.py picks them up
#                        (see config_knobs.py's module docstring -- in-process
#                        attribute mutation cannot cross the process boundary
#                        into `python run.py`; only env vars survive it)
#   __MAX_FRAMES__       500_000-1_000_000 (short run; NOT the full 3M)
#   __P0V_BASELINE__     the config's OWN measured P0v (sweep_config.py's
#                        median_P0v) -- baked in so the pass/fail check below
#                        is self-contained and never compares
#                        against a DIFFERENT config's number
#   __LABEL__            free-text tag for the save folder / job name
# Also verify (same as train_omnet_arton.sh, which this reuses UNCHANGED --
# see "WHY REUSE" below): $SCRATCH, conda env path, $OMNET_APPTAINER_SIF,
# $OMNET_HOST_CODE, $OMNET_INET_DIR, $OMNETPP_DIR match YOUR cluster.
# ============================================================================
#
# WHY REUSE train_omnet_arton.sh INSTEAD OF RE-DERIVING A LAUNCH RECIPE HERE:
# a second copy of launch/rollout logic can drift from the first (see
# sweep_lib.py's module docstring). train_omnet_arton.sh already owns: the
# entropy guard, the preflight binary/highspy checks, the seed-population watchdog,
# and the resume-checkpoint validation. This script does NOT re-implement any
# of that -- it sets the config's geometry + dual-site env vars, INVOKES
# train_omnet_arton.sh as a plain subprocess (its own #SBATCH header is inert
# here; THIS file's header is what slurm actually allocates against), and
# then runs the Stage-2-specific check train_omnet_arton.sh has no reason to
# know about: does this run's eval return end up above the config's OWN
# measured P0v baseline.
#
#SBATCH --job-name=stage2___LABEL__
#SBATCH --gres=gpu:titan_rtx:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/usr/itetnas04/data-scratch-01/eguzys/data/stage2___LABEL___%j.log
set -uo pipefail

ARCH="__ARCH__"
NUM_DRONES="__NUM_DRONES__"
LABEL="__LABEL__"
P0V_BASELINE="__P0V_BASELINE__"

# ── The config's geometry AND the OMNeT-side half of the dual-site knobs
# (drone TX/user TX/speed are baked into __EXTRA_ARGS__ too -- see
# config_knobs.build_omnet_extra_args) ──────────────────────────────────────
export OMNET_EXTRA_ARGS="__EXTRA_ARGS__"
# The Python-oracle side of the SAME three dual-site knobs, for the
# `python run.py` subprocess train_omnet_arton.sh launches below --
# max_flow_reward.py / greedy_oracle.py read these at THEIR OWN import time
# (see their module comments). A run that leaves these unset gets the
# frozen defaults.
export OMNET_DRONE_TX_mW="__DRONE_TX_MW__"
export OMNET_USER_TX_mW="__USER_TX_MW__"
# OMNET_STEP_M is read by omnet_env.py as well as greedy_oracle.py: it is the
# metres a drone moves per step at |action|=1, and OMNeTEnv.step() scales the
# unit action by it.
export OMNET_STEP_M="__STEP_M__"
# ── User-set encoder mode ────────────────────────────────────────────────────
# MUST be exported explicitly, so the run uses the intended observation path
# (learned end-to-end encoder vs the frozen 8-dim reconstruction autoencoder)
# regardless of the launcher's default.
export OMNET_ENCODER_MODE="__ENCODER_MODE__"

# Short run: prove learnability, do not pay for the full 3M-frame budget.
export OMNET_MAX_FRAMES="__MAX_FRAMES__"
export OMNET_SAVE_SUFFIX="stage2_${LABEL}"

# ── Same cluster layout as train_omnet_arton.sh's own header -- override if
# yours differs. Duplicated here (not sourced from one place) because slurm
# does not evaluate the OTHER script's env until it runs; this makes the
# preflight checks below meaningful before that point. ─────────────────────
SCRATCH="${OM2G_SCRATCH:-/usr/itetnas04/data-scratch-01/eguzys/data}"
export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export TMPDIR=$SCRATCH/tmp
export WANDB_MODE=offline
export OMNET_APPTAINER_SIF=$SCRATCH/omnet.sif
export OMNET_HOST_CODE=$SCRATCH/resilient-backbone
export OMNET_CONTAINER_CODE=/work
export OMNET_INET_DIR=/opt/inet4.6
export OMNETPP_DIR=/opt/omnetpp-6.4.0
export OMNET_STATE_HISTORY_K=1
export OMNET_TRAIN_SEED_STRIDE=1
export BATCHED_PIPE_TIMEOUT=100000

echo "=== STAGE 2: arch=$ARCH N=$NUM_DRONES label=$LABEL frames=$OMNET_MAX_FRAMES ==="
echo "=== EXTRA_ARGS: $OMNET_EXTRA_ARGS ==="
echo "=== dual-site: drone_tx=$OMNET_DRONE_TX_mW user_tx=$OMNET_USER_TX_mW step_m=$OMNET_STEP_M ==="
echo "=== encoder_mode=$OMNET_ENCODER_MODE ==="
echo "=== P0v baseline this run must beat: $P0V_BASELINE ==="

cd "$OMNET_HOST_CODE"
bash omnet_module/train_omnet_arton.sh "$ARCH" "$NUM_DRONES"
RC=$?
if [ $RC -ne 0 ]; then
  echo "=== STAGE 2 TRAINING FAILED rc=$RC -- baseline check skipped ==="
  exit $RC
fi

# ── Success check: does episode return CROSS and STAY above THIS config's
# own measured P0v baseline (never "above zero", never P2/P1 -- see the
# Tier-C brief). Self-contained: P0V_BASELINE was baked into this file at
# generation time by gen_stage2_job.py, not looked up again here. ──────────
SAVE="$OMNET_HOST_CODE/src/out/benchmarl_omnet/omnet_${ARCH}_n${NUM_DRONES}_encfrozen_stage2_${LABEL}"
echo "=== checking eval return in $SAVE against P0v=$P0V_BASELINE ==="
python3 "$OMNET_HOST_CODE/omnet_module/check_stage2_baseline.py" \
  --save-dir "$SAVE" --p0v-baseline "$P0V_BASELINE"
exit $?
