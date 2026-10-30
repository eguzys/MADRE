#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  om2g_launch_arton.sh — submit ONE OM2G training arm to Arton/Slurm
# ═══════════════════════════════════════════════════════════════════════════
#
#  ⚠️  OM2G TRACK (IEEE 802.15.4 narrowband, 250 kbps).  NOT the OMNET track.
#      OMNET runs launch from omnet_module/launch_v2_learnable.sh.
#      The two experiments' numbers are NEVER comparable.
#      See code/om2g/OM2G_README.md.
#
#  Usage (from the repo root, on the arton login node):
#      sbatch --gres=gpu:geforce_rtx_3090:1 --cpus-per-task=8 --mem=32G \
#             --time=24:00:00 --job-name=OM2G_compressed \
#             om2g/om2g_launch_arton.sh compressed st 0 500000
#
#  Args: <arm> <arch> <seed> <max_frames>
#      arm   compressed   -- LearnableUserSetEncoder, D=8,   88 B,  1 frame
#            uncompressed -- PassthroughUserSetEncoder, D=680, 2776 B, 29 frames
#
#  Every knob comes from om2g/om2g_config.py, which derives everything from
#  om2g/om2g_track.py.  Nothing is re-typed here -- that is the entire point
#  of those two files.
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail

ARM="${1:-compressed}"
ARCH="${2:-st}"
SEED="${3:-0}"
MAX_FRAMES="${4:-500000}"

case "$ARM" in
  compressed|uncompressed|nocontrol) ;;
  *) echo "FATAL: arm must be compressed|uncompressed|nocontrol, got '$ARM'"; exit 1 ;;
esac

# ── EULER SITE BLOCK (only difference from om2g_launch_arton.sh) ────────────
# No $SLURM_CONF here. The env was rebuilt on scratch after the purge wiped
# padre_env_cpu; it is a plain venv, so put its bin on PATH rather than
# activating a conda env that no longer exists.
SCRATCH=/cluster/scratch/eguzys
RB=$SCRATCH/code_git
export PATH="$SCRATCH/om2g_env/bin:$PATH"
export TMPDIR=${OM2G_TMPDIR:-$RB/tmp}
mkdir -p "$TMPDIR"
export OMNET_APPTAINER_SIF=${OMNET_APPTAINER_SIF:-$RB/omnet.sif}
export OMNET_CONTAINER_CODE=${OMNET_CONTAINER_CODE:-/cluster}
export OMNET_HOST_CODE=${OMNET_HOST_CODE:-/cluster}
cd "$RB"

# ── Every OM2G environment variable, from the single source of truth ────────
# om2g_config.py emits `export K=V` lines covering all four sites: the
# simulator (OMNET_EXTRA_ARGS), the reward oracle (OM2G_* radio constants),
# the environment (OMNET_STEP_M, encoder mode) and the model (OM2G_ENCODER,
# OMNET_USER_ENC_OUT_DIM).  Sourcing it means this script cannot drift from
# the config the docs quote.
export OM2G_TRACK=1
CFG_ENV=$(python om2g/om2g_config.py --arm "$ARM" --arch "$ARCH" --shell) || {
  echo "FATAL: om2g_config.py failed -- refusing to launch a run whose"
  echo "       physics would fall back to OMNET-track defaults."; exit 1; }
eval "$CFG_ENV"

# Fail fast if the ini and the Python constants have drifted apart.
python om2g/om2g_config.py --selftest || {
  echo "FATAL: OM2G config self-test failed"; exit 1; }

# $OM2G_RUN_TAG separates run FAMILIES that share an arm and a seed. Without it
# every family lands in one folder (src/out/OM2G/OM2G_<arm>_st_s<seed>/) and
# can only be told apart by BenchMARL's timestamp. The
# control-plane-charge runs set OM2G_RUN_TAG=cc so their checkpoints,
# telemetry and save folder are unambiguous on sight.
RUN_NAME="OM2G_${ARM}_${ARCH}_s${SEED}${OM2G_RUN_TAG:+_${OM2G_RUN_TAG}}"
export OMNET_SAVE_SUFFIX="$RUN_NAME"
export OM2G_TELEMETRY_CSV="$RB/om2g_out/telemetry_${RUN_NAME}.csv"
# BenchMARL validates that evaluation_interval and checkpoint_interval are
# EXACT MULTIPLES of collected_frames_per_batch (6000 in conf/experiment/
# omnet.yaml) and refuses to start otherwise. 150000 = 25x6000, 48000 = 8x6000.
# Round numbers like 100000/50000 are NOT multiples and fail at setup.
mkdir -p "$RB/om2g_out"

# ── Entropy: pinned, never defaulted ────────────────────────────────────────
# The launcher defaults to ent0.03 while every run on this project trains
# at 0.01 + the scale_lb sigma floor.  Leaving it to the default would mix
# entropy_coef with the arm under test, so ent0.01 is pinned on every submit.
export OMNET_ALGO="${OMNET_ALGO:-padre_mappo_norm_ent0.01}"
export OMNET_EXPECT_ENT="${OMNET_EXPECT_ENT:-0.01}"

export OMNET_MAX_FRAMES="$MAX_FRAMES"
# task.ep_length MUST equal *.rlBridge.maxSteps (om2g_config sets the latter);
# train_omnet_arton.sh has a gate that refuses the run if they disagree.
# ── Where results land ──────────────────────────────────────────────────────
# train_omnet_arton.sh builds a SAVE path beginning "omnet_...", which would
# put OM2G checkpoints under an OMNET-looking directory.  experiment.save_folder is
# appended LAST in $OMNET_HYDRA_OVERRIDES and therefore wins, so every OM2G
# artefact lives under src/out/OM2G/ and carries the track token in its path.
OM2G_SAVE="out/OM2G/${RUN_NAME}"
mkdir -p "$RB/src/$OM2G_SAVE"

# $OM2G_EXTRA_OVERRIDES appends arbitrary hydra overrides (e.g.
# algorithm.scale_mapping=biased_softplus_0.5) so a hedge config can be launched
# without editing this script. Empty by default => byte-identical to before.
export OMNET_HYDRA_EXTRA="seed=${SEED} ${OM2G_EXTRA_OVERRIDES:-}"
export OMNET_HYDRA_OVERRIDES="task.ep_length=250 \
experiment.save_folder=${OM2G_SAVE} \
experiment.evaluation_interval=150000 \
experiment.checkpoint_interval=48000 \
experiment.keep_checkpoints_num=20 ${OMNET_HYDRA_EXTRA}"

echo "═══════════════════════════════════════════════════════════════════"
echo "  OM2G TRACK — IEEE 802.15.4 narrowband (NOT the OMNET track)"
echo "  arm=$ARM  arch=$ARCH  seed=$SEED  frames=$MAX_FRAMES"
echo "  encoder=$OM2G_ENCODER  latent=$OMNET_USER_ENC_OUT_DIM"
echo "  message=$(python -c "
import sys; sys.path.insert(0,'om2g')
import om2g_track as T
if '$ARM' == 'nocontrol':
    print('4 B, never sent within an episode (beaconInterval=3600s), 0.00% of channel')
else:
    d = 8 if '$ARM'=='compressed' else T.OM2G_MAX_USERS_CAP*T.OM2G_OBS_FEATURES
    print(f\"{T.om2g_message_bytes('$ARM', d)} B in {T.om2g_frames_per_message('$ARM', d)} frame(s), \"
          f\"{T.om2g_control_plane_share('$ARM', d)*100:.2f}% of the 250 kbps channel\")")"
# The control-plane charge is the POINT of this run family, so print it. A knob
# that is set but never echoed cannot be told apart from one that is unset.
echo "  radio budget=${OM2G_RADIO_CAP_BPS:-UNSET} app-bps  demand cap=${OM2G_TAU_CAP_BPS:-UNSET} bps"
echo "  control charge=${OM2G_CONTROL_ON_AIR_BYTES:-UNSET} on-air B/msg  every ${OM2G_SYNC_INTERVAL_S:-UNSET} s  users=${OMNET_MAX_USERS_CAP:-UNSET}"
echo "  ini=$OMNET_INI  config=$OMNET_CONFIG  telemetry=$OM2G_TELEMETRY_CSV"
echo "  run=$RUN_NAME"
echo "═══════════════════════════════════════════════════════════════════"

# Reuse the OMNET launcher's body: it carries the pre-flight gates (binary
# present, highspy importable, container bind visible, episode-length
# coherence, entropy guard, seed watchdog).  All OM2G-specific state is
# already in the environment above.
# $OM2G_RESUME_FROM forwards to the trainer's 3rd arg (resume from a
# checkpoint); unset means a fresh run.
exec bash omnet_module/train_omnet_euler.sh "$ARCH" "${OM2G_DRONES:-5}" "${OM2G_RESUME_FROM:-}"
