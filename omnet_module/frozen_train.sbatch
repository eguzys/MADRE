#!/bin/bash
# frozen_train.sbatch - one RL arm trained against the FROZEN SHARED encoder.
#
#   sbatch --export=ALL,ARCH=st,TSEED=69 omnet_module/frozen_train.sbatch
#
# ARCH  st | deepsets | ffn        TSEED  BenchMARL top-level `seed`
# ENC   encoder .pt (default: the chosen mixed/sorted/drop_idtag seed-0 fit)
#
#SBATCH --job-name=frozen_rl
#SBATCH --cpus-per-task=8
# ⚠️ PIN THE GPU MODEL. padre_env's torch has no kernels below sm_75, so a
# bare `--gpus=1` lands on whatever is free and dies with
#   torch.AcceleratorError: CUDA error: no kernel image is available
# Measured 2026-08-19: titan_xp (sm_61, tikgpu02/03) and tesla_v100 (sm_70,
# tikgpu05) both FAIL; geforce_rtx_2080_ti and titan_rtx (both sm_75) work.
# geforce_rtx_2080_ti has 28 GPUs across
# artongpu01-07, so pinning costs no availability.
#SBATCH --gpus=geforce_rtx_2080_ti:1
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --output=/usr/itetnas04/data-scratch-01/eguzys/data/frozen_enc/logs/rl_%j.log
set -uo pipefail
S="${OM2G_SCRATCH:-/usr/itetnas04/data-scratch-01/eguzys/data}"
ARCH="${ARCH:?set ARCH=st|deepsets|ffn}"; TSEED="${TSEED:-69}"
ENC="${ENC:-$S/frozen_enc/models/enc_mixed_sorted_drop_idtag_s0.pt}"
[ -s "$ENC" ] || { echo "FATAL: encoder missing: $ENC"; exit 1; }
[ -s "${ENC%.pt}_cfg.json" ] || { echo "FATAL: cfg sidecar missing (latent_scale)"; exit 1; }

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export TMPDIR=$S/tmp; mkdir -p "$TMPDIR"
export WANDB_MODE=offline
export OMNET_APPTAINER_SIF=$S/omnet.sif
export OMNET_HOST_CODE=$S/resilient-backbone
export OMNET_CONTAINER_CODE=/work
export OMNET_INET_DIR=/opt/inet4.6
export OMNETPP_DIR=/opt/omnetpp-6.4.0
export OMNET_STATE_HISTORY_K=1
export OMNET_TRAIN_SEED_STRIDE=1
export BATCHED_PIPE_TIMEOUT=100000

# ── THE point of this experiment: ONE encoder, byte-identical for every arm.
# omnet_env.py refuses frozen unless BOTH of these are set (the flag alone
# would run the committed encoder, which predates the current token
# normalisation).
export OMNET_ENCODER_MODE=frozen
export OMNET_ALLOW_FROZEN_ENCODER=1
export OMNET_USER_ENC_PATH="$ENC"
# Every arm's log MUST show the same sha. That is the mechanical guarantee
# that all architectures saw the same numbers in the same state.
echo "## encoder sha256: $(sha256sum "$ENC" | cut -c1-16)  $ENC"

# Overridable, so an A/B arm with a DIFFERENT encoder gets its own experiment
# dir and tag: eval_all_arms.sh and eval_arton.sbatch both resolve checkpoints
# BY TAG, picking the highest frame count.
export OMNET_RUN_TAG="${RUN_TAG:-frozen_${ARCH}_ts${TSEED}}"
# EXTRA_HYDRA appends further Hydra overrides (e.g. capacity-matching the
# deepsets actor). Kept separate from the seed so a capacity arm is a pure
# addition rather than an edit to the shared model yaml -- padre_deepsets.yaml
# is used by the learned track too and must not change under it.
export OMNET_HYDRA_EXTRA="seed=${TSEED}${EXTRA_HYDRA:+ ${EXTRA_HYDRA}}"

source $S/miniconda3/etc/profile.d/conda.sh
conda activate $S/padre_env
cd $S/resilient-backbone || exit 1
echo "## ARCH=$ARCH TSEED=$TSEED tag=$OMNET_RUN_TAG"
exec bash omnet_module/launch_v2_learnable.sh "$ARCH" 1002000
