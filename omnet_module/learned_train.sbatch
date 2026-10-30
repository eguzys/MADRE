#!/bin/bash
# learned_train.sbatch - one RL arm in the LEARNED-encoder condition, i.e. the
# EXISTING thesis configuration, run at an extra training seed.
#
# WHY: every learned-column number in the thesis comes from ONE training run
# (the 100 eval seeds are WORLD seeds, not training seeds -- omnetFrozen S0.3).
# The frozen column has 3 training seeds per arm, so without this the factorial
# is rigorous on one half only, and the architecture comparison rests on n=1.
# Seed 69 already exists as the canonical Aug-13 runs; this adds 70 and 71.
#
#   sbatch --export=ALL,ARCH=st,TSEED=69 omnet_module/frozen_train.sbatch
#
# ARCH  st | deepsets | ffn        TSEED  BenchMARL top-level `seed`
# ENC   encoder .pt (default: the chosen mixed/sorted/drop_idtag seed-0 fit)
#
#SBATCH --job-name=learned_rl
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
#SBATCH --output=/usr/itetnas04/data-scratch-01/eguzys/data/frozen_enc/logs/learnedrl_%j.log
set -uo pipefail
S="${OM2G_SCRATCH:-/usr/itetnas04/data-scratch-01/eguzys/data}"
ARCH="${ARCH:?set ARCH=st|deepsets|ffn}"; TSEED="${TSEED:-69}"

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

# LEARNED: each architecture pools the raw per-user block with its OWN encoder,
# trained end-to-end from the RL reward. This is the launcher's default; set
# explicitly so the condition is stated in the log rather than implied.
export OMNET_ENCODER_MODE=learned

export OMNET_RUN_TAG="learned_${ARCH}_ts${TSEED}"
export OMNET_HYDRA_EXTRA="seed=${TSEED}"

source $S/miniconda3/etc/profile.d/conda.sh
conda activate $S/padre_env
cd $S/resilient-backbone || exit 1
echo "## ARCH=$ARCH TSEED=$TSEED tag=$OMNET_RUN_TAG"
exec bash omnet_module/launch_v2_learnable.sh "$ARCH" 1002000
