#!/bin/bash
# Behaviour-clone relay_blind onto the compressed actor. CPU-only -- costs no
# GPU slot. Usage: sbatch om2g/bc_run.sh
#SBATCH --job-name=bc_clone
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
set -uo pipefail
SCRATCH="${OM2G_SCRATCH:-/usr/itetnas04/data-scratch-01/eguzys/data}"
RB=$SCRATCH/resilient-backbone
export TMPDIR=$SCRATCH/tmp
source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate $SCRATCH/padre_env
export OM2G_TRACK=1 OMNET_ALLOW_CPU=1 CUDA_VISIBLE_DEVICES=""
eval "$(python $RB/om2g/om2g_config.py --arm compressed --shell)"
DONOR=$RB/src/out/OM2G/OM2G_compressed_st_s0_c9lr5e4/*/checkpoints/checkpoint_96000.pt
DONOR=$(ls $DONOR | head -1)
echo "  donor : $DONOR"
echo "  pairs : $RB/bc_data/rb  ($(ls $RB/bc_data/rb/*.json 2>/dev/null | wc -l) chunks)"
cd $RB/src
exec python $RB/om2g/bc_train_blind.py "$DONOR" "$RB/bc_data/rb" "$RB/bc_data/clone_relay_blind.pt"
