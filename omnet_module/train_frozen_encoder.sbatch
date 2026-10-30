#!/bin/bash
# train_frozen_encoder.sbatch - fit ONE frozen shared user-set encoder.
#
#   sbatch --export=ALL,TRAIN=greedy,SEED=0 omnet_module/train_frozen_encoder.sbatch
#
# Env knobs:
#   TRAIN   greedy | mixed        which dataset to FIT on
#   SEED    init seed; run >=3 -- their spread is the noise floor the
#           coverage study compares distribution shift against
#   CW      col-weight scheme: std (default) | drop_idtag | none
#   EPOCHS  default 200
#
#SBATCH --job-name=frozen_enc
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/usr/itetnas04/data-scratch-01/eguzys/data/frozen_enc/logs/train_%j.log
set -uo pipefail
S="${OM2G_SCRATCH:-/usr/itetnas04/data-scratch-01/eguzys/data}"
R=$S/resilient-backbone
TRAIN="${TRAIN:-greedy}"; SEED="${SEED:-0}"; CW="${CW:-drop_idtag}"
EPOCHS="${EPOCHS:-200}"; TGT="${TGT:-sorted}"

source $S/miniconda3/etc/profile.d/conda.sh
conda activate $S/padre_env
cd "$R/src" || exit 1
export CUDA_VISIBLE_DEVICES=""    # CPU: the model is tiny, data movement dominates
export PYTHONPATH="$R/src:${PYTHONPATH:-}"

# [2026-08-19] frac10/frac20 added for the MIX-FRACTION SWEEP. The S0.5 rule
# says "the smallest fraction that closes the gap"; 33% (= `mixed`) was simply
# what had been collected, never optimised. These are symlink views under
# $SCRATCH/frozen_enc/mix_<name>/ holding all 80 greedy shards plus a prefix of
# the random ones.
case "$TRAIN" in
  greedy) GLOB="$S/frozen_enc/greedy/*.npz" ;;
  frac10) GLOB="$S/frozen_enc/mix_frac10/*.npz" ;;
  frac20) GLOB="$S/frozen_enc/mix_frac20/*.npz" ;;
  frac33) GLOB="$S/frozen_enc/mix_frac33/*.npz" ;;
  mixed)  GLOB="$S/frozen_enc/*/*.npz" ;;   # LEGACY: overlaps the holdout, see below
  *) echo "FATAL: TRAIN must be greedy|frac10|frac20|frac33|mixed, got '$TRAIN'"; exit 1 ;;
esac
OUT="$S/frozen_enc/models/enc_${TRAIN}_${TGT}_${CW}_s${SEED}.pt"
mkdir -p "$S/frozen_enc/models"
[ -s "$OUT" ] && { echo "== already done: $OUT"; exit 0; }

echo "## TRAIN=$TRAIN TGT=$TGT CW=$CW SEED=$SEED  glob=$GLOB"
# Held-out evaluation: the full random set lies INSIDE TRAIN=mixed's own
# training data, so it cannot serve as an out-of-distribution set for that arm.
#
# 10 of the 40 random shards are reserved in random_holdout/ and NO
# encoder ever trains on them. Every arm is scored on the same never-seen
# states, so the ratios are commensurable. mix views draw only from the
# remaining 30 (random_train/).
#
# TRAIN=mixed is kept for provenance but OVERLAPS the holdout -- use frac33
# instead, which is the same 33% ratio drawn from random_train only.
python encoder_experiments/train_frozen_omnet_encoder.py \
    --npz "$GLOB" --eval-npz "$S/frozen_enc/random_holdout/*.npz" \
    --out "$OUT" --seed "$SEED" --col-weight "$CW" --epochs "$EPOCHS" \
    --target "$TGT" \
    --hidden 128 --output 8
RC=$?
[ $RC -ne 0 ] && { echo "!! training FAILED rc=$RC"; exit 1; }
echo "## done $(date '+%F %T')"
