#!/bin/bash
# eval_tonight.sh - an evaluation queue, in priority order.
#
# STRICTLY SERIAL. Each cell is 8-wide (8 python + 8 OMNeT children = 16 cores);
# two cells at once oversubscribes 2x and measured SLOWER, not faster.
# Do not "parallelise" this.
#
# Idempotent at chunk granularity - re-running skips finished chunks, so it is
# safe to kill and restart.
set -uo pipefail
cd /home/emilis/Documents/thesis/code
export CUDA_VISIBLE_DEVICES=""      # laptop GPU OOM misreports as obs-shape mismatch

E=omnet_module/eval_n100.sh

echo "################ 1. close the sigma line (already trained, cheap) ########"
# sigma05_st (init 0.5) and sigma02_st (init 0.2) are the two arms trained
# specifically to test the inverted-U. Predictions: sigma05 lands BETWEEN
# +0.430 and +0.826; sigma02 lands BELOW +0.430. Baselines at N=5 already
# exist in out/n100, so skip them.
EVAL_SKIP_BASELINES=1 EVAL_ARMS="sigma05_st sigma02_st" bash "$E"

echo "################ 2. N-transfer matrix: N=6 column #######################"
# Baselines FIRST and at their own N - hover/greedy at N=6 are different
# numbers from N=5, and the gap normalisation needs its own pair per column.
EVAL_N=6 EVAL_OUT=out/n100_N6 EVAL_ARMS="" bash "$E"
# Then the transfer cells: models trained at N=5, evaluated at N=6.
EVAL_N=6 EVAL_TRAIN_N=5 EVAL_OUT=out/n100_N6 EVAL_SKIP_BASELINES=1 \
  EVAL_ARMS="st_baseline selftok_ds" bash "$E"

echo "################ 3. N-transfer matrix: N=4 column #######################"
EVAL_N=4 EVAL_OUT=out/n100_N4 EVAL_ARMS="" bash "$E"
EVAL_N=4 EVAL_TRAIN_N=5 EVAL_OUT=out/n100_N4 EVAL_SKIP_BASELINES=1 \
  EVAL_ARMS="st_baseline selftok_ds" bash "$E"

echo "################ done $(date '+%F %T') ##################################"
echo "Still owed once ffn3/ffn4 land: the N=3 baseline pair + ffn3@N=3,"
echo "and ffn4@N=4 (its baselines are already done by step 3)."
