#!/bin/bash
# eval_followup.sh - everything eval_tonight.sh could not finish, in the order
# that matters. Run ONLY after eval_tonight.sh has exited (never two cells at
# once - 2x8-wide on 16 cores measured SLOWER, load 87).
#
#   pgrep -f eval_tonight && echo "still running, do not start"
#
# ORDER IS DELIBERATE: every cell that completes the N-transfer matrix runs
# BEFORE the sigma re-measurement. The matrix is the deliverable; sigma03 at
# n=100 is a tidy-up. If this script is killed part-way, the matrix survives.
set -uo pipefail
cd /home/emilis/Documents/thesis/code
export CUDA_VISIBLE_DEVICES=""

E=omnet_module/eval_n100.sh
RREPO="${OM2G_SCRATCH:-/usr/itetnas04/data-scratch-01/eguzys/data}/resilient-backbone"
SSH="ssh -o ConnectTimeout=30 -o BatchMode=yes eguzys@login.ee.ethz.ch"

# A cell for an arm that has not finished training yet would score a partial
# checkpoint and look like a real result. fetch_arm already refuses anything
# under 500k frames, but check explicitly so the skip is LOUD rather than a
# one-line "fetch failed" buried in the log.
trained() {  # trained <tag> -> 0 if checkpoint_1002000.pt exists on arton
  local r; r=$($SSH "ls $RREPO/src/out/benchmarl_omnet/*_${1}/*/checkpoints/checkpoint_1002000.pt 2>/dev/null | head -1" 2>/dev/null | tr -d '\r')
  [ -n "$r" ]
}

echo "############ 1. MISSED N=6 deepsets transfer cell ########################"
# The N=6 deepsets transfer cell completes the ST-vs-deepsets architecture
# comparison at N=6. need_space() requires 1 GB free to fetch the checkpoint.
EVAL_N=6 EVAL_TRAIN_N=5 EVAL_OUT=out/n100_N6 EVAL_SKIP_BASELINES=1 \
  EVAL_ARMS="selftok_ds" bash "$E"

echo "############ 2. FFN row: ffn3 at N=3 #####################################"
# ffn3/ffn4 are trained AT their own N, so EVAL_TRAIN_N must NOT be set.
# N=3 has no baselines yet - hover/greedy are N-specific and set the gap
# denominator for the whole column, so they run first.
if trained ffn3; then
  EVAL_N=3 EVAL_OUT=out/n100_N3 EVAL_ARMS="" bash "$E"
  EVAL_N=3 EVAL_OUT=out/n100_N3 EVAL_SKIP_BASELINES=1 EVAL_ARMS="ffn3" bash "$E"
else
  echo "!! ffn3 has no checkpoint_1002000.pt on arton yet - SKIPPING (re-run later)"
fi

echo "############ 3. FFN row: ffn4 at N=4 #####################################"
# N=4 baselines come from eval_tonight.sh step 3, so they are not re-run.
if trained ffn4; then
  EVAL_N=4 EVAL_OUT=out/n100_N4 EVAL_SKIP_BASELINES=1 EVAL_ARMS="ffn4" bash "$E"
else
  echo "!! ffn4 has no checkpoint_1002000.pt on arton yet - SKIPPING (re-run later)"
fi

echo "############ 4. sigma03_st at n=100 (tidy-up, LAST) ######################"
# +0.430 is an n=20 number tabulated beside n=100 sigma05 (+0.357) and sigma02
# (+0.371). n=20 is too small to order these arms, so "the low-sigma arms are
# flat" needs all three at n=100.
EVAL_SKIP_BASELINES=1 EVAL_ARMS="sigma03_st" bash "$E"

echo "############ done $(date '+%F %T') #######################################"
echo "Matrix after this: N=5 {ST,DS,FFN}  N=6 {ST,DS}  N=4 {ST,DS,FFN4}  N=3 {FFN3}"
echo "Still owed: ffn6@N=6 (trains until ~04:00), and the 9 odim arms."

# Completion marker. watchdog.sh cannot otherwise distinguish "finished
# normally" from "died", and restarting a finished queue re-downloads every
# checkpoint before skipping each cell.
touch /home/emilis/Documents/thesis/code/out/.eval_queue_done
echo "wrote done-marker out/.eval_queue_done"
