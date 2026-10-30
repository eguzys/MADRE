#!/bin/bash
# eval_odim.sh - the out_dim compression sweep at n=100, plus the two cells
# still missing from the N-transfer matrix.
#
# The sweep tests the compression bottleneck: 325 raw user numbers are masked-SUMMED into out_dim before any
# architecture sees them, and 8 was inherited from the frozen encoder, never
# chosen. The sweep goes DOWN {2,4} and UP {16} around that 8.
#
# STRICTLY SERIAL - each cell is 8-wide on 16 cores; two at once measured
# SLOWER. Idempotent at chunk granularity.
#
# ⚠️ OMNET_USER_ENC_OUT_DIM must match what the arm was TRAINED at or the model
# rebuild dies on a bare tensor-shape mismatch. One invocation per out_dim.
set -uo pipefail
cd /home/emilis/Documents/thesis/code
export CUDA_VISIBLE_DEVICES=""      # laptop GPU OOM misreports as obs-shape mismatch

E=omnet_module/eval_n100.sh

# eval_n100.sh never deletes a fetched checkpoint, and 8 cells x 405 MB =
# 3.2 GB is about the free local disk, so the queue drops each checkpoint once
# its cell has produced a CSV. The CSV is the deliverable; the checkpoint can be
# fetched again in ~3 min.
sweep() {   # sweep <out_dim> <tag...>
  local od="$1"; shift
  for tag in "$@"; do
    OMNET_USER_ENC_OUT_DIM="$od" EVAL_SKIP_BASELINES=1 EVAL_ARMS="$tag" bash "$E"
    if [ -s "out/n100/${tag}_n100.csv" ]; then
      rm -rf out/arton_ckpt/*   # cell banked, reclaim the 405 MB
      echo "   [disk] $(df -h / | awk 'NR==2{print $4}') free after $tag"
    else
      echo "!! $tag produced no CSV - leaving checkpoint in place for inspection"
    fi
  done
}

echo "############ 1. out_dim DOWN: 2 (the tightest compression) ##############"
# Baselines at N=5 already exist in out/n100 and are model-independent
# (hover/greedy never touch the user encoder), so they are shared across the
# whole sweep - out_dim=8 is the existing st_baseline/selftok_ds/selftok_ffn row.
sweep 2 odim2_st odim2_deepsets odim2_ffn

echo "############ 2. out_dim DOWN: 4 ########################################"
sweep 4 odim4_st odim4_deepsets odim4_ffn

echo "############ 3. ffn6 at N=6 - completes the FFN diagonal ###############"
# The FFN row is one model per N (ffn3@3, ffn4@4, ffn5@5); ffn6 finished
# overnight and is the last cell of the N=6 column. out_dim is the default 8,
# so this runs OUTSIDE sweep()'s env override.
EVAL_N=6 EVAL_OUT=out/n100_N6 EVAL_SKIP_BASELINES=1 EVAL_ARMS="ffn6" bash "$E"
[ -s out/n100_N6/ffn6_n100.csv ] && rm -rf out/arton_ckpt/*

echo "############ 4. out_dim UP: 16 #########################################"
# odim16_ffn was still training at 432k frames this morning (fetch_arm refuses
# anything under 500k, so it will skip LOUDLY rather than score a partial net).
# Re-run this line once it lands.
sweep 16 odim16_st odim16_deepsets odim16_ffn

echo "############ done $(date '+%F %T') #####################################"
echo "Still owed: odim16_ffn if it skipped, plus ST/DS at N=3 (needs 00fa5f10"
echo "re-fetched - the local ST reference was deleted in the 20:52 cleanup)."
touch /home/emilis/Documents/thesis/code/out/.eval_odim_done
