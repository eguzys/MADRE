#!/usr/bin/env bash
set -uo pipefail
N_EPISODES=100
COND=sinr
RESULTS_DIR=results/eval_fix2_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RESULTS_DIR"
OUT=out/benchmarl
FFN_N4=$OUT/ffn_probe/normalizedmappo_static_mlp__5540bc8e_26_05_27-01_39_19/checkpoints/checkpoint_4320000.pt
DS_N4=$OUT/deepsets_probe/normalizedmappo_static_deepsetsmodel__b743decc_26_05_31-20_06_04/checkpoints/checkpoint_3960000.pt

run_eval() {
  local tag=$1 n=$2 policy=$3 ckpt=$4
  if [[ "$policy" == "trained" && ! -f "$ckpt" ]]; then echo "!!! SKIP $tag — missing $ckpt"; return; fi
  echo "=== $tag  (N=$n, policy=$policy, n=$N_EPISODES) ==="
  python evaluation/evaluate.py "$ckpt" --policy "$policy" --condition "$COND" \
      -n "$N_EPISODES" --numb-of-nodes "$n" --no-render 2>&1 | tee "$RESULTS_DIR/${tag}.log"
  echo
}

run_eval deepsets_transfer_n5 5 trained "$DS_N4"
run_eval greedy_n3 3 greedy "$FFN_N4"
run_eval greedy_n4 4 greedy "$FFN_N4"
run_eval greedy_n5 5 greedy "$FFN_N4"
echo "ALL DONE. logs + csvs in: $RESULTS_DIR"
