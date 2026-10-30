#!/usr/bin/env bash
# eval_all.sh — evaluate every trained model + greedy at 100 episodes.
set -uo pipefail

N_EPISODES=100
COND=sinr
TASK_YAML=conf/task/padre/static.yaml
RESULTS_DIR=results/eval_all_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RESULTS_DIR"

BACKUP=$(mktemp)
cp "$TASK_YAML" "$BACKUP"
restore() { cp "$BACKUP" "$TASK_YAML"; rm -f "$BACKUP"; echo ">>> restored $TASK_YAML"; }
trap restore EXIT INT TERM

set_n() { sed -i "s/^\(\s*numbOfNodes:\).*/\1 $1/" "$TASK_YAML"; echo ">>> numbOfNodes -> $1"; }

OUT=out/benchmarl
FFN_N3=$OUT/ffn_n3/normalizedmappo_static_mlp__27659a5c_26_05_30-13_37_07/checkpoints/checkpoint_4920000.pt
FFN_N4=$OUT/ffn_probe/normalizedmappo_static_mlp__5540bc8e_26_05_27-01_39_19/checkpoints/checkpoint_4320000.pt
FFN_N5=$OUT/ffn_n5/normalizedmappo_static_mlp__45e092f1_26_05_29-15_58_45/checkpoints/checkpoint_3960000.pt
ST_N4=$OUT/st_probe/normalizedmappo_static_stmodel__ada0e8a8_26_05_28-01_09_34/checkpoints/checkpoint_4800000.pt
DS_N4=$OUT/deepsets_probe/normalizedmappo_static_deepsetsmodel__b743decc_26_05_31-20_06_04/checkpoints/checkpoint_3960000.pt

run_eval() {
  local tag=$1 n=$2 policy=$3 ckpt=$4
  if [[ "$policy" == "trained" && ! -f "$ckpt" ]]; then
    echo "!!! SKIP $tag — checkpoint not found: $ckpt"; return
  fi
  set_n "$n"
  echo "=== $tag  (N=$n, policy=$policy, n=$N_EPISODES) ==="
  python evaluation/evaluate.py "$ckpt" \
      --policy "$policy" --condition "$COND" -n "$N_EPISODES" --no-render \
      2>&1 | tee "$RESULTS_DIR/${tag}.log"
  echo
}

run_eval ffn_n3        3 trained "$FFN_N3"
run_eval ffn_n4        4 trained "$FFN_N4"
run_eval ffn_n5        5 trained "$FFN_N5"
run_eval st_n4         4 trained "$ST_N4"
run_eval deepsets_n4   4 trained "$DS_N4"

run_eval st_transfer_n3       3 trained "$ST_N4"
run_eval st_transfer_n5       5 trained "$ST_N4"
run_eval deepsets_transfer_n3 3 trained "$DS_N4"
run_eval deepsets_transfer_n5 5 trained "$DS_N4"

run_eval greedy_n3 3 greedy "$FFN_N4"
run_eval greedy_n4 4 greedy "$FFN_N4"
run_eval greedy_n5 5 greedy "$FFN_N4"

echo "ALL DONE. logs + csvs in: $RESULTS_DIR"
