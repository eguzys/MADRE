#!/bin/bash
# Submits the u=5-22 x {st,ds} x {learned,frozen} eval grid.
# Per-job script: eval_learned_sweep.sh <checkpoint> <users>
# Wall 2h: measured ~20 s/episode => n=100 is ~35 min (rendering off).
set -euo pipefail
SC=/cluster/scratch/eguzys
JOB=$SC/eval_learned_sweep.sh

declare -A CKPT=(
  [st_learned]=$SC/normalizedmappo_static_stmodel__af33a1cd_26_07_21-00_01_34/checkpoints/checkpoint_3000000.pt
  [ds_learned]=$SC/normalizedmappo_static_deepsetsmodel__582418b0_26_07_21-15_48_01/checkpoints/checkpoint_3000000.pt
  [st_frozen]=$SC/st_raw_0a58342b/checkpoints/checkpoint_3000000.pt
  [ds_frozen]=$SC/ds_raw_61c39ef7/checkpoints/checkpoint_3000000.pt
)

for cell in "${!CKPT[@]}"; do
  ck="${CKPT[$cell]}"
  if [ ! -f "$ck" ]; then echo "MISSING checkpoint for $cell: $ck" >&2; exit 1; fi
  # learned mode pads to max_users=20 (numbOfUsers upper bound); u>20 raises
  hi=22; case "$cell" in *_learned) hi=20;; esac
  for u in $(seq 5 $hi); do
    sbatch --job-name="ev_${cell}_u${u}" \
           --time=02:00:00 --cpus-per-task=2 --mem-per-cpu=2G \
           --output="$SC/eval_${cell}_u${u}_%j.out" \
           "$JOB" "$ck" "$u"
  done
done
