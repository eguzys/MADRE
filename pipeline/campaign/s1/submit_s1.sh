#!/bin/bash
# Spreading-1 evaluation: submit the S1_* families of plan_s1.tsv (plan_s1.py output) HELD, nice 3000.
# Usage: submit_s1.sh <ERE on the S1 name> [s0only]
#   rows with a reuse entry (an exact SPR1_* twin) are never submitted. s0only = submit only shard s0 (smoke test);
#   a later full call fills the remaining shards (shard-level idempotent, see s1lib.sh).
set -u
export PATH=/usr/sepp/bin:/usr/bin:/bin:$PATH SLURM_CONF=/home/sladmitet/slurm/slurm.conf
. $HOME/u100diag/c4s/s1/s1lib.sh
PLAN=$S1_W/s1/plan_s1.tsv
RX=${1:?regex}; MODE=${2:-all}
tail -n +2 $PLAN | while IFS=$'\t' read -r s1 orig kind sb a1 users ck exp mem tim logdir split reuse fpp fpw; do
  echo "$s1" | grep -Eq "$RX" || continue
  [ "$reuse" != "-" ] && { echo "REUSE $s1 = $reuse (not submitted)"; continue; }
  if [ "$MODE" = s0only ]; then
    s1_submit "$s1" "$kind" "$sb" "$a1" "$users" "$ck" "$exp" "$mem" "$tim" s0only "$logdir"
  else
    s1_submit "$s1" "$kind" "$sb" "$a1" "$users" "$ck" "$exp" "$mem" "$tim" "$split" "$logdir"
  fi
done
