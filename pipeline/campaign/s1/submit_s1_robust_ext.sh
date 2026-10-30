#!/bin/bash
# Robustness evaluations at spreading 1 for all seven message sizes (eval only, no training).
#   USPAWN (uniform drone start) for L16, L256, L512   -> S1_USPAWN_<L>s<k>  (L8/32/64/128 already in plan_s1.tsv)
#   CKFIN  (LAST checkpoint)   for L8, L16, L32, L64, L256, L512 -> S1_CKFIN_<L>s<k>  (L128 already in plan_s1.tsv)
# Seeds = the 4-seed sets (L8 1,2,3,5  L16 1,2,3,4  L32 1,2,3,5  L64 1,2,3,4  L256 1,2,3,4  L512 1,2,3,4).
# Settings identical to the existing S1_USPAWN_L64s4 / S1_CKFIN_L128s* rows of plan_s1.tsv: shard_spr.sbatch, 40 users,
# chi=1 (no SENDERPROB), 1200 m world of the cell, beacon OFF, ALL,DIAG_SPREADING=1 (+DIAG_SPAWN=uniform for USPAWN),
# 10 shards x 10 episodes (ep seeds 0..99), 32G for L512 (as the planner's rule), HELD + nice 3000 via s1lib.sh s1_submit
# (ledger_s1.tsv).
# Checkpoints (never 'best', never a u100 stage):
#   USPAWN = the explicit best-return ckpt of the 40-user training = the ckpt of the cluster-start family of the
#            same seed in plan_s1.tsv (W1ev_br12rtL16s1-3 / FS_<L>s<k>), checked = pick_best_ckpt over the single run dir;
#   CKFIN  = the highest-frame checkpoint_<N>.pt in that same normalizedmappo_*/checkpoints dir (the 40-user run).
# Usage: submit_s1_robust_ext.sh plan            # write $S1_W/s1/plan_s1_robust_ext.tsv only
#        submit_s1_robust_ext.sh submit [ERE]    # (re)write the plan and submit every row (shard-level idempotent)
set -u
export PATH=/usr/sepp/bin:/usr/bin:/bin:$PATH SLURM_CONF=/home/sladmitet/slurm/slurm.conf
. $HOME/u100diag/c4s/s1/s1lib.sh
PLAN=$S1_W/s1/plan_s1.tsv
EXT=$S1_W/s1/plan_s1_robust_ext.tsv
MODE=${1:?plan|submit}; RX=${2:-.}
L=$S1_D/u100diag/logs; SB=$S1_D/u100diag/shard_spr.sbatch

fam() { local r=$1 k=$2   # the 1200 m cluster-start family of (rung, seed) = plan_s1.py fam1200
  case "$r" in L8|L32|L128) [ "$k" -le 3 ] && { echo ADPT40_${r}s$k; return; }; [ "$k" = 5 ] && { echo X5_${r}s$k; return; };; esac
  [ "$r" = L16 ] && [ "$k" -le 3 ] && { echo W1ev_br12rtL16s${k}_SPAWNcluster_TAIL; return; }
  echo FS_${r}s$k; }

tmp=$EXT.tmp.$$
printf "s1\torig\tkind\tsb\ta1\tusers\tck\texp\tmem\ttime\tlogdir\tsplit\tref_fam\tref_ck\n" > $tmp
row() {  # kind(USPAWN|CKFIN) rung seed
  local what=$1 r=$2 k=$3 f cell ck mem cdir last
  f=$(fam $r $k)
  cell=$(awk -F'\t' -v o=$f '$2==o {print $5}' $PLAN); ck=$(awk -F'\t' -v o=$f '$2==o {print $7}' $PLAN)
  [ -n "$cell" ] && [ -n "$ck" ] || { echo "NO PLAN ROW for $f" >&2; return 1; }
  case "$ck" in */normalizedmappo_*/checkpoints/checkpoint_*.pt) ;; *) echo "odd ckpt $ck" >&2; return 1;; esac
  case "$ck" in *u100*) echo "u100 ckpt refused $ck" >&2; return 1;; esac
  mem=-; case "$cell" in *L512*) mem=32G;; esac
  if [ "$what" = USPAWN ]; then
    printf "S1_USPAWN_%ss%s\tUSPAWN_%ss%s\tlearned\t%s\t%s\t40\t%s\tALL,DIAG_SPREADING=1,DIAG_SPAWN=uniform\t%s\t-\t%s\tstd\t%s\t%s\n" \
      $r $k $r $k $SB $cell $ck $mem $L $f $ck >> $tmp
  else
    cdir=$(dirname $ck)
    last=$(ls $S1_RB/src/$cdir | sed -n 's/^checkpoint_\([0-9][0-9]*\)\.pt$/\1/p' | sort -n | tail -1)
    [ -n "$last" ] || { echo "no checkpoints in $cdir" >&2; return 1; }
    printf "S1_CKFIN_%ss%s\tCKFIN_%ss%s\tlearned\t%s\t%s\t40\t%s\tALL,DIAG_SPREADING=1\t%s\t-\t%s\tstd\t%s\t%s\n" \
      $r $k $r $k $SB $cell $cdir/checkpoint_$last.pt $mem $L $f $ck >> $tmp
  fi
}
for r in L16 L256 L512; do for k in 1 2 3 4; do row USPAWN $r $k || exit 1; done; done
for rk in L8:1,2,3,5 L16:1,2,3,4 L32:1,2,3,5 L64:1,2,3,4 L256:1,2,3,4 L512:1,2,3,4; do
  r=${rk%%:*}; for k in $(echo ${rk#*:} | tr , ' '); do row CKFIN $r $k || exit 1; done
done
# never clobber a plan that differs (rows of an earlier run must stay what was submitted)
if [ -e $EXT ] && ! cmp -s $tmp $EXT; then echo "plan differs from existing $EXT; kept old, new in $tmp" >&2; exit 1; fi
mv $tmp $EXT
echo "plan: $(($(grep -c . $EXT) - 1)) families -> $EXT"
[ "$MODE" = submit ] || exit 0
tail -n +2 $EXT | while IFS=$'\t' read -r s1 orig kind sb a1 users ck exp mem tim logdir split rf rck; do
  echo "$s1" | grep -Eq "$RX" || continue
  s1_submit "$s1" "$kind" "$sb" "$a1" "$users" "$ck" "$exp" "$mem" "$tim" "$split" "$logdir"
done
