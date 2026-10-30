# Spreading-1 evaluation: shared submit function, sourced by submit_s1.sh and submit_s1_robust_ext.sh.
# s1_submit NAME KIND SB A1 USERS CK EXP MEM TIME SPLIT LOGDIR
#   KIND learned : sbatch SB A1(cell) USERS ss ep CK NAME      (shard_spr.sbatch / shard_spr_ub.sbatch)
#   KIND baseline: sbatch SB A1(policy) USERS ss ep NAME       (baseline_shard_cl_*_spr / suite/*_spr)
#   EXP: comma list; the token UB expands to the user-beacon export (same string as every UB family).
#   SPLIT std = 10 shards x 10 eps (ss 0..90); gtm64 = as std but ss 60 split into (60,4) (64,1) (65,5), so ep
#   seed 64 of NDUB100_GTMG runs in its own 1-episode shard.
#   MEM/TIME "-" = the sbatch file's default. Every shard HELD (-H), --nice=3000, partition list widened after submit.
#   Shard-level idempotent: a shard is skipped if its log exists or its job name is queued (held jobs are queued).
# Ledger $W/ledger_s1.tsv: time NAME ss job SB A1 USERS CK EXP MEM TIME EP LOGDIR
S1_D=/usr/itetnas04/data-scratch-01/eguzys/data
S1_RB=$S1_D/resilient-backbone
S1_W=$S1_D/u100diag/c4s
S1_LEDGER=$S1_W/ledger_s1.tsv
S1_UB=OM2G_USER_BEACON=1,OM2G_USER_BEACON_INTERVAL=1.0,OM2G_USER_BEACON_BYTES=1,OMNET_HOST_CODE=$S1_D/ub_smoke,OMNET_SMOKE_DIR=$S1_D/ub_smoke/omnet_smoke

s1_shards() {  # prints "ss ep" lines
  if [ "$1" = s0only ]; then echo "0 10"
  elif [ "$1" = gtm64 ]; then printf "0 10\n10 10\n20 10\n30 10\n40 10\n50 10\n60 4\n64 1\n65 5\n70 10\n80 10\n90 10\n"
  else for ss in 0 10 20 30 40 50 60 70 80 90; do echo "$ss 10"; done; fi
}

s1_submit() {
  local NAME=$1 KIND=$2 SB=$3 A1=$4 USERS=$5 CK=$6 EXP=$7 MEM=$8 TIME=$9 SPLIT=${10} LOGDIR=${11}
  local ss ep j n=0 opts XP q
  case "$NAME" in S1_*) ;; *) echo "s1_submit: refusing non-S1 name $NAME"; return 1;; esac
  [ -f "$SB" ] || { echo "s1_submit: no sbatch file $SB"; return 1; }
  if [ "$KIND" = learned ]; then [ -s "$S1_RB/src/$CK" ] || { echo "s1_submit: $NAME ckpt missing $CK"; return 1; }; fi
  XP=$(echo "$EXP" | sed "s#\(^\|,\)UB\(,\|$\)#\1$S1_UB\2#")
  q=$(squeue -u eguzys -h -o "%j")
  opts=(-H --parsable --nice=3000)
  [ "$MEM" != "-" ] && opts+=(--mem=$MEM)
  [ "$TIME" != "-" ] && opts+=(--time=$TIME)
  while read -r ss ep; do
    [ -e "$LOGDIR/${NAME}_s$ss.log" ] && continue
    echo "$q" | grep -qx "${NAME}_s$ss" && continue
    if [ "$KIND" = learned ]; then
      j=$(cd $S1_RB && env -u OM2G_USER_BEACON -u OMNET_HOST_CODE -u OMNET_SMOKE_DIR -u DIAG_SPAWN -u DIAG_MINSEP -u DIAG_SENDERPROB -u DIAG_CKPT -u DIAG_POST -u OM2G_EVAL_POLICY \
          sbatch "${opts[@]}" --export="$XP" --job-name=${NAME}_s$ss "$SB" "$A1" "$USERS" "$ss" "$ep" "$CK" "$NAME" 2>&1 | tail -1)
    else
      j=$(cd $S1_RB && env -u OM2G_USER_BEACON -u OMNET_HOST_CODE -u OMNET_SMOKE_DIR -u DIAG_SPAWN -u DIAG_MINSEP -u DIAG_SENDERPROB -u DIAG_CKPT -u DIAG_POST -u OM2G_EVAL_POLICY \
          sbatch "${opts[@]}" --export="$XP" --job-name=${NAME}_s$ss "$SB" "$A1" "$USERS" "$ss" "$ep" "$NAME" 2>&1 | tail -1)
    fi
    case "$j" in ''|*[!0-9]*) echo "  FAIL $NAME s$ss -> $j"; continue;; esac
    scontrol update JobId=$j Partition=disco.med,cpu.normal,gpu.normal >/dev/null 2>&1
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -u +%FT%TZ)" "$NAME" "$ss" "$j" "$SB" "$A1" "$USERS" "$CK" "$EXP" "$MEM" "$TIME" "$ep" "$LOGDIR" >> $S1_LEDGER
    n=$((n+1))
  done < <(s1_shards "$SPLIT")
  echo "S1 HELD $NAME: $n shards ($KIND $(basename $SB) $A1 users=$USERS ckpt=$(basename "$CK") exp=$EXP mem=$MEM time=$TIME split=$SPLIT)"
}
