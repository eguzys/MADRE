#!/bin/bash
# watchdog_arton.sh - poll arton, requeue on infrastructure failure, evaluate on
# completion. Runs DETACHED (setsid), so it survives the terminal that started
# it. Launch with:
#     setsid nohup bash omnet_module/watchdog_arton.sh >/dev/null 2>&1 &
#
# EVALUATION IS OMNeT-TRACK ONLY. It calls omnet_module/evaluate_omnet.py, never
# src/evaluation/evaluate.py -- that one builds PADREParallelEnv with ep_length
# 1000 and would score a different world.
#
# Requeue policy: at most ONE requeue per arm, and only for an INFRASTRUCTURE
# fault (e.g. a SEPP/apptainer node fault). A config or code error is left
# stopped on purpose -- resubmitting it would only cost another 16 h.
set -uo pipefail

REPO=/home/emilis/Documents/thesis/code
PY=/home/emilis/pythonEnviroment/padre/bin/python
SSH="ssh -o ConnectTimeout=30 -o BatchMode=yes -o ServerAliveInterval=30 eguzys@login.ee.ethz.ch"
RSCRATCH="${OM2G_SCRATCH:-/usr/itetnas04/data-scratch-01/eguzys/data}"
RREPO=$RSCRATCH/resilient-backbone
REMOTE_ENV="export SLURM_CONF=/home/sladmitet/slurm/slurm.conf; export SCRATCH=$RSCRATCH;"

STATE=$REPO/out/watchdog
LOG=$STATE/watchdog.log
mkdir -p "$STATE"

# tag:jobid  -- jobid is rewritten in place on requeue
JOBS_FILE=$STATE/jobs.txt
if [ ! -f "$JOBS_FILE" ]; then
  cat > "$JOBS_FILE" <<'EOF'
ent003_st 2201422
ent001_st 2201423
selftok_ds 2201491
selftok_ffn 2201492
sigma03_st 2201685
EOF
fi

POLL=${WATCHDOG_POLL:-600}

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# ---------------------------------------------------------------------------
# Evaluate one finished arm: fetch checkpoint + config.pkl, run 20 seeds
# locally, append the paired gap to results.txt.
# ---------------------------------------------------------------------------
evaluate_arm() {
  local tag="$1"
  local done_marker="$STATE/${tag}.evaluated"
  [ -f "$done_marker" ] && return 0

  log "$tag: locating run dir on arton"
  local rundir
  rundir=$($SSH "ls -d $RREPO/src/out/benchmarl_omnet/*${tag}*/*/ 2>/dev/null | head -1" | tr -d '\r')
  if [ -z "$rundir" ]; then log "$tag: NO RUN DIR - skipping"; return 1; fi

  local ckpt="$rundir/checkpoints/checkpoint_1002000.pt"
  if ! $SSH "test -f '$ckpt'"; then
    # fall back to the highest-numbered checkpoint
    ckpt=$($SSH "ls $rundir/checkpoints/checkpoint_*.pt 2>/dev/null | sort -t_ -k2 -n | tail -1" | tr -d '\r')
    [ -z "$ckpt" ] && { log "$tag: NO CHECKPOINT - skipping"; return 1; }
    log "$tag: 1002000 absent, using $(basename "$ckpt")"
  fi

  local runname; runname=$(basename "${rundir%/}")
  local dest="$REPO/out/arton_ckpt/$runname"
  mkdir -p "$dest/checkpoints"
  log "$tag: fetching -> $dest"
  scp -q -o BatchMode=yes "eguzys@login.ee.ethz.ch:$rundir/config.pkl" "$dest/config.pkl" || { log "$tag: scp config.pkl FAILED"; return 1; }
  scp -q -o BatchMode=yes "eguzys@login.ee.ethz.ch:$ckpt" "$dest/checkpoints/" || { log "$tag: scp checkpoint FAILED"; return 1; }

  local lckpt="$dest/checkpoints/$(basename "$ckpt")"
  local outdir="$REPO/out/eval_$tag"
  mkdir -p "$outdir/parts"

  cd "$REPO" || return 1
  unset OMNET_APPTAINER_SIF OMNET_USER_TX_mW OMNET_DRONE_TX_mW OMNET_STEP_M
  export OMNET_MAX_STEPS=250 OMNET_ENCODER_MODE=learned OMNET_PEER_LINK_MASK=1
  export CUDA_VISIBLE_DEVICES=""

  log "$tag: evaluating 20 seeds (OMNeT track)"
  local jf="$outdir/parts/.jobs"; : > "$jf"
  # No trailing field: xargs -L 1 joins a line ending in whitespace to the next.
  for s in 0 2 4 6 8 10 12 14 16 18; do echo "$s 2" >> "$jf"; done
  ( export outdir lckpt PY REPO
    run_one() {
      local st="$1" ct="$2"
      local o="$outdir/parts/t_${st}.csv"
      [ -s "$o" ] && return 0
      "$PY" omnet_module/evaluate_omnet.py "$lckpt" --policy trained --num-drones 5 \
        --seed-start "$st" -n "$ct" --out "$o" > "$outdir/parts/t_${st}.log" 2>&1
    }
    export -f run_one
    xargs -a "$jf" -P 8 -L 1 bash -c 'run_one "$@"' _
  )

  local dest_csv="$outdir/trained_${tag}_n5.csv"
  echo "seed,episode_reward,steps" > "$dest_csv"
  cat "$outdir"/parts/t_*.csv 2>/dev/null | grep -v '^seed,' | sort -t, -k1,1n >> "$dest_csv"
  local n=$(( $(wc -l < "$dest_csv") - 1 ))
  if [ "$n" -lt 1 ]; then log "$tag: EVAL PRODUCED NOTHING - see $outdir/parts/*.log"; return 1; fi

  "$PY" - "$dest_csv" "$tag" >> "$STATE/results.txt" <<'PYEOF'
import csv, sys, statistics as st
def load(p): return {int(r["seed"]): float(r["episode_reward"])/250.0 for r in csv.DictReader(open(p))}
t = load(sys.argv[1]); tag = sys.argv[2]
b = "/home/emilis/Documents/thesis/code/out/omnet_v3_baselines"
g = load(f"{b}/greedy_n5.csv"); h = load(f"{b}/hover_n5.csv")
S = sorted(set(t) & set(g) & set(h))
gap = sum(t[s]-h[s] for s in S) / sum(g[s]-h[s] for s in S)
print(f"{tag:<14} n={len(S):<3} mean/step={st.mean([t[s] for s in S]):8.1f} "
      f"median={st.median([t[s] for s in S]):8.1f} gap={gap:+.3f} "
      f"beats_hover={sum(t[s]>h[s] for s in S)}/{len(S)}   (ST baseline 737.1, gap +0.766)")
PYEOF

  touch "$done_marker"
  log "$tag: EVALUATED -> $(tail -1 "$STATE/results.txt")"
  return 0
}

log "watchdog started (pid $$, poll ${POLL}s)"

while true; do
  all_done=1
  queue=$($SSH "$REMOTE_ENV squeue -u eguzys -h -o '%i %T'" 2>/dev/null | tr -d '\r')

  while read -r tag jid; do
    [ -z "$tag" ] && continue
    [ -f "$STATE/${tag}.evaluated" ] && continue
    [ -f "$STATE/${tag}.dead" ] && continue
    all_done=0

    state=$(echo "$queue" | awk -v j="$jid" '$1==j {print $2}')

    if [ -n "$state" ]; then
      continue   # still queued or running
    fi

    # Not in the queue: finished, or died.
    iter=$($SSH "tr '\r' '\n' < $RREPO/slurm-${jid}.out 2>/dev/null | grep -oE '[0-9]+/167' | tail -1" | tr -d '\r')
    if [ "$iter" = "167/167" ]; then
      log "$tag ($jid): COMPLETE $iter"
      evaluate_arm "$tag" || log "$tag: evaluation failed, will retry next pass"
    else
      # Distinguish infrastructure fault from a config/code error.
      infra=$($SSH "tr '\r' '\n' < $RREPO/slurm-${jid}.out 2>/dev/null | grep -cE 'apptainer: Command not found|SEPP Setup Error|OMNeT child died'" | tr -d '\r')
      log "$tag ($jid): GONE at ${iter:-<no iter>}, infra_markers=$infra"
      if [ "${infra:-0}" -gt 0 ] && [ ! -f "$STATE/${tag}.requeued" ]; then
        newid=$($SSH "$REMOTE_ENV cd $RREPO && sbatch --parsable --gres=gpu:titan_rtx:1 --cpus-per-task=8 --mem=64G --time=24:00:00 --job-name=omnet_${tag} submit_${tag}.sh" | tr -d '\r' | tail -1)
        if [[ "$newid" =~ ^[0-9]+$ ]]; then
          touch "$STATE/${tag}.requeued"
          sed -i "s/^${tag} .*/${tag} ${newid}/" "$JOBS_FILE"
          log "$tag: REQUEUED as $newid (infrastructure fault, one requeue only)"
        else
          log "$tag: requeue FAILED: $newid"
          touch "$STATE/${tag}.dead"
        fi
      else
        [ -f "$STATE/${tag}.requeued" ] && log "$tag: already requeued once - leaving dead" \
                                       || log "$tag: not an infra fault - leaving dead on purpose"
        touch "$STATE/${tag}.dead"
      fi
    fi
  done < "$JOBS_FILE"

  if [ "$all_done" -eq 1 ]; then
    log "all arms evaluated or dead - watchdog exiting"
    log "RESULTS:"; cat "$STATE/results.txt" >> "$LOG" 2>/dev/null
    exit 0
  fi
  sleep "$POLL"
done
