#!/bin/bash
# watchdog.sh - a REAL OS process that supervises arton + the local eval queue.
#
# WHY THIS EXISTS: a supervisor you cannot see in `ps` is not one you can leave
# running unattended. This script is a plain nohup'd bash loop you can see in
# `ps`, it writes a timestamped line to out/watchdog.log every tick, and it
# resubmits dead jobs and restarts the eval queue on its own.
#
# Watch it:   tail -f code/out/watchdog.log
# Check it:   pgrep -af watchdog.sh
# Stop it:    pkill -f omnet_module/watchdog.sh
#
# SAFETY RULES ENCODED HERE:
#  - A failed ssh returns EMPTY output, which reads as "every job is gone".
#    Auto-resubmitting on that would launch a dozen duplicate 16h jobs. So the
#    ssh exit code is checked and a non-zero rc is SSHFAIL: log it, touch
#    nothing.
#  - A job at 167/167 is DONE, not dead. Never resubmit it.
#  - Each tag is resubmitted at most $MAX_RESUB times ever. A job that dies
#    instantly and repeatedly needs inspection, and resubmitting it forever
#    just fills the queue and hides the error.
#  - Never runs two eval cells at once (2x8-wide on 16 cores measured SLOWER,
#    load 87). It only restarts the queue when NO cell is running.
#  - Read-only on arton apart from `sbatch`. Deletes nothing, anywhere.
set -uo pipefail

REPO=/home/emilis/Documents/thesis/code
RREPO="${OM2G_SCRATCH:-/usr/itetnas04/data-scratch-01/eguzys/data}/resilient-backbone"
SSH="ssh -o ConnectTimeout=30 -o BatchMode=yes eguzys@login.ee.ethz.ch"
REMOTE_ENV='export SLURM_CONF=/home/sladmitet/slurm/slurm.conf; export SCRATCH=/usr/itetnas04/data-scratch-01/eguzys/data'
LOG=$REPO/out/watchdog.log
STATE=$REPO/out/.watchdog_resub
INTERVAL=${WATCHDOG_INTERVAL:-1800}
MAX_RESUB=2

# tag -> the submit wrapper that recreates it. Only these are ever resubmitted.
TAGS="ffn3 ffn4 ffn6 odim2_st odim2_deepsets odim2_ffn odim4_st odim4_deepsets odim4_ffn odim16_st odim16_deepsets odim16_ffn"

mkdir -p "$REPO/out"; touch "$STATE"
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

resub_count() { grep -c "^$1\$" "$STATE" 2>/dev/null || echo 0; }

say "=== watchdog started (pid $$, interval ${INTERVAL}s) ==="

while true; do
  # ---------- arton ----------
  SNAP=$($SSH "$REMOTE_ENV; squeue -u eguzys -h -o '%i %j %t'" 2>/dev/null); RC=$?
  if [ $RC -ne 0 ]; then
    say "SSHFAIL rc=$RC -- cannot see arton. Taking NO action this tick."
  else
    RUNNING=$(echo "$SNAP" | awk '{print $2}' | sed 's/^omnet_//')
    NR=$(echo "$SNAP" | grep -c . )
    say "arton: $NR job(s) in queue"
    for tag in $TAGS; do
      echo "$RUNNING" | grep -qx "$tag" && continue      # still queued/running
      # Absent from squeue. Done, or dead?
      ITER=$($SSH "$REMOTE_ENV; tr '\r' '\n' < \$SCRATCH/resilient-backbone/slurm-*.out 2>/dev/null | true; \
             d=\$(ls -dt \$SCRATCH/resilient-backbone/src/out/benchmarl_omnet/*_${tag}/ 2>/dev/null | head -1); \
             [ -n \"\$d\" ] && ls \$d*/checkpoints/checkpoint_1002000.pt 2>/dev/null | head -1" 2>/dev/null)
      if [ -n "$ITER" ]; then
        say "  $tag: FINISHED (checkpoint_1002000.pt present) - no action"
        continue
      fi
      n=$(resub_count "$tag")
      if [ "$n" -ge "$MAX_RESUB" ]; then
        say "  $tag: DEAD but already resubmitted ${n}x - NOT resubmitting again, needs a human"
        continue
      fi
      if ! $SSH "test -f $RREPO/submit_${tag}.sh" 2>/dev/null; then
        say "  $tag: DEAD but submit_${tag}.sh missing - cannot resubmit"
        continue
      fi
      OUT=$($SSH "$REMOTE_ENV; cd $RREPO && sbatch --job-name=omnet_${tag} --time=1-00:00:00 \
            --cpus-per-task=8 --mem=64G --gres=gpu:titan_rtx:1 submit_${tag}.sh" 2>&1 | grep -oE '[0-9]+$')
      if [ -n "$OUT" ]; then
        echo "$tag" >> "$STATE"
        say "  $tag: DEAD -> RESUBMITTED as job $OUT (resubmit #$((n+1)))"
      else
        say "  $tag: DEAD, resubmit FAILED"
      fi
    done
  fi

  # ---------- local eval queue ----------
  # Bracket the pattern ([e]val...) so pgrep does not match the shell running
  # this very command -- an unbracketed pattern self-matches and reports 1 when
  # nothing is running, which reads as "alive".
  #
  # $QUEUE is whichever driver is current (default eval_followup.sh: the N=6
  # deepsets cell, then the FFN row, then sigma03). Restarting a different,
  # already-finished script would exit instantly and look like a successful
  # restart.
  QUEUE=${WATCHDOG_QUEUE:-omnet_module/eval_followup.sh}
  DONE_MARK=$REPO/out/.eval_queue_done
  CELLS=$(pgrep -fc "[e]valuate_omnet.py" 2>/dev/null || echo 0)
  DRIVER=$(pgrep -fc "[e]val_followup.sh" 2>/dev/null || echo 0)
  if [ -f "$DONE_MARK" ]; then
    :   # queue finished on purpose; nothing to supervise
  elif [ "$DRIVER" -gt 0 ]; then
    say "eval: driver alive, $CELLS cell process(es)"
  elif [ "$CELLS" -gt 0 ]; then
    say "eval: driver gone but $CELLS cell(s) still finishing - waiting, NOT restarting"
  else
    # A COMPLETED queue looks identical to a crashed one: both are "driver and
    # cells gone". The done-marker tells them apart; without it a finished queue
    # would be restarted, and because fetch_arm runs BEFORE run_cell's "already
    # done" check, each restart would re-download every 405 MB checkpoint.
    # As with SSHFAIL: "cannot see it" and "it is finished" are not the same
    # state as "it died".
    say "eval: driver AND cells both gone, no done-marker -> restarting $QUEUE"
    ( cd "$REPO" && nohup bash "$QUEUE" >> out/eval_followup.log 2>&1 & )
  fi

  FREE=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  [ "${FREE:-9}" -lt 2 ] && say "WARNING: only ${FREE}G free on / - fetches will start refusing"

  sleep "$INTERVAL"
done
