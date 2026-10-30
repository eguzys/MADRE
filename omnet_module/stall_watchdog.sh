#!/bin/bash
# Telemetry-stall watchdog. Backgrounded by train_omnet_*.sh, alongside
# seed_watchdog.sh.
#
# WHY THIS EXISTS: a worker process dying mid-rollout
# (an HiGHS OracleLPSolveError in the max-flow reward oracle on arton, or an
# OMNeT child exiting because squashfuse_ll ran out of threads under apptainer
# on Euler -- two DIFFERENT original causes, same downstream failure) throws
# torchrl's ParallelEnv cleanup path into a race: _check_for_faulty_process
# (torchrl/_utils.py, ~L481) calls `_p.terminate()` then `time.sleep(0.1)`
# then `_p.close()` on every surviving worker with NO try/except around
# close(), so if 100ms is not enough for one worker to actually die (it may
# still be tearing down its own OMNeT++/apptainer child), `_p.close()` raises
# ValueError -- which escapes the loop immediately, so any worker LATER in
# iteration order never even gets terminate() called and is left alive.
# Because torchrl's worker Processes are non-daemonic by default
# (batched_envs.py `daemon: bool = False`), those survivors make Python's
# multiprocessing atexit handler (`multiprocessing.util._exit_function`)
# call `.join()` on them with NO TIMEOUT when the interpreter unwinds after
# the (also uncaught) top-level exception. Net effect: the python process --
# and therefore the whole Slurm job step -- hangs FOREVER holding its CPUs.
# squeue reports RUNNING indefinitely; only a stalled telemetry CSV reveals
# it. Observed in cell .err logs on both clusters (OracleLPSolveError and
# container-mount failures as the original causes).
#
# This wrapper does NOT fix the torchrl race (that lives in the installed
# library, shared and versioned outside this repo). It is a robustness
# backstop: if the run's own progress signal (the OM2G telemetry CSV) stops
# growing for STALL_SECONDS, assume the job is a zombie and scancel it so
# Slurm frees the CPUs for the next cell instead of holding them for hours.
# False positives matter more here than false negatives -- a wrongly-killed
# healthy run only costs a resubmit, a real hang costs hours of 20-CPU
# allocation -- so the default threshold (20 min) is generous relative to the
# ~30-90s telemetry-flush cadence observed under normal training.
#
# Usage: stall_watchdog.sh <slurm_job_id> <telemetry_csv_path> [stall_seconds] [poll_seconds] [grace_seconds]
#   telemetry_csv_path  may be empty ("") for tracks that do not write one
#                        (plain OMNET-track cells, or an OM2G arm that hasn't
#                        set OM2G_TELEMETRY_CSV) -- the watchdog then exits
#                        immediately as a clean no-op, so it is always safe
#                        to background unconditionally.
set -uo pipefail

JOBID="${1:-0}"
CSV="${2:-}"
STALL="${3:-3600}"    # 60 min of silence => assume zombie.
# NOT 20 min. The slowest cells here (forwarding arms, ~8x beacon size)
# have a NATURAL telemetry cadence of 10-15 min, so a 20-min limit would sit on
# top of healthy behaviour. Real zombies sit silent for HOURS (e.g. 3h), so
# 60 min still catches them with a wide margin. This watchdog is a BACKSTOP
# for unattended hours, not the primary monitor.
POLL="${4:-60}"
GRACE="${5:-900}"     # 15 min allowed for the CSV to even be created

log() { echo "[stall_watchdog $(date '+%F %T')] $*"; }

if [ -z "$CSV" ] || [ -z "$JOBID" ] || [ "$JOBID" = "0" ]; then
  exit 0   # nothing to watch, or no job to scancel -- opt out cleanly
fi

# The process that backgrounded us (train_omnet_*.sh). Same discriminator
# seed_watchdog.sh uses: once it's gone, the run ended on its own (success
# or a clean failure) and there is nothing left for us to do.
LAUNCHER=$PPID

# Wait for the file to appear at all. Bounded, not indefinite -- a run that
# never gets this far has its own pre-flight gates to catch it; this
# watchdog only judges STALLS in an already-started run.
DEADLINE=$(( $(date +%s) + GRACE ))
while [ ! -f "$CSV" ]; do
  kill -0 "$LAUNCHER" 2>/dev/null || { log "launcher $LAUNCHER gone before $CSV ever appeared -- exiting"; exit 0; }
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    log "WARNING: $CSV never appeared within ${GRACE}s of launch -- giving up (not killing; a run that never starts has other gates)"
    exit 0
  fi
  sleep 10
done

# The watchdog may only judge silence that happened ON ITS OWN WATCH.
# A RESUMED run inherits a telemetry CSV from the previous attempt whose mtime
# is hours old, so a naive NOW-MTIME would fire within one poll on a healthy
# job. Clamping the reference to WATCH_START gives every run a full STALL window from the moment
# we start looking, whatever the file looked like before.
WATCH_START=$(date +%s)
# A RESUMED run must load a ~1.5 GB checkpoint, rebuild the envs and spawn its
# OMNeT subprocesses before it writes a single telemetry row -- observed to take
# well over STALL (20 min), which must not count as a stall. So the threshold
# is split:
# until we have seen the file grow ONCE on our own watch, the run is "starting"
# and gets STARTUP; after the first growth it is "running" and gets STALL.
STARTUP="${6:-${OM2G_STALL_STARTUP_SECONDS:-21600}}"   # chi=1.0 100-user resumes need >90 min for the first row
LAST_GROWTH=$(stat -c %Y "$CSV" 2>/dev/null || echo "$WATCH_START")
SEEN_GROWTH=0
log "watching $CSV for job $JOBID (stall=${STALL}s poll=${POLL}s, clamped to watch start)"
while :; do
  sleep "$POLL"
  kill -0 "$LAUNCHER" 2>/dev/null || { log "launcher $LAUNCHER gone -- run ended on its own, exiting"; exit 0; }

  MTIME=$(stat -c %Y "$CSV" 2>/dev/null) || { log "WARNING: $CSV unreadable this cycle, skipping"; continue; }
  NOW=$(date +%s)
  if [ "$MTIME" -gt "$LAST_GROWTH" ]; then
    LAST_GROWTH=$MTIME
    if [ "$SEEN_GROWTH" -eq 0 ]; then
      SEEN_GROWTH=1
      log "first telemetry growth seen -- switching from startup grace (${STARTUP}s) to stall threshold (${STALL}s)"
    fi
    continue
  fi
  if [ "$SEEN_GROWTH" -eq 1 ]; then
    AGE=$(( NOW - LAST_GROWTH )); LIMIT=$STALL; WHAT="has not grown"
  else
    AGE=$(( NOW - WATCH_START )); LIMIT=$STARTUP; WHAT="never produced a row since watch start"
  fi
  if [ "$AGE" -ge "$LIMIT" ]; then
    log "STALL DETECTED: $CSV $WHAT in ${AGE}s (threshold ${LIMIT}s)."
    log "Job $JOBID is almost certainly a zombie (torchrl cleanup hang -- see header comment above). Scancelling to free the allocation."
    scancel "$JOBID"
    exit 0
  fi
done
