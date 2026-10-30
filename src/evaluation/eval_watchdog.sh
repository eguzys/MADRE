#!/bin/bash
# External progress watchdog for Euler eval jobs.
# READ-ONLY: stats log mtimes. Never signals or inspects the job process.
# Reports; does not kill. Distinguishes NO-PROGRESS from NO-EVIDENCE.
# Usage: eval_watchdog.sh <log_glob> [stall_minutes] [poll_seconds]
GLOB="${1:?usage: eval_watchdog.sh <log_glob> [stall_min] [poll_s]}"
STALL_MIN="${2:-10}"
POLL_S="${3:-600}"
while true; do
  now=$(date +%s); found=0; stalled=0; alive=0
  for f in $GLOB; do
    [ -e "$f" ] || continue
    found=$((found+1))
    age=$(( (now - $(stat -c %Y "$f")) / 60 ))
    if [ "$age" -ge "$STALL_MIN" ]; then
      stalled=$((stalled+1))
      echo "STALL $(date +%H:%M) ${age}m since write: $f"
    else
      alive=$((alive+1))
    fi
  done
  if [ "$found" -eq 0 ]; then
    echo "NO-EVIDENCE $(date +%H:%M) no logs match $GLOB -- not a pass"
  else
    echo "SCAN $(date +%H:%M) files=$found writing=$alive stalled=$stalled"
  fi
  sleep "$POLL_S"
done
