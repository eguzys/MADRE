#!/bin/bash
# verify_cell.sh <cell_tag> [max_wait_s]
#
# Block until a freshly-launched cell PRINTS its configuration, then assert the
# banner matches what the cell script asked for. Run this IMMEDIATELY after
# every launch -- never at the next status check.
#
# The running job is checked, not only the cell script: om2g_config.py --shell
# is eval'd after the cell's exports and can overwrite any knob, and the job's
# own banner shows the values actually in effect.
RB="${OM2G_SCRATCH:-/usr/itetnas04/data-scratch-01/eguzys/data}/resilient-backbone"
TAG="${1:?cell tag required}"; MAX="${2:-600}"
LOG="$RB/om2g_logs/cells/${TAG}.out"
CELL="$RB/arch_out/cells/${TAG}.sh"
[ -f "$CELL" ] || { echo "  no cell script $CELL"; exit 2; }

t=0
while [ $t -lt "$MAX" ]; do
  if grep -qa "OMNeTEnv\]" "$LOG" 2>/dev/null; then break; fi
  sleep 15; t=$((t+15))
done
grep -qa "OMNeTEnv\]" "$LOG" 2>/dev/null || { echo "  $TAG: NO BANNER after ${MAX}s -- job may not have started"; exit 3; }

BANNER=$(grep -a "OMNeTEnv\]" "$LOG" | tail -1)
FAIL=0
# every knob the cell asked for must be visible in the run's own output
while read -r V; do
  K="${V%%=*}"; WANT="${V#*=}"
  case "$K" in
    OMNET_PEER_LINK_MASK)
      EXP=$([ "$WANT" = "1" ] && echo on || echo off)
      echo "$BANNER" | grep -q "PEER_LINK_MASK=$EXP" \
        && echo "  ok   $K=$WANT  (banner: PEER_LINK_MASK=$EXP)" \
        || { echo "  FAIL $K=$WANT but banner says: $(echo "$BANNER" | grep -o 'PEER_LINK_MASK=[a-z]*')"; FAIL=1; }
      ;;
    OM2G_SHAPE_FIEDLER)
      if [ "$WANT" = "0" ]; then
        echo "$BANNER" | grep -q "shaping=off" \
          && echo "  ok   shaping off" \
          || { echo "  FAIL shaping should be off; banner: $(echo "$BANNER" | grep -o 'shaping=[a-z]*')"; FAIL=1; }
      fi
      ;;
    OM2G_LATENT_DIM)
      grep -qa "latent=$WANT" "$LOG" \
        && echo "  ok   latent=$WANT" \
        || { echo "  FAIL latent should be $WANT; log: $(grep -ao 'latent=[0-9]*' "$LOG" | tail -1)"; FAIL=1; }
      ;;
  esac
done < <(grep -aoE "^export (OMNET_PEER_LINK_MASK|OM2G_SHAPE_FIEDLER|OM2G_LATENT_DIM)=[^ ]*" "$CELL" | sed 's/^export //')
[ $FAIL -eq 0 ] && echo "  $TAG: VERIFIED" || { echo "  $TAG: MISMATCH -- kill it, do not let it train"; exit 1; }
