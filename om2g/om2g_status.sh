#!/bin/bash
# OM2G status: one SSH round-trip, everything that matters.
# ⚠️ OM2G TRACK. See code/om2g/OM2G_README.md.
timeout 150 ssh -o BatchMode=yes -o ConnectTimeout=25 eguzys@login.ee.ethz.ch 'bash -s' <<'REMOTE'
export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
RB="${OM2G_SCRATCH:-/usr/itetnas04/data-scratch-01/eguzys/data}/resilient-backbone"
echo "### $(date '+%H:%M:%S')  QUEUE"
squeue -u eguzys -h -O "JobID:12,Name:26,State:10,TimeUsed:9,NodeList:14" || echo "  squeue failed"
echo "### PROGRESS  (OM2Gg_* = GATED, the real runs, 56 iters/336k; OM2G_* = ungated control, 84 iters/504k)"
cd $RB/om2g_logs 2>/dev/null || exit 0
for f in OM2Gg_compressed_s*.log OM2Gg_uncompressed_s*.log OM2G_compressed_s*.log OM2G_uncompressed_s*.log; do
  [ -f "$f" ] || continue
  it=$(grep -oE '[0-9]+/(56|84)' "$f" | tail -1)
  eta=$(grep -oE '\[[0-9:]+<[0-9:?]+' "$f" | tail -1 | tr -d '[')
  st="ok"
  grep -q "TRAIN FAILED" "$f" && st="**FAILED**"
  grep -qE "Traceback|CUDA out of memory" "$f" && st="**ERROR**"
  printf "  %-34s %-10s %-22s %s\n" "$f" "${it:-—}" "${eta:-—}" "$st"
done
echo "### TASK GATE"
for f in OM2G_task_gate_*.log; do
  [ -f "$f" ] && { printf "  %s: " "$f"; tail -n 3 "$f" | tr '\n' ' | '; echo; }
done
echo "### TELEMETRY (episodes logged / mean control share / peer availability)"
for c in $RB/om2g_out/telemetry_*.csv; do
  [ -f "$c" ] || continue
  PY_BIN=/usr/itetnas04/data-scratch-01/eguzys/data/padre_env/bin/python
  [ -x "$PY_BIN" ] || PY_BIN=$(command -v python3 || command -v python)
  "$PY_BIN" - "$c" <<'PY'
import csv, sys, os
# 10 collector workers append to ONE csv, so stray header rows are
# interleaved. Skip any row whose fields are not numeric.
def _num(r, k):
    try: return float(r[k])
    except (TypeError, ValueError, KeyError): return None
rows = [r for r in csv.DictReader(open(sys.argv[1]))
        if _num(r, "ctrl_share_of_channel") is not None]
if not rows: sys.exit()
def m(k):
    v=[x for x in (_num(r,k) for r in rows) if x is not None]
    return sum(v)/len(v) if v else float("nan")
print(f"  {os.path.basename(sys.argv[1]):<46} n={len(rows):<5} "
      f"ctrl={m('ctrl_share_of_channel')*100:6.2f}%  "
      f"peer_avail={m('peer_available_frac')*100:5.1f}%  "
      f"fresh={m('peer_fresh_frac')*100:5.1f}%  "
      f"busy={m('channel_busy')*100:5.1f}%")
PY
done
echo "### CHECKPOINTS (latest per run)"
find $RB/src/out/OM2G -name "checkpoint_*.pt" 2>/dev/null | sed 's|.*/OM2G/||;s|/normalized[^/]*/checkpoints/|  |' | sort -t" " -k1,1 -k3 | awk '{a[$1]=$0} END{for(k in a) print a[k]}' | sort
REMOTE
