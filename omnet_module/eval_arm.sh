#!/bin/bash
# eval_arm.sh <tag> - fetch a finished arton arm and score it locally, n=20.
#
# OMNeT TRACK ONLY: uses omnet_module/evaluate_omnet.py. Never
# src/evaluation/evaluate.py, which builds PADREParallelEnv at ep_length=1000
# and would score a different world.
#
# Reports the paired gap against the DETERMINISTIC baselines
# (greedy 835.34, hover 416.21 mean/step, seeds 0-19) plus a per-seed PAIRED
# comparison against the +0.766 ST reference - unpaired means at n=20 carry
# ~±0.15 SE, so only the paired difference is readable.
set -uo pipefail
cd /home/emilis/Documents/thesis/code

TAG="${1:?usage: eval_arm.sh <tag>   e.g. ent003_st}"
PY=/home/emilis/pythonEnviroment/padre/bin/python
RREPO="${OM2G_SCRATCH:-/usr/itetnas04/data-scratch-01/eguzys/data}/resilient-backbone"
SSH="ssh -o ConnectTimeout=30 -o BatchMode=yes eguzys@login.ee.ethz.ch"
OUTDIR="out/eval_$TAG"

# NEWEST run dir, not the first alphabetically. A tag can match several dirs,
# including interrupted runs that hold only an early checkpoint.
rundir=$($SSH "ls -dt $RREPO/src/out/benchmarl_omnet/*${TAG}*/*/ 2>/dev/null | head -1" | tr -d '\r')
[ -z "$rundir" ] && { echo "FATAL: no run dir for $TAG"; exit 1; }

# HIGHEST frame count, parsed numerically from the basename. `sort -t_ -k2 -n`
# on the full path splits on every underscore in the run name, so field 2 is
# not the frame count.
frames=$($SSH "ls $rundir/checkpoints/ 2>/dev/null | grep -oE '^checkpoint_[0-9]+\.pt$' | sed -E 's/checkpoint_([0-9]+)\.pt/\1/' | sort -n | tail -1" | tr -d '\r')
[ -z "$frames" ] && { echo "FATAL: no checkpoint for $TAG in $rundir"; exit 1; }
ckpt="$rundir/checkpoints/checkpoint_${frames}.pt"

# Guard: a full arm is 1002000 frames. Anything tiny is a dead-run stub and
# scoring it would silently produce a meaningless number.
if [ "$frames" -lt 500000 ]; then
  echo "FATAL: $TAG best checkpoint is only ${frames} frames ($rundir)"
  echo "       that is a dead-run stub, not a finished arm. Refusing to score it."
  exit 1
fi
echo "[$TAG] run=$(basename "${rundir%/}")  frames=$frames"

runname=$(basename "${rundir%/}")
dest="out/arton_ckpt/$runname"
mkdir -p "$dest/checkpoints" "$OUTDIR/parts"
echo "[$TAG] fetching $(basename "$ckpt")"
scp -q -o BatchMode=yes "eguzys@login.ee.ethz.ch:$rundir/config.pkl" "$dest/config.pkl" || exit 1
scp -q -o BatchMode=yes "eguzys@login.ee.ethz.ch:$ckpt" "$dest/checkpoints/" || exit 1
LCKPT="$(readlink -f "$dest/checkpoints/$(basename "$ckpt")")"

unset OMNET_APPTAINER_SIF OMNET_USER_TX_mW OMNET_DRONE_TX_mW OMNET_STEP_M
export OMNET_MAX_STEPS=250 OMNET_ENCODER_MODE=learned OMNET_PEER_LINK_MASK=1
export CUDA_VISIBLE_DEVICES=""
export OUTDIR LCKPT PY

jf="$OUTDIR/parts/.jobs"; : > "$jf"
# No trailing field: xargs -L 1 joins a line ending in whitespace to the next.
for s in 0 2 4 6 8 10 12 14 16 18; do echo "$s 2" >> "$jf"; done
run_one() {
  o="$OUTDIR/parts/t_$1.csv"; [ -s "$o" ] && return 0
  "$PY" omnet_module/evaluate_omnet.py "$LCKPT" --policy trained --num-drones 5 \
    --seed-start "$1" -n "$2" --out "$o" > "$OUTDIR/parts/t_$1.log" 2>&1
}
export -f run_one
echo "[$TAG] evaluating 20 seeds"
xargs -a "$jf" -P 8 -L 1 bash -c 'run_one "$@"' _

csv="$OUTDIR/trained_${TAG}_n5.csv"
echo "seed,episode_reward,steps" > "$csv"
cat "$OUTDIR"/parts/t_*.csv 2>/dev/null | grep -v '^seed,' | sort -t, -k1,1n >> "$csv"
n=$(( $(wc -l < "$csv") - 1 ))
[ "$n" -lt 1 ] && { echo "FATAL: eval produced nothing, see $OUTDIR/parts/*.log"; exit 1; }

"$PY" - "$csv" "$TAG" <<'PYEOF'
import csv, sys, math, statistics as st, os
def L(p): return {int(r["seed"]): float(r["episode_reward"])/250.0 for r in csv.DictReader(open(p))}
t = L(sys.argv[1]); tag = sys.argv[2]
b = "out/omnet_v3_baselines"
g, h = L(f"{b}/greedy_n5.csv"), L(f"{b}/hover_n5.csv")
S = sorted(set(t) & set(g) & set(h))
gap = sum(t[s]-h[s] for s in S)/sum(g[s]-h[s] for s in S)
r = [(t[s]-h[s])/(g[s]-h[s]) for s in S if abs(g[s]-h[s]) > 1e-9]
se = st.stdev(r)/math.sqrt(len(r))
print(f"\n=== {tag}  n={len(S)} ===")
print(f"  mean/step {st.mean([t[s] for s in S]):8.1f}   median {st.median([t[s] for s in S]):8.1f}")
print(f"  paired gap (ratio-of-sums) {gap:+.3f}   mean-of-ratios {st.mean(r):+.3f} +/- {se:.3f} SE")
print(f"  beats hover {sum(t[s]>h[s] for s in S)}/{len(S)}     ST reference: 737.1 / +0.766")
ref = "out/sigma_eval/trained_det_n5.csv"
if os.path.exists(ref):
    d0 = L(ref); C = sorted(set(t) & set(d0))
    d = [t[s]-d0[s] for s in C]
    m, dse = st.mean(d), st.stdev(d)/math.sqrt(len(d))
    verdict = "SIGNIFICANT" if abs(m) > 1.96*dse else "not significant"
    print(f"  PAIRED vs ST baseline: {m:+.1f} +/- {dse:.1f} SE per seed -> {verdict}")
    print(f"  better on {sum(x>0 for x in d)}/{len(d)} seeds")
PYEOF
