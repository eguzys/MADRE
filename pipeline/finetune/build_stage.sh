#!/bin/bash
# Build the 100-user fine-tune stage for ONE (rung, seed) and its six per-chi user-beacon (UB) stages.
# Usage: build_stage.sh <basecell> <seed> <pfx> <obs_new>     e.g. build_stage.sh br12rtL64s4 4 L64 15972
# Recipe: pick_best_ckpt on the base's single normalizedmappo dir,
# >2 GB source check, widen2.py to cap 100 with the rung's obs width, dropbuf.py,
# then <pfx>_u100sp<N>UB_s<seed> = cp -al of <pfx>_u100sp4_s<seed> for N in 05 1 2 4 6 10 (one stage per rung/chi/seed).
# Never overwrites an existing stage. Prints "STAGE_OK <pfx> s<seed> best=<k>" at the end.
set -u
B=$1; SD=$2; P=$3; OBS=$4
D=/usr/itetnas04/data-scratch-01/eguzys/data; RB=$D/resilient-backbone; PY=$D/padre_env/bin/python
cd $RB
base=$(ls -1d src/out/OM2G/OM2G_compressed_st_s${SD}_${B}/normalizedmappo_* 2>/dev/null)
nb=$(echo "$base" | grep -c .)
[ "$nb" = 1 ] || { echo "FATAL $B: expected exactly one base experiment dir, got $nb"; exit 1; }
grep -aq "=== TRAIN DONE" om2g_logs/cells/$B.out || { echo "FATAL $B: no TRAIN DONE"; exit 1; }
pick=$(timeout 600 $PY om2g/pick_best_ckpt.py $base 2>/dev/null | tail -1)
[ -s "$pick" ] || { echo "FATAL $B: pick_best gave '$pick'"; exit 1; }
case "$pick" in $base/checkpoints/checkpoint_*.pt) ;; *) echo "FATAL $B: $pick outside $base"; exit 1;; esac
best=$(basename $pick .pt | sed 's/checkpoint_//')
sz=$(stat -c %s $pick); [ $sz -gt 2000000000 ] || { echo "FATAL $B: $pick is $sz bytes (stripped?)"; exit 1; }
dst=$D/u100diag_ckpts/${P}_u100sp4_s${SD}; ck=$dst/checkpoints/checkpoint_${best}.pt
if [ -s $ck ] && [ -f $dst/WIDEN_OK ]; then echo "SKIP widen, $dst exists"
else
  [ -e $dst ] && { echo "FATAL $dst exists without WIDEN_OK marker -- inspect by hand"; exit 1; }
  echo "[$(date +%H:%M:%S)] widen $B $pick -> $dst (obs $OBS)"
  $PY $HOME/u100diag/widen2.py $pick 100 $OBS $dst || { echo "FATAL $B widen2 failed"; rm -rf $dst; exit 1; }
  $PY $HOME/u100diag/dropbuf.py $ck || { echo "FATAL $B dropbuf failed"; rm -rf $dst; exit 1; }
  touch $dst/WIDEN_OK
fi
for sp in sp05 sp1 sp2 sp4 sp6 sp10; do
  ub=$D/u100diag_ckpts/${P}_u100${sp}UB_s${SD}
  if [ -e $ub ]; then
    [ -s $ub/checkpoints/checkpoint_${best}.pt ] || { echo "FATAL $ub exists but lacks checkpoint_${best}.pt"; exit 1; }
    echo "SKIP $ub exists"; continue
  fi
  cp -al $dst $ub && rm -f $ub/WIDEN_OK || { echo "FATAL cp -al $ub"; exit 1; }
done
echo "STAGE_OK $P s$SD best=$best $(date)"
