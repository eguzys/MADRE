#!/bin/bash
# tail_eval_all.sh <cell-name-regex> [--go]
# Queues converged-metric (_TAIL) evals for every TRAINED cell matching the regex
# that does not already have them. Without --go it only prints what it would do.
#
# SPAWN SAFETY. Spawn is NOT uniform across families: v1600* trains at cluster,
# br12*/br16*/v1200*/w512*/u1000* train at explore. This reads each cell's OWN
# OM2G_SPAWN out of its cell script, forces exactly that, and names the output
# file after it. Never pass a blanket spawn here: train and eval spawn must
# match the cell.
set -uo pipefail
PAT="${1:?cell regex}"; GO="${2:-}"
RB="${OM2G_SCRATCH:-/usr/itetnas04/data-scratch-01/eguzys/data}/resilient-backbone"
export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
export PATH=/usr/sepp/bin:$PATH
cd "$RB"
n=0; sk=0
for s in arch_out/cells/*.sh; do
  c=$(basename "$s" .sh)
  echo "$c" | grep -qE "$PAT" || continue
  ls src/out/OM2G/OM2G_*_"$c"/*/checkpoints/checkpoint_300000.pt >/dev/null 2>&1 || { sk=$((sk+1)); continue; }
  spawn=$(grep -hoE 'OM2G_SPAWN=[a-z]+' "$s" | tail -1 | cut -d= -f2)
  if [ -z "$spawn" ]; then echo "NO SPAWN declared in $c, skipped"; sk=$((sk+1)); continue; fi
  suf="_SPAWN${spawn}_TAIL"
  have=$(ls om2g_out/ 2>/dev/null | grep -cE "_${c}${suf}_n5_seed")
  if [ "$have" -ge 20 ]; then sk=$((sk+1)); continue; fi
  if [ "$GO" != "--go" ]; then
    echo "WOULD RUN  $c  spawn=$spawn  suffix=$suf  have=$have"
  else
    OM2G_OUT_SUFFIX="$suf" OM2G_SPAWN_FORCE="$spawn" bash om2g/eval_fanout.sh "$c" 100 5 >/dev/null 2>&1
    echo "queued $c spawn=$spawn"
  fi
  n=$((n+1))
done
echo "cells=$n skipped=$sk"
