#!/bin/bash
# gate_resize.sh - the prerequisite for the N-transfer matrix.
#
# WHY: ST and deepsets are claimed to transfer from a single N=5 model because
# they are permutation-invariant over the peer set. That claim is only testable
# if the peer axis GENUINELY RESIZES at eval time. If the observation silently
# zero-pads (or truncates) to 5 peer tokens, the off-diagonal cells of the
# matrix would not measure transfer at all.
#
# This runs the ST N=5 checkpoint at N=4,5,6 on 3 seeds each. Two signals:
#   1. it must not crash  (shape mismatch => no resize machinery at all)
#   2. the three cells must produce DIFFERENT rewards. Identical numbers at
#      N=4 and N=6 would mean the extra/missing drones never reached the policy.
# Cheap: ~3 min total. Never commit 9 n=100 cells before this passes.
set -uo pipefail
cd /home/emilis/Documents/thesis/code

PY=/home/emilis/pythonEnviroment/padre/bin/python
CKPT=$(readlink -f out/arton_ckpt/normalizedmappo_static_stmodel__00fa5f10_26_08_09-14_03_08/checkpoints/checkpoint_1002000.pt)
OUT=out/gate_resize
mkdir -p "$OUT"

unset OMNET_APPTAINER_SIF OMNET_USER_TX_mW OMNET_DRONE_TX_mW OMNET_STEP_M
export OMNET_MAX_STEPS=250 OMNET_ENCODER_MODE=learned OMNET_PEER_LINK_MASK=1
# The laptop GPU OOMs and reports as an obs-shape mismatch, which is the
# failure this gate is trying to detect. Force CPU.
export CUDA_VISIBLE_DEVICES=""

for N in 4 5 6; do
  echo "== N=$N  $(date '+%T')"
  "$PY" omnet_module/evaluate_omnet.py "$CKPT" --policy trained \
    --num-drones "$N" --train-num-drones 5 \
    --seed-start 0 -n 3 --out "$OUT/st_at_n${N}.csv" \
    > "$OUT/st_at_n${N}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "   !! FAILED rc=$rc - last lines:"
    tail -15 "$OUT/st_at_n${N}.log" | sed 's/^/      /'
    continue
  fi
  echo "   rewards: $(tail -n +2 "$OUT/st_at_n${N}.csv" | cut -d, -f2 | tr '\n' ' ')"
done

echo
echo "=== VERDICT ==="
for N in 4 5 6; do
  [ -s "$OUT/st_at_n${N}.csv" ] && \
    echo "N=$N: $(tail -n +2 "$OUT/st_at_n${N}.csv" | cut -d, -f2 | tr '\n' ' ')"
done
echo "Cells must be non-empty AND mutually different. Identical => silent padding."
