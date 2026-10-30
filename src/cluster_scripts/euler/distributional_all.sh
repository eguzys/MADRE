#!/bin/bash
set -uo pipefail
module load stack/2024-06 python/3.11.6 2>/dev/null || true
source /cluster/scratch/eguzys/pisa_env/bin/activate

NPZ=/cluster/scratch/eguzys/pisa_data/user_sets.npz
ENC_DIR=/cluster/scratch/eguzys/pisa_data/scaling_out
SCRIPT=/cluster/scratch/eguzys/distributional_recon.py
OUTDIR=/cluster/scratch/eguzys/pisa_data/distributional_all
mkdir -p "$OUTDIR"

for spec in \
  "deepsets_h32_o8.pt   32  8" \
  "deepsets_h32_o16.pt  32 16" \
  "deepsets_h64_o8.pt   64  8" \
  "deepsets_h64_o16.pt  64 16" \
  "deepsets_h128_o8.pt  128 8" \
  "deepsets_h128_o16.pt 128 16" \
; do
  set -- $spec
  FILE=$1; H=$2; O=$3
  ENC="$ENC_DIR/$FILE"
  echo ""
  echo "### $FILE  (hidden=$H, output=$O)"
  [ -f "$ENC" ] || { echo "  MISSING $ENC -- skip"; continue; }
  python "$SCRIPT" \
    --npz "$NPZ" --encoder "$ENC" --hidden "$H" --output "$O" \
    --out "$OUTDIR/dist_${FILE%.pt}.json" \
  || echo "  FAILED $FILE (continuing)"
done
echo ""
echo "ALL DONE $(date)"
