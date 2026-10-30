#!/bin/bash
#SBATCH --job-name=per_feat_all
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --output=/cluster/scratch/eguzys/pisa_data/per_feature_all_%j.log

# Runs the per-feature reconstruction stat over EVERY pretrained encoder in
# scaling_out/. No retraining. Each encoder: ~1 min CPU. Six encoders -> ~6 min.
# Works as a plain script (bash this file) OR as an sbatch job.

set -uo pipefail   # NOTE: no -e; we want the loop to continue if one encoder fails

module load stack/2024-06 python/3.11.6 2>/dev/null || true
source /cluster/scratch/eguzys/pisa_env/bin/activate

export SWEEP_SCRIPT_DIR=/cluster/scratch/eguzys
export NPZ_X_KEY=tokens
export NPZ_M_KEY=masks

NPZ=/cluster/scratch/eguzys/pisa_data/user_sets.npz
ENC_DIR=/cluster/scratch/eguzys/pisa_data/scaling_out
SCRIPT=/cluster/scratch/eguzys/per_feature_recon.py
OUTDIR=/cluster/scratch/eguzys/pisa_data/per_feature_all
mkdir -p "$OUTDIR"

# each entry: "file hidden output"
for spec in \
  "deepsets_h32_o8.pt  32  8" \
  "deepsets_h32_o16.pt 32 16" \
  "deepsets_h64_o8.pt  64  8" \
  "deepsets_h64_o16.pt 64 16" \
  "deepsets_h128_o8.pt 128 8" \
  "deepsets_h128_o16.pt 128 16" \
; do
  set -- $spec
  FILE=$1
  H=$2
  O=$3
  ENC="$ENC_DIR/$FILE"

  echo ""
  echo "########################################################################"
  echo "### ENCODER: $FILE   (hidden=$H, output=$O)"
  echo "########################################################################"

  if [ ! -f "$ENC" ]; then
    echo "  MISSING: $ENC  -- skipping"
    continue
  fi

  python "$SCRIPT" \
    --npz     "$NPZ" \
    --encoder "$ENC" \
    --hidden  "$H" \
    --output  "$O" \
    --out     "$OUTDIR/per_feature_${FILE%.pt}.json" \
  || echo "  FAILED on $FILE (continuing)"
done

echo ""
echo "ALL DONE $(date)"
echo "per-encoder JSONs in: $OUTDIR"
