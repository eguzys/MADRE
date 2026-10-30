#!/bin/bash
# eval_baseline_arton.sbatch - score a BASELINE policy (no checkpoint) on arton.
#
#   sbatch --export=ALL,POLICY=aggregate,EN=3,OUTD=n100_N3 omnet_module/eval_baseline_arton.sbatch
#
# Exists because eval_arton.sbatch hardcodes --policy trained and resolves a
# checkpoint; aggregate/greedy/hover/random need neither. Kept separate rather
# than adding a branch to a script that is currently driving results.
#
#SBATCH --job-name=omnet_basel
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/usr/itetnas04/data-scratch-01/eguzys/data/omnet_baseline_%j.log
set -uo pipefail
S="${OM2G_SCRATCH:-/usr/itetnas04/data-scratch-01/eguzys/data}"
R=$S/resilient-backbone
POLICY="${POLICY:?set POLICY=aggregate|greedy|hover|random}"
EN="${EN:-5}"; OUTD="${OUTD:-n100}"; WIDTH="${WIDTH:-8}"

export OMNET_APPTAINER_SIF=$S/omnet.sif
export OMNET_HOST_CODE=$R
export OMNET_CONTAINER_CODE=/work
export OMNET_INET_DIR=/opt/inet4.6
export OMNETPP_DIR=/opt/omnetpp-6.4.0
export BATCHED_PIPE_TIMEOUT=100000
export TMPDIR=$S/tmp; mkdir -p "$TMPDIR"
source $S/miniconda3/etc/profile.d/conda.sh
conda activate $S/padre_env
cd "$R" || exit 1
export OMNET_MAX_STEPS=250 OMNET_ENCODER_MODE=learned OMNET_PEER_LINK_MASK=1
export CUDA_VISIBLE_DEVICES=""

OUT="$S/eval_out/$OUTD"; mkdir -p "$OUT/parts"
FINAL="$OUT/${POLICY}_n100.csv"
if [ -f "$FINAL" ] && [ "$(grep -vc '^seed,' "$FINAL")" -gt 0 ]; then
  echo "== already done ($(grep -vc '^seed,' "$FINAL") episodes)"; exit 0
fi
rm -f "$FINAL"
echo "## policy=$POLICY  eval N=$EN  -> $FINAL"

JF="$OUT/parts/.jobs_$POLICY"; : > "$JF"
for s in $(seq 0 5 95); do echo "$s 5" >> "$JF"; done
export POLICY OUT EN
run_one() {
  local o="$OUT/parts/${POLICY}_$1.csv"
  [ -s "$o" ] && return 0
  python omnet_module/evaluate_omnet.py --policy "$POLICY" \
    --num-drones "$EN" --train-num-drones "$EN" \
    --seed-start "$1" -n "$2" --out "$o" > "$OUT/parts/${POLICY}_$1.log" 2>&1
}
export -f run_one
xargs -a "$JF" -P "$WIDTH" -L 1 bash -c 'run_one "$@"' _
echo "seed,episode_reward,steps" > "$FINAL"
cat "$OUT"/parts/${POLICY}_*.csv 2>/dev/null | grep -v '^seed,' | sort -t, -k1,1n >> "$FINAL"
N_EP=$(( $(wc -l < "$FINAL") - 1 ))
echo "## -> $N_EP episodes  $(date '+%F %T')"
if [ "$N_EP" -eq 0 ]; then
  echo "!! ZERO episodes; first chunk log:"; sed -n '1,40p' "$OUT/parts/${POLICY}_0.log" 2>/dev/null; exit 1
fi
