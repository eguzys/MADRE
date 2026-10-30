#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  om2g_eval_all.sh — submit an eval job for every finished OM2G run
# ═══════════════════════════════════════════════════════════════════════════
#
#  ⚠️  OM2G TRACK.  Run ON arton.
#
#  Usage:  om2g/om2g_eval_all.sh <family> [episodes]
#            family = gated    -> the REAL runs (OM2GEnv, delivery gating on)
#            family = ungated  -> the control  (plain OMNeTEnv, no gating)
#
#  ⚠️ BOTH families write into the SAME out/OM2G/OM2G_<arm>_st_s<seed>/ folder
#     (om2g_launch_arton.sh derives RUN_NAME from arm+seed only). They are
#     told apart ONLY by BenchMARL's timestamped subfolder:
#         *_26_08_25-01_27_*  -> ungated control (one launch burst)
#         anything else        -> gated, the real runs (incl. relaunches)
#     The pattern is explicit here rather than "newest wins", so each
#     family scores its own runs.
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail

FAMILY="${1:?family required: gated|ungated}"
EPISODES="${2:-40}"

# The ungated control was launched in a single burst at 01:27; every gated run
# is anything else (including relaunches after a node failure). Ungated is
# defined exactly and gated as the complement, so a relaunch at any hour is
# still picked up.
UNGATED_PAT='_26_08_25-01_27_'
case "$FAMILY" in
  gated|ungated) ;;
  *) echo "FATAL: family must be gated|ungated, got '$FAMILY'"; exit 1 ;;
esac

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
RB="${OM2G_SCRATCH:-/usr/itetnas04/data-scratch-01/eguzys/data}/resilient-backbone"
cd "$RB"

n=0
for arm in compressed uncompressed; do
  for seed in 0 1 2; do
    DIR="src/out/OM2G/OM2G_${arm}_st_s${seed}"
    [ -d "$DIR" ] || { echo "skip: no $DIR"; continue; }
    if [ "$FAMILY" = ungated ]; then
      RUN=$(ls -d $DIR/normalizedmappo* 2>/dev/null | grep -- "$UNGATED_PAT" | tail -1)
    else
      RUN=$(ls -d $DIR/normalizedmappo* 2>/dev/null | grep -v -- "$UNGATED_PAT" | tail -1)
    fi
    [ -n "$RUN" ] || { echo "skip: no $FAMILY run under $DIR"; continue; }
    CK=$(ls "$RUN"/checkpoints/checkpoint_*.pt 2>/dev/null \
         | sed 's/.*checkpoint_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)
    [ -n "$CK" ] || { echo "skip: no checkpoint in $RUN"; continue; }
    FRAMES=$(basename "$CK" | sed 's/checkpoint_\([0-9]*\)\.pt/\1/')
    REL=${CK#src/}
    TAG="${FAMILY}_${arm}_s${seed}_${FRAMES}f"
    echo "submit $TAG  <- $REL"
    sbatch --cpus-per-task=4 --mem=16G --time=6:00:00 --job-name=OM2Gev_${TAG} \
      --output=$RB/om2g_logs/OM2Gev_${TAG}_%j.log \
      om2g/om2g_eval_arton.sh "$arm" "$REL" "$TAG" "$EPISODES" 2>&1 | grep Submitted
    n=$((n+1))
  done
done
echo "submitted $n eval job(s) for family=$FAMILY at n=$EPISODES episodes"
