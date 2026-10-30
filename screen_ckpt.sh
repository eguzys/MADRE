#!/bin/bash
# n=30 SCREEN of one checkpoint as 3 parallel chunks (~20 min). Screen only --
# n<50 is too noisy for a conclusion; confirm winners at n=100.
RB="${OM2G_SCRATCH:-/usr/itetnas04/data-scratch-01/eguzys/data}/resilient-backbone"
export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
TAG="$1"; N="$2"
C=$(find $RB/src/out/OM2G/OM2G_compressed_st_s*_${TAG} -name "checkpoint_${N}.pt" 2>/dev/null | sort | tail -1)
[ -z "$C" ] && { echo "  no checkpoint_${N}.pt for ${TAG}"; exit 1; }
REL=${C#$RB/src/}

# ── PER-CELL REQUIRED ENV ────────────────────────────────────────────────────
# A checkpoint must be scored under the same observation/message contract it
# was TRAINED under. Some mismatches fail loudly (a token-width change trips
# the exact-tail assert); others would not: scoring a p2demzero checkpoint
# without OM2G_MSG_ABLATE reports the UN-ABLATED condition, and a leftover
# exported flag would change c15a. The flags are therefore derived from the
# cell script that trained it, never assumed.
CELL_SH="$RB/arch_out/cells/${TAG}.sh"
if [ -f "$CELL_SH" ]; then
  for V in OM2G_USER_PARTNER_TAG OM2G_MSG_ABLATE OM2G_LATENT_DIM OM2G_PAIR_FLOOR OMNET_PEER_LINK_MASK OM2G_FORWARD_HOPS; do
    WANT=$(grep -aoE "^export ${V}=[^ ]*" "$CELL_SH" | tail -1 | sed "s/^export //")
    HAVE_V=$(eval echo "\$$V")
    if [ -n "$WANT" ]; then
      if [ "$WANT" != "${V}=${HAVE_V}" ]; then
        echo "REFUSING TO SCORE: ${TAG} was trained with '$WANT' but this eval has ${V}='${HAVE_V}'."
        echo "  Re-run with --export=ALL,...,$WANT"
        exit 3
      fi
    elif [ -n "$HAVE_V" ]; then
      echo "REFUSING TO SCORE: ${TAG} was trained WITHOUT $V but this eval sets ${V}='${HAVE_V}' (contamination)."
      exit 3
    fi
  done
  echo "  [cell-env] ${TAG}:flags verified against $CELL_SH"
fi

W="OM2G_DRONES=8,OMNET_NUM_DRONES=8,OM2G_USER_SPAWN=archipelago,OM2G_CLUSTERS=4,OM2G_ARENA=3800,OM2G_SPREAD=47.5,OM2G_PCROSS=0.9,OM2G_MINSEP=1465,OM2G_MAXNN=1750,OM2G_MARGIN=300,OM2G_SPAWN=cluster"
S="OM2G_SHAPE_FIEDLER=0,OM2G_SHAPE_VORONOI=0,OM2G_SHAPE_EDGE=0,OM2G_SHAPE_SPREAD=0"
K=$((N/1000))
for k in 0 10 20; do
  sbatch --export=ALL,$W,$S $RB/om2g/om2g_eval_arton.sh compressed "$REL" scr_${TAG}${K}k_${k} 10 $k 2>&1 | tail -1
done
