#!/bin/bash
#SBATCH --job-name=ent_pick
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/usr/itetnas04/data-scratch-01/eguzys/data/ent_pick_%j.log
set -uo pipefail
R="${OM2G_SCRATCH:-/usr/itetnas04/data-scratch-01/eguzys/data}/resilient-backbone"
cd $R/src/out/benchmarl_omnet
BEST=""; BEST_SCORE=-1e9
for E in 0.03 0.05 0.1 0.3; do
  d=$(ls -d omnet_st_n5_ent$E/*/ 2>/dev/null | head -1)
  [ -z "$d" ] && { echo "ent$E: no run dir -- skip"; continue; }
  S=$d$(basename ${d%/})/scalars
  ENT=$(tail -1 $S/train_drone_entropy.csv 2>/dev/null | awk -F, '{print $NF}')
  CLIP=$(tail -1 $S/train_drone_clip_fraction.csv 2>/dev/null | awk -F, '{print $NF}')
  RW=$(tail -1 $S/collection_reward_episode_reward_mean.csv 2>/dev/null | awk -F, '{print $NF}')
  [ -z "$ENT" ] || [ -z "$CLIP" ] && { echo "ent$E: no scalars -- skip"; continue; }
  # gate: entropy must not have collapsed (> -1.5) and clip must be sane (< 0.25)
  OK=$(awk -v e="$ENT" -v c="$CLIP" 'BEGIN{print (e>-1.5 && c<0.25)?1:0}')
  echo "ent$E: entropy=$ENT clip=$CLIP reward=$RW gate_pass=$OK"
  if [ "$OK" = "1" ]; then
    S2=$(awk -v r="$RW" 'BEGIN{print r+0}')
    BETTER=$(awk -v a="$S2" -v b="$BEST_SCORE" 'BEGIN{print (a>b)?1:0}')
    [ "$BETTER" = "1" ] && { BEST=$E; BEST_SCORE=$S2; }
  fi
done
if [ -z "$BEST" ]; then
  echo "=== NO ARM PASSED THE GATE -- entropy coefficient is not the lever. ==="
  echo "=== Next move is the action-std floor (scale_mapping). Not auto-submitting. ==="
  exit 0
fi
echo "=== WINNER: ent$BEST (reward=$BEST_SCORE) -- submitting 3M run ==="
cd $R
# ST N5 -- the headline run
sbatch --gres=gpu:1 --job-name=st5_ent$BEST \
  --export=ALL,OMNET_ALGO=padre_mappo_norm_ent$BEST,OMNET_MAX_FRAMES=3000000,OMNET_SAVE_SUFFIX=ent${BEST}_full \
  omnet_module/train_omnet_arton.sh st 5

# The other two cells that collapsed under entropy_coef 0.01: DeepSets N5, MLP N6.
# MLP N4 also collapsed at 0.01 (segment 20260722_01092 ends at entropy -3.11);
# the earlier 'stayed healthy' read came from segment 20260721_13191 alone, which is 25
# points and never crosses zero. All four cells are relaunched at the winning coefficient.
sbatch --gres=gpu:1 --job-name=ds5_ent$BEST \
  --export=ALL,OMNET_ALGO=padre_mappo_norm_ent$BEST,OMNET_MAX_FRAMES=3000000,OMNET_SAVE_SUFFIX=ent${BEST}_full \
  omnet_module/train_omnet_arton.sh deepsets 5

sbatch --gres=gpu:1 --job-name=mlp6_ent$BEST \
  --export=ALL,OMNET_ALGO=padre_mappo_norm_ent$BEST,OMNET_MAX_FRAMES=3000000,OMNET_SAVE_SUFFIX=ent${BEST}_full \
  omnet_module/train_omnet_arton.sh mlp 6

# MLP N5 -- other half of the matched N=5 pair; collapsed at 0.01 (eval 138k vs greedy 156k)
sbatch --gres=gpu:1 --job-name=mlp5_ent$BEST \
  --export=ALL,OMNET_ALGO=padre_mappo_norm_ent$BEST,OMNET_MAX_FRAMES=3000000,OMNET_SAVE_SUFFIX=ent${BEST}_full \
  omnet_module/train_omnet_arton.sh mlp 5

# MLP N4 -- collapsed at 0.01 (segment 20260722_01092, entropy -3.11)
sbatch --gres=gpu:1 --job-name=mlp4_ent$BEST \
  --export=ALL,OMNET_ALGO=padre_mappo_norm_ent$BEST,OMNET_MAX_FRAMES=3000000,OMNET_SAVE_SUFFIX=ent${BEST}_full \
  omnet_module/train_omnet_arton.sh mlp 4
