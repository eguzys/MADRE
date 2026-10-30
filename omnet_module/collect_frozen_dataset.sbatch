#!/bin/bash
# collect_frozen_dataset.sbatch - collect ONE shard of the frozen-encoder
# reconstruction dataset. Submit one job per (policy, seed-block); shards
# concatenate because every row carries its own episode_id.
#
#   sbatch --export=ALL,POLICY=greedy,SEED=0,EPISODES=40 omnet_module/collect_frozen_dataset.sbatch
#
# Env knobs:
#   POLICY    greedy | random | hover   (default greedy)
#   SEED      base seed; episode layouts are SEED..SEED+EPISODES-1
#   EPISODES  episodes in this shard (default 40)
#   N         drones (default 5)
#   OUTD      subdir under $SCRATCH/frozen_enc (default = POLICY)
#
#SBATCH --job-name=frozen_collect
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/usr/itetnas04/data-scratch-01/eguzys/data/frozen_enc/logs/collect_%j.log
set -uo pipefail

S="${OM2G_SCRATCH:-/usr/itetnas04/data-scratch-01/eguzys/data}"
R=$S/resilient-backbone
POLICY="${POLICY:-greedy}"; SEED="${SEED:-0}"; EPISODES="${EPISODES:-40}"
N="${N:-5}"; OUTD="${OUTD:-$POLICY}"

# ⚠️ In-container INET/omnetpp paths. They are required: the host-side INET
# check in omnet_env.py is skipped whenever OMNET_APPTAINER_SIF is set, so
# without them INET_DIR falls back to a host path and every OMNeT child stops
# at NED load.
export OMNET_APPTAINER_SIF=$S/omnet.sif
export OMNET_HOST_CODE=$R
export OMNET_CONTAINER_CODE=/work
export OMNET_INET_DIR=/opt/inet4.6
export OMNETPP_DIR=/opt/omnetpp-6.4.0
export BATCHED_PIPE_TIMEOUT=100000
export TMPDIR=$S/tmp
mkdir -p "$TMPDIR" "$S/frozen_enc/$OUTD" "$S/frozen_enc/logs"

source $S/miniconda3/etc/profile.d/conda.sh
conda activate $S/padre_env
cd "$R" || exit 1

# The collector calls config_knobs.apply_config itself and sets its own
# geometry; do NOT export OMNET_EXTRA_ARGS / OMNET_STEP_M here or an ambient
# value would race the binding. OMNET_MAX_STEPS must agree with ep_steps=250.
unset OMNET_EXTRA_ARGS OMNET_STEP_M OMNET_USER_TX_mW OMNET_DRONE_TX_mW
export OMNET_MAX_STEPS=250 OMNET_PEER_LINK_MASK=1
export CUDA_VISIBLE_DEVICES=""

OUT="$S/frozen_enc/$OUTD/${POLICY}_seed${SEED}_n${EPISODES}.npz"
if [ -s "$OUT" ]; then echo "== already done: $OUT"; exit 0; fi

echo "## policy=$POLICY seed=$SEED episodes=$EPISODES N=$N"
echo "## -> $OUT"
python omnet_module/collect_omnet_user_sets.py \
  --policy "$POLICY" --seed "$SEED" --episodes "$EPISODES" \
  --num-drones "$N" --out "$OUT"
RC=$?
if [ $RC -ne 0 ] || [ ! -s "$OUT" ]; then
  echo "!! collection FAILED (rc=$RC, out missing/empty)"; exit 1
fi
echo "## done $(date '+%F %T')  $(du -h "$OUT" | cut -f1)"
