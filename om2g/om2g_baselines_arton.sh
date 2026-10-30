#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  om2g_baselines_arton.sh — hover / random / greedy / aggregate at the OM2G
#  radio.  Without these the training reward (in kbps) has no scale.
# ═══════════════════════════════════════════════════════════════════════════
#
#  ⚠️  OM2G TRACK (IEEE 802.15.4).  None of the OMNET-track baselines transfer:
#      the reward unit alone differs by 1000x.  See code/om2g/OM2G_README.md.
#
#  Note on `greedy`: greedy_oracle is an ORACLE (it reads the training reward
#  and has 40:1 user compression), so it is an upper reference rather than a
#  matched comparator.  `aggregate` is the matched one.
#  Both are run; report both and say which is which.
#
#  Usage: sbatch ... om2g/om2g_baselines_arton.sh <policy> [episodes] [seed_start]
#
#  Default episodes is 100, not 40: any number reported as a FINAL finding
#  needs n>=100 (n=20-40 is fine for a quick sanity check, not for reporting).
#  seed_start lets a prior smaller run be TOPPED UP to n=100 -- pass the old
#  count as seed_start and (100 - old count) as episodes; the two CSVs'
#  rows concatenate into the full n=100 sample.
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail

POLICY="${1:-hover}"
EPISODES="${2:-100}"
SEED_START="${3:-0}"

export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
SCRATCH="${OM2G_SCRATCH:-/usr/itetnas04/data-scratch-01/eguzys/data}"
RB=$SCRATCH/resilient-backbone
export TMPDIR=$SCRATCH/tmp
source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate $SCRATCH/padre_env
cd "$RB"

export OM2G_TRACK=1
eval "$(python om2g/om2g_config.py --arm compressed --shell)"

# ── Each policy pays for the state exchange IT actually requires ────────────
# hover/random/spsa exchange no state. The others do:
#
#   * `oracle` reads every peer's exact position AND every user's exact
#     position -- including the ~90% of users a drone cannot hear (uplink
#     decode range 157.8 m against a 550 m user cloud). No drone can obtain
#     that by sensing, so it has to arrive by transmission, and the payload is
#     the uncompressed user block.
#   * `aggregate` needs the user cloud's global moments, which likewise have to
#     be pooled across drones -- a compressed-class message.
#
# The charge is therefore set PER POLICY, from om2g_track's own framing math:
#
#   hover / random / spsa : no charge  (genuinely send nothing)
#   greedy   (realistic)  : one-hop position beacon,  51 B  -> 0.82%
#   aggregate             : pooled moment summary,   131 B  -> 2.10%
#   oracle                : uncompressed user block, 2949 B -> 47.18%
#                           (report as an UPPER BOUND; it is unrealizable)
#
# The trained compressed arm pays 131 B / 2.10% -- i.e. more than realistic
# greedy: the policy buys better user knowledge with 1.3% more airtime than a
# beacon.
case "${POLICY:-}" in
  hover|random|spsa|scripted_relay|relay_informed)
    unset OM2G_CONTROL_ON_AIR_BYTES ;;
  relay_blind)
    # A deployable relay heuristic: it must broadcast its own position like
    # greedy does, so it pays greedy's beacon. relay_informed is an ORACLE and
    # is left uncharged deliberately -- it is an upper bound, not a protocol.
    export OM2G_CONTROL_ON_AIR_BYTES=$(python3 -c \
      "import sys;sys.path.insert(0,'$RB/om2g');import om2g_track as T;print(T.om2g_beacon_on_air_bytes())") ;;
  greedy|greedy_pooled)
    export OM2G_CONTROL_ON_AIR_BYTES=$(python3 -c \
      "import sys;sys.path.insert(0,'$RB/om2g');import om2g_track as T;print(T.om2g_beacon_on_air_bytes())") ;;
  aggregate)
    export OM2G_CONTROL_ON_AIR_BYTES=$(python3 -c \
      "import sys;sys.path.insert(0,'$RB/om2g');import om2g_track as T;print(T.om2g_message_on_air_bytes('compressed',8))") ;;
  engineered|dispatch|relay_explore)
    # PAYLOAD-MATCHED charge. The moment message is 5 floats + a count = 24 B
    # of payload; framed the same way as greedy's beacon (om2g_beacon_on_air_bytes:
    # payload + 8 UDP + 20 IP + 15 MAC/PHY) that is 24+43 = 67 B on air, one frame.
    #
    # 67 B is the cost of this message itself, rather than the learned arm's
    # 131 B. A 131 B run is available as a sensitivity row via
    # OM2G_FORCE_CHARGE_BYTES=131.
    export OM2G_CONTROL_ON_AIR_BYTES=${OM2G_FORCE_CHARGE_BYTES:-67} ;;
  oracle)
    export OM2G_CONTROL_ON_AIR_BYTES=$(python3 -c \
      "import sys;sys.path.insert(0,'$RB/om2g');import om2g_track as T;print(T.om2g_message_on_air_bytes('uncompressed'))") ;;
  *)
    echo "baselines: unknown POLICY='${POLICY:-}' -- refusing to guess a charge" >&2
    exit 2 ;;
esac
echo "[charge] POLICY=${POLICY} OM2G_CONTROL_ON_AIR_BYTES=${OM2G_CONTROL_ON_AIR_BYTES:-UNSET}"

export OMNET_APPTAINER_SIF=$SCRATCH/omnet.sif
export OMNET_HOST_CODE=$RB
export OMNET_CONTAINER_CODE=/work
export OMNET_INET_DIR=/opt/inet4.6
export OMNETPP_DIR=/opt/omnetpp-6.4.0
export CUDA_VISIBLE_DEVICES=""

mkdir -p "$RB/om2g_out"
# $OM2G_RUN_TAG keeps a re-measurement from overwriting earlier CSVs taken
# under a different reward configuration.
# `greedy` is the realistic (beacon-sharing) policy; the full-knowledge one is
# `oracle`. The greedy runs are tagged `greedy1hop`/`greedyNhop` so files of
# the two policies are never mixed.
FILETAG="$POLICY"
[ "$POLICY" = "greedy" ] && FILETAG="greedy${OM2G_GREEDY_HOPS:-1}hop"
[ "$POLICY" = "greedy_pooled" ] && FILETAG="greedypooled${OM2G_GREEDY_HOPS:-1}hop"
# engineered now honours OM2G_GREEDY_HOPS too (the Lever_sense dial): without
# the hops in the filename an h0 and an h1 run overwrite each other.
[ "$POLICY" = "engineered" ] && FILETAG="engineered${OM2G_GREEDY_HOPS:-1}hop"
[ "$POLICY" = "dispatch" ] && FILETAG="dispatch${OM2G_GREEDY_HOPS:-1}hop"
[ "$POLICY" = "relay_explore" ] && FILETAG="relayexplore"
OUT="$RB/om2g_out/OM2G_baseline_${FILETAG}_N${OM2G_DRONES:-5}_n${EPISODES}${OM2G_RUN_TAG:+_${OM2G_RUN_TAG}}_seed${SEED_START}.csv"

echo "═══════════════════════════════════════════════════════════════════"
echo "  OM2G BASELINE — policy=$POLICY  N=${OM2G_DRONES:-5}  episodes=$EPISODES  spawn=${OM2G_SPAWN:-default}"
echo "  reward unit: kbps (NOT the OMNET track's Mbps)"
echo "  out=$OUT"
echo "═══════════════════════════════════════════════════════════════════"

cd omnet_module
exec python evaluate_omnet.py \
  --policy "$POLICY" \
  --num-drones ${OM2G_DRONES:-5} \
  --episodes "$EPISODES" \
  --seed-start "$SEED_START" \
  --out "$OUT"
