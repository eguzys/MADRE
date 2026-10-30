#!/bin/bash
# eval_arton.sbatch - run ONE evaluation cell (n=100) on arton instead of the
# laptop. Submit one job per cell; they are independent and there is no
# concurrent-job cap on this account, so the whole out_dim sweep runs in
# parallel in the time a single cell takes.
#
#   sbatch --export=ALL,TAG=odim2_st,ODIM=2 omnet_module/eval_arton.sbatch
#
# Env knobs (all optional except TAG):
#   TAG   arton run-dir tag, e.g. odim2_st / ffn6 / selftok_ds
#   ODIM  OMNET_USER_ENC_OUT_DIM the arm was TRAINED at. Unset => 8 (default).
#   EN    N to evaluate at (default 5)
#   ETN   N the checkpoint was TRAINED at (default = EN; set for transfer cells)
#   OUTD  output subdir under $SCRATCH/eval_out (default n100)
#   WIDTH parallel chunks (default 8: 8 python + 8 OMNeT children = 16 cores)
#   ENC   frozen encoder .pt. Set => encoder_mode=frozen (+ the two env vars
#         omnet_env.py requires). Unset => learned. MUST match the encoder the
#         checkpoint was trained with.
#
#SBATCH --job-name=omnet_eval
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/usr/itetnas04/data-scratch-01/eguzys/data/omnet_eval_%j.log
set -uo pipefail

S="${OM2G_SCRATCH:-/usr/itetnas04/data-scratch-01/eguzys/data}"
R=$S/resilient-backbone

TAG="${TAG:?set TAG=<arton run tag>}"
EN="${EN:-5}"
ETN="${ETN:-$EN}"
OUTD="${OUTD:-n100}"
WIDTH="${WIDTH:-8}"

# ── container toolchain ──────────────────────────────────────────────────────
# ⚠️ eval_n100.sh (the LAPTOP driver) UNSETS OMNET_APPTAINER_SIF, because the
# laptop has a natively built OMNeT binary. Arton does not - omnet_env.py line
# ~1628 branches on this variable and falls back to a bare spawn that will not
# exist here. It must be SET, and its bind root must contain omnet_smoke/.
export OMNET_APPTAINER_SIF=$S/omnet.sif
export OMNET_HOST_CODE=$R
# ⚠️ INET and omnetpp live INSIDE the .sif, so these must be IN-CONTAINER paths.
# Omitting them is NOT caught by the host-side INET check (omnet_env.py ~line
# 100 skips it whenever OMNET_APPTAINER_SIF is set), so INET_DIR would fall
# back to a host path and every OMNeT child would stop at NED-load time.
# These three lines mirror the training launcher.
export OMNET_CONTAINER_CODE=/work
export OMNET_INET_DIR=/opt/inet4.6
export OMNETPP_DIR=/opt/omnetpp-6.4.0
export BATCHED_PIPE_TIMEOUT=100000
export TMPDIR=$S/tmp
mkdir -p "$TMPDIR"

source $S/miniconda3/etc/profile.d/conda.sh
conda activate $S/padre_env
cd "$R" || exit 1

# Same world knobs the laptop driver exports, minus the sif unset above.
# Encoder mode is selectable. A FROZEN-encoder checkpoint expects the 28-wide
# pooled token, while `learned` makes build_drone_tokens emit the 345-wide
# learned token, so the mode must match the checkpoint. Set ENC=<encoder.pt>
# to switch to frozen.
#
# ⚠️ The encoder MUST be the same file the arm was TRAINED with. Scoring a
# frozen policy against a different encoder silently changes its observation --
# it is the same class of error as evaluating an odim arm at the wrong out_dim,
# except that here it would not even raise, because every encoder of this shape
# produces a valid 8-dim latent. The sha is echoed below; check it against the
# arm's training log.
if [ -n "${ENC:-}" ]; then
  [ -s "$ENC" ] || { echo "!! ENC does not exist: $ENC"; exit 1; }
  [ -s "${ENC%.pt}_cfg.json" ] || { echo "!! ENC cfg sidecar missing (latent_scale): ${ENC%.pt}_cfg.json"; exit 1; }
  export OMNET_ENCODER_MODE=frozen
  export OMNET_ALLOW_FROZEN_ENCODER=1
  export OMNET_USER_ENC_PATH="$ENC"
  echo "## encoder_mode=FROZEN sha256=$(sha256sum "$ENC" | cut -c1-16) $ENC"
else
  export OMNET_ENCODER_MODE=learned
  echo "## encoder_mode=learned (each arch pools with its own trained encoder)"
fi
export OMNET_MAX_STEPS=250 OMNET_PEER_LINK_MASK=1
export CUDA_VISIBLE_DEVICES=""      # eval is CPU-only; no GPU is requested

# ⚠️ Must match the arm's TRAINING value or the model rebuild dies on a bare
# tensor-shape mismatch (user_encoder.py reads this at import time).
if [ -n "${ODIM:-}" ]; then
  export OMNET_USER_ENC_OUT_DIM="$ODIM"
  echo "## user-encoder out_dim = $ODIM"
fi

OUT="$S/eval_out/$OUTD"
mkdir -p "$OUT/parts"

# ── resolve the checkpoint ───────────────────────────────────────────────────
# Rank by FRAME COUNT, not mtime: a tag can own more than one run dir (a
# restarted allocation, or an interrupted run holding only an early
# checkpoint), and the newest mtime is not necessarily the furthest run.
scan() {  # scan <glob-expanded dirs...> -> "<frames> <dir>" lines
  for d in "$@"; do
    f=$(ls "$d"/checkpoints/ 2>/dev/null \
        | grep -oE '^checkpoint_[0-9]+\.pt$' \
        | sed -E 's/checkpoint_([0-9]+)\.pt/\1/' | sort -n | tail -1)
    [ -n "$f" ] && echo "$f $d"
  done
}
# Two naming levels. Most tags are a suffix of the EXPERIMENT dir
# (..._staging_odim2_st), but the ST reference is identified by a hash that
# only appears in the RUN dir (normalizedmappo_static_stmodel__00fa5f10_...),
# and its experiment dir is the bare ..._staging that every other tag also
# matches as a prefix. Try the experiment level first, then the run level.
best=$(scan "$R"/src/out/benchmarl_omnet/*${TAG}*/*/ | sort -n | tail -1)
[ -z "$best" ] && best=$(scan "$R"/src/out/benchmarl_omnet/*/*${TAG}*/ | sort -n | tail -1)
[ -z "$best" ] && { echo "!! no checkpoint for tag $TAG"; exit 1; }
FR="${best%% *}"; RD="${best#* }"
# A partially trained arm scores like a real result and is invisible in the
# output. Refuse anything that is not a finished run.
if [ "$FR" -lt 500000 ]; then
  echo "!! $TAG only at frame $FR (<500k) - refusing to score a partial net"; exit 1
fi
CK="$RD/checkpoints/checkpoint_${FR}.pt"
echo "## $TAG -> frame $FR"
echo "## $CK"
echo "## eval N=$EN (trained N=$ETN) -> $OUT/${TAG}_n100.csv"

FINAL="$OUT/${TAG}_n100.csv"
# ⚠️ -s is NOT enough: a header-only CSV from a failed run is 26 bytes and
# "non-empty". Require actual DATA rows.
if [ -f "$FINAL" ] && [ "$(grep -vc '^seed,' "$FINAL")" -gt 0 ]; then
  echo "== already done ($(grep -vc '^seed,' "$FINAL") episodes), nothing to do"; exit 0
fi
rm -f "$FINAL"


# ── run 20 chunks of 5 seeds, WIDTH at a time ────────────────────────────────
JF="$OUT/parts/.jobs_$TAG"; : > "$JF"
# ⚠️ no trailing whitespace on these lines: `xargs -L 1` treats a line ending
# in a blank as a CONTINUATION and silently joins it to the next.
for s in $(seq 0 5 95); do echo "$s 5" >> "$JF"; done

export TAG OUT CK EN ETN
run_one() {
  local o="$OUT/parts/${TAG}_$1.csv"
  [ -s "$o" ] && return 0
  python omnet_module/evaluate_omnet.py "$CK" --policy trained \
    --num-drones "$EN" --train-num-drones "$ETN" \
    --seed-start "$1" -n "$2" --out "$o" > "$OUT/parts/${TAG}_$1.log" 2>&1
}
export -f run_one
xargs -a "$JF" -P "$WIDTH" -L 1 bash -c 'run_one "$@"' _

echo "seed,episode_reward,steps" > "$FINAL"
cat "$OUT"/parts/${TAG}_*.csv 2>/dev/null | grep -v '^seed,' | sort -t, -k1,1n >> "$FINAL"
N_EP=$(( $(wc -l < "$FINAL") - 1 ))
echo "## -> $N_EP episodes"
echo "## done $(date '+%F %T')"
# Fail LOUDLY: without this, a job with 0 episodes would exit 0 and Slurm
# would report it as COMPLETED.
if [ "$N_EP" -eq 0 ]; then
  echo "!! ZERO episodes -- every chunk failed. First chunk log:"
  sed -n '1,60p' "$OUT/parts/${TAG}_0.log" 2>/dev/null
  exit 1
fi
