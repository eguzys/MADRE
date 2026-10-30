#!/bin/bash
# OMNeT++ realism-run launcher, D-INFK student-cluster. Adapted from
# train_omnet_arton.sh -- same math, same gates, only cluster-specific paths
# and resource flags differ (this cluster's job_submit plugin rejects
# --gres/--cpus-per-task entirely and auto-assigns 1 GPU/2 CPU/24GB per job;
# fighting that produces "Specifying TRES per node/task is not allowed").
#
# Usage: sbatch --account=projects --partition=jobs \
#        train_omnet_studcluster.sh <arch> [num_drones] [resume_checkpoint]
#SBATCH --job-name=omnet_train
#SBATCH --time=48:00:00
set -uo pipefail

# Entropy guard: the entropy_coef this run MUST train/resume at. The run
# refuses to start if the algorithm config (padre_mappo_norm*.yaml or the
# checkpoint's config.pkl) carries a different value.
EXPECT_ENT="${OMNET_EXPECT_ENT:-0.03}"

ARCH="${1:-mlp}"
# Drone count for THIS run. Default 5 = the frozen task-validation config; the
# ST/DeepSets transfer checkpoints must be trained here and nowhere else.
# N != 5 is for the per-N FFN baselines only: FFN cannot forward-pass at an N
# it was not trained at, so each transfer-table column needs its own FFN.
# The six task-validation criteria (task_validation.py) gate TRAINING cell
# selection, and no cell passes at N=4. An
# FFN trained at N=4 here is therefore trained in a world its own criteria
# reject as an anchor; it is reported as a same-world baseline, not as a
# validated training cell, and that must be said in the methods.
NUM_DRONES="${2:-5}"
# ── User-set encoder mode (round-2 addition) ─────────────────────────────────
# frozen  = the shipped reconstruction-trained DeepSets AE, pooled ENV-side
#           (conf/task/omnet/static.yaml). The round-1 configuration.
# learned = the env emits the raw padded per-user block instead and each
#           architecture pools it with its OWN encoder, trained from the RL
#           reward (conf/task/omnet/static_learnedenc.yaml).
# Both halves must agree; run.py has a coherence gate that refuses a mismatch.
# Selected by $OMNET_ENCODER_MODE so the positional args stay backward
# compatible with every existing sbatch invocation.
#
# The default is LEARNED, which is the design decision for the OMNeT
# track (the frozen 8-dim summary is dominated by raw-dBm channels; measured
# input-variance share of the peer geometry under frozen was 0.004%). frozen
# is still available but must now be asked for BY NAME, and for the set-based
# architectures it additionally requires OMNET_ALLOW_FROZEN_ENCODER=1 so that
# "I am deliberately running the round-1 configuration" cannot be confused
# with an unset variable.
# arch=mlp is BenchMARL's stock Mlp and structurally cannot own a trainable
# encoder (see the FATAL a few lines down), so its only legal mode is frozen
# and it keeps that as its default. Everything else defaults to learned.
if [ "$ARCH" = "mlp" ]; then
  OMNET_ENCODER_MODE="${OMNET_ENCODER_MODE:-frozen}"
else
  OMNET_ENCODER_MODE="${OMNET_ENCODER_MODE:-learned}"
fi
if [ "$OMNET_ENCODER_MODE" = "frozen" ] && [ "$ARCH" != "mlp" ] \
   && [ "${OMNET_ALLOW_FROZEN_ENCODER:-0}" != "1" ]; then
  echo "FATAL: refusing to train arch=$ARCH with encoder_mode=frozen."
  echo "       frozen routes the whole user set through a frozen 8-dim"
  echo "       reconstruction autoencoder (obs_layout=padre, max_users=20)."
  echo "       The OMNeT design decision is a LEARNED encoder; set this"
  echo "       variable explicitly."
  echo "       Either drop OMNET_ENCODER_MODE (you now get learned), or set"
  echo "       OMNET_ALLOW_FROZEN_ENCODER=1 to say you mean it."
  exit 1
fi
case "$OMNET_ENCODER_MODE" in
  frozen)  TASK_NAME="omnet/static";             MODEL_ENC_OVERRIDES="model.encoder_mode=frozen" ;;
  learned) TASK_NAME="omnet/static_learnedenc";  MODEL_ENC_OVERRIDES="model.encoder_mode=learned model.obs_layout=omnet model.max_users=${OMNET_MAX_USERS_CAP:-25}" ;;
  *) echo "FATAL: OMNET_ENCODER_MODE must be frozen|learned, got '$OMNET_ENCODER_MODE'"; exit 1 ;;
esac
# Every architecture here (FFN/MLP, ST, DeepSets) can take
# either a frozen or a learned encoder. The real and much narrower constraint is
# about one MODEL CLASS: arch=mlp maps to BenchMARL's STOCK Mlp
# (model=padre_mlp), which has no encoder_mode/obs_layout field to set, so Hydra
# rejects the override. The repo's OWN feed-forward model (arch=ffn ->
# model=padre_ffn) has the field and supports learned perfectly well.
#
# So this is a model-class check, not an architecture limitation: use
# arch=ffn for a learned-encoder feed-forward cell.
if [ "$OMNET_ENCODER_MODE" = "learned" ] && [ "$ARCH" = "mlp" ]; then
  echo "FATAL: arch=mlp maps to BenchMARL's stock Mlp (model=padre_mlp), which has"
  echo "       no encoder_mode/obs_layout field for Hydra to override."
  echo "       This is NOT an architecture limitation -- a feed-forward net can host"
  echo "       a learned encoder fine. Use arch=ffn (the repo's own padre_ffn) for"
  echo "       the learned-encoder feed-forward cell, or arch=st / arch=deepsets."
  exit 1
fi
case "$NUM_DRONES" in
  ''|*[!0-9]*) echo "FATAL: num-drones must be an integer, got '$NUM_DRONES'"; exit 1 ;;
esac
export OMNET_NUM_DRONES="$NUM_DRONES"

# Optional 3rd arg: checkpoint to resume from. Validated here so a bad path
# fails immediately rather than 15 minutes downstream.
RESUME_FROM="${3:-}"
if [ -n "$RESUME_FROM" ]; then
  case "$RESUME_FROM" in /*) ABS="$RESUME_FROM" ;; *) ABS="$PWD/$RESUME_FROM" ;; esac
  [ -f "$ABS" ] || { echo "FATAL: resume checkpoint not found: $ABS"; exit 1; }
  CFG="$(dirname "$(dirname "$ABS")")/config.pkl"
  [ -f "$CFG" ] || { echo "FATAL: no config.pkl beside checkpoint (expected $CFG)."; \
                     echo "       resume_experiment() rebuilds from config.pkl, not hydra."; exit 1; }
  echo "=== RESUME FROM: $ABS ==="
  echo "=== config.pkl:  $CFG ==="
fi

# [2026-08-13] gat added. It was excluded because gat_model.py had no
# user-set encoder, so it could not run the learned-encoder OMNeT track at all
# (self_flag_col indexes the reassembled token; on a raw block it points at a
# user field). The encoder is now ported, so gat is a first-class arch here.
# arch -> model=padre_$ARCH, and $MODEL_ENC_OVERRIDES supplies
# encoder_mode/obs_layout/max_users, which GATConfig now accepts.
case "$ARCH" in
  mlp|st|deepsets|ffn|gat) ;;
  *) echo "FATAL: arch must be mlp|st|deepsets|ffn|gat, got '$ARCH'"; exit 1 ;;
esac
RESUME_FROM="${3:-}"
if [ -n "$RESUME_FROM" ]; then
  [ -f "$RESUME_FROM" ] || { echo "FATAL: resume_checkpoint '$RESUME_FROM' does not exist"; exit 1; }
  RESUME_FROM="$(readlink -f "$RESUME_FROM")"  # absolute -- we cd twice before using it
fi

RB=/work/projects/eguzys/code
export TMPDIR=/work/scratch/eguzys/tmp
mkdir -p "$TMPDIR"
export WANDB_MODE=offline

# ── Option-B wrapper: the .sif is the TOOLCHAIN, the code is bind-mounted.
# INET/omnetpp live IN-CONTAINER, so these two must be in-container paths.
export OMNET_APPTAINER_SIF=$RB/omnet.sif
export OMNET_HOST_CODE=$RB
export OMNET_CONTAINER_CODE=/work
export OMNET_INET_DIR=/opt/inet4.6
export OMNETPP_DIR=/opt/omnetpp-6.4.0

# ── Seed stride: TorchRL builds every collector env in ONE process with the
# SAME reset seed, so without this all parallel envs simulate the IDENTICAL
# world (verified: 4 envs, episodeSeed=71). DORMANT unless exported here.
# Critic state history depth. K=1 = single-frame global state (sufficient
# statistic: reward is a memoryless fn of the current frame). Set explicitly so
# the value is visible in the job environment. K is not swept.
export OMNET_STATE_HISTORY_K=1

export OMNET_TRAIN_SEED_STRIDE=1
export BATCHED_PIPE_TIMEOUT=100000

source /work/projects/eguzys/miniconda3/etc/profile.d/conda.sh
conda activate /work/scratch/eguzys/padre_env
cd $RB

# ── Pre-flight gates. Each one stops a run that would otherwise start and
# produce unusable output.
BIN=omnet_smoke/out/clang-release/omnet_smoke
[ -x "$BIN" ] || { echo "FATAL: no binary at $BIN -- 'make clean && make -j4 MODE=release' in-container"; exit 1; }
python -c "import highspy" 2>/dev/null || { echo "FATAL: highspy missing -- reward would be a silent zero"; exit 1; }
apptainer exec --bind $OMNET_HOST_CODE:$OMNET_CONTAINER_CODE $OMNET_APPTAINER_SIF \
  test -x $OMNET_CONTAINER_CODE/$BIN || { echo "FATAL: container cannot see $BIN through the bind"; exit 1; }

echo "=== node: $(hostname), gpu: $(nvidia-smi -L) ==="
echo "=== arch=$ARCH  N=$NUM_DRONES  encoder=$OMNET_ENCODER_MODE  task=$TASK_NAME  sif=$OMNET_APPTAINER_SIF  stride=$OMNET_TRAIN_SEED_STRIDE ==="
echo "=== hydra overrides: ${OMNET_HYDRA_OVERRIDES:-<none>} ==="
echo "=== OMNET_EXTRA_ARGS: ${OMNET_EXTRA_ARGS:-<none>} ==="
# Episode length lives in TWO places and they must agree (see the
# OMNET_HYDRA_OVERRIDES note at the run.py call). Catch a mismatch here rather
# than after 40 h of training against a truncation the simulator disagrees with.
_ep_hydra=$(printf '%s' "${OMNET_HYDRA_OVERRIDES:-}" | grep -oE 'task\.ep_length=[0-9]+' | tail -1 | cut -d= -f2)
_ep_sim=$(printf '%s' "${OMNET_EXTRA_ARGS:-}" | grep -oE 'rlBridge\.maxSteps=[0-9]+' | tail -1 | cut -d= -f2)
if [ -n "$_ep_hydra" ] && [ -n "$_ep_sim" ] && [ "$_ep_hydra" != "$_ep_sim" ]; then
  echo "FATAL: episode-length mismatch -- task.ep_length=$_ep_hydra but"
  echo "       *.rlBridge.maxSteps=$_ep_sim. BenchMARL would truncate at one"
  echo "       length while the simulator ends the episode at another."
  exit 1
fi

cd src
SAVE=out/benchmarl_omnet/omnet_${ARCH}_n${NUM_DRONES}_enc${OMNET_ENCODER_MODE}${OMNET_SAVE_SUFFIX:+_${OMNET_SAVE_SUFFIX}}
mkdir -p "$SAVE"

# Seed-population watchdog: N envs must yield N distinct episodeSeeds.
# Backgrounded -- samples once at T+4min and scancels this job on collapse.
# See seed_watchdog.sh for why.
N_ENVS=$(python -c "import yaml; d=yaml.safe_load(open('conf/experiment/omnet.yaml')); v=d.get('on_policy_n_envs_per_worker'); assert v, 'on_policy_n_envs_per_worker absent from omnet.yaml -- the watchdog must not guess'; print(v)") || { echo "FATAL: cannot read env count for the seed watchdog"; exit 1; }
"$RB/omnet_module/seed_watchdog.sh" "${SLURM_JOB_ID:-0}" "$N_ENVS" 900 &
# Telemetry-stall watchdog: if a training run zombies out after a worker
# death (torchrl _check_for_faulty_process cleanup crash, see the
# stall_watchdog.sh header), squeue keeps reporting RUNNING forever while
# nothing trains. Backgrounded like seed_watchdog.sh; a no-op when
# OM2G_TELEMETRY_CSV is unset (plain OMNET-track cells).
"$RB/omnet_module/stall_watchdog.sh" "${SLURM_JOB_ID:-0}" "${OM2G_TELEMETRY_CSV:-}" "${OM2G_STALL_SECONDS:-1200}" &

if [ -n "$RESUME_FROM" ]; then
  echo "=== resuming from $RESUME_FROM ==="
  # task/algorithm/model/experiment.* below are only to satisfy Hydra's
  # mandatory defaults (task=???, algorithm=???) -- run.py ignores all of
  # them once +resume_from is set and rebuilds from that checkpoint's own
  # config.pkl instead (see benchmarl_padre/omnet_checkpoint.py). In
  # particular experiment.max_n_frames here is NOT used; the original run's
  # total carries over unchanged. entropy_coef likewise comes from the
  # checkpoint's config.pkl, not from the algorithm= value below -- so the
  # entropy guard below reads config.pkl directly, not this script's arg.
  #
  # Entropy guard (resume): reads config.pkl, so a checkpoint trained at a
  # different entropy_coef (e.g. bare 0.01) is refused.
  ACTUAL_ENT=$(OMNET_CONFIG_PKL="$CFG" OMNET_EXPECT_ENT="$EXPECT_ENT" python -c '
import os, pickle, sys
path = os.environ["OMNET_CONFIG_PKL"]
expect = float(os.environ["OMNET_EXPECT_ENT"])
with open(path, "rb") as f:
    pickle.load(f)  # task
    pickle.load(f)  # task_config
    algorithm_config = pickle.load(f)
actual = getattr(algorithm_config, "entropy_coef", None)
if actual is None:
    print(f"FATAL: entropy_coef missing from algorithm_config in {path}", file=sys.stderr)
    sys.exit(1)
actual = float(actual)
# sigma-floor requirement: entropy_coef=0.01 is ACCEPTED only when the
# scale_lb sigma-floor is present and > 0. This is the OMNeT-track rule
# (ent0.01 + floor learns AND avoids collapse). Bare 0.01 without a floor
# is the quarantined collapse config and must still be REFUSED even if
# OMNET_EXPECT_ENT=0.01 is set -- the floor, not the coefficient alone,
# is what makes 0.01 safe here.
scale_lb = getattr(algorithm_config, "scale_lb", None)
if abs(actual - 0.01) < 1e-9:
    if scale_lb is None or float(scale_lb) <= 0.0:
        print(f"FATAL: entropy_coef=0.01 but scale_lb floor absent/zero (scale_lb={scale_lb}) in {path} -- this is bare-0.01 collapse config, refusing", file=sys.stderr)
        sys.exit(1)
print(actual)
sys.exit(0 if abs(actual - expect) < 1e-9 else 1)
')
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "FATAL: entropy guard failed (resume): $CFG algorithm_config.entropy_coef=${ACTUAL_ENT:-<unreadable, see stderr above>}, expected $EXPECT_ENT. Refusing to resume."
    exit 1
  fi
  echo "entropy guard OK (resume): config.pkl -> ${ACTUAL_ENT}"

  # $OMNET_RESUME_MAX_FRAMES -> run.py's +resume_max_n_frames, which
  # resume_experiment() applies to experiment_config.max_n_frames.
  #
  # BenchMARL's max_n_frames is a TOTAL, not a delta, and it comes from the
  # checkpoint's own config.pkl on resume -- a checkpoint already at its budget
  # would satisfy the stop condition on entry and write "TRAIN DONE" without
  # training. Any resume that is meant to train FURTHER must raise the total here.
  RESUME_FRAMES_ARG=""
  if [ -n "${OMNET_RESUME_MAX_FRAMES:-}" ]; then
    case "$OMNET_RESUME_MAX_FRAMES" in
      ''|*[!0-9]*) echo "FATAL: OMNET_RESUME_MAX_FRAMES must be an integer, got '$OMNET_RESUME_MAX_FRAMES'"; exit 1 ;;
    esac
    echo "=== resume frame budget (TOTAL, not additive): $OMNET_RESUME_MAX_FRAMES ==="
    RESUME_FRAMES_ARG="+resume_max_n_frames=$OMNET_RESUME_MAX_FRAMES"
  else
    echo "WARNING: OMNET_RESUME_MAX_FRAMES unset -- the budget comes from"
    echo "         config.pkl. If this checkpoint already reached it, training"
    echo "         will stop immediately having done nothing."
  fi

  python run.py task=${TASK_NAME} algorithm=padre_mappo_norm_ent0.03 model=padre_${ARCH} \
    ${MODEL_ENC_OVERRIDES} \
    experiment=omnet \
    +resume_from="$ABS" \
    ${RESUME_FRAMES_ARG}
else
  ALGO="${OMNET_ALGO:-padre_mappo_norm_ent0.03}"
  ALGO_YAML="conf/algorithm/${ALGO}.yaml"
  # Entropy guard (fresh-start): resolve the algorithm actually being used
  # and hard-verify its entropy_coef before launching, instead of trusting
  # whatever ${OMNET_ALGO} happened to be set to.
  ACTUAL_ENT=$(OMNET_ALGO_YAML="$ALGO_YAML" OMNET_EXPECT_ENT="$EXPECT_ENT" python -c '
import os, sys, yaml
path = os.environ["OMNET_ALGO_YAML"]
expect = float(os.environ["OMNET_EXPECT_ENT"])
if not os.path.isfile(path):
    print(f"FATAL: no such algorithm yaml: {path}", file=sys.stderr)
    sys.exit(1)
d = yaml.safe_load(open(path))
actual = d.get("entropy_coef")
if actual is None:
    print(f"FATAL: entropy_coef missing from {path}", file=sys.stderr)
    sys.exit(1)
actual = float(actual)
print(actual)
sys.exit(0 if abs(actual - expect) < 1e-9 else 1)
')
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "FATAL: entropy guard failed (fresh): ${ALGO_YAML} has entropy_coef=${ACTUAL_ENT:-<unreadable, see stderr above>}, expected $EXPECT_ENT. Refusing to train."
    exit 1
  fi
  echo "entropy guard OK (fresh): ${ALGO} -> ${ACTUAL_ENT}"

  # $OMNET_HYDRA_OVERRIDES: free-form extra Hydra overrides, appended LAST so
  # they win. Added 2026-08-07 so a run can set task.ep_length (episode length
  # must match *.rlBridge.maxSteps in $OMNET_EXTRA_ARGS -- BenchMARL reads
  # task.ep_length for truncation, the sim reads maxSteps, and if they disagree
  # the collector and the simulator disagree about when an episode ends).
  # Echoed above so it is visible in the job log.
  python run.py task=${TASK_NAME} algorithm=${ALGO} model=padre_${ARCH} \
    ${MODEL_ENC_OVERRIDES} \
    experiment=omnet \
    experiment.max_n_frames=${OMNET_MAX_FRAMES:-3000000} \
    experiment.save_folder=$SAVE \
    ${OMNET_HYDRA_OVERRIDES}
fi
RC=$?
if [ $RC -ne 0 ]; then
  echo "=== TRAIN FAILED rc=$RC $(date) ==="
  exit $RC
fi
echo "=== TRAIN DONE $(date) ==="
