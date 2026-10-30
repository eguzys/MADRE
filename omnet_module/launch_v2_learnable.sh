#!/bin/bash
# launch_v2_learnable.sh — the first OMNeT config measured to have real
# learnable headroom. Wraps train_omnet_arton.sh with the v2 task definition.
#
# WHY THIS CONFIG (all measured):
#
#   LIVE, 20 matched seeds, 250-step episodes, jammer off:
#       hover   39.1 reward/step      <- the trivial control
#       random  56.9  (1.31x hover)
#       greedy  96.3  (2.23x hover, beats hover on 18/20 seeds)
#       strong placement (k-means + local refine, offline)  ~177
#       => hover/best = 0.22   (gate G_spawn_suboptimal wants <= 0.60)
#
#   The previous config (cfg_99) measured hover/best = 0.80 and
#   greedy/hover = 1.06x: standing still was as good as anything, so it had no
#   learnable headroom.
#
# THE THREE KNOBS THAT CHANGED, and why:
#   droneSpawnMode  cluster -> staging
#       "cluster" deals drones onto the user-cluster CENTROIDS, i.e. hands the
#       policy a near-optimal placement at t=0. Over 144 measured cells, ZERO
#       had hover/best <= 0.6. It cannot be made learnable by any other knob.
#   user_spread_eff  100 -> 250 m
#       At 100 m against r_u = 158 m, one drone blankets an entire cluster, so
#       placement does not matter and arena size is irrelevant. THE dominant
#       knob, and the one no previous sweep varied.
#   pCross  0.1 -> 0.5
#       At the 60 m/step drone motion the gap between clusters is crossable, so
#       cross-cluster commodities make the relay backbone load-bearing.
#
# Plus: jammer OFF (it deletes ~85% of reward on some seeds, nothing on others,
# with zero gradient either side, and at 20 m/step it outruns the drones);
# episodes 1000 -> 250 (greedy reaches its peak in 1-14 steps, so 1000 steps
# just averages the steady state and drowns the transient the learner needs);
# encoder LEARNED (OMNET_ENCODER_MODE is exported explicitly).
#
# Usage:   bash omnet_module/launch_v2_learnable.sh [arch] [max_frames] [resume_ckpt]
#          arch defaults to st, max_frames to 500000.
#
# resume_ckpt is forwarded to train_omnet_arton.sh's $3. It
# MUST be launched through here rather than by calling train_omnet_arton.sh
# directly, because resume rebuilds the EXPERIMENT from config.pkl but the
# SIMULATOR geometry still comes from $OMNET_EXTRA_ARGS at runtime -- resuming
# without this wrapper would continue the run in the omnetpp.ini default
# world instead of the one it was trained in.
#
# $OMNET_RUN_TAG appends a marker to the save suffix so parallel variants
# (A/B arms, seed replicates) land in distinguishable directories instead of
# stacking sibling hashes under one name.
set -uo pipefail

ARCH="${1:-st}"
MAX_FRAMES="${2:-500000}"
RESUME_CKPT="${3:-}"

# ── The task definition. Single source of truth: config_knobs.Config. ────────
N=5
L=2000.0
MAX_CLUSTERS=2
SPREAD_EFF=350.0        # EFFECTIVE metres (base is derived, see below)
P_CROSS=0.5
N_USERS=17
STEP_M=60.0             # metres per RL step at |action|=1
EP_STEPS=250
DRONE_TX=200.0
USER_TX=10.0
SPAWN_MODE=staging
JAMMER_ACTIVATE=999999  # > EP_STEPS  =>  jammer never fires

# Build $OMNET_EXTRA_ARGS from config_knobs rather than hand-writing it, so the
# Python-oracle side (max_flow_reward TX powers, greedy_oracle.STEP_M) and the
# OMNeT side cannot drift apart (see config_knobs.py's module docstring on the
# dual-site knobs).
EXTRA=$(python3 - <<PY
import sys
sys.path.insert(0, "omnet_module")
from config_knobs import Config, build_omnet_extra_args
print(build_omnet_extra_args(Config(
    n_drones=$N, L=$L, max_clusters=$MAX_CLUSTERS, p_cross=$P_CROSS,
    user_spread_eff=$SPREAD_EFF, step_m=$STEP_M, drone_tx_mW=$DRONE_TX,
    user_tx_mW=$USER_TX, ep_steps=$EP_STEPS, n_users=$N_USERS,
    drone_spawn_mode="$SPAWN_MODE", jammer_activate=$JAMMER_ACTIVATE)))
PY
)
if [ -z "$EXTRA" ]; then
  echo "FATAL: build_omnet_extra_args produced nothing. Refusing to launch a"
  echo "       run whose geometry would silently fall back to the ini defaults"
  echo "       (a different world from the v2 task)."
  exit 1
fi
export OMNET_EXTRA_ARGS="$EXTRA"

# Python-oracle halves of the dual-site knobs (read at import by the run.py
# SUBPROCESS -- in-process attribute mutation cannot cross a process boundary).
export OMNET_DRONE_TX_mW="$DRONE_TX"
export OMNET_USER_TX_mW="$USER_TX"
export OMNET_STEP_M="$STEP_M"          # omnet_env scales the action by this
export OMNET_NUM_DRONES="$N"

# Learned encoder (DEFAULT): each architecture pools raw per-user tokens with
# its OWN encoder, trained end-to-end from the RL reward.
#
# Respects the environment, like every other knob in this block, so the
# FROZEN SHARED ENCODER arm can reuse this launcher unchanged. Set
# OMNET_ENCODER_MODE=frozen together with OMNET_ALLOW_FROZEN_ENCODER=1 and
# OMNET_USER_ENC_PATH=<refitted encoder> (omnet_env.py requires all three; the
# flag alone is refused).
export OMNET_ENCODER_MODE="${OMNET_ENCODER_MODE:-learned}"

# Peer link mask: gate peer visibility on the drone radio graph's connected
# components, so an unreachable peer's position + sensed-user set are only
# visible over a radio path that exists in the scenario.
#
# MEASURED on the v2 trajectories: the swarm is fully connected on 100% of
# steps (r_d = 523 m at 200 mW; median pairwise separation 159-451 m), so
# enabling this changes NOTHING on this task -- it is a verified no-op here.
# It matters only when clusters are separated by more than r_d, which is the
# same condition needed for a multi-hop relay backbone (median drone-graph
# diameter is 1 on v2).
#
# It MUST match between training and evaluation -- export the same value in the
# eval shell. OMNeTEnv prints the setting at construction so a mismatch is
# visible in both logs.
export OMNET_PEER_LINK_MASK="${OMNET_PEER_LINK_MASK:-1}"

# Per-drone credit assignment. With the team reward broadcast unchanged to all
# five drones, every drone gets the same advantage (run aa732995 converged to a
# rigid swarm, cosine 0.99 between the drones' mean actions, spread 65 m -> 69 m
# in a 473 m user cloud). Each drone gets D_i = R(all) - R(all \ i), the same
# MARGINAL quantity the greedy baseline uses.
#
# Costs one extra LP solve per drone per step: 11.6 ms/step at N=5 against the
# global reward's 2.1 ms, using max_flow_reward's per-drone `cache_tag`
# warm-start slots (see _ORACLE_LP_CACHE).
#
# MEASURED IN TRAINING, job 2167157 vs aa732995: +38% per iteration (260 s vs
# 189 s steady state at 6000 frames/iter, i.e. 11.9 ms per env-step). Budget
# wall-clock accordingly -- the microbenchmark alone suggests +19%, which is
# too low because the real per-frame cost under parallel envs is ~31.5 ms, not
# the ~49 ms that estimate assumed.
# Set 0 for the A/B against the global (team) reward. Eval is unaffected either
# way -- evaluate_omnet.py always scores infos["global_reward"].
export OMNET_DIFFERENCE_REWARD="${OMNET_DIFFERENCE_REWARD:-1}"

export OMNET_MAX_FRAMES="$MAX_FRAMES"
# NOTE the suffix names the output dir; keep the spread in it equal to
# $SPREAD_EFF so the run dir names the world it was trained in.
export OMNET_SAVE_SUFFIX="v2_N${N}_L2000_sprd${SPREAD_EFF%.*}_pX05_staging${OMNET_RUN_TAG:+_${OMNET_RUN_TAG}}"

# task.ep_length must equal *.rlBridge.maxSteps or BenchMARL truncates at one
# length while the simulator ends the episode at another. train_omnet_arton.sh
# hard-checks this pair and refuses to launch on a mismatch.
# keep_checkpoints_num is raised so a BEST checkpoint can be chosen after the
# fact (the final step is not necessarily the best point of a run).
# $OMNET_HYDRA_EXTRA is appended LAST so an ablation arm can override a model
# field (e.g. model.use_self_token=false for the self-token A/B) without editing
# this file and thereby changing the config of every other arm too.
export OMNET_HYDRA_OVERRIDES="task.ep_length=${EP_STEPS} \
experiment.checkpoint_interval=48000 \
experiment.keep_checkpoints_num=12${OMNET_HYDRA_EXTRA:+ ${OMNET_HYDRA_EXTRA}}"

echo "=============================================================="
echo " OMNeT task v2 — first config with measured learnable headroom"
echo "   arch=$ARCH  N=$N  frames=$MAX_FRAMES  encoder=$OMNET_ENCODER_MODE"
echo "   spawn=$SPAWN_MODE  spread_eff=$SPREAD_EFF  pCross=$P_CROSS"
echo "   step_m=$STEP_M  ep_steps=$EP_STEPS  jammer=OFF"
echo "   peer_link_mask=$OMNET_PEER_LINK_MASK  (export the SAME value at eval)"
echo "   difference_reward=$OMNET_DIFFERENCE_REWARD  (per-drone credit)"
# Baselines in the spread-350 world, TEAM reward, n=20 (seeds 0-19),
# from eval_baselines_parallel.sh: greedy beats hover 19/20, which is the same
# gate criterion the spread was chosen on.
echo "   baselines to beat:  hover 416.2/step   random 308.5   greedy 835.3   (n=20)"
echo "=============================================================="
if [ -n "$RESUME_CKPT" ]; then
  echo "   RESUMING from: $RESUME_CKPT"
  echo "   resume frame budget (TOTAL): ${OMNET_RESUME_MAX_FRAMES:-<from config.pkl -- may be a no-op!>}"
fi
echo "   run tag: ${OMNET_RUN_TAG:-<none>}   hydra extra: ${OMNET_HYDRA_EXTRA:-<none>}"
echo "=============================================================="
# $OMNET_TRAINER selects the trainer script, so variants (e.g. the gat arch
# whitelist) live in a SEPARATE file and the script that running jobs hold
# open never has to change. bash reads a script incrementally by BYTE OFFSET,
# so rewriting it in place (scp truncates and rewrites) would make running
# jobs resume at a shifted offset. Push remote scripts with scp-to-temp +
# mv (atomic rename) so the running process keeps its old inode.
exec bash "${OMNET_TRAINER:-omnet_module/train_omnet_arton.sh}" "$ARCH" "$N" $RESUME_CKPT
