"""
OMNeTEnv — PettingZoo ParallelEnv that drives the OMNeT++ simulator over ZMQ.

Drop-in replacement for PADREParallelEnv. Top face (PettingZoo contract) is
identical. Bottom face is a ZMQ REP socket + subprocess relaunch per episode.

ZMQ contract (REP server = Python, REQ client = OMNeT++):
  - Python binds FIRST on tcp://127.0.0.1:5555, then OMNeT++ is launched.
  - Per step: recv obs JSON → send {"type":"action","actions":[...],"reward":r}.
  - Episode end: recv obs with done=true → send {"type":"ack"}.
"""
import atexit
import collections
import hashlib
import json
import math
import os
import shlex
import signal
import subprocess
import tempfile
import weakref

import numpy as np
import zmq
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

import max_flow_reward as _mfr
from max_flow_reward import compute_oracle_max_flow_reward, _link_capacity_bps

# ── Process-leak prevention ──────────────────────────────────────────────────
#
# Three-layer defence against orphaned OMNeT++ subprocesses (a fourth layer,
# prctl(PR_SET_PDEATHSIG) via preexec_fn, was removed: preexec_fn is NOT
# thread-safe and TorchRL's collector runs envs on threads.
# process_group=0 (see _launch_omnet) covers the group-kill case):
#   1. __del__ — backstop if close() is skipped (GC, uncaught exception).
#   2. atexit — kills all live envs on clean Python exit.
#   3. SIGTERM handler — converts SIGTERM to SystemExit so atexit fires.

_live_envs: "weakref.WeakSet" = weakref.WeakSet()


def _atexit_cleanup() -> None:
    for env in list(_live_envs):
        try:
            env._kill_proc()
        except Exception:
            pass


atexit.register(_atexit_cleanup)


def _sigterm_to_exit(signum, frame):
    raise SystemExit(128 + signum)


if signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
    signal.signal(signal.SIGTERM, _sigterm_to_exit)


# ── Module constants (edit to adapt to new machines / layouts) ──────────────

HOME = os.path.expanduser("~")
# Paths must RESOLVE, not be hardcoded -- the same code runs on the laptop, on
# Euler, and inside the Apptainer container (INET at /opt/inet4.6, project
# bind-mounted at /work). Resolution order, most specific first:
#   1. explicit OMNET_* env override
#   2. the toolchain's OWN exported vars (the container exports INET_ROOT and
#      OMNETPP_ROOT; `source setenv` sets them on the laptop too)
#   3. path relative to THIS FILE (omnet_smoke is a sibling of omnet_module)
#   4. the historical laptop layout
# Resolving at runtime lets one file run unchanged on every machine.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INET_DIR = (os.environ.get("OMNET_INET_DIR")
            or os.environ.get("INET_ROOT")
            or os.path.join(HOME, "Documents", "thesis", "inet4.6"))
OMNETPP_DIR = (os.environ.get("OMNETPP_DIR")
               or os.environ.get("OMNETPP_ROOT")
               or os.path.join(HOME, "Documents", "thesis", "omnetpp-6.4.0"))
OMNET_SMOKE_DIR = (os.environ.get("OMNET_SMOKE_DIR")
                   or os.path.join(_REPO, "omnet_smoke"))
OMNET_LOG_STDOUT = os.environ.get("OMNET_LOG_STDOUT")

if not os.path.isdir(OMNET_SMOKE_DIR):
    raise RuntimeError(
        f"OMNET_SMOKE_DIR does not exist: {OMNET_SMOKE_DIR}\n"
        f"(resolved from __file__ at {_REPO}; override with $OMNET_SMOKE_DIR)")
# Skip the host-side INET check when the apptainer wrapper is on: INET then
# lives INSIDE the container (e.g. /opt/inet4.6), not on the host, and
# OMNET_INET_DIR points at the in-container path. The code tree (OMNET_SMOKE_DIR)
# is bind-mounted so its check above still applies. Mirrors the missing-binary
# assert's `and not OMNET_APPTAINER_SIF` guard in _launch_omnet.
if not os.environ.get("OMNET_APPTAINER_SIF") and not os.path.isdir(os.path.join(INET_DIR, "src")):
    raise RuntimeError(
        f"INET not found at {INET_DIR}/src -- override with $OMNET_INET_DIR "
        f"or source INET's setenv (which exports INET_ROOT).")

OMNET_BINARY = "./out/clang-release/omnet_smoke"

# ── Apptainer wrapping (Option B: torch driver OUTSIDE, binary IN-container) ──
# The .sif is the TOOLCHAIN only: omnetpp + INET are baked in.
# padre_env (with torch) drives from outside; only the OMNeT binary runs inside.
# Bind ONLY the code tree -> the container supplies INET/omnetpp. Set
# OMNET_INET_DIR / OMNETPP_DIR to the IN-CONTAINER paths in the launcher.
# Unset OMNET_APPTAINER_SIF -> bare spawn (laptop native). Toggled per machine.
OMNET_APPTAINER_SIF = os.environ.get("OMNET_APPTAINER_SIF")
OMNET_CONTAINER_CODE = os.environ.get("OMNET_CONTAINER_CODE", "/work")
OMNET_HOST_CODE = os.environ.get(
    "OMNET_HOST_CODE",
    os.path.abspath(os.path.join(OMNET_SMOKE_DIR, "..")),
)
ZMQ_PORT = 5555  # fallback only; runtime uses a random free port per instance
# 300 s, because under node contention (several jobs x ~20 OMNeT children on
# 8 CPUs each) a healthy step can exceed 30 s of wall time while the child is
# descheduled, not hung. 300 s still catches a genuinely hung child; the pcpu
# check below catches a truly wedged one immediately regardless of the timeout.
ZMQ_RECV_TIMEOUT_MS = int(os.environ.get("OMNET_ZMQ_RECV_TIMEOUT_MS", 300000))
# Drone count. Overridable via $OMNET_NUM_DRONES so a training run can anchor
# at an N other than the frozen 5 (per-N FFN baselines: FFN cannot forward-pass
# at an N it was not trained at, so the transfer table needs an FFN trained AT
# each N to compare the zero-shot ST/DS policies against). _launch_omnet passes
# --*.numDrones={MAX_N} as a CLI override, so the ini needs no edit and there is
# no shared file to race on. evaluate_omnet.py patches this module global
# directly instead (see _patched_num_drones) since it changes N per cell within
# one process.
#
# A malformed value raises instead of falling back: it means a launcher typo,
# and the run would otherwise train at an unintended N.
_env_n = os.environ.get("OMNET_NUM_DRONES")
if _env_n is None:
    MAX_N = 5   # frozen config; must match *.numDrones in omnetpp.ini
else:
    try:
        MAX_N = int(_env_n)
    except ValueError:
        raise RuntimeError(
            f"OMNET_NUM_DRONES={_env_n!r} is not an integer.")
    if not (1 <= MAX_N <= 20):
        raise RuntimeError(
            f"OMNET_NUM_DRONES={MAX_N} outside the sane range 1..20.")
# ── Action scale: metres of commanded displacement per RL step at |action|=1 ──
#
# The action space is Box(-1, 1, (2,)). RLControlledMobility::move() interprets
# (dx, dy) as a velocity in m/s and only CLIPS at maxSpeed:
#     if (mag > maxSpeed_) { dx *= maxSpeed_/mag; ... }
#     lastPosition += lastVelocity * elapsedTime;   // elapsedTime = 1 s
# so an unscaled action would move a drone at most sqrt(2) m per step.
# config_knobs.build_omnet_extra_args expresses "drone speed" as
# --**.host[*].mobility.maxSpeed={step_m}mps and config_knobs.py:60 asserts
# "step_m == drone speed in m/s (1s tick)". maxSpeed is a CEILING, not a scale,
# so the action is multiplied by STEP_M here.
#
# At 1 m/step the best single action would move the reward by ~0.008% of its
# level and GAE's effective horizon (1/(1-gamma*lmbda) ~ 9 steps) would cover
# ~9 m against a user-uplink radius of 158 m. Four other sites in the codebase
# (config_knobs.py:60, sweep_lib.probe_random_velocity,
# sweep_lib.diffusive_reach_m, omnetpp.ini:206) assume the scaling this
# constant performs.
#
# WHY THE SCALE LIVES HERE and not in the .cc: every action that reaches the
# simulator in training and in evaluation passes through OMNeTEnv.step(), so
# one multiply covers the policy, the greedy oracle, the random baseline and
# the hover baseline identically, with no rebuild and no change to the meaning
# of a NED parameter. The C++ clip is left alone and keeps its proper job --
# an ISOTROPIC speed limit. Note that a diagonal action reaches
# sqrt(2)*STEP_M, so with maxSpeed == step_m the clip renormalises it back to
# step_m: max speed is step_m in ANY direction, which is the intended physics.
#
# Same single source of truth as greedy_oracle.STEP_M -- $OMNET_STEP_M, set by
# config_knobs.apply_python_side -- and the SAME default (20.0, which is also
# omnetpp.ini:99's maxSpeed), so the two can never disagree about how far a
# drone moves. A malformed value is a launcher typo and is refused (see
# _require_scale).
_env_step_m = os.environ.get("OMNET_STEP_M")
if _env_step_m is None:
    STEP_M = 20.0   # matches greedy_oracle.STEP_M's default AND omnetpp.ini:99
else:
    try:
        STEP_M = float(_env_step_m)
    except ValueError:
        raise RuntimeError(
            f"OMNET_STEP_M={_env_step_m!r} is not a float. It is the metres a "
            f"drone moves per RL step at |action|=1; a typo here silently "
            f"changes the physics of the task.")
    if not (0.0 < STEP_M <= 500.0):
        raise RuntimeError(
            f"OMNET_STEP_M={STEP_M} outside the sane range (0, 500] m/step.")

MAX_USERS_CAP = int(os.environ.get("OMNET_MAX_USERS_CAP", "25"))
                    # per-user token pad cap; must cover the largest configured
                    # *.numUsers (omnetpp.ini: default 17; [Config numUsers10]=10,
                    # [Config numUsers25]=25). build_raw_user_tokens() raises if
                    # a live episode ever exceeds this rather than silently truncating.
                    # $OMNET_MAX_USERS_CAP exists for the OM2G track, which runs
                    # 40 users (om2g_track.OM2G_MAX_USERS_CAP); it must be set
                    # identically on the MODEL side via model.max_users, or the
                    # padded user block and the model's view of it disagree and
                    # reconstructed_input_dim's tail-width assert fires. UNSET
                    # reproduces the OMNET track's 25 exactly.
# $OM2G_USER_PARTNER_TAG widens the per-user token from 13 to 17 by appending
# the id-tag vector of the user's CONVERSATION PARTNER (cols 13-16).
#
# WHY. The 13-col token says WHO a drone hears; it never says WHO THEY ARE
# CALLING. Measured in this world, that missing field is the largest lever
# there is: relay_blind (positions only) 123.4 vs relay_informed (knows demand)
# 206.2. A drone therefore cannot tell a bridge worth building from one nobody
# is calling across.
#
# WHY IT IS NOT ORACLE INFORMATION. A drone gets the partner's IDENTITY only
# for users it actually hears (rssi_valid), which is exactly what the
# destination address in a received packet header gives a real relay. It does
# NOT get the partner's position. Drone A learns "my user wants identity X";
# only a drone that HEARS X knows where X is. Routing therefore requires the
# two to exchange state, which makes the message causally necessary rather
# than merely useful.
#
# Opt-in: default 13 keeps every existing checkpoint and arm bit-identical.
OBS_FEATURES = 17 if os.environ.get("OM2G_USER_PARTNER_TAG") else 13
                   # for the full column layout. Change this constant + edit
                   # build_raw_user_tokens() ONLY to expand.
# Own-state width 7. The two leading columns are peer j's
# ABSOLUTE normalised position; see build_drone_tokens' column-layout comment.
# This is the ONE place the width is defined — build_drone_tokens' offsets and
# both DRONE_TOKEN* constants derive from it, and architectures/user_encoder.py's
# OBS_LAYOUTS["omnet"]["own_state_width"] must be kept equal to it (the model
# side has no way to read this module).
OWN_STATE_W = 7
DRONE_TOKEN = OWN_STATE_W + 8 + 8 + 4 + 1  # = 28; encoder_mode="frozen" — see
                   # build_drone_tokens() below for the full column layout.
                   # Change OWN_STATE_W + edit build_drone_tokens() ONLY to
                   # expand.
# per-drone raw token width, encoder_mode="learned": own_state(7) + the RAW
# padded user block (MAX_USERS_CAP*OBS_FEATURES, replacing the 8-wide frozen
# user_latent) + tail (event_latent(8) + self_flag + relay_thru +
# channel_busy + time_since_rx = 12) + mask(1). See build_drone_tokens'
# encoder_mode="learned" branch and architectures/user_encoder.py's "omnet"
# OBS_LAYOUTS entry, which this constant must stay in lockstep with.
DRONE_TOKEN_LEARNED = OWN_STATE_W + MAX_USERS_CAP * OBS_FEATURES + 8 + 4 + 1  # = 345
_TOKEN_ABSMAX = 10.0  # actor-facing columns must be O(1).
                      # After normalisation every RAW column is
                      # clipped to [-1, 1] by _normalize_dbm/_normalize_angle/
                      # _normalize_throughput_bps; the slack above 1.0 exists
                      # only for the frozen EVENT encoder's 8-dim latent, whose
                      # output is a network activation and is not bounded by
                      # construction (measured up to 1.8). Do NOT raise it to
                      # make a new column fit — normalise the column.
MAX_EVENTS_CAP = 32  # per-drone per-step raw-event pad cap — must match
                     # RLBridge.h's MAX_EVENTS_CAP (measured: peak
                     # events/drone/step is 30, p99 is 15).
EVENT_FEATURES = 12  # per-event raw token width — see build_raw_event_tokens()
                     # below for the full column layout.


# ── Centralized critic global state schema ───────────────────────────────────
#
# _build_state (below) builds the global state fed to the centralized MAPPO
# critic (PettingZooWrapper(return_state=True) -> env.state() -> tensordict
# "state" key -> Mappo.get_critic).
#
# CTDE design principle (MAPPO, Yu et al. 2022): the centralized critic must
# observe at least what the reward function observes, since the reward is a
# deterministic function of that state. Training reward is
# compute_oracle_max_flow_reward (max_flow_reward.py): an LP over a graph
# built from drone/user positions and noise_floor, scored against the
# episode's fixed commodity structure (sender_pairs). Those fields, plus
# normalized time remaining, are the sufficient statistic:
#   - positions (drones + users) and noise_floor: read fresh EVERY step —
#     drones move every step (live-verified); noise_floor is constant only
#     until the jammer starts moving (omnetpp.ini jammer_verify: holds spawn
#     steps 0-99, moves 100-499, noise_floor spikes for nearby drones from
#     step 500+), so it must not be read once and cached.
#   - sender_pairs (per-user partner index): drawn once per episode by
#     RLBridge::buildPadreSenders (called only from initialize(), never from
#     the per-step handleMessage loop) and verified constant across steps in
#     a live 5-step trace. Read ONCE at reset and cached — rebuilding it every
#     step from obs_json would just redo an identical array copy. It still
#     varies ACROSS episodes (the training distribution), so it must be in
#     the state.
#   - the radio CONNECTIVITY graph (which edges exist / their capacities) is
#     deliberately EXCLUDED: max_flow_reward._build_edges shows it is a
#     deterministic function of positions + noise_floor, so passing it too
#     is redundant and forces variable-size graph engineering for zero
#     information gain.
#
# No base-station/gateway/sink node exists in the sim to omit (confirmed
# against omnet_smoke/rlsmoke/SmokeNet.ned: drones + users + jammer only), so
# _build_edges is not missing an entity type either.
#
# Fixed width: MAX_N and MAX_USERS_CAP are each fixed per run (module
# globals), so this is a plain fixed-size vector — no set encoder needed on
# the critic side. Per-entity presence uses zero+mask, exactly like
# build_drone_tokens/build_raw_user_tokens, since a live drone/user count can
# be less than the cap even within one run (drone module absence — the rare
# finish() fallback obs — or a smaller numUsers config).
STATE_DRONE_FEATURES = 4  # [x_norm, y_norm, noise_floor_norm, mask]
STATE_USER_FEATURES = 4   # [x_norm, y_norm, partner_idx_norm, mask]
STATE_TIME_FEATURES = 1   # [steps_remaining_norm]
STATE_DIM = (MAX_N * STATE_DRONE_FEATURES
             + MAX_USERS_CAP * STATE_USER_FEATURES
             + STATE_TIME_FEATURES)
_NOISE_FLOOR_SCALE = 100.0  # dBm -> O(1): ambient -100 dBm -> -1.0
_STATE_ABSMAX = 10.0        # guard: state feeding the critic must be O(1)

# ── K-deep rolling history (Phase 2) ─────────────────────────────────────────
#
# Same override pattern as OMNET_NUM_DRONES above: a module constant read
# once at import, fixed per process/run, never silently defaulted on a
# malformed value. K is env-side (it changes what state()/_build_state's
# caller receives) and BenchMARL's stock critic Mlp infers its input width
# straight from env.state()'s actual runtime shape (Algorithm.state_spec ->
# OMNeTClass.state_spec -> env.observation_spec["state"], populated from
# torch.as_tensor(env.state()).shape by TorchRL's PettingZooWrapper — no
# Hydra/model-config field holds a critic-side width literal anywhere in this
# repo). So widening STATE_STACKED_DIM here is self-propagating: there is no
# second place that needs to be told about K.
_env_k = os.environ.get("OMNET_STATE_HISTORY_K")
if _env_k is None:
    STATE_HISTORY_K = 1  # placeholder; override via $OMNET_STATE_HISTORY_K
else:
    try:
        STATE_HISTORY_K = int(_env_k)
    except ValueError:
        raise RuntimeError(f"OMNET_STATE_HISTORY_K={_env_k!r} is not an integer.")
    if not (1 <= STATE_HISTORY_K <= 100):
        raise RuntimeError(
            f"OMNET_STATE_HISTORY_K={STATE_HISTORY_K} outside the sane range 1..100.")

# Each stacked K-slot is the Phase-1 single frame (STATE_DIM) PLUS one
# frame_valid flag (1.0 = real frame, 0.0 = zero-padded pre-fill slot before
# history is full). Padding pre-fill with zeros instead of repeating the
# current frame is deliberate: repeating would tell the critic a moving
# scene had already been stationary for K steps at episode start, which is
# false. The mask lets the critic learn to discount those slots itself.
STACKED_FRAME_DIM = STATE_DIM + 1
STATE_STACKED_DIM = STATE_HISTORY_K * STACKED_FRAME_DIM


# ── Residual-RL knobs (see the $OM2G_RESIDUAL_BASE block in OMNeTEnv.step) ──
_RESIDUAL_BASE = (os.environ.get("OM2G_RESIDUAL_BASE", "") or "").strip().lower()
_RESIDUAL_GAIN = float(os.environ.get("OM2G_RESIDUAL_GAIN", "0.3"))


def _residual_base_velocities(obs):
    """Velocities of the protocol the learned action is a correction TO.

    Imported lazily: signal_baselines imports max_flow_reward, and pulling that
    in at omnet_env import time would reorder this module's own env reads.
    """
    if obs is None:
        raise RuntimeError("no _last_obs yet; the residual base needs an observation")
    import signal_baselines as _SB
    if _RESIDUAL_BASE == "engineered":
        return _SB.engineered_velocities(
            obs, hop_limit=int(os.environ.get("OM2G_GREEDY_HOPS", 1)))
    if _RESIDUAL_BASE == "relay_explore":
        return _SB.relay_explore_velocities(obs)
    raise RuntimeError(f"unknown OM2G_RESIDUAL_BASE={_RESIDUAL_BASE!r}")


def _build_episode_partner_index(obs_json) -> np.ndarray:
    """Per-user-slot partner index + presence mask, from the episode's fixed
    sender_pairs.

    Returns shape (MAX_USERS_CAP, 2): col 0 = partner_idx / MAX_USERS_CAP
    (0.0 where absent), col 1 = mask (1.0 iff user u is a source this
    episode). Indexed explicitly by each pair's own src index rather than by
    list position — sender_pairs happens to be src-ordered in practice, but
    this does not rely on that.

    Called ONCE at reset() and cached on the env instance (see
    self._episode_partner_idx) — sender_pairs is fixed for the whole episode
    (see the module comment above STATE_DRONE_FEATURES), so recomputing this
    every step would be redundant work reproducing an identical array.
    """
    sender_pairs = obs_json.get("sender_pairs", [])
    out = np.zeros((MAX_USERS_CAP, 2), dtype=np.float32)
    for pair in sender_pairs:
        src, dst = int(pair[0]), int(pair[1])
        if 0 <= src < MAX_USERS_CAP and 0 <= dst < MAX_USERS_CAP:
            out[src, 0] = dst / MAX_USERS_CAP
            out[src, 1] = 1.0
    return out


# ── Per-user raw observation schema ──────────────────────────────────────────
#
# _build_obs (below) replaces the old drone-to-drone placeholder with a
# per-drone, per-USER token set: for each observing drone, one row per user
# THAT DRONE SENSES this step (gated by rssi_valid — has this drone actually
# received a frame from this user), padded to MAX_USERS_CAP with all-zero
# rows. This is the "sensed set" the thesis design calls for — NOT a
# god-view dump of every user's ground-truth state to every drone (RLBridge's
# JSON otherwise gives exactly that: user_positions is one global array with
# no per-drone sensing filter at all).
#
# rssi/rssi_valid are MEASURED: read straight off obs_json["user_rssi"] /
# ["user_rssi_valid"], which RLBridge fills from the actual received-power tag
# (SignalPowerInd) attached to a real arriving frame — never computed from
# positions. Absence (this drone has never received a frame from this user)
# is a genuine sensor state, not a value to be imputed: rssi_valid stays 0.0
# and rssi stays 0.0, and (per the sensed-set gate below) the whole row stays
# all-zero.
#
# Column layout (OBS_FEATURES = 13), values in natural physical units except
# col 5 (see below) — no normalization elsewhere, that is the frozen
# encoder's job in a later pass:
#   0    angle       rad, atan2(dy,dx); observer(drone)-relative bearing to user
#   1    rssi        dBm; MEASURED received power from this user's last frame
#                    this drone actually received (obs_json["user_rssi"]).
#                    Meaningless unless rssi_valid (col 2) is 1.0.
#   2    rssi_valid  1.0 iff this drone has ever measured a frame from this
#                    user, else 0.0. Also gates presence — see col 12.
#   3    interference dBm; RLBridge's analytic sum, in LINEAR mW, of received
#                    power at this observing drone from every OTHER
#                    transmitter (every user + every other drone), same
#                    log-distance model as col 4's jammer noise floor.
#                    Assumes all other transmitters are concurrently active
#                    (ignores CSMA scheduling) — an analytic estimate, not a
#                    measured value.
#   4    noise_ctx   dBm; observing drone's own noise_floor (RLBridge.cc:1024)
#   5    throughput  user_goodput[u] (bytes/sec) iff thru_valid, else 0.0;
#                    when thru_valid, log10-compressed + MinMax-normalized to
#                    [-1, 1] by _normalize_throughput_bps() — see that
#                    function for the exact transform (mirrors PADRE's
#                    throughputUserMinMaxTuple technique in
#                    src/MANET/observation.py). Raw byte counts never reach
#                    this column; it is on the same rough scale as the dB
#                    columns above.
#   6    thru_valid  1.0 iff user_to_drone[u] == i (this drone currently
#                    serves user u), else 0.0.
#   7-10 id_tag      4-dim, deterministic-per-(episode_seed, user index) SHA256
#                    hash vector, range [-1, 1]. RLBridge has no real MAC
#                    address anywhere (only a post-association IPv4) — the
#                    only identity signal available at all is the user's
#                    array index, so THAT is hashed rather than exposed raw.
#                    This is a deliberately-meaningless identity PROBE: it
#                    tests whether a permutation-invariant architecture
#                    ignores it. It is NOT a physical sensor feature. Never
#                    use the raw index anywhere — only this hash.
#   11   id_valid    1.0 iff row is present (column 12). Redundant with it in
#                    this pass since there's no independent identity-
#                    resolution signal, but kept as its own column so future
#                    work (real MAC/ARP-latency modeling) has somewhere to
#                    diverge from the presence bit.
#   12   present     1.0 iff (a) this row index < the actual numUsers this
#                    episode, AND (b) rssi_valid (col 2) — this observing
#                    drone has actually measured a frame from this user. This
#                    is THE per-drone sensed-set mask, now grounded in a real
#                    measurement instead of a computed path-loss estimate.
#                    When 0, every other column in the row is 0.0 — including
#                    angle and noise_ctx, unlike the PADRE reference
#                    (RawUserSet_OW), which leaves those two unmasked for
#                    unsensed users. The deviation is deliberate: unsensed
#                    users must be fully excluded from the token set, not
#                    partially visible.
#
# OMITTED (not emittable from this sim without new C++ instrumentation):
#   overheard_rate — no per-(drone,user) received-frame counter exists.
#   rtt            — no round-trip timing instrumentation exists anywhere.
#   per            — INET's 802.11 MAC tracks retries internally but
#                    RLBridge never subscribes to that signal.
#   sinr_var       — UserAssociationModule's own RSSI (used only to drive
#                    ASSOCIATION decisions, not this observation) is
#                    explicitly deterministic/noise-free ("gives
#                    deterministic, noise-free RSSI for training
#                    stability") — there is no real stochastic process there
#                    to compute a variance over.
# Also excluded by design: distance (redundant with rssi), association_bool
# (a decision, not a sensor), raw sortable id.


# Per-user throughput normalization: log10-compress then MinMax to [-1, 1],
# mirroring PADRE's throughputUserMinMaxTuple technique (see
# src/MANET/observation.py RawUserSet_OW.__init__/_build_user_set_for_active_
# drone: `to_dB = 10*np.log10(x + epsilon)` then normalize_and_transform to a
# tuned (lo, hi) dB range). PADRE's own bounds (30, 68) are calibrated to ITS
# traffic model (bits, solved-flow output) and don't transfer directly here —
# RLBridge instead emits raw MELPE voice bytes/sec. Lower bound (0 dB —
# zero/near-zero traffic) is the original analytical estimate and stays put.
# Upper bound was ORIGINALLY derived analytically from traffic params
# (omnetpp.ini: melpePayloadBytes=21 * packetSizeMultiplier=1,
# UserBonnTrafficApp.h: melpeIntervalMs_=67.5) plus protocol overhead
# (UDP/IP/802.11 headers), assuming a continuously-talking user tops out
# around ~1.5 KB/s (10*log10(1500) ~= 31.8 dB). Collected data (collect_omnet_
# user_sets.py) shows real user_goodput runs hotter than the single-stream
# estimate, so the upper bound is set empirically to 50.0 dB (~30x more
# linear headroom) and the busiest observed users land around +0.7-0.9
# instead of clamping. Re-tune if a new configuration clips.
_THROUGHPUT_USER_MINMAX_DB = (0.0, 50.0)  # 10*log10(bytes/sec)
_THROUGHPUT_EPSILON = 1e-9


def _normalize_throughput_bps(bytes_per_sec: float) -> float:
    """log10-compress + MinMax-normalize a bytes/sec rate to [-1, 1].

    Same shape as PADRE's normalize_and_transform(..., to_dB) — see the
    _THROUGHPUT_USER_MINMAX_DB comment above for the bound derivation.
    """
    db = 10.0 * np.log10(bytes_per_sec + _THROUGHPUT_EPSILON)
    lo, hi = _THROUGHPUT_USER_MINMAX_DB
    normed = 2.0 * (db - lo) / (hi - lo) - 1.0
    return float(np.clip(normed, -1.0, 1.0))


# ── dBm / angle column normalisation ─────────────────────────────────────────
#
# Every actor-facing column must land in [-1, 1], the same contract PADRE's
# RawUserSet_OW.get_observation enforces via normalize_and_transform() +
# initializeMinMaxTuples() (src/MANET/observation.py). PADRE is the reference
# that trains and transfers zero-shot on this exact ST architecture.
#
# Unnormalised, the column magnitudes arriving at
# LearnableUserSetEncoder.to_latent after its sum-pool over ~14 sensed users are
#     noise_floor  1432      <- a per-drone constant
#     rssi         1177
#     interference  580
#     angle          14.7    <- the ONLY column carrying direction
# i.e. the navigational signal would be ~1% of the input it has to compete
# with, and the three larger columns are per-DRONE constants replicated into
# every user row, so after a permutation-invariant sum-pool they encode
# nothing but `number of sensed users` x a constant.
#
# The ranges below are PADRE's, verbatim: signalMinMaxTuple = (-100, -20) dBm
# is shared by power AND interference there (observation.py:306-307), and the
# angle range is (-pi, pi).
_DBM_MINMAX = (-100.0, -20.0)   # PADRE signalMinMaxTuple / interferenceMinMaxTuple


def _normalize_dbm(dbm: float) -> float:
    """MinMax-normalize a dBm reading to [-1, 1], clipped.

    Mirrors PADRE's normalize_and_transform(values, signalMinMaxTuple):
    linear map over _DBM_MINMAX then np.clip to [-1, 1]. Callers must NOT
    pass a sentinel for "no reading" — an invalid slot has to stay exactly
    0.0, because 0 dBm normalises to +1.0 and would read as the strongest
    possible signal.
    """
    lo, hi = _DBM_MINMAX
    return float(np.clip(2.0 * (dbm - lo) / (hi - lo) - 1.0, -1.0, 1.0))


def _normalize_angle(rad: float) -> float:
    """Map a radian bearing to [-1, 1] (PADRE's angle MinMax over (-pi, pi)).

    This is a linear scalar with a branch cut at +-pi, so a
    permutation-invariant MEAN over it is not the circular mean (on recorded
    traces the two differ by >90 deg on 12% of (step, drone) pairs, median
    15 deg). PADRE uses the same encoding; sin/cos (2 columns) is the
    alternative.
    """
    return float(np.clip(rad / np.pi, -1.0, 1.0))


# Metres that map to 1.0 in the peer-offset columns (own-state 2, 3).
#
# Divided by the arena (2000 m), a swarm extent of ~65 m puts the offsets at
# magnitude 0.020-0.024 while the ABSOLUTE position pair next to them sits at
# 0.44-0.60, a 25x mismatch inside the same 7-column own-state block, and the
# actor then barely responds to them (|da| = 0.0017 / 0.0074 when zeroed,
# against 0.042 / 0.038 for the absolute pair). Each column is therefore
# scale-matched to its data.
#
# WHY THIS PAIR SPECIFICALLY. The ST pools peer tokens with a permutation-
# invariant PMA whose attention was measured to be uniform (0.1946 on the own
# token vs 0.2014 on the others, against 1/5 = 0.2000). Under a uniform mean the
# peer-offset column pools to
#       mean_j (jx - ix) / S  ==  (centroid_x - own_x) / S
# i.e. the vector from this drone to the swarm centroid — the ONE per-drone
# geometric quantity that survives the pool intact, and exactly the signal a
# drone needs in order to spread away from its peers. Everything else that
# distinguishes observer i (the self_flag) is diluted to 1/N by the same pool.
# So this pair is not one feature among many; it is the channel the current
# architecture can actually use, so its scale matters.
#
# SCALE CHOICE: r_d, the drone-to-drone backbone radius from the radio model
# (523 m at the frozen 200 mW). It is the natural scale for inter-drone
# geometry — a swarm spread to the edge of its own connectivity reads ~1.0,
# and the user cloud this task must cover has radius ~473 m, i.e. the same
# order. Derived from _max_range_m rather than a literal so a config with a
# different drone TX power rescales with its own radio, matching
# config_knobs.radii_for (which is also what patches task_validation.R_D).
_PEER_OFFSET_SCALE_CACHE: dict[float, float] = {}


def _peer_offset_scale() -> float:
    """Metres mapping to 1.0 in own-state cols 2, 3. See the comment above.

    Reads _mfr.DRONE_TX_mW at CALL time, like _drone_link_components does, and
    calls task_validation._max_range_m directly instead of reading tv.R_D —
    that module attribute is cached at import for the frozen 10/200 mW and is
    stale for any other TX power (task_validation.py's own docstring), whereas
    the function is a pure closed form of the TX power.
    """
    tx = float(_mfr.DRONE_TX_mW)
    scale = _PEER_OFFSET_SCALE_CACHE.get(tx)
    if scale is None:
        import task_validation as tv
        scale = float(tv._max_range_m(tx))
        if not (scale > 0.0):
            raise RuntimeError(
                f"_peer_offset_scale: _max_range_m({tx} mW) returned {scale}; "
                "the peer-offset columns would divide by zero.")
        _PEER_OFFSET_SCALE_CACHE[tx] = scale
    return scale


# ── Peer visibility gated on radio connectivity ──────────────────────────────
#
# Without this option the peer mask (the token's LAST column) means "peer j's
# drone MODULE exists this episode" (liveness, not reachability), and the token
# carries that peer's position and sensed-user set regardless of range, i.e.
# state exchange over an ideal out-of-band link.
#
# With OMNET_PEER_LINK_MASK=1, peer j is visible to observer i only if j is in
# i's CONNECTED COMPONENT of the drone radio graph, and an invisible peer's row
# is zeroed AND unmasked. Multi-hop, not one-hop: if A—B—C then A can learn C's
# state by relaying through B, which is exactly the backbone this thesis is
# about. A drone is always visible to itself.
#
# The link predicate is _link_capacity_bps from max_flow_reward — the SAME
# function the reward's graph is built from, imported rather than reimplemented
# so there is a single radio model. A link counts
# only if it carries capacity in BOTH directions, since state exchange is a
# two-way protocol; with the jammer off the two noise floors are equal and the
# distinction is moot, but it must not silently become one-way when the jammer
# returns.
#
# DEFAULT OFF. Turning it on makes the task strictly harder and changes what
# the observation means. It MUST match between training and evaluation — OMNeTEnv.__init__
# echoes the setting loudly for exactly that reason.
_env_peer_mask = os.environ.get("OMNET_PEER_LINK_MASK", "0")
if _env_peer_mask not in ("0", "1"):
    raise RuntimeError(
        f"OMNET_PEER_LINK_MASK={_env_peer_mask!r} must be '0' or '1'. It "
        f"changes what the actor can observe about its peers; a typo here "
        f"silently trains a different task than it evaluates.")
PEER_LINK_MASK = _env_peer_mask == "1"

# Minimum link rate for a peer to count as reachable. Defaults to the rate the
# configured message actually needs (on-air bytes x 8 / sync interval), so
# message SIZE structurally buys or costs visibility -- which is the mechanism
# the whole compression claim rests on, rather than a billing line.
_MASK_MIN_BPS = float(os.environ.get("OM2G_MASK_MIN_BPS", "0") or 0.0)
if PEER_LINK_MASK and _MASK_MIN_BPS <= 0.0:
    # CALIBRATED, not derived from the raw bit rate. A packet needs far more
    # link margin than its average rate: framing, per-fragment headers and
    # error rates all compound. Anchor instead on the MEASURED radio: at the
    # 131 B message the beacon telemetry shows peer_available_frac = 0.186,
    # which corresponds to the one-conversation grade (TAU_CAP_BPS, ~11.6 kbps
    # -> ~1018 m). Scale linearly with message size, so a 2949 B message needs
    # ~22x the link rate and is deliverable almost nowhere -- which is the
    # delivery collapse behind the measured 140 -> 7.7 cost cliff, now
    # structural in the observation instead of only a billing line.
    _cb = float(os.environ.get("OM2G_CONTROL_ON_AIR_BYTES", "131") or 131.0)
    # Pinned constant, NOT _mfr.TAU_CAP_BPS: that module-level value depends on
    # OM2G env vars being loaded first and is 1e9 when they are not, which
    # would give zero peer visibility. 11614.8 bps is the one-conversation grade.
    _MASK_MIN_BPS = 11614.814814814814 * (_cb / 131.0)

# ── PEER MEMORY ($OM2G_PEER_MEMORY) ─────────────────────────────────────────
# With PEER_LINK_MASK on, an unreachable peer's row is zeroed and unmasked --
# the observer forgets it COMPLETELY the instant it drops out of range. The
# actor has no history of its own (STATE_HISTORY_K feeds only the critic
# state), so every observation of a peer is discarded the moment contact is
# lost. In a swarm that is disconnected ~81% of the time, and where different
# peers are reachable at different moments, that throws away nearly all of the
# information the radio does manage to deliver.
#
# With memory on, an unreachable peer keeps its LAST KNOWN row, and the
# time-since-rx column (event_end+3) is overwritten with the true AGE of that
# row in steps/max_steps. Nothing is invented: the observer only ever sees
# something it genuinely received earlier, correctly labelled as stale. This
# is store-and-forward on the observation side, and it is sound here because
# USERS DO NOT MOVE -- a user fact learned 100 steps ago is still exactly true,
# which is precisely the payload the message carries.
_PEER_MEMORY = os.environ.get("OM2G_PEER_MEMORY") == "1"
_peer_mem_row = {}    # (observer, peer) -> last known row (np.ndarray copy)
_peer_mem_age = {}    # (observer, peer) -> step index when it was recorded


def _reset_peer_memory():
    """Clear between episodes -- memory must never leak across scenarios."""
    _peer_mem_row.clear()
    _peer_mem_age.clear()


# ── Per-drone credit assignment: difference rewards ──────────────────────────
#
# With ONE global scalar broadcast to every drone and a shared critic, every
# drone gets the SAME advantage A_t, so the policy gradient for drone i is
# grad log pi(a_i|o_i) * A_t with A_t identical across i: a good joint outcome
# reinforces whatever EVERY drone did, including the drones that contributed
# nothing, and no gradient differentiates the drones' roles.
#
# The same reasoning drives signal_baselines.greedy_velocities, which scores
# MARGINAL coverage (what this drone adds on top of the rest of the swarm)
# rather than each drone's own total signal, so that the drones do not all
# climb toward the same densest user region. The RL reward uses
#
#     D_i = R(all drones) - R(all drones except i)
#
# the standard difference reward / counterfactual baseline. Note sum_i D_i is
# NOT R -- difference rewards are not a decomposition of the team reward, they
# are a per-agent alignment of it.
#
# COST: one extra LP solve per drone per step. Measured N=5, 17 users: 35 ms per
# step total vs 1.5 ms for the global reward alone, ~80% of it the simplex
# re-solve (max_flow_reward's structure cache already makes all N+1 variants
# warm -- they share one topology and differ only in capacities). That is
# roughly +70% wall clock on an OMNeT-dominated training step.
#
# ON BY DEFAULT. Set OMNET_DIFFERENCE_REWARD=0 for the global team reward
# (the A/B). Either way the GLOBAL reward is always reported in infos under
# "global_reward" -- that, not D_i, is the quantity to score policies on, so
# eval numbers stay comparable to the baselines.
_env_diff_reward = os.environ.get("OMNET_DIFFERENCE_REWARD", "1")
if _env_diff_reward not in ("0", "1"):
    raise RuntimeError(
        f"OMNET_DIFFERENCE_REWARD={_env_diff_reward!r} must be '0' or '1'. It "
        f"changes the training signal; a typo here silently trains against a "
        f"different objective than intended.")
DIFFERENCE_REWARD = _env_diff_reward == "1"

# Potential-based reward shaping (TRAINING ONLY -- see shaping.py). Off unless
# an OM2G_SHAPE_* weight is set. D_i solves credit assignment but not
# exploration: a drone only earns a relay's D_i once it is already standing on
# the relay position, and the empty space it must cross to get there pays
# nothing. Shaping supplies the gradient across that valley without changing
# the optimal policy. Evaluation calls shaping.assert_disabled().
import shaping as _shaping  # noqa: E402  (flat import: omnet_module is not a package)


def _leave_one_out_obs(obs_json: dict, i: int) -> dict:
    """`obs_json` with drone i removed from every DRONE-indexed field.

    compute_position_oracle_reward reads exactly two of them -- `positions` and
    `noise_floor` (via _build_edges) -- and everything else it touches
    (`user_positions`, `sender_pairs`, `user_area`) is user-indexed and so is
    passed through untouched. A shallow dict copy is therefore enough, and it
    keeps the counterfactual from disturbing the caller's obs_json.

    Dropping the entry RE-INDEXES the remaining drones (drone i+1 becomes i).
    That is harmless here because the reward is a symmetric function of the
    drone set -- it never reads a drone index for identity, only positions and
    noise floors -- and it is what keeps every leave-one-out variant sharing a
    single cached LP structure.
    """
    out = dict(obs_json)
    for key in ("positions", "noise_floor"):
        seq = obs_json.get(key)
        if seq is not None:
            out[key] = [v for k, v in enumerate(seq) if k != i]
    return out


def _drone_link_components(drone_positions, noise_floor, n_live: int):
    """Connected components of the drone-to-drone radio graph.

    Returns a list `comp` of length n_live where comp[i] is i's component id.
    An edge i—j exists iff the link carries at least $OM2G_MASK_MIN_BPS in
    BOTH directions (receiver j's noise floor for i->j, receiver i's for j->i).

    ⚠️ THE THRESHOLD IS THE WHOLE POINT. `_link_capacity_bps` is STRICTLY
    POSITIVE AT EVERY DISTANCE (its own docstring: "callers that treated 0 as
    'no edge' now get a full graph"; capacity at the 5373 m arena diagonal is
    8e-6 bps), so a `> 0.0` test would yield one component and every peer
    visible. The edge therefore requires a minimum rate.

    A state message is only DELIVERED if the link can carry it, so the
    threshold defaults to the rate this arm's message actually needs:
    on-air bytes x 8 / sync interval. At 131 B / 1 s that is ~1048 bps.
    """
    adj = [[] for _ in range(n_live)]
    for i in range(n_live):
        for j in range(i + 1, n_live):
            d = math.hypot(drone_positions[i][0] - drone_positions[j][0],
                           drone_positions[i][1] - drone_positions[j][1])
            nf_i = float(noise_floor[i])
            nf_j = float(noise_floor[j])
            tx = _mfr.DRONE_TX_mW
            if (_link_capacity_bps(d, tx, nf_j) >= _MASK_MIN_BPS
                    and _link_capacity_bps(d, tx, nf_i) >= _MASK_MIN_BPS):
                adj[i].append(j)
                adj[j].append(i)

    comp = [-1] * n_live
    c = 0
    for s in range(n_live):
        if comp[s] != -1:
            continue
        stack = [s]
        comp[s] = c
        while stack:                      # BFS/DFS over the component
            v = stack.pop()
            for w in adj[v]:
                if comp[w] == -1:
                    comp[w] = c
                    stack.append(w)
        c += 1
    return comp


def _id_tag_vector(episode_seed: int, user_idx: int, dim: int = 4) -> np.ndarray:
    """Deterministic-per-(episode, user index), unsortable identity probe.

    SHA256 over the (episode_seed, user_idx) pair seeds a fresh PRNG per
    user; the resulting vector has no monotonic relationship to user_idx
    (see test_raw_user_obs.py), so it cannot be used to recover ordering.
    """
    digest = hashlib.sha256(f"{episode_seed}:{user_idx}".encode()).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=dim).astype(np.float32)


# $OM2G_PARTNER_TAG_MODE: "id" tags the partner INDIVIDUALLY (40-way match);
# "group" tags the partner's CLUSTER (4-way). Group is the learnable version:
# a sum-pooling encoder turns a 4-symbol alphabet directly into a demand
# histogram ("my users want cluster C"), whereas matching one of 40 identities
# through an 8-float bottleneck needs a bilinear match the pooling destroys.
# It is also the more defensible model: a cluster label is an address prefix,
# which is what a real relay reads off a destination address.
_PARTNER_TAG_MODE = os.environ.get("OM2G_PARTNER_TAG_MODE", "id").strip().lower()
if _PARTNER_TAG_MODE not in ("id", "group"):
    raise RuntimeError(
        f"OM2G_PARTNER_TAG_MODE={_PARTNER_TAG_MODE!r} must be 'id' or 'group'.")
_CLUSTER_TAG_OFFSET = 100000   # keep group tags disjoint from per-user id tags

# $OM2G_TAG_SWAP: REPLACE the user's own 4-dim identity probe (cols 7-10) with
# the partner's GROUP tag, instead of widening the token to 17.
#
# Why swap rather than add: widening changes the observation width AND the
# message size at once, so a slower-learning arm could not be attributed to
# either. In a measured comparison both halves of the 17-wide pair (message
# sent and message blocked) learned equally slowly, which points at the extra
# width against a fixed 8-float latent rather than at the communication.
#
# Swapping holds token width (13), latent (8) and message size (131 B) exactly
# constant, so the ONLY thing that changes is WHAT those four numbers say:
# "which user is this" (an arbitrary fingerprint) vs "which cluster is this
# user calling" (the +82.8 demand lever). That isolates what the message
# content is worth.
# "1" = full swap (all four columns become the partner's cluster tag).
# "split" = keep TWO dims of the user's own identity and give TWO to the
# partner's cluster. Motivation: the full swap wins its
# own within-pair test decisively (+23.45, t=4.22, sent vs blocked) but scores
# LOWER in absolute terms than the plain arm (142.75 vs 153.28 @48k) because it
# surrenders the identity fingerprint entirely. Two dims is ample for either
# job: the demand tag only has to separate FOUR clusters, and a 2-D random
# probe still separates 40 users well enough to be useful. This buys demand
# without paying the full identity price, and the token/latent/message size all
# stay exactly where they were.
_TAG_SWAP_MODE = (os.environ.get("OM2G_TAG_SWAP") or "").strip().lower()
if _TAG_SWAP_MODE not in ("", "0", "1", "split"):
    raise RuntimeError(
        f"OM2G_TAG_SWAP={_TAG_SWAP_MODE!r} must be unset, '1' or 'split'.")
_TAG_SWAP = _TAG_SWAP_MODE in ("1", "split")


def _user_cluster_labels(user_positions, link_m: float = 500.0):
    """Single-linkage cluster id per user.

    NOT k-means: random init split one physical cluster and merged two others,
    reporting 55% bridging where the truth was 0% (see bridge_metric.py). Users
    sit in ~25 m clumps separated by >=1465 m, so any threshold between those
    scales recovers the true clusters exactly and deterministically.
    """
    n = len(user_positions)
    lab = [-1] * n
    c = 0
    for s0 in range(n):
        if lab[s0] >= 0:
            continue
        stack, lab[s0] = [s0], c
        while stack:
            u = stack.pop()
            ux, uy = user_positions[u]
            for v in range(n):
                if lab[v] >= 0:
                    continue
                vx, vy = user_positions[v]
                if math.hypot(ux - vx, uy - vy) <= link_m:
                    lab[v] = c
                    stack.append(v)
        c += 1
    return lab


def build_raw_user_tokens(obs_json, episode_seed: int) -> np.ndarray:
    """Per-drone, per-user raw token tensor.

    Returns shape (MAX_N, MAX_USERS_CAP, OBS_FEATURES). tokens[i] is drone
    i's own sensed set: row u is user u's token as sensed by drone i
    specifically (see column-layout comment above). Rows for users drone i
    has never measured a frame from (rssi_valid == 0) or that don't exist
    this episode (u >= actual numUsers) are all-zero.
    """
    drone_positions = obs_json.get("positions", [])
    user_positions = obs_json.get("user_positions", [])
    noise_floor = obs_json.get("noise_floor", [])
    interference = obs_json.get("interference", [])
    user_to_drone = obs_json.get("user_to_drone", [])
    user_goodput = obs_json.get("user_goodput", [])
    user_rssi = obs_json.get("user_rssi", [])
    user_rssi_valid = obs_json.get("user_rssi_valid", [])
    n_users = len(user_positions)
    if n_users > MAX_USERS_CAP:
        raise RuntimeError(
            f"build_raw_user_tokens: {n_users} users exceeds MAX_USERS_CAP="
            f"{MAX_USERS_CAP}; bump the constant (see omnetpp.ini numUsers configs)."
        )

    tokens = np.zeros((MAX_N, MAX_USERS_CAP, OBS_FEATURES), dtype=np.float32)

    # user -> conversation partner, from the episode-fixed sender_pairs the
    # simulator already ships (it also feeds the critic state).
    _need_tags = (OBS_FEATURES > 13) or _TAG_SWAP
    _cluster_of = (_user_cluster_labels(user_positions)
                   if (_need_tags and (_PARTNER_TAG_MODE == "group" or _TAG_SWAP))
                   else None)
    _partner_of = [-1] * MAX_USERS_CAP
    if _need_tags:
        for _pair in obs_json.get("sender_pairs", []):
            try:
                _a, _b = int(_pair[0]), int(_pair[1])
            except (TypeError, ValueError, IndexError):
                continue
            # SENDER DIRECTION ONLY. sender_pairs is (sender, partner), and
            # the justification for this field is that a relay reads the
            # DESTINATION ADDRESS out of a packet it received from the sender.
            # Filling the reverse direction would (a) claim a receiver
            # advertises who is calling IT, which no header carries, and (b)
            # be ill-defined -- a user can be the partner of several senders,
            # so the last one would silently overwrite.
            if 0 <= _a < MAX_USERS_CAP:
                _partner_of[_a] = _b

    for i in range(MAX_N):
        if i >= len(drone_positions) or i >= len(noise_floor):
            continue  # drone module absent this step (rare finish() fallback obs)
        dx, dy = drone_positions[i]
        nf_i = float(noise_floor[i])
        interference_i = float(interference[i]) if i < len(interference) else -100.0
        rssi_row = user_rssi[i] if i < len(user_rssi) else []
        rssi_valid_row = user_rssi_valid[i] if i < len(user_rssi_valid) else []

        for u in range(n_users):
            valid = bool(rssi_valid_row[u]) if u < len(rssi_valid_row) else False
            if not valid:
                continue  # row stays all-zero: this drone has never heard this user

            ux, uy = user_positions[u]
            ddx, ddy = ux - dx, uy - dy
            angle = float(np.arctan2(ddy, ddx))
            id_vec = _id_tag_vector(episode_seed, u)

            row = tokens[i, u]
            row[0] = _normalize_angle(angle)
            row[1] = _normalize_dbm(float(rssi_row[u]))
            row[2] = 1.0  # rssi_valid
            row[3] = _normalize_dbm(interference_i)
            row[4] = _normalize_dbm(nf_i)
            if OBS_FEATURES > 13 and u < len(_partner_of):
                pu = _partner_of[u]
                if pu >= 0:
                    # Identity (or GROUP) of the partner ONLY -- never their
                    # position. In group mode this says "my user is calling
                    # cluster C", which is the +82.8 demand lever
                    # (relay_informed 206.2 vs relay_blind 123.4) and is not
                    # inferable from the observer's own sensing.
                    if _cluster_of is not None and pu < len(_cluster_of):
                        row[13:17] = _id_tag_vector(
                            episode_seed, _CLUSTER_TAG_OFFSET + _cluster_of[pu])
                    else:
                        row[13:17] = _id_tag_vector(episode_seed, pu)
            thru_valid = (u < len(user_to_drone)) and (user_to_drone[u] == i)
            raw_bps = float(user_goodput[u]) if thru_valid and u < len(user_goodput) else 0.0
            row[5] = _normalize_throughput_bps(raw_bps) if thru_valid else 0.0
            row[6] = 1.0 if thru_valid else 0.0
            if _TAG_SWAP and u < len(_partner_of) and _partner_of[u] >= 0 \
                    and _cluster_of is not None:
                _grp = _id_tag_vector(
                    episode_seed,
                    _CLUSTER_TAG_OFFSET + _cluster_of[_partner_of[u]])
                if _TAG_SWAP_MODE == "split":
                    # half identity, half demand -- keep both, pay for neither
                    row[7:9] = id_vec[0:2]
                    row[9:11] = _grp[0:2]
                else:
                    # demand, not identity: "this user is calling cluster C"
                    row[7:11] = _grp
            else:
                row[7:11] = id_vec
            row[11] = 1.0  # id_valid
            row[12] = 1.0  # present

    return tokens


# ── Per-drone raw per-event MAC/PHY observation schema ───────────────────────
#
# Raw individual MAC/PHY reception-or-drop events, one row per event, as
# emitted by RLBridge.cc's EventRecord (see that file's RLBridge.h comment for
# the authoritative column derivation). NOT yet compressed or fed to the
# policy — this is the raw set a later frozen encoder will consume, exactly
# parallel to build_raw_user_tokens() above. Column order (EVENT_FEATURES =
# 12) is contractual with the C++ side:
#   0  snir          dB; meaningless unless snir_valid
#   1  snir_valid    1.0 iff col 0 is meaningful
#   2  per           packet error rate
#   3  ber           bit error rate
#   4  size          packet size, bytes
#   5  is_drop       1.0 if this event is a drop, 0.0 if a successful reception
#   6  drop_retry    1.0 if drop reason == RETRY_LIMIT_REACHED
#   7  drop_queue    1.0 if drop reason == QUEUE_OVERFLOW
#   8  drop_corrupt  1.0 if drop reason == INCORRECTLY_RECEIVED
#   9  drop_notforus 1.0 if drop reason == NOT_ADDRESSED_TO_US
#   10 from_drone    1.0 if the sender was a drone, 0.0 if a user
#   11 present       1.0 for a real event, 0.0 for a padding slot
def build_raw_event_tokens(obs_json):
    """Per-drone, per-event raw token tensor + presence mask.

    Returns (tokens, mask): tokens has shape
    (MAX_N, MAX_EVENTS_CAP, EVENT_FEATURES); mask has shape
    (MAX_N, MAX_EVENTS_CAP), True where a row is a real event. RLBridge emits
    only real events (no padding) in obs_json["events"][i]; padding to
    MAX_EVENTS_CAP happens here. A drone with zero events this step, or one
    absent from obs_json["events"] entirely (rare finish() fallback obs),
    gets an all-zero token set with an all-False mask.
    """
    events_by_drone = obs_json.get("events", [])
    tokens = np.zeros((MAX_N, MAX_EVENTS_CAP, EVENT_FEATURES), dtype=np.float32)
    mask = np.zeros((MAX_N, MAX_EVENTS_CAP), dtype=bool)

    for i in range(MAX_N):
        if i >= len(events_by_drone):
            continue  # drone absent this step: all-zero set, all-False mask
        drone_events = events_by_drone[i]
        n = min(len(drone_events), MAX_EVENTS_CAP)
        for e in range(n):
            tokens[i, e] = drone_events[e]
            mask[i, e] = True

    return tokens, mask


# ── Per-drone raw observation schema ─────────────────────────────────────────
#
# The per-agent observation axis is the DRONE set, not the user set: the
# thesis compares architectures on permutation-invariance over DRONES (ST
# attends over drones; FFN flattens MAX_N*DRONE_TOKEN, which must actually
# change when N changes). Each drone's compressed sensed-user set is nested
# INSIDE its token via a permutation-invariant set encoder — never flattened
# raw into the drone axis.
#
# Column layout (DRONE_TOKEN = 28), values in natural physical units except
# columns 7:15 and 15:23 (opaque latents) and col 24 (see below):
#   0    x_abs        peer drone j's ABSOLUTE x, normalised to [-1, 1] over the
#                     simulator's arena_width. Every other positional column
#                     in this token is a DIFFERENCE, so this pair gives the
#                     actor an absolute frame (where in the arena it is, and
#                     where the boundary lies). Mirrors PADRE's
#                     get_node_position / posMinMaxTuple = (0, playground_size).
#   1    y_abs        as col 0, over arena_height.
#   2    dx           m; peer drone j's x minus observer i's x, /arena_width
#   3    dy           m; peer drone j's y minus observer i's y, /arena_height
#   4    rssi         dBm; MEASURED received power from peer drone j's last
#                     frame observer i actually received
#                     (obs_json["peer_rssi"]), read off the physical layer's
#                     SignalPowerInd tag on that frame — NOT computed from
#                     positions. Meaningless unless rssi_valid (col 3) is 1.0.
#   5    rssi_valid   1.0 iff observer i has ever measured a frame from peer
#                     drone j, else 0.0. Under this thesis's baseline
#                     configs drones are ~2 km apart (out of radio range of
#                     each other), so this is 0.0 most of the time — that is
#                     the correct sensor state. NOTE cols
#                     0-3 are populated regardless: peer position is modelled
#                     as always-available state exchange, not as something
#                     gated on a live radio link.
#   6    nf           dBm; peer drone j's own noise_floor
#   7:15 user_latent  8-dim; peer drone j's compressed sensed-user set,
#                     produced by OMNeTUserEncoderStub.encode() over
#                     build_raw_user_tokens()[j] (that function is the
#                     encoder's INPUT and is otherwise unrelated/unchanged).
#   15:23 event_latent 8-dim; peer drone j's compressed per-event set for this
#                     step, produced by FrozenOMNeTEventEncoder.encode() over
#                     cut_event_tokens(build_raw_event_tokens())[j]. The CUT is
#                     mandatory: the encoder (h64_o8) was trained
#                     on receptions + real-failure drops only, with
#                     NOT_ADDRESSED_TO_US excluded. Feeding the
#                     raw firehose is a distribution shift, not an error.
#   23   self_flag    1.0 iff j == i, else 0.0
#   24   relay_thru   drone_fwd[j] (bytes/sec forwarded onward by j),
#                     normalized via _normalize_throughput_bps
#   25   channel_busy fraction in [0,1]; peer drone j's own CCA duty cycle —
#                     fraction of the last RL step its radio's reception
#                     state (IRadio::ReceptionState) was BUSY or RECEIVING,
#                     i.e. a real hardware medium-occupancy measurement, not
#                     derived from packet counts. See RLBridge.cc
#                     droneBusyTime_ / receiveSignal(intval_t).
#   26   time_since_rx seconds; peer drone j's own time since its radio last
#                     completed a successful reception. Counts up from
#                     episode start if it has never received anything yet
#                     (not a sentinel — "N seconds of silence" is exactly the
#                     signal this column exists to carry).
#   27   mask         1.0 iff drone j is live (present in positions/
#                     noise_floor this step), else 0.0 — when 0, every other
#                     column in the row is 0.0. Independent of rssi_valid:
#                     this marks whether peer j's drone MODULE exists this
#                     episode, not whether i has heard it. MUST be the last
#                     column: the ST/DeepSets models (architectures/st_model.py,
#                     deepsets_model.py) hardcode `mask_col = x[..., -1]` and
#                     strip it via `x = x[..., :-1]` before encoding — same
#                     convention as PADRE's per-token layout (ObservationWrappers.py:
#                     "last column is mask").
#
# This is the encoder_mode="frozen" (default) layout, DRONE_TOKEN=28 wide.
# encoder_mode="learned" keeps every column above EXCEPT 7:15: that slot
# widens to the raw MAX_USERS_CAP*OBS_FEATURES user-token block instead of
# an 8-dim pooled latent (mask stays last; DRONE_TOKEN_LEARNED=345 wide) --
# see build_drone_tokens()'s docstring for the exact offsets.


class OMNeTUserEncoderStub:
    """Placeholder pooler standing in for a future frozen DeepSets set-encoder
    over a drone's sensed-user token set.

    Permutation-invariant and mask-respecting by construction (it ignores its
    input and returns zeros), so it is a safe drop-in for the real trained
    encoder later without changing any shape/width bookkeeping. Mirrors
    PADRE's PISAStub (src/MANET/observation.py). Do NOT invent an encoder
    here — the real one is a separate, later piece of work.
    """

    def __init__(self, out_dim: int = 8):
        self.out_dim = out_dim

    def encode(self, user_tokens: np.ndarray) -> np.ndarray:
        # user_tokens: (MAX_USERS_CAP, OBS_FEATURES) — PLACEHOLDER: zeros.
        return np.zeros(self.out_dim, dtype=np.float32)


# Real frozen per-user set encoder (h128_o8, frozen-arena winner). Loaded from
# the repo, path resolved relative to THIS file so it works on every cluster.
# Overridable via $OMNET_USER_ENC_PATH for the FROZEN SHARED ENCODER
# experiment. The default below was fitted on unnormalised dBm tokens; it is
# kept so existing paths resolve, and build_drone_tokens refuses to USE it
# (see the encoder_mode=="frozen" guard). A refitted encoder must be pointed to
# explicitly rather than overwriting the committed file, so that (a) the old
# artifact stays reproducible and (b) which encoder a run used is recoverable
# from its environment instead of from the mtime of a file.
_ENC_PATH = os.environ.get("OMNET_USER_ENC_PATH") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "src", "MANET", "encoders", "omnet", "deepsets_h128_o8.pt")


def _sha256_file(path, _n=8):
    """Short content hash of a weights file, for run provenance."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
try:
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "src", "MANET"))
    from frozen_encoder import FrozenOMNeTUserEncoder
    # $OMNET_USER_ENC_OUT_DIM is the MODEL-side latent width (see
    # architectures/user_encoder.py). It was reused here to size the ENV-side
    # FROZEN encoder during the shared-encoder experiment, where the two always
    # matched. They do not match on the OM2G track: its uncompressed arm sets
    # the model-side width to 325 (the passthrough "encoder" emits the raw
    # padded user block), while this frozen encoder is a fixed .pt file whose
    # to_latent layer is 8 wide -- so sizing it from that variable tries to load
    # an 8-wide tensor into a 325-wide layer and the import dies.
    #
    # The frozen user encoder is unused on any encoder_mode="learned" run
    # (build_drone_tokens' "learned" branch never calls it; only the EVENT
    # encoder stays frozen), and OM2G is always "learned" -- so pinning it to
    # the file's own width there costs nothing and un-couples the two.
    if os.environ.get("OM2G_TRACK") == "1":
        _USER_ENC_OUT_DIM = 8
    else:
        _USER_ENC_OUT_DIM = int(os.environ.get("OMNET_USER_ENC_OUT_DIM", "8"))
    _user_encoder = FrozenOMNeTUserEncoder(_ENC_PATH, dim=13, hidden=128,
                                           out_dim=_USER_ENC_OUT_DIM)
    # ALL architectures must pool with the SAME weights -- that identity is the
    # entire control in the frozen experiment. Printed at import so every run
    # log carries which encoder it used; a mismatch between arms is then visible
    # in the logs rather than inferred from a filename.
    _USER_ENC_SHA = _sha256_file(os.path.abspath(_ENC_PATH))
    # This runs at IMPORT, before any env exists, so it fires even when the run
    # is encoder_mode='learned' and the encoder is never applied
    # (build_drone_tokens only uses user_latents in the 'frozen' branch), so
    # the line says which branch will actually consume it. The
    # "[frozen-enc]" prefix is load-bearing: build_manifest.py greps for it.
    _ENC_MODE_HINT = os.environ.get("OMNET_ENCODER_MODE", "").strip() or "unset"
    _ENC_APPLIED = ("APPLIED" if _ENC_MODE_HINT == "frozen"
                    else f"LOADED BUT NOT APPLIED (encoder_mode={_ENC_MODE_HINT})")
    print(f"[frozen-enc] user encoder sha256={_USER_ENC_SHA} "
          f"out_dim={_USER_ENC_OUT_DIM} path={os.path.abspath(_ENC_PATH)} "
          f"-- {_ENC_APPLIED}",
          flush=True)
except Exception as _e:
    raise RuntimeError(
        f"failed to load frozen OMNeT user encoder from {_ENC_PATH}: {_e}\n"
        f"Expected the h128_o8 DeepSets AE committed at "
        f"src/MANET/encoders/omnet/deepsets_h128_o8.pt (force-added past "
        f".gitignore). This is NOT the stub -- if you want zeros, that is a bug.")


_EV_ENC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "src", "MANET", "encoders", "omnet",
                            "event_deepsets_h64_o8.pt")
try:
    from frozen_encoder import FrozenOMNeTEventEncoder
    _event_encoder = FrozenOMNeTEventEncoder(_EV_ENC_PATH, dim=12, hidden=64,
                                             out_dim=8)
except Exception as _e:
    raise RuntimeError(
        f"failed to load frozen OMNeT event encoder from {_EV_ENC_PATH}: {_e}")

DROP_NOTFORUS_COL = 9   # event token col 9 -- see build_raw_event_tokens()
EVENT_PRESENT_COL = 11


def cut_event_tokens(tokens, mask):
    """Apply the CUT decision to a live event token set.

    Drops NOT_ADDRESSED_TO_US events, which a real 802.11 MAC discards before
    they reach the processor. The frozen event encoder was trained on the cut
    set; feeding it the raw firehose (87% not-for-us) is a distribution
    shift, so this MUST run before _event_encoder.encode().

    NOT applied inside build_raw_event_tokens(): that function is shared with
    collect_omnet_sets.py, and the collected npz files must keep recording the
    raw stream so the KEEP option stays reachable without a re-collection.

    Returns (tokens, mask) of the same shape, with cut rows zeroed and
    unmasked, then compacted left so real events occupy the leading slots.
    """
    out_t = np.zeros_like(tokens)
    out_m = np.zeros_like(mask)
    for i in range(tokens.shape[0]):
        keep = mask[i] & (tokens[i, :, DROP_NOTFORUS_COL] == 0.0)
        n = int(keep.sum())
        if n:
            out_t[i, :n] = tokens[i][keep]
            out_m[i, :n] = True
    return out_t, out_m


def _require_scale(obs_json, key):
    """Read a normalisation scale from the simulator's obs. Never defaults.

    These exist ONLY in the ini/RLBridge. A Python-side copy would be a second
    implementation of a simulator parameter that could disagree with the
    first. If the key is absent the binary predates the emit, so the call
    raises instead of scaling by a default.
    """
    v = obs_json.get(key)
    if v is None:
        raise RuntimeError(
            f"obs JSON has no '{key}'. The OMNeT binary predates the "
            "scale emit. Rebuild: make clean && make -j4 MODE=release. "
            "Refusing to guess a scale -- a wrong one trains plausibly and "
            "silently.")
    v = float(v)
    if not (v > 0 and np.isfinite(v)):
        raise RuntimeError(f"obs JSON '{key}' = {v!r}; must be finite and > 0.")
    return v


def drone_token_width(encoder_mode: str) -> int:
    """Per-drone token width for a given encoder_mode. See DRONE_TOKEN /
    DRONE_TOKEN_LEARNED above. Never guess this from a magic number --
    OMNeTEnv's obs space and build_drone_tokens() both call this so the two
    can never silently disagree."""
    if encoder_mode == "frozen":
        return DRONE_TOKEN
    if encoder_mode == "learned":
        return DRONE_TOKEN_LEARNED
    raise ValueError(f"encoder_mode must be 'frozen' or 'learned', got {encoder_mode!r}")


def build_drone_tokens(obs_json, episode_seed: int, encoder_mode: str = "frozen") -> np.ndarray:
    """Per-drone, per-peer-drone raw token tensor.

    Returns shape (MAX_N, MAX_N, drone_token_width(encoder_mode)). tokens[i]
    is drone i's own view: row j is peer drone j's token as sensed by
    observer i specifically (see column-layout comment above). Rows for
    drones not live this episode (index >= actual live drone count) are
    all-zero, mask=0.

    build_raw_user_tokens() (unchanged in both modes) supplies each peer
    drone's own sensed-user set.
      encoder_mode="frozen" (default): that set is compressed by the frozen
        _user_encoder into an 8-dim user_latent nested at columns 5:13 --
        this path is byte-identical to the pre-encoder_mode code.
      encoder_mode="learned": columns 5:5+MAX_USERS_CAP*OBS_FEATURES instead
        carry the RAW padded per-user token block verbatim, so a model-side
        LearnableUserSetEncoder (architectures/user_encoder.py, obs_layout=
        "omnet") can pool it with real gradients from the RL loss. The event
        encoder stays FROZEN in both modes -- only the user encoder moves
        off the env side.
    """
    if encoder_mode not in ("frozen", "learned"):
        raise ValueError(
            f"encoder_mode must be 'frozen' or 'learned', got {encoder_mode!r}")

    drone_positions = obs_json.get("positions", [])
    noise_floor = obs_json.get("noise_floor", [])
    drone_fwd = obs_json.get("drone_fwd", [])
    channel_busy = obs_json.get("channel_busy", [])
    time_since_rx = obs_json.get("time_since_rx", [])
    # Scales come from the simulator, never from a Python-side constant. A
    # missing key means an OLD BINARY -- fail loudly rather than silently
    # falling back to a default that would quietly mis-scale every observation.
    arena_w = _require_scale(obs_json, "arena_width")
    arena_h = _require_scale(obs_json, "arena_height")
    max_steps = _require_scale(obs_json, "max_steps")
    step_idx = int(obs_json.get("step", 0) or 0)
    peer_rssi = obs_json.get("peer_rssi", [])
    peer_rssi_valid = obs_json.get("peer_rssi_valid", [])
    n_live = min(len(drone_positions), len(noise_floor))

    user_tokens = build_raw_user_tokens(obs_json, episode_seed)

    if encoder_mode == "frozen":
        # _user_encoder is a PRETRAINED .pt whose weights were fitted against
        # unnormalised dBm/radian user tokens. build_raw_user_tokens normalises
        # those columns, so its input distribution does not match its training
        # distribution. Refused unless explicitly re-enabled.
        # Two independent conditions, because the ALLOW flag alone proves
        # nothing: it says "I meant to pick frozen", not "the encoder on disk
        # was refitted on normalised tokens". Requiring an EXPLICIT
        # $OMNET_USER_ENC_PATH as well means the committed default can
        # never be reached by setting one env var.
        if os.environ.get("OMNET_ALLOW_FROZEN_ENCODER") == "1" \
                and not os.environ.get("OMNET_USER_ENC_PATH"):
            raise RuntimeError(
                "OMNET_ALLOW_FROZEN_ENCODER=1 was set but OMNET_USER_ENC_PATH "
                "was not: that combination would run the STALE committed "
                "encoder (fitted on raw dBm) while asserting it is fine. Point "
                "OMNET_USER_ENC_PATH at an encoder refitted on normalised "
                "tokens.")
        if os.environ.get("OMNET_ALLOW_FROZEN_ENCODER") != "1":
            raise RuntimeError(
                "encoder_mode='frozen' is invalid with the token "
                "normalisation: the frozen user encoder was fitted on RAW dBm "
                "tokens and build_raw_user_tokens now emits normalised ones. "
                "Use encoder_mode='learned', or refit the frozen encoder on "
                "normalised tokens and set OMNET_ALLOW_FROZEN_ENCODER=1.")
        user_latents = np.zeros((MAX_N, _user_encoder.out_dim), dtype=np.float32)
        for j in range(min(MAX_N, n_live)):
            user_latents[j] = _user_encoder.encode(user_tokens[j])
        user_block_width = _user_encoder.out_dim
    else:
        # No env-side pooling at all: the raw (MAX_USERS_CAP, OBS_FEATURES)
        # block reaches the model verbatim, one flattened row per peer drone.
        user_block_width = MAX_USERS_CAP * OBS_FEATURES

    # Event latent: cut first (see cut_event_tokens), then pool. present col
    # carries the mask into encode(), same contract as the user encoder.
    # UNCHANGED in both modes -- only the USER encoder becomes learnable.
    _ev_tok, _ev_mask = build_raw_event_tokens(obs_json)
    _ev_tok, _ev_mask = cut_event_tokens(_ev_tok, _ev_mask)
    event_latents = np.zeros((MAX_N, _event_encoder.out_dim), dtype=np.float32)
    for j in range(min(MAX_N, n_live)):
        event_latents[j] = _event_encoder.encode(_ev_tok[j])

    token_width = drone_token_width(encoder_mode)
    tokens = np.zeros((MAX_N, MAX_N, token_width), dtype=np.float32)

    # Column offsets, derived from OWN_STATE_W so there is one place to change.
    user_end = OWN_STATE_W + user_block_width
    event_end = user_end + _event_encoder.out_dim

    # Peer visibility gated on the drone radio graph's connected components,
    # so an unreachable peer's position + sensed-user set are not observed.
    # OFF by default; see PEER_LINK_MASK above.
    comp = (_drone_link_components(drone_positions, noise_floor, n_live)
            if (PEER_LINK_MASK and n_live > 0) else None)

    # Hoisted — constant for the whole token build (and cached across
    # calls per TX power), so the inner loop does no import and no radio math.
    peer_scale = _peer_offset_scale()

    for i in range(MAX_N):
        if i >= n_live:
            continue  # row stays all-zero: observer i is absent this episode
        ix, iy = drone_positions[i]
        rssi_row = peer_rssi[i] if i < len(peer_rssi) else []
        rssi_valid_row = peer_rssi_valid[i] if i < len(peer_rssi_valid) else []

        for j in range(MAX_N):
            if j >= n_live:
                continue  # row stays all-zero: peer j is absent, mask=0
            if (comp is not None and j != i and comp[j] != comp[i]
                    and _PEER_MEMORY and (i, j) in _peer_mem_row):
                # Unreachable, but we have heard this peer before: replay the
                # remembered row and stamp its true age (a received
                # observation, labelled with its age).
                tokens[i, j, :] = _peer_mem_row[(i, j)]
                age = max(0, int(step_idx) - _peer_mem_age[(i, j)])
                tokens[i, j, event_end + 3] = min(1.0, age / max(1, max_steps))
                continue
            if comp is not None and j != i and comp[j] != comp[i]:
                # Peer j is in a different component: observer i cannot reach
                # it by any number of hops. Row stays all-zero with mask=0, so
                # the ST's key_padding_mask drops it from attention AND the
                # data is genuinely absent (an arch that ignores the mask, e.g.
                # FFN's flatten, must not be able to read it either).
                continue

            jx, jy = drone_positions[j]

            row = tokens[i, j]
            # Peer j's ABSOLUTE position, normalised to [-1, 1] over the
            # simulator's own arena. The other positional columns are
            # DIFFERENCES (dx, dy here; the user tokens' bearing), so this pair
            # is the actor's absolute frame (the critic has the same through
            # _build_state's x_norm/y_norm). PADRE, the reference that trains and transfers, feeds
            # every node's absolute position normalised by playground size
            # (ObservationWrappers.get_node_position / posMinMaxTuple).
            row[0] = 2.0 * jx / arena_w - 1.0
            row[1] = 2.0 * jy / arena_h - 1.0
            # Kept alongside the absolute pair: the relative offset is the
            # quantity the perm-invariant peer aggregator actually wants, and
            # subtracting it from two normalised absolutes costs the network a
            # layer of precision it does not need to spend.
            #
            # Scaled by r_d (the inter-drone radio range), NOT by the arena:
            # a ~65 m swarm over a 2000 m arena would put this pair at 0.02
            # next to an absolute pair at 0.5.
            # See _peer_offset_scale for the measurements and why this is the
            # one per-drone channel that survives the PMA's uniform pooling.
            # Clipped to the _TOKEN_ABSMAX contract: two drones further apart
            # than r_d cannot hear each other anyway, so saturating past 1.0
            # discards no actionable geometry.
            row[2] = float(np.clip((jx - ix) / peer_scale, -1.0, 1.0))
            row[3] = float(np.clip((jy - iy) / peer_scale, -1.0, 1.0))
            # Normalise, same contract as build_raw_user_tokens. An
            # invalid/absent peer RSSI must stay EXACTLY 0.0 and never go
            # through _normalize_dbm — 0 dBm would normalise to +1.0, the
            # strongest possible reading. On recorded traces 23% of live peer
            # slots carry no valid RSSI.
            _pv = float(rssi_valid_row[j]) if j < len(rssi_valid_row) else 0.0
            row[4] = (_normalize_dbm(float(rssi_row[j]))
                      if (_pv > 0.0 and j < len(rssi_row)) else 0.0)
            row[5] = _pv
            row[6] = _normalize_dbm(float(noise_floor[j]))
            if encoder_mode == "frozen":
                row[OWN_STATE_W:user_end] = user_latents[j]
            else:
                row[OWN_STATE_W:user_end] = user_tokens[j].reshape(-1)
            row[user_end:event_end] = event_latents[j]
            row[event_end] = 1.0 if j == i else 0.0
            fwd_bps = float(drone_fwd[j]) if j < len(drone_fwd) else 0.0
            row[event_end + 1] = _normalize_throughput_bps(fwd_bps)
            row[event_end + 2] = float(channel_busy[j]) if j < len(channel_busy) else 0.0
            row[event_end + 3] = (float(time_since_rx[j]) / max_steps) if j < len(time_since_rx) else 0.0

            # Remember what was genuinely received, for replay when this peer
            # later drops out of the connected component (see _PEER_MEMORY).
            if _PEER_MEMORY and j != i:
                _peer_mem_row[(i, j)] = tokens[i, j].copy()
                _peer_mem_age[(i, j)] = step_idx
            row[-1] = 1.0  # mask: drone j is live — MUST stay the last column

    # POSITIVE-LISTENING ABLATION (evaluation only; off unless OM2G_MSG_ABLATE
    # is set). Lowe et al., "On the Pitfalls of Measuring Emergent
    # Communication" (AAMAS 2019): agents routinely show positive SIGNALLING
    # without positive LISTENING -- messages carry information about the sender
    # while the receiver ignores them. Any claimed communication win must be
    # shown to collapse when the received messages are destroyed, otherwise a
    # reviewer will attribute it to the shared encoder rather than the channel.
    #
    # tokens[i, j] is observer i's view of peer j, so rows j != i ARE the
    # received messages. The self row is left untouched: this ablates what a
    # drone HEARS, never what it senses itself.
    #   zero    -> received messages carry nothing
    #   shuffle -> messages are real but come from the wrong peer (destroys the
    #              sender-identity binding while preserving message statistics,
    #              which is the sharper of the two controls)
    # What each control can and cannot show:
    #  * `shuffle` permutes WHICH PEER sits in which row. ST and deepsets are
    #    set encoders -- permutation-invariant over peers by construction -- so
    #    this is a no-op by design, kept as a negative control that must
    #    return the un-ablated score.
    #  * `zero` blanks the whole peer row, including the peer's POSITION, so it
    #    shows whether drones need to know about each other at all; it does not
    #    separate the LEARNED MESSAGE from plain peer state.
    #
    # `payload_zero` / `payload_shuffle` ablate ONLY the transmitted-message
    # columns [OWN_STATE_W:event_end] -- the pooled user block and the event
    # latent, i.e. exactly what goes on the air -- and leave the peer's
    # position and own-state intact. payload_shuffle takes each peer's payload
    # from a DIFFERENT observer's view, so the message stays statistically
    # realistic while no longer describing the peer it is attached to; that
    # breaks the content binding without touching peer identity, which is what
    # permutation-invariance made impossible above.
    _abl = os.environ.get("OM2G_MSG_ABLATE", "off").strip().lower()
    # "mask_out" is the no-peers mode. "zero" writes tokens[i,j,:-1], which
    # PRESERVES the final mask column -- and st_model builds
    # key_padding_mask = mask_col < 0.5, so a "zeroed" peer is still ATTENDED
    # as a live peer at arena centre with zero offset and mid-scale noise
    # floor. mask_out zeroes the row INCLUDING the mask column, so the model's
    # own absent-peer path (fully_masked, st_model.py:348) drops them from
    # attention entirely.
    # "position_zero" blanks ONLY the peer position columns and keeps the
    # message, separating the effect of peer GEOMETRY from that of message
    # CONTENT: payload_zero keeps positions and blanks content; this does the
    # opposite.
    # Cols 0-1 are the peer's absolute normalized position, 2-3 the r_d-scaled
    # relative offset; everything from OWN_STATE_W on is the transmitted
    # payload and is left untouched.
    _ABL_MODES = ("off", "zero", "shuffle", "payload_zero", "payload_shuffle",
                  "mask_out", "position_zero")
    if _abl not in _ABL_MODES:
        raise RuntimeError(
            f"OM2G_MSG_ABLATE={_abl!r} must be one of {_ABL_MODES}. A typo "
            f"would silently report an un-ablated score as the ablation.")
    if _abl != "off":
        _rng = np.random.default_rng(episode_seed)
        _pay = slice(OWN_STATE_W, event_end)
        for i in range(MAX_N):
            peers = [j for j in range(MAX_N) if j != i]
            if _abl == "position_zero":
                for j in peers:
                    tokens[i, j, 0:4] = 0.0        # geometry out, message kept
            elif _abl == "mask_out":
                for j in peers:
                    tokens[i, j, :] = 0.0          # INCLUDING the mask column
            elif _abl == "zero":
                for j in peers:
                    tokens[i, j, :-1] = 0.0        # keep the live/mask column
            elif _abl == "shuffle":
                perm = _rng.permutation(peers)
                tokens[i, peers, :-1] = tokens[i, perm, :-1].copy()
            elif _abl == "payload_zero":
                for j in peers:
                    tokens[i, j, _pay] = 0.0
            else:  # payload_shuffle
                # Shuffling a peer's payload across OBSERVERS would be a no-op,
                # because peer j's transmitted message is the same object no
                # matter who is looking at it (only the RSSI columns are
                # observer-specific).
                #
                # The payload is permuted across PEERS instead, holding each peer's
                # position row fixed. That is detectable where the plain
                # row-`shuffle` was not: permuting whole rows is invisible to a
                # set encoder, but permuting only the payload breaks the
                # binding between WHERE a drone is and WHAT it reports, which
                # is precisely the content a listener must be using.
                perm = _rng.permutation(peers)
                tokens[i, peers, _pay] = tokens[i, perm, _pay].copy()

    # cols [user_end:event_end] carry the frozen event-encoder's latent (an
    # unbounded learned output, not a physical dBm/radian value the guard
    # below is designed for). It can overshoot |_TOKEN_ABSMAX| rarely, so
    # only this slice is clipped -- the physical columns stay guarded at full
    # strength below.
    np.clip(tokens[..., user_end:event_end], -_TOKEN_ABSMAX, _TOKEN_ABSMAX,
            out=tokens[..., user_end:event_end])

    # O(1) GUARD. Every column reaching the actor must be O(1). This catches
    # an unscaled column loudly, at the source, instead of as a JSON parse
    # error inside the sim.
    if not np.isfinite(tokens).all():
        bad = sorted(set(np.argwhere(~np.isfinite(tokens))[:, 2].tolist()))
        raise RuntimeError(f"drone token has non-finite values in columns {bad}")
    hot = np.argwhere(np.abs(tokens) > _TOKEN_ABSMAX)
    if hot.size:
        cols = sorted(set(hot[:, 2].tolist()))
        raise RuntimeError(
            f"drone token columns {cols} exceed |{_TOKEN_ABSMAX}| "
            f"(max seen {np.abs(tokens).max():.4g}). Unnormalised input reaching "
            "the actor can produce NaN actions. In encoder_mode="
            f"{encoder_mode!r} this includes the raw per-user token block "
            f"(cols 5:{user_end}, dBm/radians -- see build_raw_user_tokens) as "
            "well as the pooled latents -- scale the column or raise "
            "_TOKEN_ABSMAX deliberately.")

    return tokens


class OMNeTEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv backed by an OMNeT++ subprocess over ZMQ.

    Contract mirrors PADREParallelEnv exactly:
    - possible_agents = [drone_0 .. drone_{MAX_N-1}], fixed at construction
    - reset() obs keyed by ALL possible_agents (TorchRL _reset contract)
    - step() obs keyed by LIVE agents only (TorchRL _step live-signal contract)
    - rewards broadcast the same scalar to ALL possible_agents

    OMNeT++ runs ONE episode per launch. reset() kills the old process and
    spawns a fresh one with an incrementing episodeSeed (never 0).
    """

    metadata = {"render_modes": [], "name": "omnet_parallel_v0"}

    # Distinct seed stride per env instance. See __init__.

    def __init__(self, *, worker_index: int, encoder_mode: str = "frozen"):
        # encoder_mode: "frozen" (default) keeps the per-user set encoder
        # env-side, exactly as before -- byte-identical observations.
        # "learned" instead emits the raw padded user block so a model-side
        # LearnableUserSetEncoder can train it from the RL reward (see
        # build_drone_tokens' docstring). Threaded in from OMNeTClass.
        # get_env_fun via the task config -- see omnet_common.py.
        if encoder_mode not in ("frozen", "learned"):
            raise ValueError(
                f"encoder_mode must be 'frozen' or 'learned', got {encoder_mode!r}")
        self._encoder_mode = encoder_mode

        self.possible_agents = [f"drone_{i}" for i in range(MAX_N)]
        self.agents = list(self.possible_agents)

        # Training-only potential-based shaping; a no-op unless OM2G_SHAPE_* is
        # set. Held per-env because Phi(s) must persist between steps.
        self._shaper = _shaping.Shaper(MAX_N)

        self._obs_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(MAX_N, drone_token_width(encoder_mode)),
            dtype=np.float32,
        )
        self._state_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(STATE_STACKED_DIM,),
            dtype=np.float32,
        )
        self._act_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
        )

        # SEED STRIDE -- one per env INSTANCE, not one per process.
        #
        # Process base (separates concurrent RUNS) + instance offset
        # (separates envs WITHIN a run). Stride 100_000 >> any episode count.
        # A pid-only base would give every collector env in one process the
        # SAME world (TorchRL constructs serial collector envs in one process),
        # and a class counter would restart at 0 in every forked worker under
        # parallel_collection. The instance identity is therefore PASSED IN
        # (worker_index) and never inferred.
        assert isinstance(worker_index, int) and worker_index >= 0, (
            f"worker_index must be a non-negative int, got {worker_index!r}. "
            "It is supplied by OMNeTClass.get_env_fun; a missing value means "
            "the env was constructed outside the sanctioned path.")
        self._worker_index = worker_index
        # Echo the observation-semantics switches, once per env, on worker 0.
        # A train/eval mismatch on either of these scores a policy against an
        # observation it never trained on, and neither is visible in a
        # checkpoint, so the log records them.
        if worker_index == 0:
            print(f"[OMNeTEnv] encoder_mode={encoder_mode} "
                  f"PEER_LINK_MASK={'on' if PEER_LINK_MASK else 'off'} "
                  f"STEP_M={STEP_M} token_width={drone_token_width(encoder_mode)} "
                  f"reward={'difference (D_i)' if DIFFERENCE_REWARD else 'global'}",
                  # Shaping SELF-REPORTS: the fingerprint is printed by the
                  # code that actually consumes the knobs.
                  _shaping.describe(),
                  flush=True)
        # base counter for the seed=None (free-running training) path
        self._episode_counter = ((os.getpid() % 2000) * 1_000_000
                                 + (self._worker_index + 1) * 100_000)
        self._proc = None
        self._log_fh = None
        # Default to a tempfile (not DEVNULL) so a dead child's output always
        # exists to read back when it dies -- overridable via OMNET_LOG_STDOUT.
        if OMNET_LOG_STDOUT:
            self._log_path = OMNET_LOG_STDOUT
        else:
            fd, self._log_path = tempfile.mkstemp(prefix="omnet_stdout_", suffix=".log")
            os.close(fd)
        self._last_obs = None
        # Cached once per episode at reset() -- see _build_episode_partner_index
        # and the module comment above STATE_DRONE_FEATURES for why this must
        # not be rebuilt every step.
        self._episode_partner_idx = None
        # Rolling K-deep single-frame history for the centralized critic's
        # stacked state (Phase 2). Cleared at the start of every reset() --
        # a new episode must not inherit frames from the previous (killed)
        # process. oldest-first order; deque's maxlen enforces len <= K by
        # construction.
        self._state_history = collections.deque(maxlen=STATE_HISTORY_K)
        # Peer memory must never leak across episodes/scenarios.
        _reset_peer_memory()
        # Actual endpoint assigned by OS (random free port) — set in _bind_socket.
        self._endpoint = None

        # ZMQ context lives for the lifetime of the env; sockets are recycled
        # per episode with linger=0 to avoid "address already in use" crashes.
        # Each instance binds a random port so multiple BenchMARL env instances
        # (test_env + collector envs) can coexist without port conflicts.
        self._ctx = zmq.Context()
        self._sock = None

        _live_envs.add(self)


    # ── helpers ──────────────────────────────────────────────────────────────

    def _refresh_live_agents(self, obs_json):
        n_live = len(obs_json.get("positions", []))
        self.agents = [f"drone_{i}" for i in range(n_live)]

    def _attach_event_tokens(self, obs_json):
        """Parse obs_json["events"] into the padded raw event set + mask.

        Stashed onto obs_json itself (not returned) — this is the raw,
        uncompressed per-event MAC/PHY observation. It is NOT wired into the
        drone token or the policy observation (self._build_obs); that comes
        later, behind a frozen encoder, exactly like build_raw_user_tokens().
        """
        event_tokens, event_mask = build_raw_event_tokens(obs_json)
        obs_json["event_tokens"] = event_tokens
        obs_json["event_mask"] = event_mask

    def _build_obs(self, obs_json) -> np.ndarray:
        """
        Build the per-drone raw drone-token tensor, shape
        (MAX_N, MAX_N, drone_token_width(self._encoder_mode)). tensor[i] is
        drone i's own view of every peer drone (including itself) — see the
        column-layout comment above build_drone_tokens() (module level) for
        the full feature schema.

        Each agent i receives tensor[i]. DESIGN INTENT: expanding real
        features requires changing DRONE_TOKEN/DRONE_TOKEN_LEARNED and
        editing build_drone_tokens() only. No other code may hardcode
        feature width — always read from drone_token_width() or tensor.shape.
        """
        return build_drone_tokens(obs_json, self._episode_counter, self._encoder_mode)

    def _build_state(self, obs_json) -> np.ndarray:
        """
        Centralized MAPPO critic's global ("god-mode") state vector, shape
        (STATE_DIM,). See the module comment above STATE_DRONE_FEATURES for
        the field-selection rationale.

        Layout (index -> meaning):
          [0 : 4*MAX_N)                                  drone blocks, block i
              = [x_norm, y_norm, noise_floor_norm, mask]
          [4*MAX_N : 4*MAX_N + 4*MAX_USERS_CAP)           user blocks, block u
              = [x_norm, y_norm, partner_idx_norm, mask]
          [-1]                                            steps_remaining_norm
        """
        assert self._episode_partner_idx is not None, (
            "_build_state called before reset() populated "
            "self._episode_partner_idx -- sender_pairs must be cached at "
            "episode start, see reset().")

        drone_positions = obs_json.get("positions", [])
        noise_floor = obs_json.get("noise_floor", [])
        user_positions = obs_json.get("user_positions", [])
        arena_w = _require_scale(obs_json, "arena_width")
        arena_h = _require_scale(obs_json, "arena_height")
        max_steps = _require_scale(obs_json, "max_steps")
        step = float(obs_json.get("step", 0))

        n_live_drones = min(len(drone_positions), len(noise_floor))
        n_users = len(user_positions)
        if n_users > MAX_USERS_CAP:
            raise RuntimeError(
                f"_build_state: {n_users} users exceeds MAX_USERS_CAP="
                f"{MAX_USERS_CAP}; bump the constant (see omnetpp.ini "
                "numUsers configs).")

        state = np.zeros(STATE_DIM, dtype=np.float32)

        # ── Drone block: [x_norm, y_norm, noise_floor_norm, mask] * MAX_N ──
        for i in range(MAX_N):
            if i >= n_live_drones:
                continue  # drone module absent this step -- row stays zero, mask=0
            base = i * STATE_DRONE_FEATURES
            x, y = drone_positions[i]
            state[base + 0] = x / arena_w
            state[base + 1] = y / arena_h
            state[base + 2] = float(noise_floor[i]) / _NOISE_FLOOR_SCALE
            state[base + 3] = 1.0

        # ── User block: [x_norm, y_norm, partner_idx_norm, mask] * MAX_USERS_CAP ──
        user_base = MAX_N * STATE_DRONE_FEATURES
        partner_idx = self._episode_partner_idx
        for u in range(MAX_USERS_CAP):
            if u >= n_users:
                continue  # row stays zero, mask=0
            base = user_base + u * STATE_USER_FEATURES
            x, y = user_positions[u]
            state[base + 0] = x / arena_w
            state[base + 1] = y / arena_h
            state[base + 2] = partner_idx[u, 0]
            state[base + 3] = partner_idx[u, 1]

        # ── Time: steps remaining / horizon ──
        state[-1] = max(0.0, max_steps - step) / max_steps

        # ── GUARDS: shape-correct-but-constant/NaN must raise ──
        assert state.shape == (STATE_DIM,), \
            f"state shape {state.shape} != ({STATE_DIM},)"
        assert np.isfinite(state).all(), "state has NaN/Inf"
        assert not np.all(state == 0.0), \
            "state is all-zero -- obs_json not read"
        assert np.abs(state).max() < _STATE_ABSMAX, (
            f"state has |value| >= {_STATE_ABSMAX} "
            f"(max={np.abs(state).max():.4g}); normalization is broken")

        return state

    def _push_state_frame(self, obs_json):
        """Build this step's single frame and append it to the rolling
        history (Phase 2). Called exactly once per obs_json -- at the same
        point self._last_obs is set in reset()/step() -- so a frame is never
        pushed twice for one simulator step."""
        self._state_history.append(self._build_state(obs_json))

    def _build_stacked_state(self) -> np.ndarray:
        """Concatenate the rolling K-deep history into one fixed-width
        vector, shape (STATE_STACKED_DIM,) = K * (STATE_DIM + 1).

        Oldest->newest order, stable every call. Missing history at episode
        start (fewer than K frames pushed yet) is zero-padded at the FRONT
        (oldest end) with frame_valid=0.0 -- NOT filled by repeating the
        current frame (see the module comment above STACKED_FRAME_DIM).
        """
        history = self._state_history
        n_valid = len(history)
        assert n_valid <= STATE_HISTORY_K, (
            f"state history has {n_valid} frames > K={STATE_HISTORY_K} -- "
            "deque maxlen invariant violated")
        n_pad = STATE_HISTORY_K - n_valid

        frames = np.zeros((STATE_HISTORY_K, STACKED_FRAME_DIM), dtype=np.float32)
        for slot, frame in enumerate(history):  # oldest -> newest
            frames[n_pad + slot, :STATE_DIM] = frame
            frames[n_pad + slot, STATE_DIM] = 1.0  # frame_valid

        stacked = frames.reshape(-1)

        assert stacked.shape == (STATE_STACKED_DIM,), \
            f"stacked state shape {stacked.shape} != ({STATE_STACKED_DIM},)"
        assert np.isfinite(stacked).all(), "stacked state has NaN/Inf"
        return stacked

    def _bind_socket(self):
        """Close the old socket with linger=0, then bind a fresh REP socket on a free port."""
        if self._sock is not None:
            self._sock.setsockopt(zmq.LINGER, 0)
            self._sock.close()
        self._sock = self._ctx.socket(zmq.REP)
        # Never block forever on recv(): a dead/hung OMNeT child must surface
        # as a loud error within ZMQ_RECV_TIMEOUT_MS, not a silent hang.
        self._sock.setsockopt(zmq.RCVTIMEO, ZMQ_RECV_TIMEOUT_MS)
        # Bind to a random free port; OS assigns it to avoid conflicts when
        # BenchMARL creates multiple env instances (test_env + collector envs).
        port = self._sock.bind_to_random_port("tcp://127.0.0.1")
        self._endpoint = f"tcp://127.0.0.1:{port}"

    def _tail_log(self, n: int = 40) -> str:
        """Last n lines of this instance's OMNeT stdout log, best-effort."""
        try:
            with open(self._log_path, "r", errors="replace") as f:
                return "".join(f.readlines()[-n:])
        except OSError as e:
            return f"(failed to read {self._log_path}: {e})"

    def _recv_obs(self) -> bytes:
        """recv() on the REP socket with a hard timeout.

        A hung ZMQ recv (dead or stuck OMNeT child) would block forever,
        indistinguishable from a slow-but-alive run. Instead: time out, check
        whether the child is actually dead, and raise a loud, specific error.
        """
        try:
            return self._sock.recv()
        except zmq.Again:
            rc = self._proc.poll() if self._proc is not None else None
            if rc is not None:
                raise RuntimeError(
                    f"OMNeT child exited (code {rc}) while Python was waiting "
                    f"for a ZMQ reply.\n"
                    f"--- last 40 lines of {self._log_path} ---\n"
                    f"{self._tail_log(40)}\n"
                    f"--- end log ---\n"
                    "OMNeT child died. Did you `make clean && make -j4`?"
                ) from None
            pid = self._proc.pid if self._proc else None
            cpu = "?"
            if pid is not None:
                try:
                    import subprocess as _sp
                    cpu = _sp.run(["ps", "-p", str(pid), "-o", "pcpu="],
                                  capture_output=True, text=True,
                                  timeout=5).stdout.strip() or "?"
                except Exception:
                    cpu = "?"
            raise RuntimeError(
                f"No ZMQ reply from OMNeT within "
                f"{ZMQ_RECV_TIMEOUT_MS / 1000:.0f}s, but the child process "
                f"(pid {pid}) is still alive. It is stuck/mute, not dead.\n"
                f"child CPU%% = {cpu}. If ~0, it is genuinely hung. If nonzero, "
                f"it is CPU-STARVED and this timeout is a false positive -- "
                f"check node load and raise OMNET_ZMQ_RECV_TIMEOUT_MS."
            ) from None

    def _kill_proc(self):
        """Kill the OMNeT++ process group so no children survive."""
        if self._proc is not None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            try:
                self._proc.wait(timeout=5)
            except Exception:
                pass
            self._proc = None
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    def _launch_omnet(self):
        """Spawn a fresh OMNeT++ process with the current episode seed.

        OMNET_APPTAINER_SIF set -> binary runs INSIDE the container via
        `apptainer exec`; this Python process (padre_env, torch) stays OUTSIDE
        (Option B). Only the CODE tree is bound; the container supplies
        INET/omnetpp, so OMNET_INET_DIR/OMNETPP_DIR must point at IN-CONTAINER
        paths on the cluster. Unset -> bare spawn (laptop native).
        """
        env = os.environ.copy()
        ld = ":".join(filter(None, [
            os.path.join(INET_DIR, "src"),
            os.path.join(OMNETPP_DIR, "lib"),
            env.get("LD_LIBRARY_PATH", ""),
        ]))
        inet_src = os.path.join(INET_DIR, "src")
        # ── Track guard + ini selection (OM2G) ───────────────────────────
        # $OMNET_INI picks the simulator config FILE. Default "omnetpp.ini" is
        # the OMNET track (802.11g, network SmokeNet). "om2g.ini" is the OM2G
        # track (IEEE 802.15.4, network OM2GNet) -- a different experiment
        # whose numbers are NOT comparable; see code/om2g/OM2G_README.md.
        #
        # The two must never be crossed. $OM2G_TRACK=1 is the OM2G opt-in
        # (om2g_track.require_om2g checks the same variable), so the pairing
        # is asserted here, at the ONE place the simulator is actually spawned.
        # A mismatch is a correctness error, not a preference: it would score an
        # 802.15.4 policy against an 802.11g world or vice versa.
        _ini = os.environ.get("OMNET_INI", "omnetpp.ini")
        _is_om2g_ini = "om2g" in _ini.lower()
        _om2g_optin = os.environ.get("OM2G_TRACK") == "1"
        if _is_om2g_ini and not _om2g_optin:
            raise RuntimeError(
                f"OMNET_INI={_ini!r} selects the OM2G track but OM2G_TRACK=1 "
                "is not exported. Export it, or use omnetpp.ini.")
        if _om2g_optin and not _is_om2g_ini:
            raise RuntimeError(
                f"OM2G_TRACK=1 is exported but OMNET_INI={_ini!r} is an OMNET "
                "track config. These are two different experiments and their "
                "results are not comparable -- see code/om2g/OM2G_README.md. "
                "Set OMNET_INI=om2g.ini, or unset OM2G_TRACK.")
        cmd_tail = [
            OMNET_BINARY,
            "-u", "Cmdenv",
            "-c", os.environ.get("OMNET_CONFIG", "validate"),
            "-n", f".:{inet_src}",
            _ini,
            f"--*.rlBridge.episodeSeed={self._episode_counter}",
            f"--*.rlBridge.maxSteps={os.environ.get('OMNET_MAX_STEPS', '1000')}",
            f'--*.rlBridge.pythonEndpoint="{self._endpoint}"',
            "--sim-time-limit=1010s",
            # CLI overrides always win over the ini file in OMNeT++, so this
            # keeps *.numDrones in lockstep with MAX_N (the Python-side drone
            # count) WITHOUT editing the shared omnetpp.ini on disk -- an
            # on-disk edit is a race across concurrent instances at different
            # N (two evaluate_omnet.py runs at N=4 and N=6 stomp on each
            # other's restore). MAX_N is a module global precisely so a
            # caller can override it per run (see evaluate_omnet.py).
            f"--*.numDrones={MAX_N}",
        ]
        # OMNET_EXTRA_ARGS -- same override pattern as --*.numDrones just
        # above, generalised: a sweep over arena size / cluster cap needs to
        # vary --*.rlBridge.arenaWidth, --*.rlBridge.arenaHeight,
        # --**.mobility.constraintAreaMaxX/Y, --*.rlBridge.maxClusters, ...
        # per run, and editing the shared omnetpp.ini on disk has the exact
        # same race as the numDrones case (two sweep cells at different L
        # running concurrently would stomp each other's ini edit). Appended
        # LAST so these win: OMNeT resolves a repeated key by last-match
        # (same rule as duplicate ini entries), so anything in
        # OMNET_EXTRA_ARGS overrides even the fixed entries above it in this
        # list if a caller ever needs to. Empty/unset by default, so every
        # existing caller (which never sets this var) is byte-for-byte
        # unaffected.
        extra_args = os.environ.get("OMNET_EXTRA_ARGS", "")
        if extra_args:
            cmd_tail.extend(shlex.split(extra_args))
        _bin = os.path.join(OMNET_SMOKE_DIR, OMNET_BINARY)
        if not os.path.isfile(_bin) and not OMNET_APPTAINER_SIF:
            raise RuntimeError(
                f"simulator binary missing: {_bin}\n"
                f"Build it: make clean && make -j4 MODE=release  (in {OMNET_SMOKE_DIR})\n"
                f"A missing/corrupt build otherwise surfaces as Python blocking "
                f"forever on ZMQ -- the zombie.")
        self._log_fh = open(self._log_path, "a")

        if OMNET_APPTAINER_SIF:
            cwd = None
            env["APPTAINERENV_LD_LIBRARY_PATH"] = ld
            cmd = [
                "apptainer", "exec",
                "--bind", f"{OMNET_HOST_CODE}:{OMNET_CONTAINER_CODE}",
                "--pwd", os.path.join(OMNET_CONTAINER_CODE, "omnet_smoke"),
                OMNET_APPTAINER_SIF,
                *cmd_tail,
            ]
        else:
            cwd = OMNET_SMOKE_DIR
            env["LD_LIBRARY_PATH"] = ld
            cmd = cmd_tail

        self._proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            process_group=0,
        )

    # ── PettingZoo contract ───────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        # 1. Kill any previous OMNeT++ process; rebind socket (linger=0)
        self._kill_proc()
        self._bind_socket()

        # 2. Episode seed.
        #    seed=N given   -> use it verbatim. REQUIRED for evaluation: two
        #                      policies at "seed 7" must see the SAME episode,
        #                      or paired stats are meaningless.
        #    seed=None      -> increment the counter (training: just needs a
        #                      different layout each episode).
        #    Never 0: RLBridge.cc treats episodeSeed_==0 as "draw a
        #    non-reproducible layout from OMNeT's own RNG" (RLBridge.cc:279).
        #    Also int32-bounded.
        if seed is not None:
            # Collector (parallel training) calls reset(seed=S) with the
            # SAME S on every env -> without a per-instance offset all envs
            # simulate the SAME world (verified: 4 envs, all episodeSeed=71).
            # Evaluator also calls reset(seed=S) but needs S VERBATIM so two
            # policies at "seed 7" see the SAME episode (paired
            # stats). Training launcher sets OMNET_TRAIN_SEED_STRIDE=1;
            # eval never does. Default = verbatim (safe for eval).
            if os.environ.get("OMNET_TRAIN_SEED_STRIDE") == "1":
                self._episode_counter = int(seed) + 1 + (self._worker_index + 1) * 100_000
            else:
                self._episode_counter = int(seed) + 1      # +1 so seed=0 is legal
        else:
            self._episode_counter += 1
        assert 0 < self._episode_counter < 2_147_000_000, \
            f"episodeSeed {self._episode_counter} out of int32 range"

        # SEED-UNIQUENESS TRIPWIRE. Two envs on one seed would simulate one
        # world. Each env claims its seed by
        # creating a file exclusively -- a second claimant is a hard stop, not
        # a warning. Scoped to the run via the collector's parent pid so
        # concurrent runs and reused seeds across episodes never false-fire.
        _claim_dir = os.path.join(
            tempfile.gettempdir(), f"omnet_seeds_{os.getppid()}")
        os.makedirs(_claim_dir, exist_ok=True)
        _claim = os.path.join(_claim_dir, str(self._episode_counter))
        # TorchRL resets each env TWICE at the same seed: once in the parent
        # for spec discovery (batched_envs.py:692 _get_metadata -> the wrapper's
        # _init_env -> _reset_parallel) and once for real in the worker. An env
        # re-claiming its OWN seed is therefore legal; only a DIFFERENT env
        # claiming it is an error. Identity, not mere existence, is the test.
        _me = f"{self._worker_index}"
        try:
            fd = os.open(_claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, _me.encode())
            os.close(fd)
        except FileExistsError:
            with open(_claim) as _fh:
                _owner = _fh.read().strip()
            if _owner == _me:
                pass          # same env, second reset -- expected
            else:
                raise RuntimeError(
                    f"episodeSeed {self._episode_counter} claimed by "
                    f"worker_index={_owner}, now also by "
                    f"worker_index={self._worker_index}. Two envs simulating "
                    "one world: it trains to "
                    "completion on correlated copies with plausible curves. "
                    "Check OMNeTClass.get_env_fun is handing out distinct "
                    "worker_index values.") from None

        # 3. Launch fresh OMNeT++ (REQ client connects to our REP socket)
        self._launch_omnet()

        # 4. Receive first obs
        msg = self._recv_obs()
        obs_json = json.loads(msg.decode("utf-8"))
        self._attach_event_tokens(obs_json)
        # sender_pairs is fixed for the whole episode (see module comment
        # above STATE_DRONE_FEATURES) -- cache it now, once, rather than
        # rebuilding it every _build_state call.
        self._episode_partner_idx = _build_episode_partner_index(obs_json)
        self._last_obs = obs_json
        # New episode -- history must not carry frames over from the
        # previous (killed) process. Clear, then push this episode's first
        # frame (deque length becomes 1, the rest zero-padded until K steps
        # have been pushed).
        self._state_history.clear()
        self._push_state_frame(obs_json)
        self._refresh_live_agents(obs_json)

        # Baseline Phi(s_0) for potential-based shaping. Seeding it here (rather
        # than leaving it None) means the first step's shaping term is measured
        # against the spawn, so a drone that leaves a crowded spawn is credited
        # for it. No-op unless OM2G_SHAPE_* is set.
        self._shaper.reset(obs_json)

        # 5. Return obs keyed by ALL possible_agents (TorchRL _reset contract)
        tensor = self._build_obs(obs_json)
        obs = {f"drone_{i}": tensor[i].copy() for i in range(MAX_N)}
        infos = {agent: {} for agent in self.possible_agents}
        return obs, infos

    def step(self, actions):
        # Order: send action FIRST, then recv new obs, then compute reward
        # from the post-action state — matching the RL (s, a, r, s') convention.

        # 1. Stack MAX_N actions into [[dx,dy], ...] (or [[dx,dy,gate], ...]
        #    when self._act_space has a 3rd, OM2G-learned-gate dimension --
        #    see OM2GEnv.__init__. Every other caller's _act_space stays
        #    shape (2,), so this is a no-op for them.
        #    reward field in the message is 0.0 (OMNeT++ uses it only for logging).
        _n_act = self._act_space.shape[0]
        _default_action = [0.0] * _n_act
        action_list = [
            list(map(float, actions.get(f"drone_{i}", _default_action)))
            for i in range(MAX_N)
        ]
        # A NaN action serialises as bare `NaN`, which is not legal JSON, and
        # nlohmann would fail with a parse error inside the SIMULATOR, far from
        # the actual fault. Fail here, at the fault, with the values visible.
        if not all(np.isfinite(v) for row in action_list for v in row):
            raise RuntimeError(
                f"Non-finite action at step {self._step_count if hasattr(self, '_step_count') else '?'}: "
                f"{action_list}\nThe policy emitted NaN/Inf. This is a MODEL "
                "fault, not a simulator fault -- the sim never saw it. Common "
                "cause: an empty or fully-masked set reaching the encoder "
                "(PADRE gotcha 56 class).")

        # ── $OM2G_RESIDUAL_BASE: learn a CORRECTION to a deployed protocol ──
        # action = base_policy(obs) + gain * learned_action, clipped to the
        # action box. The arm starts AT the baseline instead of at a random
        # walk, so it cannot fall far below it, and the only thing left to
        # learn is the correction the baseline structurally cannot make
        # (demand-aware relaying -- what the message payload carries, VOI
        # 0.466 at this world).
        #
        # WHY THE GAIN EXISTS. The natural phrasing is "delta initialised near
        # zero", but the policy's own action scale at init is sigma ~1.4
        # (measured 2026-09-02 from the checkpoints) -- the full width of the
        # [-1, 1] box. Added raw, that drowns the base and the arm starts at
        # "baseline plus a random walk twice its size", not at the baseline.
        # The gain is what makes the phrase true.
        #
        # Default OFF and byte-identical when unset.
        if _RESIDUAL_BASE:
            try:
                base = _residual_base_velocities(self._last_obs)
            except Exception as exc:      # never silently degrade to a control
                raise RuntimeError(
                    f"$OM2G_RESIDUAL_BASE={_RESIDUAL_BASE} failed to produce "
                    f"base velocities: {exc}. Fix or unset it -- falling back "
                    "to a pure learned action would make this arm a relabelled "
                    "control.") from exc
            for _i in range(min(len(base), len(action_list))):
                _bx, _by = float(base[_i][0]), float(base[_i][1])
                action_list[_i][0] = max(-1.0, min(1.0, _bx + _RESIDUAL_GAIN * action_list[_i][0]))
                action_list[_i][1] = max(-1.0, min(1.0, _by + _RESIDUAL_GAIN * action_list[_i][1]))

        # Scale the unit action to metres/step. See STEP_M at the top of this
        # module for why this multiply has to exist and why it lives here.
        # Checked AFTER the finite-value guard so a NaN is still reported in
        # the model's own action units, not in scaled ones. Only dx/dy are
        # spatial -- a 3rd (OM2G learned-gate) element passes through
        # unscaled, it's a [-1,1] send/skip decision, not a distance.
        if _n_act == 3:
            action_list = [[dx * STEP_M, dy * STEP_M, gate] for dx, dy, gate in action_list]
        else:
            action_list = [[dx * STEP_M, dy * STEP_M] for dx, dy in action_list]

        self._sock.send_string(json.dumps({"type": "action", "actions": action_list, "reward": 0.0}))

        # 2. Receive post-action obs
        msg = self._recv_obs()
        obs_json = json.loads(msg.decode("utf-8"))
        self._attach_event_tokens(obs_json)
        done = bool(obs_json.get("done", False))

        # 3. Compute reward from the NEW (post-action) obs
        #
        # NO bare `except Exception: pass` here. This line computes the ONLY
        # training signal in the system, and a swallowed exception is
        # indistinguishable from a genuinely disconnected network (e.g. a
        # missing LP solver would return 0.0 for every reward).
        #
        # An environment/solver fault is not a zero-reward state -- it is a
        # broken install and must crash. max_flow_reward already raises
        # OracleLPSolveError rather than returning 0.0 for the same reason.
        try:
            reward = float(compute_oracle_max_flow_reward(obs_json))
            per_agent = self._per_agent_rewards(obs_json, reward)
        except (ImportError, ModuleNotFoundError) as e:
            raise RuntimeError(
                f"reward solver unavailable: {e}. This env cannot produce a "
                f"training signal -- fix the install (pip install highspy), "
                f"do not fall back to zero.") from e
        except Exception as e:
            raise RuntimeError(
                f"reward computation failed at step {obs_json.get('step', '?')}: "
                f"{type(e).__name__}: {e}. Refusing to substitute a zero "
                f"reward -- see the note above this line.") from e

        self._last_obs = obs_json
        self._push_state_frame(obs_json)

        # Done path — send ack, clear agents, return immediately.
        #   The ack IS this turn's send (REP alternation: recv→send, done).
        #   No further socket I/O happens after this point on the done step.
        if done:
            self._sock.send_string(json.dumps({"type": "ack"}))
            live_before_clear = list(self.agents)
            self.agents = []
            tensor = self._build_obs(obs_json)
            obs = {f"drone_{i}": tensor[i].copy() for i in range(len(live_before_clear))}
            rewards = dict(per_agent)
            terms = {agent: True for agent in live_before_clear}
            truncs = {agent: False for agent in live_before_clear}
            infos = {agent: {"global_reward": reward}
                     for agent in self.possible_agents}
            return obs, rewards, terms, truncs, infos

        self._refresh_live_agents(obs_json)

        # 4. obs: LIVE agents only (TorchRL uses presence as live/dead signal)
        tensor = self._build_obs(obs_json)
        obs = {f"drone_{i}": tensor[i].copy() for i in range(len(self.agents))}

        # rewards: per-drone difference reward, or the global scalar broadcast
        # to ALL possible_agents when DIFFERENCE_REWARD is off. See the
        # DIFFERENCE_REWARD block above.
        rewards = dict(per_agent)

        # terms/truncs: keyed by currently-live agents
        terms = {agent: False for agent in self.agents}
        truncs = {agent: False for agent in self.agents}

        # infos: all possible_agents. `global_reward` is the TEAM objective and
        # is what policies must be SCORED on -- D_i is a training signal, not a
        # performance number, and summing it would not equal the team reward.
        infos = {agent: {"global_reward": reward}
                 for agent in self.possible_agents}

        return obs, rewards, terms, truncs, infos

    def _per_agent_rewards(self, obs_json: dict, global_reward: float) -> dict:
        """Reward per possible_agent: difference reward, or the global scalar.

        D_i = R(all) - R(all \\ i): what the swarm would lose if drone i were
        not there. See the DIFFERENCE_REWARD block above for the rationale.

        Absent drones (index >= the live count) get 0.0, which is their
        difference reward by definition -- removing a drone that is not there
        changes nothing. Under the global broadcast they keep receiving the
        team scalar.
        """
        if not DIFFERENCE_REWARD:
            return self._shaper.shape(
                {agent: global_reward for agent in self.possible_agents}, obs_json)

        n_live = len(obs_json.get("positions", []))
        out = {}
        for i in range(MAX_N):
            if i >= n_live:
                out[f"drone_{i}"] = 0.0
                continue
            # A one-drone swarm has no counterfactual: removing the last drone
            # leaves no network at all, so R(all \ i) is 0 and D_i == R.
            # cache_tag=i keeps drone i's counterfactual LP in its OWN
            # warm-start slot. Without it all N variants collide on one key
            # (same nd, same topology) and each solve starts from the basis
            # another drone's removal left behind: measured 38.0 ms/step vs
            # 13.5 ms with per-drone slots. Purely a cache hint -- the returned
            # value is identical either way.
            counterfactual = (
                float(compute_oracle_max_flow_reward(
                    _leave_one_out_obs(obs_json, i), cache_tag=i))
                if n_live > 1 else 0.0
            )
            out[f"drone_{i}"] = global_reward - counterfactual
        return self._shaper.shape(out, obs_json)

    def state(self):
        if self._last_obs is None:
            return np.zeros(STATE_STACKED_DIM, dtype=np.float32)
        return self._build_stacked_state()

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def render(self):
        pass

    def __del__(self):
        # Must not leave the ZMQ context to pyzmq's own finalizer: its
        # Context.__del__ calls destroy() -> term(), which BLOCKS until every
        # socket is closed. Finalizer ordering is not guaranteed, so a context
        # collected before its socket can deadlock the interpreter. Tear down
        # explicitly, linger=0.
        try:
            self.close()
        except Exception:
            pass

    def close(self):
        self._kill_proc()
        if self._sock is not None:
            self._sock.setsockopt(zmq.LINGER, 0)
            self._sock.close()
            self._sock = None
        if self._ctx is not None:
            # destroy(linger=0) closes any straggler sockets and returns
            # immediately; bare term() blocks forever on an unclosed socket.
            self._ctx.destroy(linger=0)
            self._ctx = None
