#!/usr/bin/env python3
"""
evaluate_omnet.py — OMNeT++ zero-shot cardinality transfer eval harness.

Drives `n` seeded episodes at a fixed drone
count N in the frozen arena (L=1700, max_clusters=2, droneSpawnMode=cluster —
unchanged), for one of four policies, and writes a per-episode CSV with a
populated seed column (seeds 0..n-1; never 100-199 — that band is reserved).

    python evaluate_omnet.py [CHECKPOINT] --policy {trained,greedy,random,hover} \\
        --num-drones N -n 100 --out results.csv

`hover` is the trivial-control baseline — action == 0 for the whole
episode. Run it alongside random: at the real step size a random walk scores
BELOW hovering, so hover is the floor a learned policy has to clear.

CHECKPOINT is positional and optional — required only for --policy trained.
Checkpoint loading is NOT reimplemented here: see
evaluation.policies.load_trained_policy.

Non-goals: no plotting, no cross-cell aggregation, no
statistics. Median/zeros/Mann-Whitney happen later, from the CSVs this
script writes.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import os
import sys
from pathlib import Path

import numpy as np

# Make omnet_module importable regardless of CWD (mirrors test_omnet_env.py).
_OMNET_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if _OMNET_MODULE_DIR not in sys.path:
    sys.path.insert(0, _OMNET_MODULE_DIR)

# src/ holds the evaluation package (policies.load_trained_policy) plus
# everything it in turn imports (architectures, benchmarl_padre, ...).
_SRC_DIR = os.path.abspath(os.path.join(_OMNET_MODULE_DIR, "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# NOTE ON IMPORT ORDER: omnet_env is deliberately NOT imported here. It reads
# STEP_M and MAX_N from the environment AT IMPORT TIME, so it must not be
# imported until _apply_eval_config() has populated those variables. main()
# imports it (and the helpers below re-import it, which is free once cached).
# Importing it at module scope would pin the eval to the ini defaults -- see
# _apply_eval_config's docstring.

# TWO functions named `greedy_velocities` live in this directory and differ:
# greedy_oracle's probes the GLOBAL position-oracle reward; signal_baselines'
# probes each drone's own LOCAL signal only.
#
# The signal_baselines one is imported: compute_position_oracle_reward IS the
# training reward, so a baseline built on it would be handed the exact quantity
# the policy is scored on. Aliased so the two cannot be swapped by accident.
#
# ⚠️ greedy_velocities reads obs["user_positions"], i.e. all 25 users exactly;
# every learned architecture first compresses that 25x13 block 40:1 into 8
# numbers via a masked SUM (architectures/user_encoder.py
# LEARNED_USER_ENCODER_OUT_DIM=8), from which no individual user is
# recoverable. greedy therefore acts on MORE user information than any policy
# it is compared against.
#   * greedy    -> a reference with exact user positions.
#   * aggregate -> the information-MATCHED comparator (user set reduced to its
#                  first two moments).
from signal_baselines import greedy_velocities as _signal_greedy_velocities  # noqa: E402
from signal_baselines import aggregate_velocities as _aggregate_velocities  # noqa: E402
from signal_baselines import engineered_velocities as _engineered_velocities  # noqa: E402
from signal_baselines import engineered_exact_velocities as _engineered_exact_velocities  # noqa: E402
from signal_baselines import reset_baseline_memory as _reset_baseline_memory  # noqa: E402
from signal_baselines import dispatch_velocities as _dispatch_velocities  # noqa: E402
from signal_baselines import relay_explore_velocities as _relay_explore  # noqa: E402

# ── One radio, one number ────────────────────────────────────────────────────
# signal_baselines is the BELIEF model (what a scripted policy thinks the radio
# does); max_flow_reward is the SCORING model (what the reward actually pays
# for). They must be the same radio (same drone TX power), so any drift
# between the two constants is fatal here.
import signal_baselines as _sb  # noqa: E402
import max_flow_reward as _mfr_chk  # noqa: E402
if abs(_sb.DRONE_TX_mW - _mfr_chk.DRONE_TX_mW) > 1e-9:
    raise RuntimeError(
        "radio constant drift: signal_baselines.DRONE_TX_mW="
        f"{_sb.DRONE_TX_mW} but max_flow_reward.DRONE_TX_mW="
        f"{_mfr_chk.DRONE_TX_mW}. The baselines would PLAN in one radio and be "
        "SCORED in another. Set $OMNET_DRONE_TX_mW (om2g_config.py --shell "
        "exports it) before importing, or fix the constant.")
from signal_baselines import scripted_relay_velocities as _scripted_relay_velocities  # noqa: E402
from signal_baselines import relay_ladder_velocities as _relay_ladder_velocities  # noqa: E402
from signal_baselines import (  # noqa: E402
    greedy_realistic_velocities as _greedy_realistic_velocities,
    greedy_pooled_velocities as _greedy_pooled_velocities,
    spsa_velocities as _spsa_velocities,
)


# ---------------------------------------------------------------------------
# N override — MAX_N (omnet_env.py:116) is the Python-side drone count;
# *.numDrones (omnetpp.ini:21) is the simulator-side one, and the two must
# agree (the comment on omnet_env.py:116 says so). They're kept in lockstep
# by patching MAX_N here and having _launch_omnet() pass "--*.numDrones=
# {MAX_N}" as a command-line override — CLI overrides always win over the
# ini file in OMNeT++, so omnetpp.ini itself is never touched on disk.
#
# omnetpp.ini is never rewritten: it is SHARED mutable state, and two
# evaluate_omnet.py processes running concurrently at different N would race
# on it. The CLI-override approach has no shared file to race on.
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _patched_num_drones(n: int):
    """Run the enclosed block with omnet_env rebound to n drones.

    Patching MAX_N alone is NOT enough. Three module constants are DERIVED
    from MAX_N at import time:

        STATE_DIM         = MAX_N*STATE_DRONE_FEATURES + MAX_USERS_CAP*4 + 1
        STACKED_FRAME_DIM = STATE_DIM + 1
        STATE_STACKED_DIM = STATE_HISTORY_K * STACKED_FRAME_DIM

    _build_state allocates np.zeros(STATE_DIM) but computes
    `user_base = MAX_N * STATE_DRONE_FEATURES` from the PATCHED value, so all
    three must follow MAX_N or the drone and user blocks of the critic state
    would be misaligned. Recompute all three.
    """
    import omnet_env
    saved = {k: getattr(omnet_env, k) for k in
             ("MAX_N", "STATE_DIM", "STACKED_FRAME_DIM", "STATE_STACKED_DIM")}
    omnet_env.MAX_N = n
    omnet_env.STATE_DIM = (n * omnet_env.STATE_DRONE_FEATURES
                           + omnet_env.MAX_USERS_CAP * omnet_env.STATE_USER_FEATURES
                           + omnet_env.STATE_TIME_FEATURES)
    omnet_env.STACKED_FRAME_DIM = omnet_env.STATE_DIM + 1
    omnet_env.STATE_STACKED_DIM = (omnet_env.STATE_HISTORY_K
                                   * omnet_env.STACKED_FRAME_DIM)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(omnet_env, k, v)


# ---------------------------------------------------------------------------
# Task geometry — ONE source of truth, shared with training
# ---------------------------------------------------------------------------

def _eval_task_config(n_drones: int):
    """The v2 task definition, byte-identical to launch_v2_learnable.sh.

    Every field here mirrors that script's task block. If the two ever need to
    differ, that is a task change and belongs in config_knobs.Config, not in a
    second hand-written copy.
    """
    from config_knobs import Config
    # ── OM2G track: the TX powers are NOT the OMNET ones ─────────────────────
    # _apply_eval_config feeds this Config to config_knobs.apply_config, which
    # EXPORTS every field over whatever the caller already set. On the OM2G
    # track every world parameter (TX powers 6.59 / 0.3295 mW, i.e. r_u = 158 m
    # and r_d = 523 m; population; geometry; episode length; spawn; layout;
    # step) is therefore read from om2g_track, so baselines and policies are
    # scored in the same world they were trained in. The OMNET track keeps its
    # own values below.
    _om2g = os.environ.get("OM2G_TRACK") == "1"
    if _om2g:
        import sys as _sys
        _d = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "om2g")
        if _d not in _sys.path:
            _sys.path.insert(0, _d)
        import om2g_track as _T
        _drone_tx, _user_tx = _T.OM2G_DRONE_TX_MW, _T.OM2G_USER_TX_MW
        _n_users = _T.OM2G_NUM_USERS
        _L, _spread = _T.OM2G_ARENA_M, _T.OM2G_USER_SPREAD_EFF
        _ep_steps = _T.OM2G_EP_STEPS
        _spawn = _T.OM2G_SPAWN_MODE
        _clusters, _pcross = _T.OM2G_MAX_CLUSTERS, _T.OM2G_P_CROSS
        _step_m = _T.OM2G_STEP_M   # $OM2G_STEP_M; default 60.0
    else:
        _drone_tx, _user_tx = 200.0, 10.0
        _n_users = 17          # the OMNET track's population, unchanged
        _L, _spread = 2000.0, 350.0
        _ep_steps = 250        # the OMNET track's episode length, unchanged
        _spawn = "staging"     # the OMNET track's spawn, unchanged
        _clusters, _pcross = 2, 0.5   # the OMNET track's layout, unchanged
        _step_m = 60.0                # the OMNET track's step, unchanged
    return Config(
        n_drones=n_drones,
        # The geometry is a track property (read, never restated).
        L=_L,
        # Cluster count and cross-traffic follow $OM2G_CLUSTERS / $OM2G_PCROSS,
        # so a policy is scored in the cluster layout it was trained in.
        max_clusters=_clusters,
        p_cross=_pcross,
        # OMNET track: 350, chosen by a 20-seed sweep over
        # {150, 200, 250, 350, 500} under the sum-of-throughput reward.
        # Below ~250 the staging spawn is already close to optimal, so the
        # greedy baseline cannot separate from HOVER (13-14/20 at every value
        # <= 250). 350 is the smallest value where the intended ordering holds
        # on the gate: greedy beats random 20/20 and hover 19/20. 500 also
        # passes (19/20 both) but moves the task further than needed.
        user_spread_eff=_spread,
        # Step size follows $OM2G_STEP_M, so a step-size sweep is scored at the
        # step it was trained with.
        step_m=_step_m,
        drone_tx_mW=_drone_tx,   # OM2G: 6.59 mW; OMNET: 200 mW -- see above
        user_tx_mW=_user_tx,     # OM2G: 0.3295 mW; OMNET: 10 mW
        # Episode length is a track property -- read it, never restate it.
        ep_steps=_ep_steps,
        # The user population is a track property, read from the track.
        n_users=_n_users,
        # Spawn mode follows $OM2G_SPAWN on the OM2G track.
        drone_spawn_mode=_spawn,
        jammer_activate=999999,   # > ep_steps => jammer never fires
        label="eval_v2",
    )


def _apply_eval_config(n_drones: int) -> dict:
    """Bind the eval world to the TRAINING world, before importing omnet_env.

    Without it, $OMNET_EXTRA_ARGS would be unset and the simulator would use
    omnetpp.ini for all of its geometry (maxSpeed 20 mps, droneSpawnMode
    "cluster", userSpread eff 25.5 m, arena 1700) while the Python side picks
    up whatever is exported in the calling shell.

    apply_config sets $OMNET_EXTRA_ARGS (read fresh per launch) and
    $OMNET_STEP_M / $OMNET_NUM_DRONES (read ONCE at omnet_env import time),
    which is why this must run before that import.
    """
    from config_knobs import apply_config
    applied = apply_config(_eval_task_config(n_drones))
    # Eval scores infos["global_reward"], never the
    # per-drone difference reward, so the N counterfactual LP solves per step
    # would be pure cost here -- measured 35 ms/step vs 1.5 ms. Turned off
    # explicitly rather than by omission: omnet_env reads this at IMPORT time,
    # and this function is the one place guaranteed to run before that import.
    # This does NOT make the eval world differ from training -- D_i is a
    # training signal only, and the team reward is identical either way.
    os.environ["OMNET_DIFFERENCE_REWARD"] = "0"
    # Same FATAL-on-empty-geometry guard launch_v2_learnable.sh has: a run
    # whose geometry falls back to the ini defaults would produce
    # plausible-looking numbers from a different world rather than an error.
    if not applied.get("extra_args") or not os.environ.get("OMNET_EXTRA_ARGS"):
        raise SystemExit(
            "FATAL: apply_config produced no geometry. Refusing to run an eval "
            "whose world would silently fall back to the omnetpp.ini defaults "
            "(a different arena from the task's).")
    return applied


def _close_leaked_omnet_envs() -> int:
    """Close every OMNeTEnv still tracked in omnet_env._live_envs.

    load_trained_policy's reload_from_file path constructs a real BenchMARL
    Experiment (test_env + collector env(s)), which spawns real OMNeT++
    subprocesses. Those envs are never exposed back to this caller, but every
    OMNeTEnv instance self-registers in omnet_env._live_envs on construction
    (the process-leak-prevention WeakSet at the top of omnet_env.py) — walking
    it is the only way to reach and close them before building our own eval
    env. Returns the number of envs closed.
    """
    import omnet_env
    closed = 0
    for env in list(omnet_env._live_envs):
        try:
            env.close()
        except Exception:
            try:
                env._kill_proc()
            except Exception:
                pass
        closed += 1
    return closed


# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------

def _select_actions(policy: str, action_fn, obs_dict: dict, num_drones: int, env) -> dict:
    if policy in ("relay_blind", "relay_informed"):
        # The RELAY LADDER. `relay_blind` is a hand-designed relay heuristic
        # that chooses its bridge from peer POSITIONS alone -- a legitimate
        # baseline rung, and the answer to "just add connectivity to greedy".
        # `relay_informed` chooses by cross-cluster DEMAND, which only the
        # transmitted user state can supply -- an ORACLE, not a baseline.
        # The gap between them is the cash value of the payload's content.
        vels = _relay_ladder_velocities(
            env._last_obs, mode="blind" if policy == "relay_blind" else "informed")
        if len(vels) != num_drones:
            raise RuntimeError(
                f"relay_ladder_velocities returned {len(vels)} velocities, "
                f"expected {num_drones}")
        return {f"drone_{i}": vels[i] for i in range(num_drones)}
    if policy == "scripted_relay":
        # NOT a baseline -- an omniscient hand-placed FALSIFIER. It parks drones
        # in empty space to bridge, which no coverage hill-climb will do, so its
        # score lower-bounds the achievable optimum. If it cannot reach the
        # target, no trained policy will. Never report it as a baseline.
        vels = _scripted_relay_velocities(env._last_obs)
        if len(vels) != num_drones:
            raise RuntimeError(
                f"scripted_relay_velocities returned {len(vels)} velocities, "
                f"expected {num_drones}")
        return {f"drone_{i}": vels[i] for i in range(num_drones)}
    if policy == "engineered":
        # The DEPLOYABLE engineered protocol and the only fair opponent for the
        # learned message: each drone summarises the users IT HEARS into two
        # moments (~24 B), broadcasts them one hop, and hill-climbs on the fused
        # stand-in cloud. Unlike `aggregate` it is neither all-seeing nor free,
        # so it must be charged exactly like the learned arm.
        vels = _engineered_velocities(
            env._last_obs, hop_limit=int(os.environ.get("OM2G_GREEDY_HOPS", 1)))
        if len(vels) != num_drones:
            raise RuntimeError(
                f"engineered_velocities returned {len(vels)} velocities, "
                f"expected {num_drones}")
        return {f"drone_{i}": vels[i] for i in range(num_drones)}
    if policy == "engineered_exact":
        # `engineered`, but peers exchange EXACT heard-user positions
        # instead of a moment summary. Same charge as engineered.
        vels = _engineered_exact_velocities(
            env._last_obs, hop_limit=int(os.environ.get("OM2G_GREEDY_HOPS", 1)))
        if len(vels) != num_drones:
            raise RuntimeError(
                f"engineered_exact_velocities returned {len(vels)} velocities, "
                f"expected {num_drones}")
        return {f"drone_{i}": vels[i] for i in range(num_drones)}
    if policy == "relay_explore":
        # Relay bar with local sensing plus the cost of SEARCHING for the
        # clusters before it can bridge them. The default relay arm reads the
        # global user array (oracle), and the local-sensing one sees 1.2/4
        # clusters on average and so does not bridge.
        vels = _relay_explore(env._last_obs)
        if len(vels) != num_drones:
            raise RuntimeError(
                f"relay_explore returned {len(vels)} velocities, expected {num_drones}")
        return {f"drone_{i}": vels[i] for i in range(num_drones)}
    if policy == "dispatch":
        # Same local sensing as `engineered` (own heard users + peers it can
        # actually hear), but a MULTI-STEP actuator: it picks a cluster and
        # flies there. Every other scripted arm re-decides every step over a
        # 60 m neighbourhood, so it cannot trade a local loss for a distant
        # gain and cannot cash information about anywhere it isn't already.
        # Charged exactly like engineered so the comparison is like for like.
        vels = _dispatch_velocities(
            env._last_obs, hop_limit=int(os.environ.get("OM2G_GREEDY_HOPS", 1)))
        if len(vels) != num_drones:
            raise RuntimeError(
                f"dispatch_velocities returned {len(vels)} velocities, "
                f"expected {num_drones}")
        return {f"drone_{i}": vels[i] for i in range(num_drones)}
    if policy == "aggregate":
        # Information-MATCHED comparator: same hill-climb as greedy but the user
        # set is reduced to its first two moments, which is the class of user
        # information a learned policy can actually recover through the 40:1
        # sum-pool in architectures/user_encoder.py. See signal_baselines.
        vels = _aggregate_velocities(env._last_obs)
        if len(vels) != num_drones:
            raise RuntimeError(
                f"aggregate_velocities returned {len(vels)} velocities, "
                f"expected {num_drones}")
        return {f"drone_{i}": vels[i] for i in range(num_drones)}
    if policy == "greedy":
        # REALISTIC greedy: heard users only + one-hop beaconed peers, and it
        # PAYS for the beacon. This is the deployable baseline. The
        # full-information hill-climb is --policy oracle.
        vels = _greedy_realistic_velocities(
            env._last_obs, hop_limit=int(os.environ.get("OM2G_GREEDY_HOPS", 1)))
        if len(vels) != num_drones:
            raise RuntimeError(
                f"greedy_realistic_velocities returned {len(vels)} velocities, "
                f"expected {num_drones}")
        return {f"drone_{i}": vels[i] for i in range(num_drones)}
    if policy == "greedy_pooled":
        # DIAGNOSTIC ablation: realistic greedy stripped of per-user access,
        # holding the heard-user set fixed. Discriminates "the policy is
        # blindfolded by the 40:1 sum-pool" from "the policy is under-trained".
        vels = _greedy_pooled_velocities(
            env._last_obs, hop_limit=int(os.environ.get("OM2G_GREEDY_HOPS", 1)))
        if len(vels) != num_drones:
            raise RuntimeError(
                f"greedy_pooled_velocities returned {len(vels)} velocities, "
                f"expected {num_drones}")
        return {f"drone_{i}": vels[i] for i in range(num_drones)}
    if policy == "spsa":
        # Zero-communication control: local gradient only, no peer knowledge.
        # Brackets the local-information range from below (see signal_baselines).
        vels = _spsa_velocities(env._last_obs, np.random.default_rng(0))
        if len(vels) != num_drones:
            raise RuntimeError(
                f"spsa_velocities returned {len(vels)} velocities, "
                f"expected {num_drones}")
        return {f"drone_{i}": vels[i] for i in range(num_drones)}
    if policy == "oracle":
        # greedy_velocities needs the RAW simulator obs (positions, user
        # positions, noise floor), not the encoded drone-token tensor obs_dict
        # carries — env._last_obs is exactly the json OMNeTEnv received over
        # ZMQ. It reads ONLY fields the policy's own observation also carries;
        # it never touches the reward.
        vels = _signal_greedy_velocities(env._last_obs)
        if len(vels) != num_drones:
            raise RuntimeError(
                f"greedy_velocities returned {len(vels)} velocities, "
                f"expected {num_drones} (live drone count disagrees with "
                "--num-drones)"
            )
        return {f"drone_{i}": vels[i] for i in range(num_drones)}
    return action_fn(obs_dict, num_drones)


def _perturb(actions: dict, sigma: float, rng) -> dict:
    """Add iid N(0, sigma) to every action component.

    Used to score a NOISE-FREE policy (greedy, hover) under the same execution
    noise a trained policy explores with, so the two are comparable. Unlike an
    offline estimate (perturbing final placements once and rescoring), this is
    CLOSED-LOOP -- noise compounds over 250 steps and greedy re-plans against
    where it actually ended up, which is the regime that matters.
    """
    return {k: v + rng.normal(0.0, sigma, size=np.shape(v)) for k, v in actions.items()}


_DUMP_TRACE = [] if os.environ.get("OM2G_DUMP_TRACE") else None
LAST_TAIL125 = 0.0
LAST_TAIL50 = 0.0


def run_episode(env, policy: str, action_fn, seed: int, num_drones: int,
                trace: list | None = None, action_noise: float = 0.0) -> tuple:
    """Run one seeded episode to completion. Returns (episode_reward, steps).

    No exception handling here beyond what's needed for cleanup elsewhere:
    a reward-computation fault must crash the run, not be
    swallowed as a zero — env.step() already raises rather than returning 0.0.

    `trace`, when given, receives one (seed, step, global_reward) row per step.
    THE EPISODE MEAN CONFLATES TWO DIFFERENT SKILLS: how good a
    configuration the swarm converges to, and how fast it gets there. Greedy
    peaks in 1-14 steps of a 250-step episode, so the mean is ~95% steady
    state and a policy can lose on either axis while the single number says
    only "lower". convergence.py splits the trace back into those two.
    """
    obs, _infos = env.reset(seed=seed)
    # The scripted baselines remember where they last HEARD each user. That
    # memory is about the previous episode's users, who no longer exist.
    _reset_baseline_memory()
    # Re-seed / reset any stateful action source for this episode, so matched
    # seeds hold for the actions as well as for the env (see
    # policies.make_random_action_fn).
    # Heuristics that carry per-episode state (GreedyHeuristic) expose the same
    # reset(env, seed) signature; the stateless ones simply lack the attribute.
    reset_hook = getattr(action_fn, "reset", None)
    if reset_hook is not None:
        reset_hook(env, seed)
    # Seeded off the episode seed so a noisy run is reproducible and so matched
    # seeds compare like with like across policies.
    noise_rng = np.random.default_rng(seed) if action_noise > 0.0 else None
    total_reward = 0.0
    # $OM2G_TAIL_STEPS: the swarm spends the first part of every episode flying
    # into position from an identical start, so a whole-episode mean dilutes any
    # policy difference with a transient every arm shares. Keep the per-step
    # rewards so a converged-window mean can be reported next to the total.
    # NOTE: OMNeT also has its own warm-up transient (a fixed policy once showed
    # a 63% within-episode "gain"), so the tail is valid for comparing ARMS,
    # which all share it, and not as evidence that the swarm improved.
    _step_rewards = []
    steps = 0
    while True:
        actions = _select_actions(policy, action_fn, obs, num_drones, env)
        if noise_rng is not None:
            actions = _perturb(actions, action_noise, noise_rng)
        obs, rewards, _terms, _truncs, infos = env.step(actions)
        # SCORE ON THE TEAM REWARD, ALWAYS. `rewards` is the per-drone
        # DIFFERENCE reward D_i = R(all) - R(all \ i) when
        # OMNET_DIFFERENCE_REWARD is on -- a training signal, not a performance
        # number. infos carries the global reward, which every policy and
        # baseline is scored on.
        # drone_0 is always present in both dicts regardless of done.
        r_step = float(infos["drone_0"]["global_reward"])
        # $OM2G_DUMP_TRACE: record WHERE the swarm actually goes. A policy that
        # loses to RANDOM is placing drones somewhere specific, and the episode
        # mean cannot show where. Records
        # drone positions, user positions and how many users each drone can
        # actually HEAR (r_u = 158 m), which is the input the policy steers on.
        if _DUMP_TRACE is not None:
            _o = getattr(env, "_last_obs", None)
            if _o is not None:
                import numpy as _np
                dp = _np.asarray(_o.get("positions", []), dtype=float)
                up = _np.asarray(_o.get("user_positions", []), dtype=float)
                if dp.size and up.size:
                    d = _np.linalg.norm(dp[:, None, :] - up[None, :, :], axis=-1)
                    # $OM2G_DUMP_OBS: the ENCODED observation drone 0 actually
                    # receives, next to the TRUE world state, so the two can be
                    # checked against each other directly.
                    if steps < 3 and os.environ.get("OM2G_DUMP_OBS"):
                        try:
                            _ob = np.asarray(obs["drone_0"], dtype=float)
                            _DUMP_TRACE.append(dict(
                                seed=int(seed), step=int(steps), kind="obs",
                                obs=_ob.tolist(),
                                true_pos=dp.tolist(),
                                true_users=up.tolist(),
                                arena=float(os.environ.get("OM2G_ARENA", "2000")),
                            ))
                        except Exception as _e:
                            print(f"[dump-obs] {_e}", flush=True)
                    _acts = None
                    try:
                        _acts = [[float(v) for v in np.asarray(actions[k]).ravel()]
                                 for k in sorted(actions)]
                    except Exception:
                        pass
                    _DUMP_TRACE.append(dict(
                        seed=int(seed), step=int(steps),
                        reward=r_step,
                        act=_acts,
                        pos=dp.tolist(),
                        n_heard=[int(x) for x in (d <= 157.82).sum(axis=1)],
                        d_nearest=[float(x) for x in d.min(axis=1)],
                        users=up.tolist() if steps == 0 else None,
                        # The conversation pairing, needed to reconstruct WHICH
                        # bridge carried the demand. Without it the
                        # choice-agreement instrument cannot be computed
                        # post-hoc, and that instrument is the strongest
                        # evidence that the transmitted payload steers
                        # decisions rather than merely existing.
                        pairs=(env._last_obs.get("sender_pairs")
                               if steps == 0 else None),
                        # Per-step unit bearing from each drone to ITS nearest
                        # user. Without this cos(action, bearing) -- the single
                        # diagnostic that named the failure -- can only be
                        # computed on step 0, i.e. 15 of 3750 drone-steps.
                        # Emitting the bearing (2 floats/drone) instead of the
                        # whole user array keeps the dump small.
                        b_nearest=[[float((up[j][0] - dp[i][0]) / max(float(d[i][j]), 1e-9)),
                                    float((up[j][1] - dp[i][1]) / max(float(d[i][j]), 1e-9))]
                                   for i, j in enumerate(d.argmin(axis=1))],
                    ))
        total_reward += r_step
        _step_rewards.append(r_step)
        steps += 1
        if trace is not None:
            trace.append({"seed": seed, "step": steps, "reward": r_step})
        if not env.agents:
            break
    # Published on the module rather than returned, so run_episode keeps its
    # two-value signature and probe_action_scale / placement_analysis /
    # reward_variance keep working unchanged.
    _tail_n = int(os.environ.get("OM2G_TAIL_STEPS", "125"))
    _t125 = _step_rewards[-_tail_n:] or [0.0]
    _t50 = _step_rewards[-50:] or [0.0]
    global LAST_TAIL125, LAST_TAIL50
    LAST_TAIL125 = sum(_t125) / len(_t125)
    LAST_TAIL50 = sum(_t50) / len(_t50)
    return total_reward, steps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="OMNeT++ zero-shot cardinality transfer eval.")
    p.add_argument("checkpoint", nargs="?", default=None,
                   help="BenchMARL .pt checkpoint. Required only for --policy trained.")
    # "hover" (action == 0 for the whole episode) is the trivial-control
    # baseline: spawn_staging places the swarm on the largest user cluster, so
    # holding position is already a good placement, and at the real step size
    # a random walk scores WORSE than hovering. Report both; "beats random"
    # alone is not a claim. "oracle" uses exact per-user state; "aggregate" is
    # the information-matched comparator. See the import-order note at the top.
    p.add_argument("--policy",
                   choices=["trained", "greedy", "greedy_pooled", "oracle", "engineered_exact",
                            "aggregate", "engineered", "dispatch",
                            "relay_explore", "scripted_relay",
                            "relay_blind", "relay_informed",
                            "spsa", "random", "hover"],
                   default="trained")
    # Beacon flood radius for --policy greedy is $OM2G_GREEDY_HOPS (default 1 =
    # one-hop HELLO, RFC 6130). Kept as an env var rather than a CLI flag so it
    # travels with the charge that om2g_baselines_arton.sh derives from it --
    # a flag here and a charge there could drift apart silently.
    p.add_argument("--num-drones", type=int, required=True, dest="num_drones",
                   help="N for this eval cell (3, 4, 5, or 6). "
                        "Patches MAX_N + *.numDrones for the run, restores both after.")
    p.add_argument("--report-sigma", action="store_true", dest="report_sigma",
                   help="Measure the policy's REALISED action sigma over the "
                        "episodes and print it at the end. Realised sigma is "
                        "logged nowhere (not the BenchMARL json, not the wandb "
                        "summary), so every sigma number so far is the INIT "
                        "(scale_mapping=biased_softplus_b) -- and sigma DRIFTS "
                        "during training (0.3 -> 0.61 measured). Ordering arms "
                        "on an x-axis of init sigma is therefore wrong.")
    p.add_argument("--train-num-drones", type=int, default=None,
                   dest="train_num_drones",
                   help="N the CHECKPOINT WAS TRAINED AT, when it differs from "
                        "--num-drones (the cardinality-transfer matrix: an N=5 "
                        "ST/deepsets model evaluated at N=4 or 6). Defaults to "
                        "--num-drones, i.e. unchanged behaviour. See the "
                        "cross-N note in main().")
    p.add_argument("-n", "--episodes", type=int, default=100,
                   help="Number of seeded episodes (default 100).")
    p.add_argument("--seed-start", type=int, default=0, dest="seed_start",
                   help="First seed; episodes use seed_start..seed_start+n-1 "
                        "(default 0). Exists to shard ONE cell across array "
                        "tasks -- each task takes a disjoint seed slice and the "
                        "CSVs concatenate, since every row carries its own seed. "
                        "SEED DISCIPLINE (data.md): 0-99 is the "
                        "evaluation band and the only band a reported number may "
                        "come from; 100-199 is tuning-only. Straying past 99 is "
                        "a protocol violation, so it is refused rather than "
                        "silently allowed.")
    p.add_argument("--out", type=str, required=True, help="Output per-episode CSV path.")
    p.add_argument("--trace", type=str, default=None,
                   help="Optional per-STEP CSV (seed,step,reward). Feeds "
                        "convergence.py, which reports the value a policy "
                        "converges to and how fast it gets there -- the two "
                        "quantities --out's episode mean averages together.")
    p.add_argument("--device", type=str, default=None,
                   help="Compute device for --policy trained (default: cuda if available else cpu).")
    p.add_argument("--action-noise", type=float, default=0.0, dest="action_noise",
                   help="Add iid N(0, SIGMA) to every action before stepping. Lets a "
                        "noise-free policy (greedy/hover) be scored under the same "
                        "execution noise a trained policy explores with.")
    p.add_argument("--stochastic", action="store_true",
                   help="Sample actions at the policy's learned sigma instead of "
                        "executing the distribution mean. Scores the policy in the "
                        "noise regime it was TRAINED under (--policy trained only).")
    a = p.parse_args()
    if a.seed_start < 0:
        p.error(f"--seed-start {a.seed_start} is negative.")
    _last = a.seed_start + a.episodes - 1
    if _last > 99:
        p.error(
            f"seeds {a.seed_start}..{_last} reach past 99 into the tuning band "
            "(100-199). Reported numbers come from seeds 0-99 only; mixing the "
            "bands invalidates the cell. Reduce -n or --seed-start.")
    return a


def _write_csv(path: str, rows: list) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "episode_reward", "steps",
                                            "tail125_mean_step", "tail50_mean_step"])
        writer.writeheader()
        writer.writerows(rows)


def _write_dump_trace():
    if _DUMP_TRACE is None:
        return
    import json as _json
    path = os.environ["OM2G_DUMP_TRACE"]
    with open(path, "w") as fh:
        _json.dump(_DUMP_TRACE, fh)
    print(f"[dump] wrote {len(_DUMP_TRACE)} rows -> {path}", flush=True)


def main():
    args = parse_args()
    import atexit as _atexit
    _atexit.register(_write_dump_trace)

    # Reward shaping is a TRAINING-only signal. A shaped evaluation score is not
    # comparable to any baseline, and a run name could then assert a config the
    # code did not apply -- so fail loudly here rather than emit a number. See omnet_module/shaping.py.
    import shaping as _shaping
    _shaping.assert_disabled("evaluation")

    if args.episodes > 100:
        raise SystemExit(
            f"--episodes={args.episodes} would use seeds 0..{args.episodes - 1}, "
            "spilling into the reserved 100-199 band. Seeds 0-99 are the "
            "evaluation band; n=100 per cell."
        )

    # ------------------------------------------------------------------
    # Bind the eval world to the training world. MUST happen before the first
    # `import omnet_env` anywhere in this process (that module reads STEP_M
    # and MAX_N at import time), hence before load_trained_policy below, which
    # imports it transitively.
    # ------------------------------------------------------------------
    # CROSS-N LOAD. reload_from_file rebuilds the WHOLE experiment
    # -- actor AND centralised critic -- from the constants installed right now,
    # then force-loads the checkpoint's tensors into it. The MAPPO critic reads
    # the global state, whose width is 4*N + 102, so binding the world to the
    # EVAL N before the load makes a cross-N load fail on a shape mismatch in a
    # module that eval never even uses (get_policy_for_loss returns the actor;
    # the critic is discarded). Measured: N=4 built 118, checkpoint had 122.
    #
    # So bind at the TRAINING N for the load, then rebind at the eval N for the
    # rollout. The actor is the only thing that survives, and whether IT can
    # take a different peer count is exactly the question the transfer matrix
    # asks -- a permutation-invariant encoder (ST, deepsets) should load and
    # run; a position-dependent one (FFN) should fail, and that failure is the
    # RESULT. Do not pad an FFN to make a cross-N load succeed.
    _train_n = (args.train_num_drones if args.train_num_drones is not None
                else args.num_drones)
    applied = _apply_eval_config(_train_n)
    if _train_n != args.num_drones:
        print(f"[config] CROSS-N: loading checkpoint at its training N="
              f"{_train_n}, evaluating at N={args.num_drones}", flush=True)
    print(f"[config] eval world bound via config_knobs.apply_config"
          f"{'  [TRACK=OM2G]' if os.environ.get('OM2G_TRACK') == '1' else ''}",
          flush=True)
    print(f"[config]   OMNET_EXTRA_ARGS = {applied['extra_args']}", flush=True)
    print(f"[config]   step_m={applied.get('step_m')} "
          f"drone_tx_mW={applied.get('drone_tx_mW')} "
          f"user_tx_mW={applied.get('user_tx_mW')} "
          f"r_u={applied.get('r_u')} r_d={applied.get('r_d')}", flush=True)

    import omnet_env  # noqa: F401  (import-time env reads now satisfied)

    # ------------------------------------------------------------------
    # Build the action source. Loading happens BEFORE the N patch below:
    # a trained checkpoint's own (possible) env construction during load
    # must see the TRAINING N (module default = 5), not this eval's N.
    # ------------------------------------------------------------------
    action_fn = None
    if args.policy == "trained":
        if args.checkpoint is None:
            sys.exit("error: --policy trained requires a checkpoint argument")
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        from evaluation.policies import load_trained_policy

        ckpt_path = str(Path(args.checkpoint).resolve())
        print(f"Loading checkpoint: {ckpt_path}", flush=True)
        print(f"[eval] action mode = "
              f"{'STOCHASTIC (sampled at learned sigma)' if args.stochastic else 'deterministic (distribution mean)'}",
              flush=True)
        _sigma_sink = [] if args.report_sigma else None
        action_fn, task_config = load_trained_policy(ckpt_path, device,
                                                     stochastic=args.stochastic,
                                                     sigma_sink=_sigma_sink)
        print(f"Policy loaded. Training-time task config: {task_config}", flush=True)

        n_closed = _close_leaked_omnet_envs()
        print(f"[load] closed {n_closed} OMNeT env(s) spawned while loading "
              "the checkpoint", flush=True)

    elif args.policy == "random":
        from evaluation.policies import make_random_action_fn
        # Seeded per episode by run_episode's reset hook, so the stream is
        # reproducible at matched seeds; this initial seed only covers the
        # (impossible) case of an episode loop that never calls reset.
        action_fn = make_random_action_fn(args.num_drones, seed=args.seed_start)

    elif args.policy == "hover":
        from evaluation.policies import make_hover_action_fn
        action_fn = make_hover_action_fn(args.num_drones)

    # policy == "greedy": action_fn stays None; _select_actions special-cases it.

    # ------------------------------------------------------------------
    # Drive n seeded episodes at --num-drones, in the frozen arena.
    # ------------------------------------------------------------------
    results = []
    trace: list | None = [] if args.trace else None
    with _patched_num_drones(args.num_drones):
        _enc_mode = os.environ.get("OMNET_ENCODER_MODE", "frozen")
        # OM2G track: the policy was TRAINED against OM2GEnv, whose obs carries
        # per-peer message slots plus (in e2e codec mode) the inbox -- 6296 wide
        # at L=2 vs the base env's 700, so an OM2G checkpoint must be evaluated
        # in OM2GEnv (the width assert refuses the base env). Mirrors the
        # training path in src/benchmarl_padre/omnet_common.py.
        if os.environ.get("OM2G_TRACK") == "1":
            import sys as _sys
            _om2g_dir = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "om2g")
            if _om2g_dir not in _sys.path:
                _sys.path.insert(0, _om2g_dir)
            from om2g_env import OM2GEnv  # noqa: E402
            env = OM2GEnv(worker_index=0, encoder_mode=_enc_mode)
        else:
            env = omnet_env.OMNeTEnv(worker_index=0, encoder_mode=_enc_mode)
        try:
            for seed in range(args.seed_start, args.seed_start + args.episodes):
                # --stochastic draws from torch's global RNG, so pin it per
                # episode: without this the arm is unreproducible and depends
                # on how seeds were chunked across parallel workers.
                if args.stochastic:
                    import torch as _torch
                    _torch.manual_seed(seed)
                reward, steps = run_episode(env, args.policy, action_fn, seed,
                                            args.num_drones, trace=trace,
                                            action_noise=args.action_noise)
                results.append({"seed": seed, "episode_reward": reward, "steps": steps,
                                "tail125_mean_step": LAST_TAIL125,
                                "tail50_mean_step": LAST_TAIL50})
                print(f"  ep seed={seed:4d}  reward={reward:12.4f}  steps={steps:4d}", flush=True)
        finally:
            env.close()

    if args.report_sigma:
        _ss = locals().get("_sigma_sink")
        if _ss:
            _mean = sum(_ss) / len(_ss)
            _var = sum((x - _mean) ** 2 for x in _ss) / max(len(_ss) - 1, 1)
            print(f"\nREALISED SIGMA: mean={_mean:.4f} sd={_var ** 0.5:.4f} "
                  f"min={min(_ss):.4f} max={max(_ss):.4f} n={len(_ss)} forward passes",
                  flush=True)
        else:
            print("\nREALISED SIGMA: unavailable -- no scale-like key found in the "
                  "policy output (see _sigma_from_td). NOT reporting a guess.",
                  flush=True)

    _write_csv(args.out, results)
    print(f"\nWrote {len(results)} episode(s) to {args.out}", flush=True)

    if trace is not None:
        Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
        with open(args.trace, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["seed", "step", "reward"])
            w.writeheader()
            w.writerows(trace)
        print(f"Wrote {len(trace)} step(s) to {args.trace}", flush=True)


if __name__ == "__main__":
    main()
