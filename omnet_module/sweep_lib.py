#!/usr/bin/env python3
"""
sweep_lib.py — shared probe helpers for the arena/config sweeps.

WHY THIS EXISTS
----------------
sweep_arena.py (commit d376bb2e) defined P0v (random-velocity baseline),
the greedy-trace wrapper, and the cluster/void-geometry helpers as private
functions. The Tier-B config sweep (sweep_config.py) needs EXACTLY the same
logic against a wider knob set, and two copies of rollout logic in two
files can drift apart. So this module is the single copy; both
sweep_arena.py and sweep_config.py import it instead of each carrying their
own.

Extracted VERBATIM from sweep_arena.py (commit d376bb2e) -- no behaviour
change, this is a pure move.
"""
from __future__ import annotations

import math
import random

import numpy as np

# std of a single Uniform(-1,1) draw = 1/sqrt(3) ~= 0.5774. Named so the
# 0.5774 in diffusive_reach_m below is traceable to its source, not a magic
# number that looks suspiciously like it was typed from memory.
DIFFUSIVE_STD_UNIFORM = 1.0 / math.sqrt(3.0)


def diffusive_reach_m(step_m: float, n_steps: int) -> float:
    """Expected net-displacement SCALE of an n_steps 2-D random walk whose
    per-tick velocity is Uniform(-1,1)^2 * step_m (independent per axis).

    Per-axis displacement after n_steps is a sum of n_steps iid Uniform(-step_m,
    step_m) draws; its std is step_m * DIFFUSIVE_STD_UNIFORM * sqrt(n_steps).
    Used as the "can random wandering plausibly cross the void between two
    clusters in one episode" yardstick (R1/R2 in sweep_arena.py, A1/B-gates
    in the Tier A/B config sweep) -- NOT a rigorous first-passage-time bound,
    just the natural length scale of the walk.
    """
    return step_m * DIFFUSIVE_STD_UNIFORM * math.sqrt(n_steps)


def probe_random_velocity(rng: random.Random, start_positions: list[list[float]],
                          users: list[list[float]], areas: list[int],
                          pairs: list[list[int]], L: float,
                          steps: int = 1000, step_m: float = 20.0,
                          eval_every: int = 10, tv=None,
                          collect_records: bool = False):
    """P0v — from the real episode start (spawn_staging), every drone takes
    an INDEPENDENT uniform-random velocity in [-1,1]^2 * step_m each tick,
    clipped to the arena box, for `steps` ticks (1000 = one full training
    episode). This is the actual RL random baseline: the number a random-
    action policy earns over an episode, time-averaged -- not the score of
    one random final placement (that is P0, task_validation.probe_random).

    Reward is evaluated only every `eval_every` ticks: compute_position_
    oracle_reward's LP solve dominates cost, not the position update, so
    scoring all 1000 ticks would be ~10x the LP-solve cost for a mean
    estimate that does not need per-tick resolution.

    `tv` is the task_validation module (passed in, not imported here) so
    callers that have mutated tv.score's underlying reward constants for a
    non-default config (dual-site TX-power / speed knobs) are scored
    against THEIR OWN state, not a fresh import that would re-read the
    frozen defaults.

    collect_records: False (the
    default) reproduces the ORIGINAL 2-tuple return exactly, byte for byte
    -- every existing caller (sweep_arena.py) is unaffected. True additionally
    calls tv.compute_position_oracle_reward directly (not tv.score, which
    discards per_commodity_tau) at each evaluated tick and returns a THIRD
    element: a list of (reward, connected_count) pairs, where
    connected_count = the number of commodities with tau_k > 0 at that tick.
    This is what variance_decomposition() below needs to separate "reward
    moved because a NEW commodity connected" from "reward moved because an
    ALREADY-connected commodity's throughput changed" -- the distinction
    B3_gradient_rich now gates on (see sweep_config.py). No extra LP solves
    either way: per_commodity_tau is already computed inside the SAME call
    compute_position_oracle_reward makes; tv.score just does not expose it.

    Returns (time_averaged_reward, fraction_of_evaluated_ticks_with_zero_reward)
    if collect_records is False, else that pair PLUS the per-tick records list.
    """
    pos = [p[:] for p in start_positions]
    n = len(pos)
    rewards = []
    records: list[tuple[float, int]] = []
    for t in range(steps):
        for i in range(n):
            vx = rng.uniform(-1.0, 1.0)
            vy = rng.uniform(-1.0, 1.0)
            pos[i][0] = min(L, max(0.0, pos[i][0] + vx * step_m))
            pos[i][1] = min(L, max(0.0, pos[i][1] + vy * step_m))
        if t % eval_every == 0:
            if collect_records:
                obs = {"positions": [p[:] for p in pos], "user_positions": users,
                      "user_area": areas, "sender_pairs": pairs}
                result = tv.compute_position_oracle_reward(obs)
                r = result["reward"]
                connected = sum(1 for tau in result["per_commodity_tau"].values() if tau > 0)
                records.append((r, connected))
                rewards.append(r)
            else:
                rewards.append(tv.score(pos, users, areas, pairs))
    if not rewards:
        return (0.0, 0.0, []) if collect_records else (0.0, 0.0)
    mean_r = float(np.mean(rewards))
    frac_zero = float(np.mean([r == 0.0 for r in rewards]))
    if collect_records:
        return mean_r, frac_zero, records
    return mean_r, frac_zero


def variance_decomposition(records: list[tuple[float, int]]) -> dict:
    """Decompose Var(reward) over a P0v rollout into a CONNECTIVITY
    component (how much of the variance comes from the number of connected
    commodities changing) and a THROUGHPUT component (how much comes from
    reward moving while connectivity is HELD FIXED) -- law of total
    variance, exact (up to sampling), with `connected_count` as the
    conditioning variable:

        Var(reward) = Var(E[reward | connected_count])       <- connectivity
                    + E[Var(reward | connected_count)]        <- throughput

    This is the direct, non-proxy version of "is there a continuous slope to
    climb, or a plateau plus a cliff" (the ORIGINAL frac_zero_steps_P0v gate
    tried to answer this indirectly and could not -- see sweep_config.py's
    B3_gradient_rich comment for the measurement that showed why). A config
    dominated by the connectivity term is ~all cliff (reward changes because
    a NEW commodity connects, in ~18.5-unit quanta); a config with a healthy
    throughput share has real within-connectivity-level gradient for a
    policy to climb.

    Returns {"connectivity_var", "throughput_var", "total_var",
    "throughput_frac"}. throughput_frac = throughput_var / total_var, in
    [0, 1] by construction (both components are non-negative and sum to
    total_var exactly). 0.0 for every field if fewer than 2 records (nothing
    to compute a variance over).
    """
    if len(records) < 2:
        return {"connectivity_var": 0.0, "throughput_var": 0.0,
                "total_var": 0.0, "throughput_frac": 0.0}
    from collections import defaultdict
    groups: dict[int, list[float]] = defaultdict(list)
    for r, c in records:
        groups[c].append(r)
    all_r = np.array([r for r, _ in records], dtype=float)
    n_total = len(all_r)
    total_var = float(np.var(all_r))
    grand_mean = float(np.mean(all_r))
    within = 0.0   # E[Var(reward | connected_count)]  -- THROUGHPUT
    between = 0.0  # Var(E[reward | connected_count])  -- CONNECTIVITY
    for rs in groups.values():
        rs_arr = np.array(rs, dtype=float)
        w = len(rs_arr) / n_total
        within += w * float(np.var(rs_arr))
        between += w * (float(np.mean(rs_arr)) - grand_mean) ** 2
    throughput_frac = (within / total_var) if total_var > 0 else 0.0
    return {"connectivity_var": between, "throughput_var": within,
           "total_var": total_var, "throughput_frac": throughput_frac}


def coefficient_of_variation_connected(records: list[tuple[float, int]]) -> float:
    """std(reward | reward>0) / mean(reward | reward>0) over a P0v rollout --
    candidate (a) from the B3_gradient_rich redesign (sweep_config.py's
    module docstring has the full before/after measurement). Near 0 means
    every "connected" tick scores about the same reward (binary/plateau,
    no slope); a healthy value means reward keeps moving meaningfully even
    while at least one commodity is connected -- a real gradient to climb.

    [2026-08-05] Measured against candidate (b) (variance_decomposition's
    throughput_frac, ALSO computable from these same records): (b) came out
    near-zero (<0.02) for EVERY config tested, including a config known to
    be learnable, because log1p's compression makes the CONNECTIVITY
    component of reward variance dominate the THROUGHPUT component by
    construction (a connection is worth ~18.5 units, a throughput
    doubling ~0.69) -- so (b) does not discriminate ANY config
    from any other under this reward, not just these two. (a) measures a
    related but different thing -- the spread of reward AMONG connected
    ticks, not decomposed by exact connected-count -- and DOES discriminate
    (measured medians ~0.30 vs ~0.60 for a known-bad vs known-good config).
    variance_decomposition() is kept in this module and still reported as a
    diagnostic column precisely because ITS failure to discriminate is
    itself the evidence for why (a) was chosen instead.

    0.0 if fewer than 2 positive-reward records (nothing to compute a
    spread over).
    """
    rewards = np.array([r for r, _ in records], dtype=float)
    positive = rewards[rewards > 0]
    if len(positive) < 2 or np.mean(positive) <= 0:
        return 0.0
    return float(np.std(positive) / np.mean(positive))


def probe_greedy_traced(tv, obs: dict, n_drones: int, L: float,
                        steps: int = 60, step_m: float = 20.0
                        ) -> tuple[list[list[float]], list[float]]:
    """Runs task_validation.probe_greedy UNCHANGED (never re-derive its
    convergence/patience loop here, so there is one copy of that logic)
    while ALSO recording the per-step
    post-commit reward trace, which probe_greedy itself does not return but
    which frac_zero_steps_P1 needs.

    Technique: monkeypatch the NAME `compute_position_oracle_reward` bound in
    task_validation's OWN module globals for the duration of the call, then
    restore it. Python functions resolve global names from their defining
    module's globals dict at CALL time, not def time, so this intercepts
    every call probe_greedy's body makes through that name -- which is
    exactly ONE per rollout step (the `r = compute_position_oracle_reward(o)
    ["reward"]` line that scores the position AFTER committing a move).
    greedy_velocities' own internal 8-direction probes call a SEPARATELY
    bound name in greedy_oracle.py's module globals (`from max_flow_reward
    import compute_position_oracle_reward as compute_reward`) and are
    untouched by this patch.

    `tv` is the task_validation module, passed in explicitly (not imported
    here) for the same dual-site-state reason as probe_random_velocity above.
    """
    trace: list[float] = []
    orig = tv.compute_position_oracle_reward

    def _traced(o):
        result = orig(o)
        trace.append(result["reward"])
        return result

    tv.compute_position_oracle_reward = _traced
    try:
        final_pos = tv.probe_greedy(obs, n_drones, L, steps=steps, step_m=step_m)
    finally:
        tv.compute_position_oracle_reward = orig
    return final_pos, trace


def cluster_centroids(users: list[list[float]], areas: list[int]) -> dict[int, list[float]]:
    groups: dict[int, list[list[float]]] = {}
    for u, a in zip(users, areas):
        groups.setdefault(a, []).append(u)
    return {
        a: [float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))]
        for a, pts in groups.items()
    }


def inter_cluster_dist_m(users: list[list[float]], areas: list[int]) -> float:
    """Max pairwise distance between cluster centroids. 0.0 with 1 cluster."""
    cents = list(cluster_centroids(users, areas).values())
    if len(cents) < 2:
        return 0.0
    return max(
        math.dist(cents[i], cents[j])
        for i in range(len(cents))
        for j in range(i + 1, len(cents))
    )


def decompose_intra_cross_reward(tv, positions: list[list[float]],
                                 users: list[list[float]], areas: list[int],
                                 pairs: list[list[int]]) -> tuple[float, float]:
    """Decompose P2's Sigma_k tau_k by whether commodity k's two endpoints
    share a user_area ("intra") or not ("cross" -- the multi-hop,
    cross-cluster portion of the reward, the thing the relay task exists
    for). Uses per_commodity_tau directly (compute_position_oracle_reward's
    cheap, already-computed-per-call return) and re-applies the reward's own
    definition rather than re-scoring per-commodity sub-problems one at a time.

    The reward is the plain sum of throughput in Mbps (not the earlier
    Sigma_k log1p(tau_k) objective), so the decomposition is
    tau / REWARD_UNIT_BPS, in the same units as the reward.

    Because the objective is LINEAR, intra + cross sums back to the LP's
    own "reward" key exactly.
    """
    # REWARD_UNIT_BPS is a fixed unit conversion, not one of the radio
    # constants a caller sweeps by mutating `tv`, so importing it directly is
    # safe here (unlike compute_position_oracle_reward, which is taken off
    # `tv` on purpose -- see this module's own note above).
    import max_flow_reward as _mfr

    obs = {"positions": positions, "user_positions": users,
          "user_area": areas, "sender_pairs": pairs}
    result = tv.compute_position_oracle_reward(obs)
    intra = 0.0
    cross = 0.0
    for (si, dj), tau in result["per_commodity_tau"].items():
        r = tau / _mfr.REWARD_UNIT_BPS
        if areas[si] == areas[dj]:
            intra += r
        else:
            cross += r
    return intra, cross
