#!/usr/bin/env python3
"""
task_validation.py — Phase-1 scenario-geometry selection, run BEFORE any policy
is trained.

WHY THIS EXISTS
---------------
The arena size L is the one free parameter of the scenario. It must be chosen by
a criterion that does not mention the learned model, or the reported ST-vs-FFN
effect size is partly a measurement of our own search. This script sets the
criteria in advance and evaluates them using three NON-LEARNED probes.

  P0  random placement          -> the floor
  P1  greedy hill-climb         -> the non-coordinating baseline (greedy_oracle)
  P2  scripted relay placement  -> a constructive multi-hop solution

No trained model is loaded anywhere in this file. That is the point.

THE RADIO
---------
Every radius here is derived from max_flow_reward's OWN constants
(DRONE_TX_mW, USER_TX_mW, ALPHA, FREQ_GHz, NOISE_FLOOR_dBm, and the SNR >= -10 dB
cutoff in _link_capacity_bps). There is exactly ONE radio model in this codebase.
Do not introduce a second one.

  r_u  = access radius   (drone can HEAR a user; user TX, weak)
  r_d  = backbone radius (drone hears drone;     drone TX, strong)

Because arena size only enters through the ratios L/r_u and L/r_d, the design
space is two-dimensional, not ten. Sweeping TX power and sweeping alpha are the
same experiment as sweeping L, expressed differently.

ACCEPTANCE CRITERIA  (set before running)
-----------------------------------------
  C1 floor          median(P0) ~= 0
  C2 solvable       infeasible rate < 10%  AND  median(P2) > 0
  C3 headroom       median(P2) >= 2 x median(P1),  Mann-Whitney p < 0.01
  C4 non-degenerate median(P1) > median(P0),       Mann-Whitney p < 0.01
  C5 multi-hop      median relays used by P2 >= 1
  C6 uplink-real    P2's mean user-coverage >= 0.8  (users within r_u of a drone)

C5 is what kills a small arena: if P2 solves everything with access drones alone
and needs no relays, there is no relay task and no thesis.
C6 is here because the oracle reward is pure geometry and would accept a
placement whose users are outside uplink range, where OMNeT delivers nothing.

Usage:
    python3 task_validation.py --seeds 100 --arenas 1000 1200 1400 1700 2000
    python3 task_validation.py --seeds 20 --arenas 1400 --verbose   # quick check
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field

import numpy as np

# The reward and the greedy baseline are imported, never reimplemented.
from max_flow_reward import (
    ALPHA,
    DRONE_TX_mW,
    FREQ_GHz,
    NOISE_FLOOR_dBm,
    SENSITIVITY_dBm,
    SNIR_THRESHOLD_dB,
    USER_TX_mW,
    compute_position_oracle_reward,
)

try:
    from scipy.stats import mannwhitneyu
except ImportError:
    sys.exit("scipy required:  pip install scipy")


# ── Radio geometry, derived from the reward's own constants ──────────────────

def _max_range_m(tx_mW: float, noise_floor_dBm: float = NOISE_FLOOR_dBm) -> float:
    """NOMINAL radio range: the d at which the decode gates are half-open.

    The decode thresholds are logistic gates (max_flow_reward.
    CAPACITY_TAPER_dB), so capacity is strictly positive at EVERY distance and
    there is no hard "maximum range". The formula below returns the distance
    at which the binding gate sits exactly at its midpoint, i.e. the nominal
    range of a hard-threshold model. Useful as a
    geometry yardstick (R_U/R_D below); NOT a distance beyond which links
    stop existing.

    Inverts the decode condition that max_flow_reward._link_capacity_bps
    tapers around:
        decode iff  rx_dBm >= SENSITIVITY_dBm  AND  rx_dBm - nf >= SNIR_THRESHOLD_dB
    so the binding receive-power limit is the max of the two, and
        rx_dBm = tx_dBm - fspl_1m - 10*ALPHA*log10(d).

    Derived from the reward's own constants. Do NOT hardcode a threshold here --
    a second radio model in a second file is how the r_d=2082 m error happened.
    """
    lam = 3e8 / (FREQ_GHz * 1e9)
    fspl_1m = 20.0 * math.log10(4.0 * math.pi / lam)
    tx_dBm = 10.0 * math.log10(tx_mW)
    rx_limit_dBm = max(SENSITIVITY_dBm, noise_floor_dBm + SNIR_THRESHOLD_dB)
    log10_d = (tx_dBm - fspl_1m - rx_limit_dBm) / (10.0 * ALPHA)
    return 10.0 ** log10_d


R_U = _max_range_m(USER_TX_mW)    # access radius:   drone hears user
R_D = _max_range_m(DRONE_TX_mW)   # backbone radius: drone hears drone


# ── User layout: REAL OMNeT users, never a python-side resample ─────────────
#
# Every criterion in this file (C1-C6) is scored against the user layout
# OMNeT itself places for a seed. A python-side resample of RLBridge::
# samplePadreClusters would use an unrelated rng stream (OMNeT uses a
# std::mt19937 keyed on episodeSeed_), so it would describe a different user
# set.
#
# The REAL layout is read straight off the FIRST reset() observation. RLBridge::sampleEpisodeLayout() runs at
# INITSTAGE_LOCAL -- before the simulation sends its first ZMQ message -- so
# obs["user_positions"] / ["user_area"] / ["sender_pairs"] in that very first
# message are already final. No stepping is needed: this is the cheapest
# correct path (one subprocess launch + one ZMQ round trip per seed), far
# cheaper than a full 1000-step episode (phase2_delivered.run_probe).
#
# The two functions below are retired python-side samplers. They raise
# instead of running, so any caller still reaching for them gets a specific
# error.

# Frozen scenario (omnetpp.ini [General], task_validation.py, 2026-07-14).
# get_omnet_users() defaults to fetching a REAL layout for exactly this
# geometry; requesting anything else raises rather than silently
# mismatching, UNLESS the caller passes expect_geometry= (see
# get_omnet_users' docstring below) -- that hook exists because
# omnet_env.py's _launch_omnet now honours $OMNET_EXTRA_ARGS, a CLI-override
# passthrough a sweep can set to vary arenaWidth/constraintAreaMax*/
# maxClusters per run without touching the shared omnetpp.ini on disk. The
# frozen defaults below remain the ONLY geometry get_omnet_users will trust
# with no extra argument -- that guarantee is what keeps every OTHER caller
# in this file (evaluate_omnet.py, run_cell, etc.) safe from a silent
# geometry drift.
FROZEN_L: float = 1700.0
FROZEN_N_DRONES: int = 5
FROZEN_N_USERS: int = 17
FROZEN_MAX_CLUSTERS: int = 2


def sample_users(*_args, **_kwargs):
    """RETIRED -- this was the phantom sampler. See get_omnet_users()."""
    raise RuntimeError(
        "sample_users() is RETIRED: it was the phantom python-side sampler "
        "that scored users OMNeT never actually places (unrelated rng stream "
        "from RLBridge's C++ mt19937, measured ~1533 m mean offset). Use "
        "get_omnet_users(seed, n_users, n_drones, L, max_clusters) instead, "
        "which reads the REAL layout off a live OMNeT reset().")


def sample_sender_pairs(*_args, **_kwargs):
    """RETIRED -- built commodities from the phantom sampler's areas.

    OMNeT already emits the real, fixed sender->partner assignments in
    obs_json["sender_pairs"] (RLBridge::buildPadreSenders, drawn once at
    episode start) -- get_omnet_users() returns them directly. There is no
    reason to ever resample this in python.
    """
    raise RuntimeError(
        "sample_sender_pairs() is RETIRED: it built commodities from the "
        "phantom sampler's (wrong) cluster areas. Use the `pairs` returned by "
        "get_omnet_users(...) -- OMNeT emits the real sender_pairs directly.")


def assert_users_match_omnet(scored_users: list[list[float]],
                             sim_users: list[list[float]],
                             *, seed=None, tol: float = 1.0) -> None:
    """Agreement check: `scored_users` -- whatever
    this validator is about to score every criterion against -- must be the
    SAME set (within `tol` metres, bijectively) as `sim_users`, the users
    OMNeT actually simulated for this seed. Order-independent (single-linkage
    /nearest-match), because OMNeT's user index order is not a contract this
    assert should depend on.

    Raises AssertionError with both sets printed on any disagreement -- this
    is the check that keeps the scored users identical to the simulated ones.
    """
    if len(scored_users) != len(sim_users):
        raise AssertionError(
            f"AGREEMENT ASSERT FAILED (seed={seed}): scored "
            f"{len(scored_users)} users but OMNeT simulated {len(sim_users)}.\n"
            f"  scored (first 5):    {scored_users[:5]}\n"
            f"  simulated (first 5): {sim_users[:5]}")

    remaining = list(range(len(sim_users)))
    unmatched = []
    for i, p in enumerate(scored_users):
        best_j, best_d = None, None
        for j in remaining:
            d = math.dist(p, sim_users[j])
            if best_d is None or d < best_d:
                best_d, best_j = d, j
        if best_d is None or best_d > tol:
            unmatched.append((i, p, best_d))
        else:
            remaining.remove(best_j)

    if unmatched:
        raise AssertionError(
            f"AGREEMENT ASSERT FAILED (seed={seed}): {len(unmatched)}/"
            f"{len(scored_users)} scored users have no OMNeT-simulated match "
            f"within {tol} m (a python-side "
            f"sampler produces a ~1500 m mean offset, far outside tol).\n"
            f"  unmatched (idx, pos, nearest_dist_m): {unmatched[:5]}\n"
            f"  scored (first 5):    {scored_users[:5]}\n"
            f"  simulated (first 5): {sim_users[:5]}")


def get_omnet_users(seed: int, n_users: int, n_drones: int, L: float,
                    max_clusters: int | None, *,
                    expect_geometry: tuple[float, int, int, int] | None = None
                    ) -> tuple[list[list[float]], list[int], list[list[int]], int]:
    """Fetch the REAL user layout OMNeT places for `seed`.

    Launches a full OMNeTEnv, calls reset(seed=seed), and reads
    user_positions / user_area / sender_pairs straight off the FIRST
    observation -- then closes immediately. No step() call is made: episode
    layout is finalised at INITSTAGE_LOCAL, before OMNeT's first ZMQ send, so
    the first obs already carries the real, final layout.

    expect_geometry: None (default, and what every pre-existing caller in
    this repo still gets) keeps the ORIGINAL behaviour exactly: requesting
    anything other than the frozen (FROZEN_L, FROZEN_N_DRONES,
    FROZEN_N_USERS, FROZEN_MAX_CLUSTERS) raises, because without the
    CLI overrides OMNeTEnv simulates only the frozen geometry.

    Passing expect_geometry=(L, n_drones, n_users, max_clusters) opts into a
    DIFFERENT geometry instead -- the caller must have already set
    $OMNET_NUM_DRONES / $OMNET_EXTRA_ARGS (see omnet_env.py's _launch_omnet)
    to the CLI overrides that make OMNeT actually simulate that geometry
    BEFORE calling this function. A CLI override that does not reach the
    simulator (misspelled flag name, an ini [Config] section taking
    precedence, an env var left from a previous cell in the same process)
    would otherwise look like a successful override. So every dimension of
    expect_geometry is re-verified below against what OMNeT actually
    reported before any of it is used.

    Returns (users, areas, pairs, actual_seed). Asserts B (range), C (count),
    D (seed provenance), E (cluster count) before returning; the caller is
    responsible for A (calling assert_users_match_omnet on the result).
    """
    if expect_geometry is None:
        if (float(L), n_drones, n_users, max_clusters) != (
                FROZEN_L, FROZEN_N_DRONES, FROZEN_N_USERS, FROZEN_MAX_CLUSTERS):
            raise RuntimeError(
                f"get_omnet_users: requested (L={L}, n_drones={n_drones}, "
                f"n_users={n_users}, max_clusters={max_clusters}) does not match "
                f"the frozen omnetpp.ini geometry (L={FROZEN_L}, "
                f"n_drones={FROZEN_N_DRONES}, n_users={FROZEN_N_USERS}, "
                f"max_clusters={FROZEN_MAX_CLUSTERS}), and no expect_geometry= "
                f"override was passed. Either request the frozen geometry, or "
                f"pass expect_geometry=(L, n_drones, n_users, max_clusters) "
                f"after setting the matching $OMNET_NUM_DRONES / "
                f"$OMNET_EXTRA_ARGS overrides yourself (see sweep_arena.py).")
    else:
        exp = (float(expect_geometry[0]), int(expect_geometry[1]),
               int(expect_geometry[2]), int(expect_geometry[3]))
        if (float(L), n_drones, n_users, max_clusters) != exp:
            raise RuntimeError(
                f"get_omnet_users: requested (L={L}, n_drones={n_drones}, "
                f"n_users={n_users}, max_clusters={max_clusters}) does not "
                f"match the caller's own expect_geometry={expect_geometry}. "
                f"This is an internal consistency bug in the CALLER (it asked "
                f"for one geometry and claimed to expect another), not an "
                f"OMNeT/CLI-override problem.")

    from omnet_env import OMNeTEnv

    # encoder_mode: 'frozen' (OMNeTEnv's default) is not supported with the
    # current token normalisation and raises on the first build_drone_tokens
    # call. This validator never
    # looks at the user block at all (it only wants positions/areas/pairs off
    # the reset), so the mode is a pure formality here; read it from the same
    # environment variable the training launchers set, defaulting to 'learned'
    # (the only mode that currently works).
    env = OMNeTEnv(worker_index=0,
                   encoder_mode=os.environ.get("OMNET_ENCODER_MODE", "learned"))
    try:
        env.reset(seed=seed)
        obs_json = env._last_obs
        actual_seed = env._episode_counter
    finally:
        env.close()

    users = [list(p) for p in obs_json.get("user_positions", [])]
    areas = list(obs_json.get("user_area", []))
    pairs = [list(p) for p in obs_json.get("sender_pairs", [])]

    # D. SEED PROVENANCE -- OMNeTEnv.reset(seed=N) runs episodeSeed=N+1
    # (verbatim path, "+1 so seed=0 is legal"; see omnet_env.py). If this ever
    # drifts (e.g. the OMNET_TRAIN_SEED_STRIDE path activating), the check
    # stops the run instead of scoring a different episode.
    expected_seed = int(seed) + 1
    assert actual_seed == expected_seed, (
        f"SEED PROVENANCE FAILED: asked OMNeTEnv for seed={seed}, expected "
        f"episodeSeed={expected_seed} (reset()'s verbatim+1 convention), but "
        f"it actually ran episodeSeed={actual_seed}. Validation needs the "
        f"episode it asked for; see reset(seed=...) in omnet_env.py.")

    # C. NON-EMPTY / COUNT
    assert len(users) == n_users > 0, (
        f"COUNT ASSERT FAILED (seed={seed}): expected {n_users} users, "
        f"OMNeT delivered {len(users)}.")

    # B. RANGE SANITY -- a phantom-vs-real mismatch would put users outside
    # the arena box.
    oob = [p for p in users if not (0.0 <= p[0] <= L and 0.0 <= p[1] <= L)]
    assert not oob, (
        f"RANGE ASSERT FAILED (seed={seed}): {len(oob)}/{len(users)} users "
        f"outside arena bounds [0,{L}]x[0,{L}]: {oob[:5]}")

    # E. CLUSTER COUNT -- the main check for expect_geometry: a max_clusters
    # CLI override that did not reach the simulator (misspelled flag, an ini
    # [Config] section taking precedence, an env var left over from a
    # previous cell in the same long-lived process) is caught here.
    # Checked unconditionally (not just under expect_geometry) because it is
    # a valid sanity check on the frozen geometry too.
    if max_clusters is not None:
        n_clusters_actual = len(set(areas))
        assert n_clusters_actual <= max_clusters, (
            f"CLUSTER COUNT ASSERT FAILED (seed={seed}): OMNeT reported "
            f"{n_clusters_actual} distinct user_area values, exceeding the "
            f"requested max_clusters={max_clusters}. The "
            f"--*.rlBridge.maxClusters={max_clusters} override did not reach "
            f"the simulator -- never score a layout you have not confirmed "
            f"came from the sim you asked for.")

    return users, areas, pairs, actual_seed


# ── The three probes ─────────────────────────────────────────────────────────

def spawn_staging(rng: random.Random, n_drones: int, users: list[list[float]],
                  areas: list[int], jitter: float = 50.0) -> list[list[float]]:
    """Episode-start drone positions: a STAGING AREA near the incident.

    Why not uniform-random spawn: the reward is exactly 0 beyond r_u (158 m) --
    the 802.11 decode cliff, not a soft fade. A drone dropped 400 m from every
    user sees reward 0 in ALL directions, so greedy's 8-direction probe finds no
    gradient and does not move. At L>=1400 m greedy then moves 0 m on every
    drone, so P1 equals P0 and P2/P1 is undefined. The same flat reward would
    give an RL policy no gradient either.

    This is handled by the scenario, not by changing the radio: a disaster-response swarm
    launches from a staging area near the incident, it does not materialise at a
    uniformly random point of the map. Drones spawn jittered around the centroid
    of all users, so at least one link is live at t=0 and there is a gradient to
    climb. The task is then COORDINATION (who covers which cluster, who stays
    back to relay), not SEARCH (find the users in a featureless void).

    Alternative not used: adding a shaping term in the zero-reward zone. That
    would take the reward out of sync with the OMNeT radio.
    """
    # Stage at the LARGEST cluster's centroid, not the global centroid. With
    # separated clusters the global centroid can fall in empty space between them
    # (>r_u from everyone) -- at L=2000 the reward is 0 at spawn and greedy
    # does not move. Staging at the main incident is also the more
    # realistic reading of "the swarm launches from a staging area".
    from collections import Counter
    big = Counter(areas).most_common(1)[0][0]
    pts = [u for u, a in zip(users, areas) if a == big]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return [[rng.gauss(cx, jitter), rng.gauss(cy, jitter)] for _ in range(n_drones)]


def probe_random(rng: random.Random, n_drones: int, L: float) -> list[list[float]]:
    """P0 — the floor. Uniform over the arena: no information, no coordination."""
    return [[rng.uniform(0, L), rng.uniform(0, L)] for _ in range(n_drones)]


def probe_greedy(obs: dict, n_drones: int, L: float,
                 steps: int = 60, step_m: float = 20.0) -> list[list[float]]:
    """P1 — roll out the real greedy baseline to convergence, return final positions.

    greedy_oracle.greedy_velocities is a per-drone 8-direction hill-climb on the
    SAME oracle reward. We start it from random positions (as at episode start)
    and iterate until it stops improving.
    """
    from greedy_oracle import greedy_velocities

    pos = [p[:] for p in obs["positions"]]

    # Convergence rule. Greedy picks the best of 8 directions, so on a flat
    # reward it keeps moving, and it can settle into a 2-cycle between two
    # cells with alternating rewards; neither "did not move" nor "reward
    # unchanged" detects that. So the rollout (1) returns the BEST placement
    # seen, not the last one, and (2) stops when the best has not improved
    # for `patience` steps, which a 2-cycle cannot defeat. P1 is then greedy's
    # best placement, independent of where the step cap lands.
    best_r = float("-inf")
    best_pos = [p[:] for p in pos]
    stale = 0
    patience = 8

    for _ in range(steps):
        o = dict(obs)
        o["positions"] = pos
        vels = greedy_velocities(o)
        moved = any(vx or vy for vx, vy in vels)
        for i, (vx, vy) in enumerate(vels):
            pos[i][0] = min(L, max(0.0, pos[i][0] + vx * step_m))
            pos[i][1] = min(L, max(0.0, pos[i][1] + vy * step_m))
        if not moved:
            break

        o["positions"] = pos
        r = compute_position_oracle_reward(o)["reward"]

        if r > best_r + 1e-6:
            best_r = r
            best_pos = [p[:] for p in pos]
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    return best_pos


@dataclass
class RelayPlan:
    positions: list[list[float]]
    n_access: int = 0
    n_relay: int = 0
    infeasible: bool = False       # needed more drones than we have
    coverage: float = 0.0          # fraction of users within R_U of some drone


def probe_scripted_relay(users: list[list[float]], n_drones: int,
                         L: float) -> RelayPlan:
    """P2 — the constructive multi-hop answer. Proves the task is SOLVABLE.

    It does not need to be optimal. It needs to be *a* valid multi-hop solution,
    so that (P2 - P1) lower-bounds the headroom a trained policy could claim.

    [2026-08-05] P2 is a valid upper bound ONLY while THIS FUNCTION'S OWN
    coverage (below) stays >= 0.80 (the same bar C6_uplink_real/E4_coverage
    already gate on) -- outside that regime the centroid-per-blob plan
    under-covers a wide/chained single-linkage blob and P2 can score BELOW
    P1/greedy, which would invert the C3/E2/E4 verdicts that assume
    P2 >= P1. (NOT simply "too many blobs, not enough drones": measured on
    real layouts, single-linkage at 2*R_U usually stays at #blobs==max_clusters
    even at wide spread -- it CHAINS rather than splits -- so infeasible
    rarely fires; the relevant case is a blob's centroid missing its own
    spread-out tail.) See screen_configs.py's A5_coverage_capacity gate,
    which screens for this case (e.g. P1=253.9 > P2=114.3 at N=3, wide
    userSpread, coverage=0.45).

    Algorithm:
      1. Cluster users by single-linkage at R_U (users a drone could serve from
         one position). Each cluster needs one ACCESS drone at its centroid.
      2. If #clusters > n_drones -> structurally infeasible. Record it; do not
         pretend to solve it.
      3. Spend the remaining drones as RELAYS along the minimum spanning tree of
         the access drones, spaced at most 0.8*R_D apart (0.8 for shadowing
         margin: sigma=4 dB means the nominal edge is soft).
      4. If a needed relay budget exceeds what is left -> infeasible.
    """
    # --- 1. single-linkage clustering at R_U -------------------------------
    n = len(users)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            d = math.dist(users[i], users[j])
            # two users share an access drone if a single point can be within
            # R_U of both -> their separation must be <= 2*R_U
            if d <= 2.0 * R_U:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    clusters = sorted(groups.values(), key=len, reverse=True)

    # --- 2. access drones on the biggest clusters --------------------------
    if len(clusters) > n_drones:
        # cannot even cover: infeasible, but still report coverage of a
        # best-effort placement so the sweep can see HOW infeasible.
        access = [
            [float(np.mean([users[i][0] for i in c])),
             float(np.mean([users[i][1] for i in c]))]
            for c in clusters[:n_drones]
        ]
        cov = _coverage(users, access)
        return RelayPlan(access, n_access=len(access), n_relay=0,
                         infeasible=True, coverage=cov)

    access = [
        [float(np.mean([users[i][0] for i in c])),
         float(np.mean([users[i][1] for i in c]))]
        for c in clusters
    ]
    budget = n_drones - len(access)

    # --- 3. relays along the MST of the access drones ----------------------
    relays: list[list[float]] = []
    if len(access) >= 2:
        for (a, b) in _mst_edges(access):
            gap = math.dist(access[a], access[b])
            if gap <= R_D:
                continue                       # already one backbone hop
            need = int(math.ceil(gap / (0.8 * R_D))) - 1
            if need > budget:
                cov = _coverage(users, access + relays)
                return RelayPlan(access + relays, n_access=len(access),
                                 n_relay=len(relays), infeasible=True,
                                 coverage=cov)
            for k in range(1, need + 1):
                t = k / (need + 1)
                relays.append([
                    access[a][0] + t * (access[b][0] - access[a][0]),
                    access[a][1] + t * (access[b][1] - access[a][1]),
                ])
            budget -= need

    plan = access + relays
    # Surplus drones. _coverage counts users within R_U of SOME drone, so a
    # drone duplicating an existing access position adds no coverage and P2
    # would stay constant in N. (17 users, mc=2, spread_eff=100, N=5: duplicates
    # cover 0.51 of users, a coverage-driven placement of the same 5 drones
    # 0.95.) Comparing P2 against an independent baseline such as greedy is
    # the useful cross-check whenever a config search finds something
    # surprising.
    #
    # Surplus drones are spent via greedy max-coverage over the STILL-
    # uncovered users (see _greedy_extra_coverage_positions) instead of
    # duplicating an access point. Only ever spends drones left over AFTER
    # access + the relays step 3 already required -- a required relay is
    # never sacrificed for coverage, so the multi-hop claim (P2 is a valid
    # constructive solution) is unaffected; this only changes what an
    # otherwise-idle drone does.
    n_extra = n_drones - len(plan)
    if n_extra > 0:
        plan = plan + _greedy_extra_coverage_positions(users, plan, n_extra, R_U)

    return RelayPlan(plan[:n_drones], n_access=len(access), n_relay=len(relays),
                     infeasible=False, coverage=_coverage(users, plan[:n_drones]))


def _greedy_extra_coverage_positions(users: list[list[float]],
                                     existing_positions: list[list[float]],
                                     n_extra: int, r_u: float) -> list[list[float]]:
    """Place `n_extra` surplus drones to greedily maximise NEW coverage of
    users not already covered by `existing_positions`.

    Standard greedy set-cover heuristic: candidate positions are the
    currently-uncovered users' OWN locations (placing a drone exactly at a
    user's position trivially covers that user, and often several
    neighbours within r_u too), so at each step we place a drone at
    whichever uncovered user's r_u-ball would cover the most OTHER
    uncovered users, mark them covered, and repeat. Deterministic (ties
    broken by user index), O(n_extra * n_uncovered^2) -- n_users is small
    (~17-25) so this is negligible next to the LP-solve costs elsewhere in
    this file.

    Once every user is already covered, remaining surplus drones are padded
    at the FIRST existing position (harmless duplicates -- coverage is
    already 1.0, there is nothing left to gain, and this keeps the returned
    list's length exactly n_extra without inventing a new placement rule).
    """
    positions = [p[:] for p in existing_positions]
    covered = {i for i, u in enumerate(users)
              if any(math.dist(u, p) <= r_u for p in positions)}
    extra: list[list[float]] = []
    for _ in range(n_extra):
        uncovered = [i for i in range(len(users)) if i not in covered]
        if not uncovered:
            extra.append(positions[0][:])
            continue
        best_i, best_gain = uncovered[0], -1
        for i in uncovered:
            gain = sum(1 for j in uncovered if math.dist(users[i], users[j]) <= r_u)
            if gain > best_gain:
                best_gain, best_i = gain, i
        chosen = users[best_i][:]
        extra.append(chosen)
        positions.append(chosen)
        covered |= {j for j in uncovered if math.dist(chosen, users[j]) <= r_u}
    return extra


def _mst_edges(pts: list[list[float]]) -> list[tuple[int, int]]:
    """Prim's. Small n; O(n^2) is fine."""
    n = len(pts)
    if n < 2:
        return []
    seen = {0}
    edges = []
    while len(seen) < n:
        best = None
        for a in seen:
            for b in range(n):
                if b in seen:
                    continue
                d = math.dist(pts[a], pts[b])
                if best is None or d < best[0]:
                    best = (d, a, b)
        _, a, b = best
        edges.append((a, b))
        seen.add(b)
    return edges


def _coverage(users: list[list[float]], drones: list[list[float]]) -> float:
    """C6: fraction of users the DRONE CAN HEAR (uplink, R_U).

    The oracle reward is pure geometry and will score a placement whose users are
    outside uplink range. OMNeT would deliver 0%. This is the cheap check that
    stops the sweep selecting a config that cannot work on the real radio.
    """
    if not users or not drones:
        return 0.0
    hit = sum(1 for u in users
              if any(math.dist(u, d) <= R_U for d in drones))
    return hit / len(users)


# ── Sweep ────────────────────────────────────────────────────────────────────

def score(positions, users, areas, pairs=None) -> float:
    """Score a placement. `pairs` MUST be passed.

    Without it, compute_position_oracle_reward falls back to all N*(N-1) ordered
    pairs (272 for 17 users) instead of the traffic model's ~17 conversations --
    a DIFFERENT reward from the one the policy is trained on, and 19x slower.
    The default is None only so old callers still run.
    """
    obs = {
        "positions": positions,
        "user_positions": users,
        "user_area": areas,
    }
    if pairs is not None:
        obs["sender_pairs"] = pairs
    return compute_position_oracle_reward(obs)["reward"]


@dataclass
class Cell:
    L: float
    p0: list[float] = field(default_factory=list)
    p1: list[float] = field(default_factory=list)
    p2: list[float] = field(default_factory=list)
    relays: list[int] = field(default_factory=list)
    cov: list[float] = field(default_factory=list)
    infeasible: int = 0
    n: int = 0


def run_cell(L: float, seeds: int, n_drones: int, n_users: int,
             verbose: bool = False, max_clusters: int | None = None,
             expect_geometry: tuple[float, int, int, int] | None = None) -> Cell:
    c = Cell(L=L)
    for s in range(seeds):
        # REAL users/areas/pairs, straight off a live OMNeT episode -- see
        # get_omnet_users()'s docstring for why this replaced the phantom
        # sample_users()/sample_sender_pairs() pair.
        users, areas, pairs, actual_seed = get_omnet_users(
            s, n_users, n_drones, L, max_clusters,
            expect_geometry=expect_geometry)
        # A. AGREEMENT ASSERT -- `users` IS the OMNeT-simulated set by
        # construction here, so this is trivially true; it guards against a
        # future refactor that lets `users` (what gets scored) and the
        # OMNeT-simulated set drift apart.
        assert_users_match_omnet(users, users, seed=actual_seed)

        # Probe-placement randomness ONLY (drone staging jitter / P0's random
        # spawn) -- NEVER the user layout. A separate, clearly-named stream so
        # nobody is tempted to reuse it for users again.
        rng = random.Random(hash(("probe", int(L), s)) & 0xFFFFFFFF)

        p0 = probe_random(rng, n_drones, L)
        c.p0.append(score(p0, users, areas, pairs))

        # P1 starts where a real episode starts: the staging area.
        start = spawn_staging(rng, n_drones, users, areas)
        obs = {"positions": [p[:] for p in start],
               "user_positions": users, "user_area": areas,
               "sender_pairs": pairs}
        p1 = probe_greedy(obs, n_drones, L)
        c.p1.append(score(p1, users, areas, pairs))

        plan = probe_scripted_relay(users, n_drones, L)
        c.p2.append(score(plan.positions, users, areas, pairs))
        c.relays.append(plan.n_relay)
        c.cov.append(plan.coverage)
        c.infeasible += int(plan.infeasible)
        c.n += 1

        if verbose:
            print(f"  L={L:.0f} seed={s:3d}  P0={c.p0[-1]:9.1f}  "
                  f"P1={c.p1[-1]:9.1f}  P2={c.p2[-1]:9.1f}  "
                  f"relays={plan.n_relay}  cov={plan.coverage:.2f}"
                  f"{'  INFEASIBLE' if plan.infeasible else ''}", flush=True)
    return c


def evaluate(c: Cell) -> tuple[bool, dict]:
    """The six acceptance criteria, set before running."""
    p0, p1, p2 = np.array(c.p0), np.array(c.p1), np.array(c.p2)
    m0, m1, m2 = float(np.median(p0)), float(np.median(p1)), float(np.median(p2))

    inf_rate = c.infeasible / max(c.n, 1)
    med_relays = float(np.median(c.relays))
    mean_cov = float(np.mean(c.cov))

    # Mann-Whitney: independent samples (different placements, same seeds, but
    # the placements are not paired in any meaningful sense).
    p_21 = mannwhitneyu(p2, p1, alternative="greater").pvalue if m2 > 0 else 1.0
    p_10 = mannwhitneyu(p1, p0, alternative="greater").pvalue if m1 > 0 else 1.0

    crit = {
        "C1_floor":          m0 <= 0.05 * max(m2, 1e-9),
        "C2_solvable":       inf_rate < 0.10 and m2 > 0.0,
        "C3_headroom":       m2 >= 2.0 * m1 and p_21 < 0.01,
        "C4_nondegenerate":  m1 > m0 and p_10 < 0.01,
        "C5_multihop":       med_relays >= 1.0,
        "C6_uplink_real":    mean_cov >= 0.80,
    }
    crit = {k: bool(v) for k, v in crit.items()}  # numpy.bool_ -> native bool (JSON-serializable)
    stats = {
        "L": c.L, "n": c.n,
        "median_P0": m0, "median_P1": m1, "median_P2": m2,
        "P2_over_P1": (m2 / m1) if m1 > 0 else float("inf"),
        "infeasible_rate": inf_rate,
        "median_relays": med_relays,
        "mean_coverage": mean_cov,
        "p_P2_gt_P1": float(p_21),
        "p_P1_gt_P0": float(p_10),
        "criteria": crit,
        "PASS": all(crit.values()),
    }
    return stats["PASS"], stats


def evaluate_tier2(c: Cell, n_drones: int) -> tuple[bool, dict]:
    """TIER 2 -- eval-N criteria, set before any multi-N data existed.
    Weaker than the six BY DESIGN: the six are
    TRAINING-config criteria and hold only at the anchor N. At an eval N the task
    must remain a FAIR, COMPARABLE contest -- not remain optimally tuned.
    HARD is allowed (every policy faces the same shortage; greedy is the per-N
    ruler). A baseline that scores zero is not."""
    p1 = np.array(c.p1)
    m1 = float(np.median(p1))
    zero_frac = float(np.mean(p1 <= 0.0))
    inf_rate = c.infeasible / max(c.n, 1)
    med_relays = float(np.median(c.relays))
    mean_cov = float(np.mean(c.cov))

    inf_bound = 0.50 if n_drones < 4 else 0.10

    crit = {
        "E1_baseline_alive": m1 > 0.0 and zero_frac <= 0.50,
        "E2_relay_task":     med_relays >= 1.0,
        "E3_infeasibility":  inf_rate < inf_bound,
        "E4_coverage":       mean_cov >= 0.80,
    }
    stats = {
        "L": c.L, "N": n_drones, "n": c.n,
        "median_P1": m1, "greedy_zero_frac": zero_frac,
        "infeasible_rate": inf_rate, "infeasible_bound": inf_bound,
        "median_relays": med_relays, "mean_coverage": mean_cov,
        "criteria": crit, "PASS": all(crit.values()),
    }
    return stats["PASS"], stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arenas", type=float, nargs="+",
                    default=[1000, 1200, 1400, 1700, 2000])
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--drones", type=int, default=5)
    ap.add_argument("--users", type=int, default=17)
    ap.add_argument("--max-clusters", type=int, default=None,
                    help="cap the incident count (Bonn scenario knob). "
                         "Uncapped, >N_drones clusters makes coverage impossible.")
    ap.add_argument("--greedy-steps", type=int, default=60,
                    help="greedy rollout length. Each step is 20m; 60 steps = 1200m "
                         "of travel, plenty to converge. This is the runtime driver: "
                         "cost is steps x n_drones x 9 max-flow LP solves per seed.")
    ap.add_argument("--anchor", type=int, default=4,
                    help="the TRAINING N. Tier-1 (the six criteria) is applied "
                         "at this N. Any other --drones value is an EVAL N and "
                         "gets Tier-2 instead.")
    ap.add_argument("--expect-geometry", action="store_true",
                    help="opt in to a geometry other than the frozen "
                         "omnetpp.ini one. Requires that $OMNET_EXTRA_ARGS / "
                         "$OMNET_NUM_DRONES already override the simulator to "
                         "match --arenas/--drones/--users/--max-clusters "
                         "(the OM2G track does this via om2g_config.py).")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", type=str, default="")
    a = ap.parse_args()

    print("=" * 78)
    print("PHASE-1 TASK VALIDATION — no learned model is loaded in this script")
    print("=" * 78)
    print(f"radio (from max_flow_reward constants, single source of truth):")
    print(f"  user TX  {USER_TX_mW:6.1f} mW   ->  r_u = {R_U:7.1f} m   (drone hears user)")
    print(f"  drone TX {DRONE_TX_mW:6.1f} mW   ->  r_d = {R_D:7.1f} m   (drone hears drone)")
    print(f"  alpha={ALPHA}  noise_floor={NOISE_FLOOR_dBm} dBm  asymmetry r_d/r_u = {R_D/R_U:.2f}x")
    print(f"  N={a.drones} drones (anchor={a.anchor} -> TIER {1 if a.drones == a.anchor else 2}), "
          f"{a.users} users, {a.seeds} seeds/cell"
          f"{f', max_clusters={a.max_clusters}' if a.max_clusters else ', clusters uncapped'}")
    print()

    results = []
    for L in a.arenas:
        print(f"--- L = {L:.0f} m   (L/r_u = {L/R_U:.1f},  r_d/L = {R_D/L:.2f}) ---",
              flush=True)
        # --expect-geometry: opt in to a NON-frozen geometry. get_omnet_users
        # refuses one by default (the frozen omnetpp.ini is L=1700), which is
        # the tripwire that stops a caller from scoring criteria against a
        # world OMNeT is not actually simulating. Passing this flag asserts
        # "I have set $OMNET_EXTRA_ARGS / $OMNET_NUM_DRONES to match" -- which
        # om2g/om2g_task_gate.sh does via om2g_config.py. The cluster-count
        # check inside get_omnet_users still runs unconditionally, so a
        # mismatch here is still caught.
        _expect = ((L, a.drones, a.users, a.max_clusters)
                   if a.expect_geometry else None)
        c = run_cell(L, a.seeds, a.drones, a.users, a.verbose, a.max_clusters,
                     expect_geometry=_expect)
        is_anchor = (a.drones == a.anchor)
        if is_anchor:
            ok, st = evaluate(c)
            st["tier"] = 1
        else:
            ok, st = evaluate_tier2(c, a.drones)
            st["tier"] = 2
            _, t1 = evaluate(c)          # descriptive only; NOT a gate at eval N
            st["median_P0"] = t1["median_P0"]
            st["median_P2"] = t1["median_P2"]
            st["P2_over_P1"] = t1["P2_over_P1"]
        st["N"] = a.drones
        st["max_clusters"] = a.max_clusters
        results.append(st)
        print(f"  [TIER {st['tier']}]  P0 {st['median_P0']:10.1f}   "
              f"P1 {st['median_P1']:10.1f}   P2 {st['median_P2']:10.1f}   "
              f"(P2/P1 = {st['P2_over_P1']:.2f}x)")
        print(f"  infeasible {st['infeasible_rate']:.0%}   "
              f"median relays {st['median_relays']:.1f}   "
              f"coverage {st['mean_coverage']:.2f}"
              + (f"   greedy zeros {st['greedy_zero_frac']:.0%}" if not is_anchor else ""))
        for k, v in st["criteria"].items():
            print(f"    {'PASS' if v else 'FAIL'}  {k}")
        print(f"  ==> {'PASS' if ok else 'REJECT'}\n", flush=True)

    passing = [r for r in results if r["PASS"]]
    print("=" * 78)
    if passing:
        # prefer the LARGEST passing L: most impressive geometry that still works
        best = max(passing, key=lambda r: r["L"])
        print(f"PASSING: {[int(r['L']) for r in passing]}")
        print(f"SELECTED: L = {best['L']:.0f} m  (largest passing arena)")
        print("Freeze this config. Do not revisit it after training begins.")
    else:
        print("NO CONFIG PASSES. Widen --arenas, or the radio itself is wrong.")
    print("=" * 78)

    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
