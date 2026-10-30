"""Training-only reward shaping for the OMNeT/OM2G drone task.

WHY THIS EXISTS
---------------
Difference rewards (D_i = R(all) - R(all \\ i), omnet_env._per_agent_rewards)
already solve CREDIT assignment: once a drone stands on a load-bearing relay
position, D_i is large and the gradient is unambiguous. They do not solve
EXPLORATION. To reach that position a drone must first leave a cluster it is
covering and cross a stretch of empty space where it covers nobody, relays
nothing, and D_i is ~0. Under an undirected Gaussian policy that valley is
crossed only by chance, and the probability falls off with its width. That is
the mechanism behind "the drones never learn to be a relay".

Shaping supplies a gradient across the valley. Every term here is
POTENTIAL-BASED,

    F_i(s, s') = gamma * Phi_i(s') - Phi_i(s)

which by Ng, Harada & Russell (1999) leaves the optimal policy of the
underlying MDP UNCHANGED. So shaping can only change what the learner
*discovers*, never what the task *rewards* -- which is exactly the licence
under which it is used here.

THE HARD RULE
-------------
Shaping is a TRAINING signal. Evaluation must score the true unshaped team
reward (`infos[...]["global_reward"]`), with every weight below at zero.
`assert_disabled()` is called by evaluate_omnet.py so a shaped evaluation
cannot happen unnoticed: a filename alone does not prove which config the code
applied.

TERMS (all default 0.0 == off; set a weight to enable)
------------------------------------------------------
OM2G_SHAPE_VORONOI  Phi_i = share of users for which drone i is the nearest
                    drone, distance-discounted. This is the ALLOCATION term.
                    Two drones sitting on one cluster split its users and each
                    scores ~half; a drone that departs for an unclaimed cluster
                    strictly increases its own Phi the moment it becomes the
                    nearest drone to those users. That builds a monotone ramp
                    out of the over-concentrated local optimum -- the case
                    allocsearch.py is built to detect.

OM2G_SHAPE_FIEDLER  Phi_i = algebraic connectivity (Fiedler value) of the
                    soft drone-drone graph, shared by all drones. This is the
                    RELAY term. Link weights are a smooth sigmoid of range, so
                    the value rises continuously as a drone moves into a gap
                    between two components -- a gradient where the true reward
                    is flat. Shared rather than per-drone because connectivity
                    is a property of the graph; D_i still differentiates who
                    actually earned it.

OM2G_SHAPE_EDGE     Phi_i = -(how far drone i has pushed into the boundary
                    margin). Keeps drones off the arena walls, where the
                    position clip makes an action a no-op and the policy can
                    park without penalty.

OM2G_SHAPE_SPREAD   Phi_i = distance to the nearest peer, saturating at the
                    link range. A blunt anti-clumping prior; prefer VORONOI,
                    which is the same instinct expressed against the actual
                    demand rather than against the other drones.
"""
from __future__ import annotations

import math
import os
import numpy as np

_F = float

# Users are static within an episode, so the centroid set is computed once and
# reused; recomputing single-linkage every step would dominate the step cost.
_ANCHOR_LINK_M = float(os.environ.get("OM2G_ANCHOR_LINK_M", "500.0"))
_anchor_cache = {}


def _cluster_centroids(U, link_m):
    """One centroid per user-cluster, by single linkage at `link_m`."""
    key = (U.shape[0], float(U[0, 0]), float(U[0, 1]), float(U[-1, 0]), link_m)
    hit = _anchor_cache.get(key)
    if hit is not None:
        return hit
    n = len(U)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(n):
        d = np.linalg.norm(U[a + 1:] - U[a], axis=1)
        for off in np.nonzero(d < link_m)[0]:
            ra, rb = find(a), find(a + 1 + int(off))
            if ra != rb:
                parent[ra] = rb
    groups = {}
    for a in range(n):
        groups.setdefault(find(a), []).append(a)
    cents = np.array([U[list(g)].mean(axis=0) for g in groups.values()],
                     dtype=np.float64)
    if len(_anchor_cache) > 64:
        _anchor_cache.clear()
    _anchor_cache[key] = cents
    return cents


W_VORONOI = _F(os.environ.get("OM2G_SHAPE_VORONOI", "0") or 0)
W_FIEDLER = _F(os.environ.get("OM2G_SHAPE_FIEDLER", "0") or 0)
W_EDGE    = _F(os.environ.get("OM2G_SHAPE_EDGE",    "0") or 0)
W_SPREAD  = _F(os.environ.get("OM2G_SHAPE_SPREAD",  "0") or 0)

GAMMA      = _F(os.environ.get("OM2G_SHAPE_GAMMA", "0.99"))

# Both fall back through the world's own variables, so shaping always
# describes the world the simulator is running:
#   * ARENA_M also reads $OM2G_ARENA, which the world exports; otherwise at
#     arena 1600 the EDGE term would penalise a wall at 1080-1200 m and give
#     ZERO gradient at the real one.
#   * LINK_R_M follows $OM2G_RANGE_DRONE; otherwise under a shorter range the
#     soft adjacency would call drones "connected" at distances where the
#     radio cannot decode, and the FIEDLER bridging gradient would point
#     nowhere.
ARENA_M    = _F(os.environ.get("OM2G_SHAPE_ARENA")
                or os.environ.get("OM2G_ARENA_M")
                or os.environ.get("OM2G_ARENA", "1200"))
LINK_R_M   = _F(os.environ.get("OM2G_SHAPE_LINK_R")
                or os.environ.get("OM2G_RANGE_DRONE", "523.06"))  # drone-drone hop
EDGE_MARGIN_M = _F(os.environ.get("OM2G_SHAPE_EDGE_MARGIN", "120"))
VOR_TAU_M  = _F(os.environ.get("OM2G_SHAPE_VOR_TAU", "300"))  # coverage falloff
FIEDLER_TAU_M = _F(os.environ.get("OM2G_SHAPE_FIEDLER_TAU", "80"))

# ── $OM2G_SHAPE_KERNEL: what counts as a GOOD hop ────────────────────────────
# "exp" (default, unchanged) -- W = exp(-d / LINK_R_M). A proximity kernel with
# no relation to whether the radio can actually decode.
# "p"   -- W = Phi(rx - sensitivity) * Phi(snr - snir_threshold), the same
#          decode probability `max_flow_reward._link_capacity_bps` computes
#          under $OM2G_PHYSICAL_METRIC.
#
# WHY THIS EXISTS (measured 2026-09-02, bridge_quality.py, n=32 seeds/arm).
# The learned policies join user-clusters through radio-usable links 34-58% of
# steps -- at or above relay_explore's 42.5% -- yet carry no more traffic. The
# difference is the bottleneck decode probability on the chain: 0.11-0.26 for
# the learned arms against 0.737 for relay_explore. The reward charges ARQ 1/p
# at the transmitter, so those hops cost 4-9x the airtime per delivered bit.
#
# The cause is a disagreement between the two signals. The objective pays 1/p;
# the potential paid exp(-d/R). A drone maximises exp(-d/R) by merely being
# near, so stretching a hop to the edge of range is nearly free in the shaping
# and ruinous in the reward -- and the policy sat on the argmax of the signal it
# was given, as with the single-drone bridge potential before the widest-path
# form. Making the kernel p(d) puts both signals on the same
# quantity, so every metre that tightens the weakest hop is paid.
#
# Deliberately NOT clamped at _P_USABLE_MIN the way the reward's LP is: the
# clamp exists there for solver sanity, and reintroducing a hard floor here
# would recreate the zero-gradient dead zone that CAPACITY_TAPER_dB removed.
# Phi*Phi is smooth, strictly positive and strictly decreasing over the arena.
#
# Constants are read from the SAME env vars max_flow_reward reads, so the two
# cannot drift; ALPHA is a module constant there and mirrored here.
SHAPE_KERNEL = (os.environ.get("OM2G_SHAPE_KERNEL", "exp") or "exp").lower()


def _envf(name, default, fallback=None):
    """Float from env, treating "" and the literal "None" as unset.

    om2g_config.om2g_env_vars() legitimately yields Python None for knobs that
    are off, and the shell export turns that into the STRING "None" -- which
    float() then raises on.
    """
    for k in (name, fallback):
        if not k:
            continue
        v = os.environ.get(k)
        if v not in (None, "", "None"):
            return _F(v)
    return _F(default)


_K_ALPHA      = 2.5   # mirrors max_flow_reward.ALPHA (same propagation model)
_K_FREQ_GHZ   = _envf("OM2G_FREQ_GHZ", "2.4")
_K_NOISE_DBM  = _envf("OM2G_NOISE_FLOOR_DBM", "-90.0")
_K_SENS_DBM   = _envf("OM2G_SENSITIVITY_DBM", "-85.0")
_K_SNIR_DB    = _envf("OM2G_SNIR_THRESHOLD_DB", "4.0")
_K_SIGMA_DB   = _envf("OM2G_PATHLOSS_SIGMA", "4.0")
_K_TX_DRONE   = _envf("OMNET_DRONE_TX_mW", "200.0")
_K_TX_USER    = _envf("OMNET_USER_TX_mW", "200.0", fallback="OMNET_DRONE_TX_mW")
_K_FSPL_1M_DB = 20.0 * math.log10(4 * math.pi * (_K_FREQ_GHZ * 1e9) / 3e8)

# Long-range gradient floor. p(d) saturates to 0 a few hundred metres past the
# decode edge, and the archipelago's clusters are >=1465 m apart, so a pure-p
# kernel would hand a drone that starts far from every anchor exactly zero
# gradient -- the dead zone CAPACITY_TAPER_dB was introduced to remove, back
# again by another door. Blending in a small exp tail keeps the potential
# strictly decreasing over the whole 3800 m arena while leaving p dominant in
# the 200-500 m band, which is where "tighten the weakest hop" is decided.
_K_EPS = _envf("OM2G_SHAPE_KERNEL_EPS", "0.05")

# ── $OM2G_SHAPE_VOR_KERNEL: the same kernel, applied to the ACCESS link ──────
# Separate flag from $OM2G_SHAPE_KERNEL on purpose. They are two different
# interventions and combining them in one run makes neither attributable.
#
# WHY (measured 2026-09-02, feature_vs_reward.py, n=160 episodes, arm-demeaned):
#   feature   r      t     partial r | qual     t
#   reach   0.698  12.25          0.585      9.06   <- STRONGEST, and it is not
#   qual    0.574   8.82            —          —       merely a proxy for qual
#   span   -0.469  -6.67         -0.363     -4.90
#   nn     -0.066  -0.83         -0.077     -0.98
# `reach` -- the fraction of drones within earshot of at least one user -- beats
# `qual` as a predictor of episode reward AND survives controlling for it. So
# coverage is the bigger lever, and the voronoi term is the one that aims at it.
#
# The reason is the same as for the hop kernel. VOR_TAU_M defaults to 300 m and
# the kernel is exp(-d/tau), but the USER radio at 0.33 mW is effectively deaf
# past ~250 m (p_user = 0.18 at 200 m, 0.011 at 300 m). So a drone 600 m from a
# user -- which cannot hear it at all -- still books exp(-2) = 0.135 of the
# credit. The potential pays for approximate proximity; the reward pays only for
# users actually heard.
#
# Note this is the OPPOSITE call from the fiedler ANCHOR hop, where exp() was
# kept deliberately because a p-shaped anchor pays "sit on a cluster" and
# reproduces hover. For coverage, paying "sit on a cluster" is the entire point.
VOR_KERNEL = (os.environ.get("OM2G_SHAPE_VOR_KERNEL", "exp") or "exp").lower()


_ERF = np.vectorize(math.erf)


def _decode_p(d_m, tx_mW):
    """P(decode) for a link of length d_m at tx_mW. Vectorised over d_m."""
    d = np.maximum(np.asarray(d_m, dtype=float), 1.0)
    rx = 10.0 * math.log10(max(tx_mW, 1e-12)) - (
        _K_FSPL_1M_DB + 10.0 * _K_ALPHA * np.log10(d))
    snr = rx - _K_NOISE_DBM
    s = _K_SIGMA_DB * math.sqrt(2.0)
    # np has no erf and scipy is not a dependency of the training worker; the
    # arrays here are 8x8 at most, so vectorising math.erf costs nothing.
    return (0.5 * (1.0 + _ERF((rx - _K_SENS_DBM) / s))
            * 0.5 * (1.0 + _ERF((snr - _K_SNIR_DB) / s)))


def _kern(d_m, tx_mW):
    """Link strength in [0, 1] under the configured kernel."""
    tail = np.exp(-np.asarray(d_m, dtype=float) / max(LINK_R_M, 1e-9))
    if SHAPE_KERNEL == "p":
        return (1.0 - _K_EPS) * _decode_p(d_m, tx_mW) + _K_EPS * tail
    return tail

_WEIGHTS = {
    "voronoi": W_VORONOI,
    "fiedler": W_FIEDLER,
    "edge":    W_EDGE,
    "spread":  W_SPREAD,
}


def active() -> bool:
    """True if any shaping term is enabled."""
    return any(abs(w) > 0.0 for w in _WEIGHTS.values())


def describe() -> str:
    on = {k: v for k, v in _WEIGHTS.items() if abs(v) > 0.0}
    if not on:
        return "shaping=off"
    # The kernel is printed unconditionally, including the default. An
    # exported variable is not necessarily an applied one, and a run's own log
    # is the only place that distinguishes "kernel=p was set" from "kernel=p
    # took effect".
    return ("shaping=on " + " ".join(f"{k}={v:g}" for k, v in sorted(on.items()))
            + f" gamma={GAMMA:g} kernel={SHAPE_KERNEL} vor_kernel={VOR_KERNEL}"
            + (f" kernel_eps={_K_EPS:g} link_r={LINK_R_M:g}"
               f" tx_drone={_K_TX_DRONE:g}mW sens={_K_SENS_DBM:g}dBm"
               if SHAPE_KERNEL == "p" else ""))


def assert_disabled(context: str = "evaluation") -> None:
    """Refuse to continue if shaping is on. Called from the eval entry point.

    Shaping changes the LEARNING signal only; a score produced with it on is
    not comparable to a baseline and must never be reported.
    """
    if active():
        raise RuntimeError(
            f"Reward shaping is ENABLED during {context}: {describe()}. "
            "Shaping is a training-only signal -- evaluation must score the "
            "true unshaped team reward. Unset OM2G_SHAPE_* and re-run."
        )


def _potentials(obs_json: dict, n_agents: int) -> np.ndarray:
    """Per-drone potential Phi_i(s). Returns zeros when shaping is off."""
    phi = np.zeros(n_agents, dtype=np.float64)
    if not active():
        return phi

    P = np.asarray(obs_json.get("positions", []), dtype=np.float64)
    if P.ndim != 2 or len(P) == 0:
        return phi
    n = min(len(P), n_agents)

    if W_VORONOI:
        U = np.asarray(obs_json.get("user_positions", []), dtype=np.float64)
        if U.ndim == 2 and len(U):
            # d[i, u] = distance from drone i to user u
            d = np.linalg.norm(P[:n, None, :] - U[None, :, :], axis=2)
            owner = np.argmin(d, axis=0)               # nearest drone per user
            # Distance discount: a user only counts for the drone that owns it,
            # and counts less the further away that drone is. Smooth in P, so
            # the ramp toward an unclaimed cluster is continuous.
            _dv = d[owner, np.arange(len(U))]
            if VOR_KERNEL == "p":
                # P(the owning drone actually hears this user), plus the same
                # small exp tail the hop kernel uses so a drone that is far from
                # every user still has a gradient pointing at one.
                val = ((1.0 - _K_EPS) * _decode_p(_dv, _K_TX_USER)
                       + _K_EPS * np.exp(-_dv / max(VOR_TAU_M, 1e-9)))
            else:
                val = np.exp(-_dv / max(VOR_TAU_M, 1e-9))
            share = np.zeros(n, dtype=np.float64)
            np.add.at(share, owner, val)
            phi[:n] += W_VORONOI * share / max(len(U), 1)

    if W_FIEDLER and n >= 2:
        # ANCHOR-BRIDGING potential (replaces the Fiedler value).
        #
        # MEASURED and REJECTED, twice:
        #   * sigmoid adjacency  -> sigma = 0.00000, no gradient at all
        #   * log(lambda2) + anchors -> PREFERRED A HUDDLE over a bridge,
        #     11.24 vs 6.72, while the task prefers the bridge 123.6 vs 19.2.
        #     Near-zero connectivity makes small absolute changes into huge
        #     RELATIVE ones, so log() rewarded a redundant drone in a tight
        #     clump more than a relay in a gap. The log fixed the scale and
        #     inverted the direction.
        #
        # What the task actually pays for is CLUSTER-TO-CLUSTER reachability
        # through the drones, so measure exactly that: for every pair of
        # cluster anchors, the strength of the best relay path between them,
        #
        #     Phi = sum over anchor pairs (a,b) of
        #             max over drones i of  min( A(a,i), A(i,b) )
        #
        # i.e. the bottleneck strength of the best one-drone bridge. A drone
        # parked between two clusters raises this a lot; eight drones stacked
        # on one cluster raise it not at all. Per-drone credit is the
        # leave-one-out drop, so the drone actually holding the bridge is the
        # one paid -- consistent with the difference reward.
        U_a = np.asarray(obs_json.get("user_positions", []), dtype=np.float64)
        if U_a.ndim == 2 and len(U_a) >= 2:
            # Anchors must be ONE PER USER-CLUSTER. np.linspace over the user
            # INDEX would assume users are ordered uniformly across clusters --
            # false under Dirichlet cluster sizes (it misses at least one
            # cluster in 7 of 12 sampled episodes and can place two anchors
            # INSIDE one cluster, which pays a drone to sit still).
            #
            # Single-linkage, not k-means: k-means with random init can report
            # a 55% bridge rate where the truth is 0%.
            anc = _cluster_centroids(U_a, _ANCHOR_LINK_M)

            def _span(idxs):
                if len(idxs) == 0:
                    return 0.0
                Q = P[list(idxs)]
                da = np.linalg.norm(anc[:, None, :] - Q[None, :, :], axis=2)
                # ANCHOR hop stays exp() under both kernels, deliberately.
                # The measured limitation (bridge_quality.py) is in the
                # drone-DRONE bottleneck, not in access. And p(d) at the user's
                # 0.33 mW is already 0.011 at 300 m, so a p anchor term would be
                # a near-cliff that pays "sit on a cluster" far more than
                # "relay" -- pushing toward the coverage-dominated hover the
                # archipelago world exists to avoid.
                Aa = np.exp(-da / max(LINK_R_M, 1e-9))          # anchors x drones
                tot = 0.0
                # WIDEST PATH, not best single hop.
                #
                # The best single hop, max_i min(A(a,i), A(i,b)), is the
                # strength of the best ONE-DRONE bridge: a lone drone at the
                # midpoint scores Phi 0.512 and a COMPLETE CHAIN also scores
                # 0.512, so finishing the chain would earn exactly zero.
                #
                # Widest path takes the bottleneck of the best route through
                # ANY number of drones, so each relay that shortens the weakest
                # hop is paid. Iterative max-min relaxation (Floyd-Warshall for
                # the bottleneck semiring); nd is 8, so the cube is free.
                nD = Q.shape[0]
                dd = np.linalg.norm(Q[:, None, :] - Q[None, :, :], axis=2)
                W = _kern(dd, _K_TX_DRONE)                # drone-drone strength
                np.fill_diagonal(W, 1.0)
                for k in range(nD):          # bottleneck Floyd-Warshall
                    W = np.maximum(W, np.minimum(W[:, k][:, None],
                                                 W[k, :][None, :]))
                for x in range(len(anc)):
                    for y in range(x + 1, len(anc)):
                        # best over entry drone i and exit drone j of
                        # min(anchor->i, widest i..j, j->anchor)
                        m = np.minimum(Aa[x][:, None], W)
                        m = np.minimum(m, Aa[y][None, :])
                        tot += float(m.max())
                return tot

            allx = list(range(n))
            full = _span(allx)
            for i in range(n):
                phi[i] += W_FIEDLER * (full - _span([j2 for j2 in allx if j2 != i]))

    if W_EDGE:
        m = max(EDGE_MARGIN_M, 1e-9)
        # How far each drone has intruded into the boundary margin, in [0, 1].
        intrusion = np.clip(
            (m - np.minimum(np.minimum(P[:n, 0], P[:n, 1]),
                            np.minimum(ARENA_M - P[:n, 0], ARENA_M - P[:n, 1]))) / m,
            0.0, 1.0)
        phi[:n] -= W_EDGE * intrusion

    if W_SPREAD and n >= 2:
        dd = np.linalg.norm(P[:n, None, :] - P[None, :n, :], axis=2)
        np.fill_diagonal(dd, np.inf)
        nearest = dd.min(axis=1)
        phi[:n] += W_SPREAD * np.clip(nearest / max(LINK_R_M, 1e-9), 0.0, 1.0)

    return phi


class Shaper:
    """Holds Phi(s) between steps so F = gamma*Phi(s') - Phi(s) can be formed.

    One instance per environment. `reset()` at episode start: the shaping term
    on the first transition uses Phi(s_0) as the baseline, and the terminal
    transition is left unshaped, which is what keeps the sum telescoping to
    gamma^T*Phi(s_T) - Phi(s_0) and the optimal policy invariant.
    """

    __slots__ = ("_prev", "_n")

    def __init__(self, n_agents: int):
        self._n = int(n_agents)
        self._prev = None

    def reset(self, obs_json: dict | None = None) -> None:
        self._prev = None if obs_json is None else _potentials(obs_json, self._n)

    def shape(self, rewards: dict, obs_json: dict, done: bool = False) -> dict:
        """Add the potential-based term to a per-agent reward dict."""
        if not active():
            return rewards
        phi = _potentials(obs_json, self._n)
        prev = self._prev
        self._prev = phi
        if prev is None or done:
            # No shaping across the episode boundary -- see the class docstring.
            return rewards
        f = GAMMA * phi - prev
        out = dict(rewards)
        for i in range(self._n):
            key = f"drone_{i}"
            if key in out:
                out[key] = float(out[key]) + float(f[i])
        return out
