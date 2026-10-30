#!/usr/bin/env python3
"""
screen_configs.py — TIER A analytic pre-screen for the learnable-task search.

No OMNeT, no LP solve. Pure closed-form + a few-thousand-draw Monte Carlo for
the one quantity (expected inter-cluster centroid distance) that has no
closed form. Screens ~100k configs in seconds, so Tier B (real OMNeT layouts,
one slurm task per surviving config) only has to touch the configs this
stage cannot already rule out.

RADIO: r_u/r_d are recomputed FRESH per config via a BISECTION on
max_flow_reward._link_capacity_bps itself (never task_validation.R_U/R_D,
which are cached at import time for the FROZEN 10mW/200mW pair and do not
apply to any other TX power -- see config_knobs.py's module docstring for the
exact mechanism). Bisection (not task_validation's closed-form inversion) is
used here deliberately, as an INDEPENDENT numerical cross-check against that
closed form -- see _max_range_bisect's docstring.

GATES (all thresholds CLI-tunable; defaults justified in each argparse help
string):
  A1_reachable          void_traverse_m < diffusive_reach_m
  A2_relay_task_alive   expected_blobs >= 2  AND  1 <= blob_hops <= N-expected_blobs
                        [relay hops are computed from blob_stats()'s own
                        single-linkage blob centroids, the same structure
                        A5/A6 use, not from NOMINAL max_clusters centre
                        separation: at L=1000 nominal separation implies a
                        relay, but users chain into ONE blob and need none.]
  A3_gradient_rich      0.5 <= spread_ratio <= 2.0
  A4_coverable          max_clusters <= N
  A5_coverage_capacity  expected_coverage_at_n >= 0.80   [grades whether
                        probe_scripted_relay's OWN constructive plan (Tier B's
                        P2) actually COVERS the users, which A3 and A4 do not
                        check; this directly predicts task_validation's own
                        C6_uplink_real/E4_coverage gate. A raw blob COUNT
                        (<=N) does not track it on real layouts -- see
                        blob_stats()'s docstring.]
  A6_nondegenerate      NOT (expected_blobs<=1 AND expected_coverage_at_1
                        >= 0.80)   [catches the opposite case: everyone
                        collapses into one blob a SINGLE drone already
                        covers -- no coordination task left]

Usage:
  # sanity check against the CURRENT frozen config -- must FAIL A1 and A3:
  python3 screen_configs.py --report

  # full screen over the search space, ranked survivors to JSON/CSV:
  python3 screen_configs.py --screen --n-samples 20000 --top 500 \\
      --out survivors.json --csv survivors.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from max_flow_reward import _link_capacity_bps, NOISE_FLOOR_dBm  # noqa: E402
from config_knobs import Config, user_spread_base  # noqa: E402
# Reused, not re-derived: task_validation._greedy_extra_coverage_positions
# and _mst_edges are pure geometry functions (no reward/LP dependency), so
# blob_stats() below imports them directly rather than carrying a second
# copy of the surplus-drone-placement / relay-spacing rules, so the gates and
# the real probe share one geometry path. A2 also uses blob_stats' own blob
# centroids; see blob_stats' docstring.
from task_validation import _greedy_extra_coverage_positions, _mst_edges  # noqa: E402


# ── Radio: independent bisection cross-check ────────────────────────────────

def _max_range_bisect(tx_mW: float, noise_floor_dBm: float = NOISE_FLOOR_dBm,
                      lo: float = 0.5, hi: float = 20000.0, iters: int = 60) -> float:
    """Largest d (m) at which _link_capacity_bps(d, tx_mW, noise_floor_dBm) is
    still > 0, found by bisection directly against the reward's OWN capacity
    function -- not by inverting its formula symbolically (that is
    task_validation._max_range_m, which this function is an independent
    cross-check against, not a replacement for: "there is exactly ONE radio
    model in this codebase" per task_validation.py's docstring; this
    bisection calls THAT SAME model, it just checks its closed-form algebra
    independently).

    _link_capacity_bps(d) is monotonically non-increasing in d (path loss
    only grows with distance), so bisection on "capacity > 0" is well-posed.
    hi=20000m is comfortably above any radius this reward's constants can
    produce (measured r_d ~ 523m at the frozen 200mW).
    """
    if _link_capacity_bps(lo, tx_mW, noise_floor_dBm) <= 0.0:
        return 0.0
    if _link_capacity_bps(hi, tx_mW, noise_floor_dBm) > 0.0:
        raise RuntimeError(
            f"_max_range_bisect: capacity still > 0 at hi={hi}m for tx_mW={tx_mW} "
            f"-- widen `hi`, the true range exceeds the search bracket.")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _link_capacity_bps(mid, tx_mW, noise_floor_dBm) > 0.0:
            lo = mid
        else:
            hi = mid
    return lo


# ── The one Monte-Carlo quantity: expected inter-cluster centroid distance ──

_D_INTER_CACHE: dict[tuple[int, float], float] = {}


def expected_d_inter(max_clusters: int, L: float, *, n_draws: int = 4000,
                     rng: random.Random | None = None) -> float:
    """Expected MAX pairwise distance between `max_clusters` cluster centres
    drawn i.i.d. uniform in [0,L]^2 -- the Tier-A stand-in for the
    inter_cluster_dist_m a real OMNeT layout would report (sweep_lib.
    inter_cluster_dist_m). No closed form for max-pairwise-distance-of-n-
    uniform-points, hence Monte Carlo; a few thousand draws is enough for a
    SCREENING quantity (Tier B re-measures this on real layouts for whatever
    survives here). With max_clusters<=1 there is no inter-cluster distance
    at all (the whole task collapses to a single blob) -- returns 0.0, same
    convention as sweep_lib.inter_cluster_dist_m.

    Cached per (max_clusters, L): this function is called once per config
    during a >>10k-config screen, and recomputing the SAME (mc, L) draw
    every time would make the Monte Carlo cost dominate a supposedly-cheap
    screen for no accuracy gain (the quantity depends on nothing else).
    """
    if max_clusters < 2:
        return 0.0
    key = (max_clusters, float(L))
    if key in _D_INTER_CACHE:
        return _D_INTER_CACHE[key]
    r = rng or random.Random(1234567 + max_clusters * 97 + int(L))
    maxd = []
    for _ in range(n_draws):
        pts = [(r.uniform(0, L), r.uniform(0, L)) for _ in range(max_clusters)]
        m = max(
            math.dist(pts[i], pts[j])
            for i in range(max_clusters)
            for j in range(i + 1, max_clusters)
        )
        maxd.append(m)
    val = float(np.mean(maxd))
    _D_INTER_CACHE[key] = val
    return val


# ── A5/A6: single-linkage-at-2*r_u blob count AND access-COVERAGE ──────────
# "Gradient-rich" (A3) and "coverable" (A4) are coupled through
# probe_scripted_relay's OWN clustering rule (task_validation.py: users i,j
# share ONE access drone iff dist(i,j) <= 2*R_U). A smoke test (N=3,
# spread_eff=150: P1=253.9 > P2=114.3, E2/E4 both FAIL) shows the coupling,
# since P2 is meant to be an upper bound on P1.
#
# Mechanism: widening spread does NOT split a cluster into more blobs than N
# on REAL OMNeT layouts. Measured (N=3, L=1000,
# mc=2, 20 real seeds per spread_eff, via task_validation.probe_scripted_
# relay on tv.get_omnet_users layouts):
#     spread_eff   median_coverage   median_n_access   infeasible
#           25.5             1.00               2.0        3/20
#           80.0             0.88               2.0        2/20
#          100.0             0.74               2.0        3/20
#          150.0             0.56               2.0        1/20
# n_access stays ~2 (at mc=2) and infeasible almost never fires -- single-
# linkage at 2*R_U does NOT split into more blobs as spread grows here, it
# CHAINS: users far out in the Gaussian tail stay within 2*R_U of an
# intermediate neighbour, so the whole nominal cluster still counts as ONE
# blob under single-linkage even while its DIAMETER grows well past what a
# single centroid access point can serve. So the limiting case is NOT "too
# many blobs, not enough drones" -- it is "few blobs, but a blob's centroid
# does not COVER a blob spread out this wide" -- exactly the C6_uplink_real /
# E4_coverage gate (mean_coverage >= 0.80), which the smoke test above does
# not pass (coverage=0.45). A pure blob
# COUNT does not predict this; blob_stats() below instead re-simulates
# probe_scripted_relay's OWN clustering + centroid + coverage pipeline via
# Monte Carlo (same style as expected_d_inter -- cheap, no OMNeT) and reports
# the coverage that pipeline would achieve, which is what A5 actually gates.
_BLOB_CACHE: dict[tuple, dict] = {}

# Same threshold task_validation.evaluate's C6_uplink_real / E4_coverage use
# (task_validation.py: mean_cov >= 0.80) -- A5 below is DIRECTLY predicting
# that gate, not inventing a new bar.
COVERAGE_MIN = 0.80


def blob_stats(max_clusters: int, n_users: int, L: float, user_spread_eff: float,
               r_u: float, n_drones: int, r_d: float, *, n_draws: int = 800,
               rng: random.Random | None = None) -> dict:
    """Monte-Carlo re-simulation of task_validation.probe_scripted_relay's
    OWN clustering+centroid+coverage+relay pipeline: `max_clusters` centres
    uniform in [0,L]^2 (same style as expected_d_inter), `n_users` split
    evenly across them, each user drawn Gaussian(centre, user_spread_eff)
    per axis and clipped to [0,L] (mirrors RLBridge.cc::samplePadreClusters's
    per-axis Gaussian + clip, minus its random cluster-COUNT/Dirichlet-SIZE
    machinery -- a Tier-A SCREEN proxy, not a re-simulation; Tier B measures
    the real thing). Single-linkage union-find at 2*r_u (task_validation.
    probe_scripted_relay's EXACT threshold, not r_u alone), then EXACTLY
    probe_scripted_relay's own next steps: sort blobs by size, take the
    largest min(#blobs, n_drones) as access points at each blob's centroid,
    spend any REMAINING drones via task_validation._greedy_extra_coverage_
    positions (imported, not re-derived), compute coverage = fraction of
    users within r_u of SOME access point (task_validation._coverage's own
    definition), AND run task_validation._mst_edges (imported, not
    re-derived) over ALL blob centroids at 0.8*r_d spacing to count relay
    hops needed to bridge them -- probe_scripted_relay's own step 3, applied
    to the REAL blob set rather than to nominal max_clusters centres.

    Surplus drones (beyond one per blob) are spent via the same imported
    greedy-coverage function probe_scripted_relay uses, so this proxy
    places every drone the real plan places.

    A2_relay_task_alive (in screen_one) takes its relay hops from the SAME
    blob centroids this function computes, not from expected_d_inter (the
    separation of NOMINAL max_clusters centres): at L=1000, nominal centre
    separation (~502m) implies 1 relay hop, but the users actually CHAIN
    into ONE blob (median_blobs=1), so probe_scripted_relay places one access
    drone and needs ZERO relays.

    Returns:
      median_blobs           -- for "needs more drones than N exist" and
                                 "is there more than one access
                                 point to bridge at all"
      median_coverage_at_n   -- coverage using min(#blobs, n_drones) access
                                 points -- what A5_coverage_capacity gates
      median_coverage_at_1   -- coverage using ONLY the single largest
                                 blob's centroid -- what A6_nondegenerate
                                 uses to detect "this is trivial for ONE
                                 drone", independent of the config's own N
      median_blob_hops       -- relay hops needed to bridge ALL
                                 blob centroids via MST at 0.8*r_d spacing --
                                 what A2_relay_task_alive now gates, instead
                                 of the nominal-cluster-centre d_inter path

    Cached per (max_clusters, n_users, L, user_spread_eff, r_u, n_drones, r_d).
    """
    key = (max_clusters, n_users, float(L), float(user_spread_eff), float(r_u), n_drones,
          float(r_d))
    if key in _BLOB_CACHE:
        return _BLOB_CACHE[key]
    r = rng or random.Random(7654321 + max_clusters * 131 + int(L) + int(user_spread_eff * 10)
                             + n_drones * 17)
    base_n = n_users // max_clusters
    rem = n_users - base_n * max_clusters
    sizes = [base_n + (1 if i < rem else 0) for i in range(max_clusters)]
    thresh = 2.0 * r_u
    blob_counts, cov_at_n, cov_at_1, blob_hops_list = [], [], [], []
    for _ in range(n_draws):
        centers = [(r.uniform(0, L), r.uniform(0, L)) for _ in range(max_clusters)]
        pts: list[tuple[float, float]] = []
        for c_idx, c in enumerate(centers):
            for _ in range(sizes[c_idx]):
                x = min(L, max(0.0, r.gauss(c[0], user_spread_eff)))
                y = min(L, max(0.0, r.gauss(c[1], user_spread_eff)))
                pts.append((x, y))
        n = len(pts)
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i in range(n):
            for j in range(i + 1, n):
                if math.dist(pts[i], pts[j]) <= thresh:
                    ra, rb = find(i), find(j)
                    if ra != rb:
                        parent[rb] = ra
        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        comps = sorted(groups.values(), key=len, reverse=True)
        blob_counts.append(len(comps))

        def centroid(idxs: list[int]) -> list[float]:
            return [sum(pts[i][0] for i in idxs) / len(idxs),
                   sum(pts[i][1] for i in idxs) / len(idxs)]

        def coverage(access_pts: list[list[float]]) -> float:
            hit = sum(1 for p in pts if any(math.dist(p, c) <= r_u for c in access_pts))
            return hit / n

        pts_as_lists = [list(p) for p in pts]

        access_n = [centroid(g) for g in comps[: min(len(comps), n_drones)]]
        n_extra = n_drones - len(access_n)
        if n_extra > 0:
            access_n = access_n + _greedy_extra_coverage_positions(
                pts_as_lists, access_n, n_extra, r_u)
        cov_at_n.append(coverage(access_n))
        cov_at_1.append(coverage([centroid(comps[0])]))

        # Relay hops to bridge ALL blob centroids (not capped at n_drones --
        # this is a property of the LAYOUT, "does the relay task exist at
        # all", answered before any drone-budget question) -- mirrors
        # probe_scripted_relay's own step-3 loop over its access list,
        # applied to the full blob centroid set.
        full_access = [centroid(g) for g in comps]
        hops = 0
        if len(full_access) >= 2:
            for (a, b) in _mst_edges(full_access):
                gap = math.dist(full_access[a], full_access[b])
                if gap <= r_d:
                    continue
                hops += max(0, int(math.ceil(gap / (0.8 * r_d))) - 1)
        blob_hops_list.append(hops)

    result = {
        "median_blobs": float(np.median(blob_counts)),
        "median_coverage_at_n": float(np.median(cov_at_n)),
        "median_coverage_at_1": float(np.median(cov_at_1)),
        "median_blob_hops": float(np.median(blob_hops_list)),
    }
    _BLOB_CACHE[key] = result
    return result


# ── Per-config analytic quantities + gates ──────────────────────────────────

def screen_one(cfg: Config, *, a3_lo: float = 0.5, a3_hi: float = 2.0,
               n_draws: int = 4000, n_blob_draws: int = 800) -> dict:
    """All intermediate quantities + gate verdicts for one config. Never
    mutates any module-global radio state (unlike config_knobs.apply_config)
    -- this stays a PURE function of cfg, since a >>10k-config screen cannot
    afford apply_config's task_validation-patching side effects, and does
    not need them: bisection reads max_flow_reward._link_capacity_bps
    directly with EXPLICIT tx_mW arguments, never a module global.
    """
    r_u = _max_range_bisect(cfg.user_tx_mW)
    r_d = _max_range_bisect(cfg.drone_tx_mW)

    # d_inter/void_traverse/relay_hops_needed (nominal max_clusters CENTRE
    # separation) remain A1_reachable's own basis -- A1 asks "can a diffusive
    # random walk physically reach the farthest incident", a question about
    # raw distance that does not care whether users later chain into one
    # blob. relay_hops_needed (nominal) is kept as a REPORTED column only
    # (below) -- it is NOT what A2 gates on any more; see blob_hops.
    d_inter = expected_d_inter(cfg.max_clusters, cfg.L, n_draws=n_draws)
    void_traverse = max(0.0, d_inter - 2.0 * r_u)
    diffusive_reach = cfg.step_m * (1.0 / math.sqrt(3.0)) * math.sqrt(cfg.ep_steps)
    relay_hops_needed = max(0, math.ceil(d_inter / (0.8 * r_d)) - 1) if r_d > 0 else 0
    spread_ratio = cfg.user_spread_eff / r_u if r_u > 0 else float("inf")
    bstats = blob_stats(cfg.max_clusters, cfg.n_users, cfg.L, cfg.user_spread_eff,
                       r_u, cfg.n_drones, r_d, n_draws=n_blob_draws)
    blobs = bstats["median_blobs"]
    coverage_at_n = bstats["median_coverage_at_n"]
    coverage_at_1 = bstats["median_coverage_at_1"]
    blob_hops = bstats["median_blob_hops"]

    gates = {
        "A1_reachable": void_traverse < diffusive_reach,
        # A2: uses blob_stats' OWN single-linkage blob centroids (the same
        # geometry A5/A6 use), not nominal max_clusters CENTRE separation
        # (d_inter/relay_hops_needed above): at L=1000 nominal separation
        # implies a relay while the real layout chains into ONE blob and
        # needs none. Both halves of the check: blobs>=2 (something to bridge at all)
        # and blob_hops in [1, N-blobs] (the actual MST relay count at
        # 0.8*R_D spacing, budgeted against drones left over after one
        # access point per blob -- mirrors probe_scripted_relay's own
        # `budget = n_drones - len(access)` before its relay loop).
        "A2_relay_task_alive": (blobs >= 2)
                              and (1 <= blob_hops <= (cfg.n_drones - blobs)),
        "A3_gradient_rich": a3_lo <= spread_ratio <= a3_hi,
        "A4_coverable": cfg.max_clusters <= cfg.n_drones,
        # A5: does probe_scripted_relay's OWN plan (min(#blobs, N) access
        # points at each blob's centroid) actually COVER the users -- this
        # is what predicts task_validation's C6_uplink_real/E4_coverage gate
        # (>=0.80), not merely "#blobs <= N" (see this section's header
        # comment: blob count alone does not catch a wide, chained blob a
        # single centroid fails to cover).
        "A5_coverage_capacity": coverage_at_n >= COVERAGE_MIN,
        # A6: catches the OPPOSITE degenerate case -- single-linkage
        # collapses to ONE blob AND that one blob is compact enough that a
        # SINGLE drone already covers everyone (coverage_at_1 high): no
        # coordination/relay task is left to learn. Deliberately independent
        # of A5/the config's own N.
        "A6_nondegenerate": not (blobs <= 1 and coverage_at_1 >= COVERAGE_MIN),
    }
    return {
        "n_drones": cfg.n_drones, "L": cfg.L, "max_clusters": cfg.max_clusters,
        "p_cross": cfg.p_cross, "user_spread_eff": cfg.user_spread_eff,
        "step_m": cfg.step_m, "drone_tx_mW": cfg.drone_tx_mW,
        "user_tx_mW": cfg.user_tx_mW, "ep_steps": cfg.ep_steps,
        "n_users": cfg.n_users,
        "r_u": r_u, "r_d": r_d, "d_inter": d_inter,
        "void_traverse_m": void_traverse, "diffusive_reach_m": diffusive_reach,
        "relay_hops_needed": relay_hops_needed, "spread_ratio": spread_ratio,
        "expected_blobs": blobs,
        "expected_coverage_at_n": coverage_at_n,
        "expected_coverage_at_1": coverage_at_1,
        "blob_hops": blob_hops,
        "gates": gates,
        "n_gates_passed": sum(gates.values()),
        "all_pass": all(gates.values()),
        # margins used for ranking survivors (larger = more comfortably inside the gate)
        "margin_A1": diffusive_reach - void_traverse,
        "margin_A2": min(blobs - 2, blob_hops - 1, (cfg.n_drones - blobs) - blob_hops),
        "margin_A3": min(spread_ratio - a3_lo, a3_hi - spread_ratio),
        "margin_A5": coverage_at_n - COVERAGE_MIN,
    }


FROZEN_CONFIG = Config(n_drones=5, L=1700.0, max_clusters=2, p_cross=0.5,
                       user_spread_eff=25.5, step_m=20.0, drone_tx_mW=200.0,
                       user_tx_mW=10.0, ep_steps=1000, n_users=17,
                       label="frozen")


def print_report(res: dict, title: str = "") -> None:
    print("=" * 78)
    print(f"TIER A REPORT{': ' + title if title else ''}")
    print("=" * 78)
    print(f"  N={res['n_drones']}  L={res['L']:.0f}  max_clusters={res['max_clusters']}  "
          f"pCross={res['p_cross']}  userSpread_eff={res['user_spread_eff']}  "
          f"step_m={res['step_m']}  drone_tx={res['drone_tx_mW']}mW  "
          f"user_tx={res['user_tx_mW']}mW  ep_steps={res['ep_steps']}")
    print(f"  r_u={res['r_u']:.1f}m  r_d={res['r_d']:.1f}m  "
          f"d_inter={res['d_inter']:.1f}m  spread_ratio={res['spread_ratio']:.3f}")
    print(f"  void_traverse_m={res['void_traverse_m']:.1f}  "
          f"diffusive_reach_m={res['diffusive_reach_m']:.1f}  "
          f"relay_hops_needed(nominal)={res['relay_hops_needed']}  "
          f"expected_blobs={res['expected_blobs']:.1f}  "
          f"blob_hops={res['blob_hops']:.1f}  "
          f"expected_coverage_at_n={res['expected_coverage_at_n']:.2f}  "
          f"expected_coverage_at_1={res['expected_coverage_at_1']:.2f}")
    for k, v in res["gates"].items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")
    n_gates = len(res["gates"])
    print(f"  ==> {res['n_gates_passed']}/{n_gates} gates pass  "
          f"({'ALL PASS' if res['all_pass'] else 'REJECT'})")
    print("=" * 78)


# ── Search-space definition ─────────────────────────────────────────────────
# Search ranges; user_spread_eff has 6 points (see that key's own comment
# below). The FULL product is
# 3*5*2*3*6*3*3*2*1 = 9,720 configs -- large but not "sample instead of
# enumerate" large for a screen this cheap (each screen_one call is a handful
# of arithmetic ops plus a cached Monte-Carlo lookup, not a fresh draw every
# time -- see expected_d_inter's cache). We therefore ENUMERATE the full grid
# rather than Sobol/random-sample it: enumeration is exhaustive (no risk of
# missing a narrow surviving pocket the way a broad space + finite random
# sample could), and at ~58k points x ~1us/point this finishes in well under
# a minute even in pure Python. Random/Sobol sampling is left available via
# --n-samples for anyone who wants to widen a dimension until the grid gets
# too large to enumerate.
DEFAULT_SPACE = {
    "n_drones": [3, 4, 5],
    "L": [800, 1000, 1200, 1400, 1700],
    "max_clusters": [2, 3],
    "p_cross": [0.1, 0.3, 0.5],
    # Includes 80 and 125 in addition to [25, 60, 100, 150]: with surplus
    # drones placed by greedy max-coverage (blob_stats, matching
    # task_validation.probe_scripted_relay), a fine continuous scan finds an
    # all-6-gate band at spread_eff~80-90 (spread_ratio~0.5-0.6) that the
    # 4-point grid {25,60,100,150} straddles but never samples.
    "user_spread_eff": [25, 60, 80, 100, 125, 150],
    "step_m": [20, 40, 60],
    "user_tx_mW": [10, 50, 100],
    "drone_tx_mW": [200, 500],
    "ep_steps": [1000],
}


def iter_grid(space: dict, n_users: int = 17) -> "itertools.chain":
    keys = list(space.keys())
    for combo in itertools.product(*(space[k] for k in keys)):
        d = dict(zip(keys, combo))
        yield Config(n_drones=d["n_drones"], L=float(d["L"]),
                    max_clusters=d["max_clusters"], p_cross=d["p_cross"],
                    user_spread_eff=float(d["user_spread_eff"]),
                    step_m=float(d["step_m"]), drone_tx_mW=float(d["drone_tx_mW"]),
                    user_tx_mW=float(d["user_tx_mW"]), ep_steps=d["ep_steps"],
                    n_users=n_users)


def iter_random(space: dict, n_samples: int, n_users: int = 17,
                seed: int = 0) -> "itertools.chain":
    rng = random.Random(seed)
    keys = list(space.keys())
    for _ in range(n_samples):
        d = {k: rng.choice(space[k]) for k in keys}
        yield Config(n_drones=d["n_drones"], L=float(d["L"]),
                    max_clusters=d["max_clusters"], p_cross=d["p_cross"],
                    user_spread_eff=float(d["user_spread_eff"]),
                    step_m=float(d["step_m"]), drone_tx_mW=float(d["drone_tx_mW"]),
                    user_tx_mW=float(d["user_tx_mW"]), ep_steps=d["ep_steps"],
                    n_users=n_users)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true",
                    help="print the Tier-A report for the CURRENT frozen config "
                         "(N=5,L=1700,mc=2,pCross=0.5,spread_eff=25.5) and exit. "
                         "Sanity check: expected to FAIL A1 and A3.")
    ap.add_argument("--screen", action="store_true",
                    help="run the full grid screen and rank survivors.")
    ap.add_argument("--n-samples", type=int, default=0,
                    help="if >0, RANDOM-sample this many configs instead of "
                         "enumerating the full grid (see DEFAULT_SPACE's "
                         "enumeration-vs-sampling comment for why the default "
                         "is full enumeration).")
    ap.add_argument("--top", type=int, default=500,
                    help="cap on survivors written out / Tier-B shard count "
                         "(200-500 survivors is the intended range).")
    ap.add_argument("--a3-lo", type=float, default=0.5,
                    help="A3_gradient_rich lower bound on spread_ratio = "
                         "userSpread_eff/r_u. Below this, users within a "
                         "cluster are packed tighter than the access radius, "
                         "so a drone anywhere near the cluster centroid "
                         "already hears everyone -- no INTRA-cluster "
                         "positioning gradient (the frozen config's own "
                         "failure mode: 25.5/157.8=0.16).")
    ap.add_argument("--a3-hi", type=float, default=2.0,
                    help="A3_gradient_rich upper bound. Above this, users "
                         "within one nominal cluster are spread wider than "
                         "the access radius itself, so 'one cluster' is "
                         "barely a cluster -- covering it starts to need "
                         "MULTIPLE access drones just for the intra-cluster "
                         "spread, conflating the intra- and inter-cluster "
                         "tasks the gate is trying to keep separate.")
    ap.add_argument("--out", type=str, default="", help="ranked survivors JSON path.")
    ap.add_argument("--csv", type=str, default="", help="ranked survivors CSV path.")
    ap.add_argument("--n-draws", type=int, default=4000,
                    help="Monte-Carlo draws for expected_d_inter per (max_clusters, L).")
    ap.add_argument("--n-blob-draws", type=int, default=800, dest="n_blob_draws",
                    help="Monte-Carlo draws for expected_blobs (A5/A6) per "
                         "(max_clusters, n_users, L, user_spread_eff, r_u).")
    ap.add_argument("--n-drones-list", type=int, nargs="+", default=None,
                    dest="n_drones_list",
                    help="OVERRIDE DEFAULT_SPACE['n_drones'] for this run only "
                         "(a reported VARIANT, e.g. --n-drones-list 3 4 5 6 7 to "
                         "test whether more drones opens the A5 bottleneck) -- "
                         "the committed default grid (n_drones=[3,4,5]) is "
                         "untouched unless you pass this.")
    a = ap.parse_args()

    if not a.report and not a.screen:
        ap.error("pass --report and/or --screen")

    if a.report:
        res = screen_one(FROZEN_CONFIG, a3_lo=a.a3_lo, a3_hi=a.a3_hi, n_draws=a.n_draws,
                         n_blob_draws=a.n_blob_draws)
        print_report(res, title="frozen config (task_validation.FROZEN_*)")
        expect_fail = (not res["gates"]["A1_reachable"]) and (not res["gates"]["A3_gradient_rich"])
        print(f"\nsanity check (expect FAIL A1 and A3): "
              f"{'AS EXPECTED' if expect_fail else 'UNEXPECTED'}")

    if a.screen:
        space = dict(DEFAULT_SPACE)
        if a.n_drones_list is not None:
            print(f"*** VARIANT: n_drones overridden to {a.n_drones_list} "
                 f"(default grid is {DEFAULT_SPACE['n_drones']}) ***")
            space["n_drones"] = a.n_drones_list
        configs = (iter_random(space, a.n_samples) if a.n_samples > 0
                  else iter_grid(space))
        n_total = 0
        results = []
        for cfg in configs:
            n_total += 1
            results.append(screen_one(cfg, a3_lo=a.a3_lo, a3_hi=a.a3_hi, n_draws=a.n_draws,
                                      n_blob_draws=a.n_blob_draws))

        n_gates = len(results[0]["gates"]) if results else 0
        survivors = [r for r in results if r["all_pass"]]
        survivors.sort(key=lambda r: (r["margin_A1"] + r["margin_A2"] + r["margin_A3"]
                                     + r["margin_A5"]),
                      reverse=True)

        print("=" * 78)
        print(f"TIER A SCREEN: {n_total} configs scanned, {len(survivors)} survive all {n_gates} gates")
        print("=" * 78)

        # which single gate eliminates the most configs (the binding constraint)
        gate_names = list(results[0]["gates"].keys()) if results else []
        fail_counts = {g: sum(1 for r in results if not r["gates"][g]) for g in gate_names}
        print("Failure counts by gate (a config can fail more than one):")
        for g, c in sorted(fail_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {g}: {c}/{n_total} fail ({c / n_total:.0%})")
        if fail_counts:
            binding = max(fail_counts, key=fail_counts.get)
            print(f"BINDING CONSTRAINT (eliminates the most configs): {binding}")

        if survivors:
            print(f"\nTop {min(5, len(survivors))} survivors "
                 f"(by margin_A1+margin_A2+margin_A3+margin_A5):")
            for r in survivors[:5]:
                print(f"  N={r['n_drones']} L={r['L']:.0f} mc={r['max_clusters']} "
                     f"pCross={r['p_cross']} spread_eff={r['user_spread_eff']:.0f} "
                     f"step_m={r['step_m']:.0f} userTX={r['user_tx_mW']:.0f} "
                     f"droneTX={r['drone_tx_mW']:.0f}  "
                     f"void={r['void_traverse_m']:.0f} reach={r['diffusive_reach_m']:.0f} "
                     f"spread_ratio={r['spread_ratio']:.2f} "
                     f"expected_blobs={r['expected_blobs']:.1f} blob_hops={r['blob_hops']:.1f} "
                     f"coverage_at_n={r['expected_coverage_at_n']:.2f}")
        else:
            print(f"\nNO CONFIG SURVIVES ALL {n_gates} GATES. Do not force a pass; see the "
                 "binding constraint printed above.")

        capped = survivors[: a.top]
        if a.out:
            with open(a.out, "w") as f:
                json.dump({"n_total": n_total, "n_survivors": len(survivors),
                          "fail_counts": fail_counts, "survivors": capped}, f, indent=2)
            print(f"\nwrote {a.out} ({len(capped)} survivor(s), capped at --top={a.top})")
        if a.csv:
            import csv as _csv
            fields = ["n_drones", "L", "max_clusters", "p_cross", "user_spread_eff",
                      "step_m", "drone_tx_mW", "user_tx_mW", "ep_steps", "r_u", "r_d",
                      "d_inter", "void_traverse_m", "diffusive_reach_m",
                      "relay_hops_needed", "spread_ratio", "expected_blobs", "blob_hops",
                      "expected_coverage_at_n", "expected_coverage_at_1",
                      "margin_A1", "margin_A2", "margin_A3", "margin_A5"]
            with open(a.csv, "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                for r in capped:
                    w.writerow(r)
            print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
