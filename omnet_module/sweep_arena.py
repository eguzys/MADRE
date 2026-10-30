#!/usr/bin/env python3
"""
sweep_arena.py — ONE-CELL arena/cluster sweep driver, against REAL OMNeT
layouts (task_validation.get_omnet_users), not a python-side sampler.

WHY THIS EXISTS
----------------
The sweep scores REAL OMNeT layouts (get_omnet_users) and starts the greedy
probe from the real episode start (spawn_staging). out/sweep_multiN.log
(2026-07-14) used a python-side Gaussian-cluster sampler that does not match
RLBridge's own C++ mt19937 (~1533 m mean offset), so this script supersedes
it. It also adds P0v, the actual random-ACTION rollout a trained policy is
compared against (P0 alone scores a single random FINAL placement, not an
agent that has to accumulate reward over an episode the way training does).

ONE CELL PER INVOCATION -- this is what makes the 24-cell grid (N in
{3,4,5} x L in {1000,1200,1400,1700} x max_clusters in {2,3}) shardable
across a slurm array (see sweep_arena.sbatch): each task owns exactly one
(N, L, max_clusters) triple and writes its own CSV + JSON, so results
harvest incrementally instead of all-or-nothing.

CRITICAL IMPORT ORDER: the $OMNET_NUM_DRONES / $OMNET_EXTRA_ARGS overrides
below MUST be set before task_validation.get_omnet_users() is ever CALLED --
omnet_env.py reads $OMNET_NUM_DRONES into its module-global MAX_N exactly
ONCE, at import time (see omnet_env.py's "Never silently falls back"
comment), and _launch_omnet() reads $OMNET_EXTRA_ARGS fresh per-launch but
there is no reason to race that either. We set both at the very top of this
file, before importing task_validation, so this is never fragile to a future
refactor that imports omnet_env eagerly.

Usage:
    python3 sweep_arena.py --n-drones 5 --arena 1700 --max-clusters 2 \\
        --seeds 20 --seed-start 0 --out sweep_N5_L1700_mc2.csv \\
        --json sweep_N5_L1700_mc2.json --verbose
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys

import numpy as np


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-drones", type=int, required=True, dest="n_drones")
    ap.add_argument("--arena", type=float, required=True, dest="L",
                    help="arena side length L, metres (square arena).")
    ap.add_argument("--max-clusters", type=int, required=True, dest="max_clusters")
    ap.add_argument("--users", type=int, default=17,
                    help="numUsers -- stays 17 across the grid (verified fact).")
    ap.add_argument("--seeds", type=int, required=True,
                    help="number of seeds this invocation scores.")
    ap.add_argument("--seed-start", type=int, default=0, dest="seed_start",
                    help="first seed; scores seed_start..seed_start+seeds-1. "
                         "Exists so an array task can take a disjoint seed "
                         "slice of one cell -- CSVs concatenate since every "
                         "row carries its own seed.")
    ap.add_argument("--out", type=str, required=True, help="per-seed CSV path.")
    ap.add_argument("--json", type=str, default="", help="cell-level JSON path.")
    ap.add_argument("--greedy-steps", type=int, default=60,
                    help="P1 greedy rollout length (task_validation.probe_greedy's own default).")
    ap.add_argument("--rv-steps", type=int, default=1000,
                    help="P0v rollout length in ticks. 1000 = a full training episode.")
    ap.add_argument("--rv-eval-every", type=int, default=10,
                    help="P0v reward-evaluation stride -- the LP solve inside "
                         "compute_position_oracle_reward, not the position "
                         "update, is the expensive part, so we do not score "
                         "every one of the 1000 ticks.")
    ap.add_argument("--anchor", type=int, default=4,
                    help="the TRAINING N (task_validation's Tier-1 anchor). "
                         "Any other --n-drones is scored with Tier-2 "
                         "(evaluate_tier2) instead, exactly as task_validation.py does.")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


_ARGS = _parse_args()

# ── Set the CLI-override env vars BEFORE importing task_validation ─────────
# See the module docstring's ordering note. --*.rlBridge.arenaWidth/Height
# are PLAIN doubles (no unit suffix; omnetpp.ini has "1700.0"); the mobility
# constraintAreaMax* pair needs a "m" suffix (omnetpp.ini has "1700m").
# RLBridge.cc:149's arenaWidth_ = par("arenaWidth") is a SEPARATE parameter
# from mobility's constraintAreaMax* -- cluster sampling (RLBridge.cc:415,
# the userSpread scale-by-arena term) reads arenaWidth_, so BOTH must be
# overridden or the user layout will not actually change even though the
# drones' allowed roaming box did.
os.environ["OMNET_NUM_DRONES"] = str(_ARGS.n_drones)
os.environ["OMNET_EXTRA_ARGS"] = (
    f"--*.rlBridge.arenaWidth={_ARGS.L} "
    f"--*.rlBridge.arenaHeight={_ARGS.L} "
    f"--**.mobility.constraintAreaMaxX={_ARGS.L}m "
    f"--**.mobility.constraintAreaMaxY={_ARGS.L}m "
    f"--*.rlBridge.maxClusters={_ARGS.max_clusters}"
)

# Make omnet_module importable regardless of CWD (mirrors evaluate_omnet.py).
_OMNET_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if _OMNET_MODULE_DIR not in sys.path:
    sys.path.insert(0, _OMNET_MODULE_DIR)

import task_validation as tv  # noqa: E402  (after the env vars above, see docstring)
from greedy_oracle import STEP_M as GREEDY_STEP_M  # noqa: E402  -- the one true step size (20.0 m)

# ── P0v / greedy-trace / cluster-geometry helpers ───────────────────────────
# These five helpers live in sweep_lib.py, shared verbatim with
# sweep_config.py (the Tier-B config sweep), so there is one copy of this
# logic. `tv` (this module's task_validation import, above) is threaded
# through explicitly since sweep_lib.py itself imports neither task_validation
# nor greedy_oracle -- see each function's docstring for why.
from sweep_lib import (  # noqa: E402
    diffusive_reach_m,
    probe_random_velocity as _probe_random_velocity_impl,
    probe_greedy_traced,
    cluster_centroids as _cluster_centroids,
    inter_cluster_dist_m as _inter_cluster_dist_m,
    decompose_intra_cross_reward as _decompose_intra_cross_reward_impl,
)


def probe_random_velocity(rng: random.Random, start_positions: list[list[float]],
                          users: list[list[float]], areas: list[int],
                          pairs: list[list[int]], L: float,
                          steps: int = 1000, step_m: float = 20.0,
                          eval_every: int = 10) -> tuple[float, float]:
    """Thin wrapper binding sweep_lib's probe_random_velocity to THIS module's
    `tv` (task_validation), so every existing call site below is unchanged."""
    return _probe_random_velocity_impl(rng, start_positions, users, areas, pairs, L,
                                       steps=steps, step_m=step_m, eval_every=eval_every, tv=tv)


def _probe_greedy_traced(obs: dict, n_drones: int, L: float,
                         steps: int = 60, step_m: float = 20.0
                         ) -> tuple[list[list[float]], list[float]]:
    """Thin wrapper binding sweep_lib's probe_greedy_traced to THIS module's `tv`."""
    return probe_greedy_traced(tv, obs, n_drones, L, steps=steps, step_m=step_m)


def _decompose_intra_cross_reward(positions: list[list[float]],
                                  users: list[list[float]], areas: list[int],
                                  pairs: list[list[int]]) -> tuple[float, float]:
    """Thin wrapper binding sweep_lib's decompose_intra_cross_reward to THIS module's `tv`."""
    return _decompose_intra_cross_reward_impl(tv, positions, users, areas, pairs)


FIELDNAMES = [
    "seed", "P0", "P0v", "P1", "P2", "relays", "coverage", "infeasible",
    "n_clusters_actual", "inter_cluster_dist_m", "void_traverse_m",
    "diffusive_reach_m", "reach_ok", "intra_reward_P2", "cross_reward_P2",
    "frac_cross_reward", "frac_zero_steps_P0v", "frac_zero_steps_P1",
    "P1_start_reward", "P1_net_displacement_m",
]


def run_cell(a: argparse.Namespace) -> tuple["tv.Cell", list[dict]]:
    cell = tv.Cell(L=a.L)
    rows: list[dict] = []
    reach_val = diffusive_reach_m(GREEDY_STEP_M, a.rv_steps)

    for s in range(a.seed_start, a.seed_start + a.seeds):
        # REAL users/areas/pairs off a live OMNeT episode, at THIS cell's
        # geometry -- expect_geometry is the loud-verification hook added to
        # get_omnet_users for exactly this sweep (task_validation.py).
        users, areas, pairs, actual_seed = tv.get_omnet_users(
            s, a.users, a.n_drones, a.L, a.max_clusters,
            expect_geometry=(a.L, a.n_drones, a.users, a.max_clusters))
        tv.assert_users_match_omnet(users, users, seed=actual_seed)

        # Probe-placement randomness ONLY, never the user layout -- same
        # discipline as task_validation.run_cell, folded into one stream
        # keyed on (L, n_drones, max_clusters, seed) since this driver sweeps
        # more dimensions than task_validation.py's own run_cell does.
        rng = random.Random(
            hash(("probe", int(a.L), a.n_drones, a.max_clusters, s)) & 0xFFFFFFFF)

        p0_pos = tv.probe_random(rng, a.n_drones, a.L)
        p0 = tv.score(p0_pos, users, areas, pairs)

        start = tv.spawn_staging(rng, a.n_drones, users, areas)
        p1_start_reward = tv.score(start, users, areas, pairs)

        p0v_mean, p0v_frac_zero = probe_random_velocity(
            rng, start, users, areas, pairs, a.L,
            steps=a.rv_steps, step_m=GREEDY_STEP_M, eval_every=a.rv_eval_every)

        obs = {"positions": [p[:] for p in start], "user_positions": users,
              "user_area": areas, "sender_pairs": pairs}
        p1_pos, p1_trace = _probe_greedy_traced(
            obs, a.n_drones, a.L, steps=a.greedy_steps, step_m=GREEDY_STEP_M)
        p1 = tv.score(p1_pos, users, areas, pairs)
        # Edge case: greedy_velocities found no improving move on step 0 and
        # broke before ever calling the traced scoring line, so the trace is
        # empty and p1_pos == start. Fall back to the (rare) single-point
        # judgement of the final score itself rather than dividing by zero.
        frac_zero_p1 = (float(np.mean([r == 0.0 for r in p1_trace]))
                       if p1_trace else float(p1 == 0.0))
        p1_disp = float(np.mean([math.dist(sp, fp) for sp, fp in zip(start, p1_pos)]))

        plan = tv.probe_scripted_relay(users, a.n_drones, a.L)
        p2 = tv.score(plan.positions, users, areas, pairs)
        intra, cross = _decompose_intra_cross_reward(plan.positions, users, areas, pairs)
        frac_cross = cross / (intra + cross) if (intra + cross) > 0 else 0.0

        n_clusters_actual = len(set(areas))
        icd = _inter_cluster_dist_m(users, areas)
        # The stretch of the journey between clusters with EXACTLY zero
        # reward gradient: two clusters separated by less than 2*R_U can
        # already both be served from points inside r_u of each other's edge,
        # so there is no dead zone at all in that case (hence the max(0, ...)).
        void = max(0.0, icd - 2.0 * tv.R_U)
        reach_ok = void <= reach_val

        cell.p0.append(p0)
        cell.p1.append(p1)
        cell.p2.append(p2)
        cell.relays.append(plan.n_relay)
        cell.cov.append(plan.coverage)
        cell.infeasible += int(plan.infeasible)
        cell.n += 1

        row = {
            "seed": s, "P0": p0, "P0v": p0v_mean, "P1": p1, "P2": p2,
            "relays": plan.n_relay, "coverage": plan.coverage,
            "infeasible": int(plan.infeasible),
            "n_clusters_actual": n_clusters_actual,
            "inter_cluster_dist_m": icd, "void_traverse_m": void,
            "diffusive_reach_m": reach_val, "reach_ok": reach_ok,
            "intra_reward_P2": intra, "cross_reward_P2": cross,
            "frac_cross_reward": frac_cross,
            "frac_zero_steps_P0v": p0v_frac_zero,
            "frac_zero_steps_P1": frac_zero_p1,
            "P1_start_reward": p1_start_reward,
            "P1_net_displacement_m": p1_disp,
        }
        rows.append(row)

        if a.verbose:
            print(f"  seed={s:4d}  P0={p0:9.1f}  P0v={p0v_mean:9.1f}  "
                 f"P1={p1:9.1f}  P2={p2:9.1f}  relays={plan.n_relay}  "
                 f"clusters={n_clusters_actual}  void={void:7.0f}m  "
                 f"reach_ok={reach_ok}"
                 f"{'  INFEASIBLE' if plan.infeasible else ''}", flush=True)

    return cell, rows


def _write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(a: argparse.Namespace, cell: "tv.Cell", rows: list[dict]) -> dict:
    """Same visual style as out/sweep_multiN.log, so this cell's output is
    comparable at a glance against that log -- plus the P0v/R1/R2 lines."""
    print("=" * 78)
    print("ARENA SWEEP CELL — real OMNeT layouts (get_omnet_users), P0v added")
    print("=" * 78)
    print(f"radio (from max_flow_reward constants, single source of truth):")
    print(f"  user TX  {tv.USER_TX_mW:6.1f} mW   ->  r_u = {tv.R_U:7.1f} m")
    print(f"  drone TX {tv.DRONE_TX_mW:6.1f} mW   ->  r_d = {tv.R_D:7.1f} m   "
          f"asymmetry r_d/r_u = {tv.R_D / tv.R_U:.2f}x")
    is_anchor = (a.n_drones == a.anchor)
    print(f"  N={a.n_drones} drones (anchor={a.anchor} -> TIER {1 if is_anchor else 2}), "
          f"L={a.L:.0f} m, max_clusters={a.max_clusters}, {a.users} users, "
          f"seeds {a.seed_start}..{a.seed_start + a.seeds - 1}")
    print()

    if is_anchor:
        ok, st = tv.evaluate(cell)
        st["tier"] = 1
    else:
        ok, st = tv.evaluate_tier2(cell, a.n_drones)
        st["tier"] = 2
        _, t1 = tv.evaluate(cell)          # descriptive only; not a gate at eval N
        st["median_P0"] = t1["median_P0"]
        st["median_P2"] = t1["median_P2"]
        st["P2_over_P1"] = t1["P2_over_P1"]
    st["N"] = a.n_drones
    st["max_clusters"] = a.max_clusters

    p0v = np.array([r["P0v"] for r in rows])
    void = np.array([r["void_traverse_m"] for r in rows])
    reach_ok_arr = np.array([bool(r["reach_ok"]) for r in rows])
    frac_cross = np.array([r["frac_cross_reward"] for r in rows])
    frac_zero_p0v = np.array([r["frac_zero_steps_P0v"] for r in rows])
    frac_zero_p1 = np.array([r["frac_zero_steps_P1"] for r in rows])
    n_clusters_actual = np.array([r["n_clusters_actual"] for r in rows])

    st["median_P0v"] = float(np.median(p0v))
    st["median_void_traverse_m"] = float(np.median(void))
    st["diffusive_reach_m"] = float(reach_val := diffusive_reach_m(GREEDY_STEP_M, a.rv_steps))
    st["median_frac_cross_reward"] = float(np.median(frac_cross))
    st["median_frac_zero_steps_P0v"] = float(np.median(frac_zero_p0v))
    st["median_frac_zero_steps_P1"] = float(np.median(frac_zero_p1))
    st["median_n_clusters_actual"] = float(np.median(n_clusters_actual))

    # R1 -- median reach_ok is True (majority of seeds have a void the
    # diffusive walk could plausibly cross in one episode).
    st["R1_reachable"] = bool(np.median(reach_ok_arr.astype(float)) >= 0.5)
    # R2 -- the same comparison at the level of medians directly (per spec):
    # is the TYPICAL void within the TYPICAL diffusive reach.
    st["R2_prize_reachable"] = bool(st["median_void_traverse_m"] <= reach_val)

    print(f"  [TIER {st['tier']}]  P0 {st['median_P0']:9.1f}   P0v {st['median_P0v']:9.1f}   "
          f"P1 {st['median_P1']:9.1f}   P2 {st['median_P2']:9.1f}   "
          f"(P2/P1 = {st['P2_over_P1']:.2f}x)")
    print(f"  infeasible {st['infeasible_rate']:.0%}   median relays {st['median_relays']:.1f}   "
          f"coverage {st['mean_coverage']:.2f}"
          + (f"   greedy zeros {st.get('greedy_zero_frac', 0.0):.0%}" if not is_anchor else ""))
    print(f"  median clusters {st['median_n_clusters_actual']:.1f}   "
          f"median void_traverse {st['median_void_traverse_m']:7.0f} m   "
          f"diffusive_reach {reach_val:7.0f} m")
    print(f"  frac_cross_reward (median) {st['median_frac_cross_reward']:.2f}   "
          f"frac_zero_steps_P0v (median) {st['median_frac_zero_steps_P0v']:.2f}   "
          f"frac_zero_steps_P1 (median) {st['median_frac_zero_steps_P1']:.2f}")
    for k, v in st["criteria"].items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")
    print(f"    {'PASS' if st['R1_reachable'] else 'FAIL'}  R1_reachable")
    print(f"    {'PASS' if st['R2_prize_reachable'] else 'FAIL'}  R2_prize_reachable")
    overall = bool(ok and st["R1_reachable"] and st["R2_prize_reachable"])
    st["overall_PASS"] = overall
    print(f"  ==> {'PASS' if overall else 'REJECT'}")
    print("=" * 78, flush=True)
    return st


def main() -> None:
    a = _ARGS
    cell, rows = run_cell(a)
    _write_csv(a.out, rows)
    print(f"wrote {len(rows)} row(s) to {a.out}")
    st = print_summary(a, cell, rows)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(st, f, indent=2)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
