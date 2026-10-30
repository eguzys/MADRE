#!/usr/bin/env python3
"""
sweep_config.py — TIER B real-layout validation, ONE CONFIG PER INVOCATION.

Sibling to sweep_arena.py (not a rewrite of it -- see sweep_lib.py for the
probe helpers shared between the two, extracted specifically so this file
does not carry a second, subtly-different copy of that logic). Where
sweep_arena.py sweeps three knobs (N, L, max_clusters) by hand-building
$OMNET_EXTRA_ARGS itself, this script accepts EVERY knob in config_knobs.
Config as a CLI arg and drives both the OMNeT-side override string and the
Python-oracle module constants through config_knobs.apply_config -- the
single function that keeps the three dual-site knobs (drone speed, drone TX,
user TX) in sync between what the simulator will actually do and what the
probes' reward model assumes.

CRITICAL IMPORT ORDER (same discipline as sweep_arena.py's own docstring):
config_knobs.apply_config() is called at the very top of this file, BEFORE
task_validation is ever imported -- omnet_env.py reads $OMNET_NUM_DRONES into
its module-global MAX_N exactly ONCE, at import time, and omnet_env.py is
only ever imported lazily (inside task_validation.get_omnet_users), but
there is no reason to race that ordering either.

GATE 4 (signal-vs-noise) -- THE ONE THIS SCRIPT ADDS THAT SWEEP_ARENA.PY DOES
NOT HAVE. Definition (defined here only, do not redefine elsewhere):
  For each seed s, collect the three probe values {P0v_s, P1_s, P2_s} (the
  "policies" a config's real layout produces at that seed).
    policy_std  = mean over seeds of std({P0v_s, P1_s, P2_s})   -- the spread
                  ACROSS POLICIES, typical size, at a fixed layout.
    seed_std    = std over seeds of mean({P0v_s, P1_s, P2_s})  -- the spread
                  ACROSS LAYOUTS of the per-seed average, i.e. how much the
                  measured level moves just from which layout you drew.
    snr_ratio   = policy_std / seed_std
  Gate B4_signal_over_noise: snr_ratio >= --b4-min-snr (default 1.0). Below
  1.0, a training curve's seed-to-seed layout variance is LARGER than the
  gap between policies it is trying to measure -- as measured at the frozen
  config (snr_ratio ~ 1/8.6 = 0.12).

B3_gradient_rich. A zero-reward-fraction form, median(frac_zero_
steps_P0v) <= 0.10 ("mostly not sitting at zero reward"), is NOT used.
Measured (8 seeds each, fixed geometry, step_m swept):

    geometry                       step_m=20   step_m=40   step_m=60
    frozen (L=1700,spread=25.5)         0.22        0.37        0.53
    good candidate (L=1200,spread=100)  0.00        0.08        0.23

Two independent problems, both confirmed by this table:
  (1) DOES NOT DISCRIMINATE: at each config's own OPERATING step_m (20 for
      frozen, 60 for the good candidate -- step_m=60 is what A1_reachable
      needs at that geometry), the two measure ~0.22-0.26 vs ~0.23-0.25 --
      indistinguishable, even though one is known-unlearnable and the other
      is a genuine Tier-A survivor.
  (2) STRUCTURALLY FIGHTS A1: at FIXED geometry, frac_zero_steps_P0v rises
      monotonically and steeply with step_m alone (0.00->0.08->0.23; 0.22->
      0.37->0.53) -- it is dominated by how far a RANDOM walk wanders off
      the users, which is exactly what A1_reachable (Tier A) needs turned UP.
      That form would reject good configs at Tier B, the expensive stage,
      for satisfying a DIFFERENT, already-passed gate.

TWO CANDIDATE REPLACEMENTS WERE MEASURED (both computable from the SAME P0v
rollout records, no extra LP solves -- sweep_lib.py):

  (b) variance_decomposition(): law of total variance splits Var(reward)
      into a CONNECTIVITY component (reward moving because a commodity
      connects/disconnects, the log1p-quantised ~18.5-unit step)
      and a THROUGHPUT component (reward moving while
      connectivity is HELD FIXED, grouped by connected_count). DISCARDED
      after measuring it (10 seeds each): median throughput_frac was
      0.0004-0.02 for BOTH the frozen config AND the good L=1200 candidate
      -- near zero either way, and if anything slightly LOWER for the good
      candidate. Reason: log1p compresses a connection's worth
      (~18.5) so far above a throughput doubling's (~0.69) that the
      connectivity term swamps the throughput term BY CONSTRUCTION, for any
      config, under this reward -- (b) cannot discriminate anything. Kept
      as a reported diagnostic (median_throughput_frac) precisely because
      its own failure to discriminate is the evidence for choosing (a).

  (a) coefficient_of_variation_connected(): std(reward | reward>0) /
      mean(reward | reward>0) over the SAME rollout -- CHOSEN. Measured
      (10 seeds each, same two configs):
          frozen config      (L=1700, spread=25.5, step_m=20, N=5): median 0.30
                              (per-seed range 0.15-0.46)
          good L=1200 candidate (spread=100, step_m=60, N=5):       median 0.60
                              (per-seed range 0.44-0.88)
      A clean, consistent separation (the two ranges barely overlap at their
      tails). Default --b3-min-cv-connected=0.45 sits at the midpoint of the
      two medians (0.30+0.60)/2 -- auditable, not magic; re-measure before
      trusting this threshold on a materially different geometry.

median_frac_zero_steps_P0v is still REPORTED every run (it remains a useful
diagnostic of "how often is this config at zero reward"), it is simply not a
gate -- same treatment now applies to median_throughput_frac (b), reported
but not gated, for the reason above.

CAVEAT: P2 (tv.probe_scripted_relay) is only a valid UPPER BOUND
on P1 while its OWN coverage stays >= 0.80 (screen_configs.py's
A5_coverage_capacity gate predicts this DIRECTLY, not via a blob-count proxy
-- see blob_stats()'s docstring there: single-linkage at 2*R_U usually
CHAINS rather than splits on real layouts, so "#blobs <= N" looked fine even
when coverage was not). Outside that regime P2's centroid-per-blob plan
under-covers a wide blob and can score BELOW P1, so the C3/E2/E4 verdicts
this script reports do not apply to such a config. Example from a Tier-B run (N=3, spread_eff=150): P1=253.9 > P2=114.3,
coverage=0.45. Only trust this script's C3/E2/E4 columns for configs that
passed Tier A's A5 gate.

Usage:
    python3 sweep_config.py --n-drones 4 --arena 1000 --max-clusters 2 \\
        --p-cross 0.1 --user-spread-eff 150 --step-m 60 \\
        --user-tx-mw 10 --drone-tx-mw 200 --ep-steps 1000 \\
        --seeds 40 --out cfg_0001.csv --json cfg_0001.json --verbose
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
    ap.add_argument("--arena", type=float, required=True, dest="L")
    ap.add_argument("--max-clusters", type=int, required=True, dest="max_clusters")
    ap.add_argument("--p-cross", type=float, default=0.5, dest="p_cross")
    ap.add_argument("--user-spread-eff", type=float, default=25.5, dest="user_spread_eff",
                    help="EFFECTIVE user spread (metres); config_knobs converts "
                         "to the ini's base-then-scaled-by-arena parameter for you.")
    ap.add_argument("--step-m", type=float, default=20.0, dest="step_m",
                    help="drone speed (m/s, 1s tick) == greedy_oracle.STEP_M. "
                         "Dual-site knob -- see config_knobs.py.")
    ap.add_argument("--user-tx-mw", type=float, default=10.0, dest="user_tx_mW")
    ap.add_argument("--drone-tx-mw", type=float, default=200.0, dest="drone_tx_mW")
    ap.add_argument("--ep-steps", type=int, default=1000, dest="ep_steps")
    ap.add_argument("--users", type=int, default=17, dest="n_users")
    ap.add_argument("--seeds", type=int, default=40,
                    help="seeds this invocation scores. 30-50 is the "
                         "documented right range given the measured "
                         "seed-to-seed layout variance (see gate 4).")
    ap.add_argument("--seed-start", type=int, default=0, dest="seed_start")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--greedy-steps", type=int, default=60)
    ap.add_argument("--rv-eval-every", type=int, default=10, dest="rv_eval_every")
    ap.add_argument("--anchor", type=int, default=4,
                    help="TRAINING N for Tier-1 (task_validation.evaluate); any "
                         "other --n-drones uses Tier-2 (evaluate_tier2), same "
                         "convention as sweep_arena.py / task_validation.py.")
    ap.add_argument("--b2-min-ratio", type=float, default=1.3,
                    help="B2_nontrivial: median(P1)/median(P0v) must be >= this.")
    ap.add_argument("--b3-min-cv-connected", type=float, default=0.45,
                    dest="b3_min_cv_connected",
                    help="B3_gradient_rich: median(cv_connected) must be >= this. "
                         "See the module docstring's B3 note for how "
                         "0.45 was calibrated (frozen config ~0.30 vs a good "
                         "candidate ~0.60 -- this sits at their midpoint).")
    ap.add_argument("--b4-min-snr", type=float, default=1.0,
                    help="B4_signal_over_noise: snr_ratio (policy_std/seed_std, "
                         "see module docstring) must be >= this.")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


_ARGS = _parse_args()

# ── THE single dual-site setter call. Must run before task_validation (and
# therefore before omnet_env) is ever imported -- see module docstring. ─────
_OMNET_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if _OMNET_MODULE_DIR not in sys.path:
    sys.path.insert(0, _OMNET_MODULE_DIR)

from config_knobs import Config, apply_config  # noqa: E402

_CFG = Config(
    n_drones=_ARGS.n_drones, L=_ARGS.L, max_clusters=_ARGS.max_clusters,
    p_cross=_ARGS.p_cross, user_spread_eff=_ARGS.user_spread_eff,
    step_m=_ARGS.step_m, drone_tx_mW=_ARGS.drone_tx_mW, user_tx_mW=_ARGS.user_tx_mW,
    ep_steps=_ARGS.ep_steps, n_users=_ARGS.n_users,
)
_APPLIED = apply_config(_CFG)

import task_validation as tv  # noqa: E402  (after apply_config, see docstring)
from sweep_lib import (  # noqa: E402
    diffusive_reach_m, probe_random_velocity, probe_greedy_traced,
    inter_cluster_dist_m, decompose_intra_cross_reward, variance_decomposition,
    coefficient_of_variation_connected,
)

FIELDNAMES = [
    "seed", "P0", "P0v", "P1", "P2", "relays", "coverage", "infeasible",
    "n_clusters_actual", "inter_cluster_dist_m", "void_traverse_m",
    "diffusive_reach_m", "reach_ok", "intra_reward_P2", "cross_reward_P2",
    "frac_cross_reward", "frac_zero_steps_P0v", "frac_zero_steps_P1",
    "P1_start_reward", "P1_net_displacement_m", "policy_std_this_seed",
    "policy_mean_this_seed", "cv_connected", "throughput_frac",
    "connectivity_var", "throughput_var",
]


def run_config(a: argparse.Namespace) -> tuple["tv.Cell", list[dict]]:
    cell = tv.Cell(L=a.L)
    rows: list[dict] = []
    reach_val = diffusive_reach_m(a.step_m, a.ep_steps)

    for s in range(a.seed_start, a.seed_start + a.seeds):
        users, areas, pairs, actual_seed = tv.get_omnet_users(
            s, a.n_users, a.n_drones, a.L, a.max_clusters,
            expect_geometry=(a.L, a.n_drones, a.n_users, a.max_clusters))
        tv.assert_users_match_omnet(users, users, seed=actual_seed)

        rng = random.Random(
            hash(("probe-cfg", int(a.L), a.n_drones, a.max_clusters,
                 a.p_cross, a.user_spread_eff, a.step_m, a.user_tx_mW,
                 a.drone_tx_mW, s)) & 0xFFFFFFFF)

        p0_pos = tv.probe_random(rng, a.n_drones, a.L)
        p0 = tv.score(p0_pos, users, areas, pairs)

        start = tv.spawn_staging(rng, a.n_drones, users, areas)
        p1_start_reward = tv.score(start, users, areas, pairs)

        p0v_mean, p0v_frac_zero, p0v_records = probe_random_velocity(
            rng, start, users, areas, pairs, a.L,
            steps=a.ep_steps, step_m=a.step_m, eval_every=a.rv_eval_every, tv=tv,
            collect_records=True)
        vardecomp = variance_decomposition(p0v_records)
        cv_connected = coefficient_of_variation_connected(p0v_records)

        obs = {"positions": [p[:] for p in start], "user_positions": users,
              "user_area": areas, "sender_pairs": pairs}
        p1_pos, p1_trace = probe_greedy_traced(
            tv, obs, a.n_drones, a.L, steps=a.greedy_steps, step_m=a.step_m)
        p1 = tv.score(p1_pos, users, areas, pairs)
        frac_zero_p1 = (float(np.mean([r == 0.0 for r in p1_trace]))
                       if p1_trace else float(p1 == 0.0))
        p1_disp = float(np.mean([math.dist(sp, fp) for sp, fp in zip(start, p1_pos)]))

        plan = tv.probe_scripted_relay(users, a.n_drones, a.L)
        p2 = tv.score(plan.positions, users, areas, pairs)
        intra, cross = decompose_intra_cross_reward(tv, plan.positions, users, areas, pairs)
        frac_cross = cross / (intra + cross) if (intra + cross) > 0 else 0.0

        n_clusters_actual = len(set(areas))
        icd = inter_cluster_dist_m(users, areas)
        void = max(0.0, icd - 2.0 * tv.R_U)
        reach_ok = void <= reach_val

        policy_vals_this_seed = np.array([p0v_mean, p1, p2])
        policy_std_this_seed = float(np.std(policy_vals_this_seed, ddof=0))
        policy_mean_this_seed = float(np.mean(policy_vals_this_seed))

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
            "policy_std_this_seed": policy_std_this_seed,
            "policy_mean_this_seed": policy_mean_this_seed,
            "cv_connected": cv_connected,
            "throughput_frac": vardecomp["throughput_frac"],
            "connectivity_var": vardecomp["connectivity_var"],
            "throughput_var": vardecomp["throughput_var"],
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
    print("=" * 78)
    print("CONFIG SWEEP (Tier B) — real OMNeT layouts, full knob set, gate 4 (SNR)")
    print("=" * 78)
    print(f"applied dual-site: {_APPLIED['extra_args']}")
    print(f"radio (recomputed for THIS config's own TX powers):")
    print(f"  user TX  {a.user_tx_mW:6.1f} mW   ->  r_u = {tv.R_U:7.1f} m")
    print(f"  drone TX {a.drone_tx_mW:6.1f} mW   ->  r_d = {tv.R_D:7.1f} m")
    is_anchor = (a.n_drones == a.anchor)
    print(f"  N={a.n_drones} (anchor={a.anchor} -> TIER {1 if is_anchor else 2}), "
          f"L={a.L:.0f}, mc={a.max_clusters}, pCross={a.p_cross}, "
          f"spread_eff={a.user_spread_eff}, step_m={a.step_m}, "
          f"seeds {a.seed_start}..{a.seed_start + a.seeds - 1}")
    print()

    if is_anchor:
        ok, st = tv.evaluate(cell)
        st["tier"] = 1
    else:
        ok, st = tv.evaluate_tier2(cell, a.n_drones)
        st["tier"] = 2
        _, t1 = tv.evaluate(cell)
        st["median_P0"] = t1["median_P0"]
        st["median_P2"] = t1["median_P2"]
        st["P2_over_P1"] = t1["P2_over_P1"]
    st["N"] = a.n_drones
    st["L"] = a.L
    st["max_clusters"] = a.max_clusters
    st["p_cross"] = a.p_cross
    st["user_spread_eff"] = a.user_spread_eff
    st["step_m"] = a.step_m
    st["user_tx_mW"] = a.user_tx_mW
    st["drone_tx_mW"] = a.drone_tx_mW
    st["ep_steps"] = a.ep_steps
    st["r_u"] = tv.R_U
    st["r_d"] = tv.R_D

    p0v = np.array([r["P0v"] for r in rows])
    p1 = np.array([r["P1"] for r in rows])
    p2 = np.array([r["P2"] for r in rows])
    void = np.array([r["void_traverse_m"] for r in rows])
    reach_ok_arr = np.array([bool(r["reach_ok"]) for r in rows])
    frac_cross = np.array([r["frac_cross_reward"] for r in rows])
    frac_zero_p0v = np.array([r["frac_zero_steps_P0v"] for r in rows])
    frac_zero_p1 = np.array([r["frac_zero_steps_P1"] for r in rows])
    n_clusters_actual = np.array([r["n_clusters_actual"] for r in rows])
    throughput_frac = np.array([r["throughput_frac"] for r in rows])
    cv_connected = np.array([r["cv_connected"] for r in rows])

    st["median_P0v"] = float(np.median(p0v))
    st["median_P1"] = float(np.median(p1))
    st["median_P2"] = float(np.median(p2))
    st["P1_over_P0v"] = (st["median_P1"] / st["median_P0v"]) if st["median_P0v"] > 0 else float("inf")
    st["P2_over_P1"] = (st["median_P2"] / st["median_P1"]) if st["median_P1"] > 0 else float("inf")
    st["median_void_traverse_m"] = float(np.median(void))
    st["diffusive_reach_m"] = float(reach_val := diffusive_reach_m(a.step_m, a.ep_steps))
    st["median_frac_cross_reward"] = float(np.median(frac_cross))
    st["median_frac_zero_steps_P0v"] = float(np.median(frac_zero_p0v))
    st["median_frac_zero_steps_P1"] = float(np.median(frac_zero_p1))
    st["median_n_clusters_actual"] = float(np.median(n_clusters_actual))
    st["median_throughput_frac"] = float(np.median(throughput_frac))
    st["median_cv_connected"] = float(np.median(cv_connected))

    st["R1_reachable"] = bool(np.median(reach_ok_arr.astype(float)) >= 0.5)
    st["R2_prize_reachable"] = bool(st["median_void_traverse_m"] <= reach_val)

    # ── GATE 4: signal-vs-noise (see module docstring for the exact definition) ─
    per_seed_std = np.array([r["policy_std_this_seed"] for r in rows])
    per_seed_mean = np.array([r["policy_mean_this_seed"] for r in rows])
    st["policy_std"] = float(np.mean(per_seed_std))
    st["seed_std"] = float(np.std(per_seed_mean, ddof=0))
    st["snr_ratio"] = (st["policy_std"] / st["seed_std"]) if st["seed_std"] > 0 else float("inf")

    st["B2_nontrivial"] = bool(st["P1_over_P0v"] >= a.b2_min_ratio)
    # B3: cv_connected (candidate (a), coefficient_of_variation_connected,
    # sweep_lib.py). The zero-fraction form (median frac_zero_steps_P0v <=
    # 0.10) is kept only as the diagnostic column below: it does not
    # discriminate the frozen config from a good candidate and structurally
    # works against A1_reachable (see module docstring). throughput_frac
    # (variance_decomposition, candidate (b)) is near-zero (<0.02) for BOTH
    # configs, because log1p's compression makes the connectivity component
    # of reward variance dominate the throughput component by construction.
    # cv_connected discriminates: measured medians ~0.30 (frozen)
    # vs ~0.60 (good candidate) -- see module docstring for the full numbers.
    st["B3_gradient_rich"] = bool(st["median_cv_connected"] >= a.b3_min_cv_connected)
    st["B4_signal_over_noise"] = bool(st["snr_ratio"] >= a.b4_min_snr)

    print(f"  [TIER {st['tier']}]  P0 {st['median_P0']:9.1f}   P0v {st['median_P0v']:9.1f}   "
          f"P1 {st['median_P1']:9.1f}   P2 {st['median_P2']:9.1f}   "
          f"(P1/P0v={st['P1_over_P0v']:.2f}x  P2/P1={st['P2_over_P1']:.2f}x)")
    print(f"  infeasible {st['infeasible_rate']:.0%}   median relays {st['median_relays']:.1f}   "
          f"coverage {st['mean_coverage']:.2f}")
    print(f"  median clusters {st['median_n_clusters_actual']:.1f}   "
          f"median void_traverse {st['median_void_traverse_m']:7.0f} m   "
          f"diffusive_reach {reach_val:7.0f} m")
    print(f"  frac_cross_reward (median) {st['median_frac_cross_reward']:.2f}   "
          f"frac_zero_steps_P0v (median, DIAGNOSTIC ONLY) {st['median_frac_zero_steps_P0v']:.2f}   "
          f"frac_zero_steps_P1 (median) {st['median_frac_zero_steps_P1']:.2f}")
    print(f"  GATE 3 (replaces old B3): cv_connected (median) "
          f"{st['median_cv_connected']:.3f}  (need >= {a.b3_min_cv_connected})  "
          f"[diagnostic-only throughput_frac (median) {st['median_throughput_frac']:.4f} "
          f"-- see module docstring for why this is NOT the gate]")
    print(f"  GATE 4: policy_std={st['policy_std']:.2f}  seed_std={st['seed_std']:.2f}  "
          f"snr_ratio={st['snr_ratio']:.3f}  (need >= {a.b4_min_snr})")
    for k, v in st["criteria"].items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")
    for k in ("R1_reachable", "R2_prize_reachable", "B2_nontrivial",
             "B3_gradient_rich", "B4_signal_over_noise"):
        print(f"    {'PASS' if st[k] else 'FAIL'}  {k}")
    overall = bool(ok and st["R1_reachable"] and st["R2_prize_reachable"]
                  and st["B2_nontrivial"] and st["B3_gradient_rich"]
                  and st["B4_signal_over_noise"])
    st["overall_PASS"] = overall
    print(f"  ==> {'PASS' if overall else 'REJECT'}")
    print("=" * 78, flush=True)
    return st


def main() -> None:
    a = _ARGS
    cell, rows = run_config(a)
    _write_csv(a.out, rows)
    print(f"wrote {len(rows)} row(s) to {a.out}")
    st = print_summary(a, cell, rows)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(st, f, indent=2)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
