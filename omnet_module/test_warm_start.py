"""
test_warm_start.py — Parity and benchmark test for LP warm-starting.

Generates a synthetic greedy episode (no OMNeT++ needed), then:
  Phase 1: Parity — warm-start vs CBC backup on ALL edge-change steps + 50 sampled stable steps.
  Phase 2: Benchmark — 200 calls, warm-start vs non-warm (cache cleared), report ms/call.
"""

import sys
import os
import time
import copy
import random
import importlib.machinery
import types as _types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import max_flow_reward as mfr_new

_bak_path = os.path.join(os.path.dirname(__file__), "max_flow_reward.py.cbc_bak")
_loader = importlib.machinery.SourceFileLoader("mfr_cbc", _bak_path)
mfr_cbc = _types.ModuleType("mfr_cbc")
mfr_cbc.__file__ = _bak_path
_loader.exec_module(mfr_cbc)

from signal_baselines import greedy_velocities

STEP_M = 14.0
ARENA_W = 2000.0
ARENA_H = 2000.0

print("=" * 65)
print("Phase 1: generating synthetic greedy episode ...")

# Fixed user layout: 4 users in 2 areas.
USER_POSITIONS = [[200.0, 800.0], [400.0, 1200.0],   # area 0
                  [1600.0, 800.0], [1800.0, 1200.0]]  # area 1
USER_AREA = [0, 0, 1, 1]

# Drones start spread across the arena.
drone_pos = [[500.0, 1000.0], [1000.0, 1000.0], [1500.0, 1000.0]]

rng = random.Random(42)

def make_obs(d_pos):
    return {
        "positions":      [list(p) for p in d_pos],
        "user_positions": [list(p) for p in USER_POSITIONS],
        "user_area":      list(USER_AREA),
    }

N_STEPS = 300

step_obs = [make_obs(drone_pos)]
for _ in range(N_STEPS):
    obs = make_obs(drone_pos)
    vels = greedy_velocities(obs)
    new_pos = []
    for i, (dp, v) in enumerate(zip(drone_pos, vels)):
        nx = max(0.0, min(ARENA_W, dp[0] + v[0] * STEP_M))
        ny = max(0.0, min(ARENA_H, dp[1] + v[1] * STEP_M))
        new_pos.append([nx, ny])
    drone_pos = new_pos
    step_obs.append(make_obs(drone_pos))

total_steps = len(step_obs)
print(f"  Done: {total_steps} synthetic observations")

# ── Identify edge-change steps ────────────────────────────────────────────────

def get_edge_set(o):
    nd = len(o["positions"])
    nu = len(o["user_positions"])
    return frozenset(
        mfr_new._build_edges(nd, nu, o["positions"], o["user_positions"]).keys()
    )

edge_sets = [get_edge_set(o) for o in step_obs]
edge_change_idxs = [i for i in range(1, total_steps) if edge_sets[i] != edge_sets[i - 1]]
print(f"  Edge-change steps: {len(edge_change_idxs)} / {total_steps} "
      f"({100 * len(edge_change_idxs) / max(total_steps - 1, 1):.1f}%)")

change_set = set(edge_change_idxs)
hit_idxs = [i for i in range(total_steps) if i not in change_set]
sample_hits = rng.sample(hit_idxs, min(50, len(hit_idxs)))
compare_idxs = sorted(change_set | set(sample_hits))
print(f"  Comparing {len(compare_idxs)} steps "
      f"({len(edge_change_idxs)} edge-change + {len(sample_hits)} cache-hit samples)")

# ── Phase 2: parity comparison ────────────────────────────────────────────────

print("\nPhase 2: parity comparison (warm-start vs CBC backup) ...")

mfr_new._ORACLE_LP_CACHE.clear()

# NOTE: max_flow_reward.py.cbc_bak is a "checkpoint before scipy reward swap"
# snapshot (git log) -- it predates the log1p/PWL objective AND the HiGHS
# solver migration, so it still computes the OLD linear reward Sigma w_k*tau_k
# via PuLP+CBC. Phase 2 below therefore compares two different reward
# formulas, not two solvers on the same formula, and WILL report mismatches
# on every step regardless of the uniform-weighting change --
# this is a pre-existing condition, not something introduced here. Weights
# are still pinned to uniform (1.0/1.0) on the mfr_cbc side so at least the
# weighting dimension of the comparison is apples-to-apples.
_W_CROSS, _W_LOCAL = 1.0, 1.0

mismatches = []
for i in compare_idxs:
    r_new = mfr_new.compute_position_oracle_reward(step_obs[i])["reward"]
    r_cbc = mfr_cbc.compute_position_oracle_reward(step_obs[i], w_cross=_W_CROSS, w_local=_W_LOCAL)["reward"]
    denom = abs(r_cbc) + 1e-12
    rel_err = abs(r_new - r_cbc) / denom
    if rel_err > 1e-6:
        kind = "EDGE-CHANGE" if i in change_set else "cache-hit"
        mismatches.append((i, kind, r_new, r_cbc, rel_err))
        print(f"  MISMATCH [{kind}] step {i}: "
              f"new={r_new:.8f}  cbc={r_cbc:.8f}  rel_err={rel_err:.2e}")

edge_change_mismatches = sum(1 for m in mismatches if m[1] == "EDGE-CHANGE")
parity_ok = not mismatches
print(f"  {'PASS' if parity_ok else 'FAIL'} — "
      f"{len(compare_idxs)} steps compared, "
      f"{len(mismatches)} mismatches "
      f"({edge_change_mismatches} on edge-change steps)")

# ── Phase 3: benchmark ────────────────────────────────────────────────────────

print("\nPhase 3: benchmark (200 calls) ...")

bench_obs = step_obs[total_steps // 2]

# Warm up both paths
mfr_new._ORACLE_LP_CACHE.clear()
mfr_new.compute_position_oracle_reward(bench_obs)   # cold build + warm prime
mfr_new.compute_position_oracle_reward(bench_obs)   # confirm warm

N_BENCH = 200

# Warm-start: cache primed, hit path for all 200 calls
t0 = time.perf_counter()
for _ in range(N_BENCH):
    mfr_new.compute_position_oracle_reward(bench_obs)
warm_ms = (time.perf_counter() - t0) * 1000 / N_BENCH

# Non-warm: clear highs before every call (forces full rebuild each time)
def _clear_highs():
    # [2026-08-09] _ORACLE_LP_CACHE is an LRU of per-structure slots, not a
    # single flat slot, so dropping the top-level "highs"/"key" entries no
    # longer exists as an operation. Clearing evicts every cached structure,
    # which is what this benchmark wants: force the cold rebuild path.
    mfr_new._ORACLE_LP_CACHE.clear()

_clear_highs()
mfr_new.compute_position_oracle_reward(bench_obs)   # warm up cold path

t0 = time.perf_counter()
for _ in range(N_BENCH):
    _clear_highs()
    mfr_new.compute_position_oracle_reward(bench_obs)
nonwarm_ms = (time.perf_counter() - t0) * 1000 / N_BENCH

speedup = nonwarm_ms / warm_ms if warm_ms > 0 else float("inf")

print(f"  Warm-start:  {warm_ms:.2f} ms/call")
print(f"  Non-warm:    {nonwarm_ms:.2f} ms/call")
print(f"  Speedup:     {speedup:.1f}x")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("WARM-START TEST SUMMARY")
print(f"  Steps generated:     {total_steps}")
print(f"  Edge-change steps:   {len(edge_change_idxs)} "
      f"({100 * len(edge_change_idxs) / max(total_steps - 1, 1):.1f}%)")
print(f"  Steps compared:      {len(compare_idxs)}")
print(f"  Parity result:       {'ALL MATCH (rel < 1e-6)' if parity_ok else f'FAIL — {len(mismatches)} mismatches'}")
print(f"  Warm-start ms/call:  {warm_ms:.2f}")
print(f"  Non-warm ms/call:    {nonwarm_ms:.2f}")
print(f"  Speedup:             {speedup:.1f}x")
print("=" * 65)

if not parity_ok:
    sys.exit(1)
if speedup < 1.0:
    print("\nWARNING: warm-start is NOT faster — negative result.")
    sys.exit(2)
