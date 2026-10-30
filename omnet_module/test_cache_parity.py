"""
Correctness test for the LP structure cache in compute_position_oracle_reward.

Phase 1: Run a single greedy episode collecting every step's raw obs_json.
Phase 2: Replay offline through (a) the cached version and (b) the CBC backup.
         Compare on ALL edge-change steps + a 50-step random sample of cache-hit steps.
Phase 3: Benchmark 200 calls on a fixed obs, cached vs. non-cached.
"""

import sys
import os
import time
import importlib.util
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

# ── Load the two versions ──────────────────────────────────────────────────────

import max_flow_reward as mfr_new   # patched (with cache)

import importlib.machinery, types as _types
_bak_path = os.path.join(os.path.dirname(__file__), "max_flow_reward.py.cbc_bak")
_loader = importlib.machinery.SourceFileLoader("mfr_cbc", _bak_path)
mfr_cbc = _types.ModuleType("mfr_cbc")
mfr_cbc.__file__ = _bak_path
_loader.exec_module(mfr_cbc)

# ── Phase 1: collect observations from a greedy episode ───────────────────────

print("=" * 65)
print("Phase 1: collecting greedy episode observations ...")

from omnet_env import OMNeTEnv
from signal_baselines import greedy_velocities

env = OMNeTEnv(worker_index=0)
obs, _ = env.reset()
step_obs = [env._last_obs]   # initial obs

done = False
step_count = 0
while not done:
    vels = greedy_velocities(env._last_obs)
    nd = len(env._last_obs.get("positions", []))
    actions = {f"drone_{i}": vels[i] for i in range(nd)}
    obs, rewards, terms, truncs, infos = env.step(actions)
    step_obs.append(env._last_obs)
    step_count += 1
    done = any(terms.values())
    if step_count % 100 == 0:
        print(f"  ... {step_count} steps collected")

try:
    env._kill_proc()
except Exception:
    pass

total_steps = len(step_obs)
print(f"  Done: {total_steps} observations collected")

# ── Phase 2: identify edge-change steps ───────────────────────────────────────

print("\nPhase 2: identifying edge-change steps ...")

def get_edge_set(o):
    nd = len(o.get("positions", []))
    nu = len(o.get("user_positions", []))
    return frozenset(
        mfr_new._build_edges(
            nd, nu,
            o.get("positions", []),
            o.get("user_positions", []),
            o.get("noise_floor", []),
        ).keys()
    )

edge_sets = [get_edge_set(o) for o in step_obs]
edge_change_idxs = [
    i for i in range(1, total_steps) if edge_sets[i] != edge_sets[i - 1]
]

print(f"  Edge-change steps: {len(edge_change_idxs)} / {total_steps} "
      f"({100 * len(edge_change_idxs) / max(total_steps - 1, 1):.1f}%)")

# ── Phase 3: select comparison set ────────────────────────────────────────────

change_set = set(edge_change_idxs)
hit_idxs = [i for i in range(total_steps) if i not in change_set]
rng = random.Random(42)
sample_hits = rng.sample(hit_idxs, min(50, len(hit_idxs)))
compare_idxs = sorted(change_set | set(sample_hits))

print(f"  Comparing {len(compare_idxs)} steps "
      f"({len(edge_change_idxs)} edge-change + {len(sample_hits)} cache-hit samples)")

# Reset new-version cache so first call in comparison is a fresh cold start.
mfr_new._ORACLE_LP_CACHE.clear()

# ── Phase 4: parity comparison ────────────────────────────────────────────────

print("\nPhase 3: running parity comparison ...")

mismatches = []
for i in compare_idxs:
    r_new = mfr_new.compute_position_oracle_reward(step_obs[i])["reward"]
    r_cbc = mfr_cbc.compute_position_oracle_reward(step_obs[i])["reward"]
    denom = abs(r_cbc) + 1e-12
    rel_err = abs(r_new - r_cbc) / denom
    if rel_err > 1e-6:
        kind = "EDGE-CHANGE" if i in change_set else "cache-hit"
        mismatches.append((i, kind, r_new, r_cbc, rel_err))
        print(f"  MISMATCH [{kind}] step {i}: "
              f"new={r_new:.8f}  cbc={r_cbc:.8f}  rel_err={rel_err:.2e}")

edge_change_mismatches = sum(1 for m in mismatches if m[1] == "EDGE-CHANGE")
print(f"\n  {'PASS' if not mismatches else 'FAIL'} — "
      f"{len(compare_idxs)} steps compared, "
      f"{len(mismatches)} mismatches "
      f"({edge_change_mismatches} on edge-change steps)")

# ── Phase 5: benchmark ────────────────────────────────────────────────────────

print("\nPhase 4: benchmark (200 calls on fixed obs) ...")

bench_obs = step_obs[len(step_obs) // 2]   # stable mid-episode obs

# Warm up
mfr_new.compute_position_oracle_reward(bench_obs)
mfr_cbc.compute_position_oracle_reward(bench_obs)

N_BENCH = 200

t0 = time.perf_counter()
for _ in range(N_BENCH):
    mfr_new.compute_position_oracle_reward(bench_obs)
cached_ms = (time.perf_counter() - t0) * 1000 / N_BENCH

# Non-cached: clear cache before every call
t0 = time.perf_counter()
for _ in range(N_BENCH):
    mfr_new._ORACLE_LP_CACHE.clear()
    mfr_new.compute_position_oracle_reward(bench_obs)
noncached_ms = (time.perf_counter() - t0) * 1000 / N_BENCH

print(f"  Cached:     {cached_ms:.2f} ms/call")
print(f"  Non-cached: {noncached_ms:.2f} ms/call")
print(f"  Speedup:    {noncached_ms / cached_ms:.1f}x")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SUMMARY")
print(f"  Total steps collected:  {total_steps}")
print(f"  Edge-change steps:      {len(edge_change_idxs)} "
      f"({100 * len(edge_change_idxs) / max(total_steps - 1, 1):.1f}%)")
print(f"  Steps compared:         {len(compare_idxs)}")
print(f"  Parity result:          {'ALL MATCH (rel tol < 1e-6)' if not mismatches else f'FAIL — {len(mismatches)} mismatches'}")
print(f"  Cached ms/call:         {cached_ms:.2f}")
print(f"  Non-cached ms/call:     {noncached_ms:.2f}")
print(f"  Speedup:                {noncached_ms / cached_ms:.1f}x")
print("=" * 65)

if mismatches:
    sys.exit(1)
