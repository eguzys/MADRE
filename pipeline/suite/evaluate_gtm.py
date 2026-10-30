#!/usr/bin/env python3
"""GTM baseline wrapper. Adds PADRE's Greedy Traffic
Maximizer (Leuenberger, Di Maio, Braun, Wireless Days 2025, Algorithm 1) to
evaluate_omnet.py WITHOUT editing the checkout: it imports the harness, wraps
_select_actions, and runs main().

GTM: every step, every drone samples the eight compass directions at the
maximum step kappa (= OMNET_STEP_M) and moves in the one that raises its
objective the most, others frozen at their current positions (simultaneous,
as in Algorithm 1). The objective is computed from the GLOBAL max-flow
solution -- the same LP as the reward -- which is what PADRE's GTM assumes
(free, exact global state).

  SUITE_GTM_OBJ=hosted   g(m_i) = traffic hosted by drone i (PADRE-faithful)
  SUITE_GTM_OBJ=global   Sigma tau (team objective; strictly more informed)

Run from omnet_module/ with `--policy hover` (any scripted choice works, the
wrapper intercepts before the harness looks at it) and SUITE_POLICY=gtm.
"""
import math
import os
import sys

sys.path.insert(0, os.getcwd())
import evaluate_omnet as E  # noqa: E402
import max_flow_reward as M  # noqa: E402

STEP = float(os.environ.get("OMNET_STEP_M", 60.0))
OBJ = os.environ.get("SUITE_GTM_OBJ", "global").strip().lower()
DIRS = [(math.cos(2 * math.pi * k / 8), math.sin(2 * math.pi * k / 8)) for k in range(8)]
_orig = E._select_actions
_stats = {"steps": 0, "moved": 0}


def _solve(obs, positions):
    o = dict(obs)
    o["positions"] = positions
    return M.compute_position_oracle_reward(o)


def _objective(res, i, nd):
    if OBJ == "global":
        return float(res["reward"])
    # hosted traffic of drone i: every unit of flow that enters node i
    tot = 0.0
    for edge_flows in res.get("per_commodity_flow", {}).values():
        for (a, b), f in edge_flows.items():
            if b == i:
                tot += float(f)
    return tot


def gtm(obs):
    pos = [[float(p[0]), float(p[1])] for p in obs.get("positions", [])]
    nd = len(pos)
    base = _solve(obs, pos)
    vels = []
    for i in range(nd):
        best, bv = _objective(base, i, nd), (0.0, 0.0)
        for dx, dy in DIRS:
            cand = [list(p) for p in pos]
            cand[i] = [pos[i][0] + dx * STEP, pos[i][1] + dy * STEP]
            r = _objective(_solve(obs, cand), i, nd)
            if r > best:
                best, bv = r, (dx, dy)
        if bv != (0.0, 0.0):
            _stats["moved"] += 1
        vels.append([bv[0], bv[1]])
    _stats["steps"] += 1
    if _stats["steps"] % 50 == 0:
        print(f"[gtm:{OBJ}] step {_stats['steps']} moved-drone-steps={_stats['moved']}", flush=True)
    return vels


def _sel(policy, action_fn, obs_dict, num_drones, env):
    if os.environ.get("SUITE_POLICY") == "gtm":
        vels = gtm(env._last_obs)
        if len(vels) != num_drones:
            raise RuntimeError(f"gtm returned {len(vels)} velocities, expected {num_drones}")
        return {f"drone_{i}": vels[i] for i in range(num_drones)}
    return _orig(policy, action_fn, obs_dict, num_drones, env)


E._select_actions = _sel
print(f"[gtm] objective={OBJ} step={STEP} m", flush=True)
E.main()
