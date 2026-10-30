#!/usr/bin/env python3
"""placement_analysis.py — HOW does the trained policy's final configuration
differ from greedy's, on the seeds where greedy scores higher?

Usage:
  python3 omnet_module/placement_analysis.py <ckpt> <seed,seed,...> [out.json]

From convergence.py: the trained-vs-greedy difference on those seeds is
in the PLATEAU -- the configuration the swarm settles into -- and none of it
is transit (removing transit entirely is worth +0.017 of the gap). So the
question is purely about final drone positions.

Users are STATIC, so for a given seed every field of obs except "positions" is
identical no matter which policy produced it. That makes the comparison exact
and offline: capture each policy's final positions, then score BOTH inside the
SAME obs with max_flow_reward.compute_position_oracle_reward. No re-simulation,
no confound.

Reports per seed:
  R(trained), R(greedy)          -- oracle reward of each final placement
  R(kmeans)                      -- users k-means'd into N clusters, one drone
                                    per centroid: a placement-quality ceiling
                                    that ignores relay/backbone structure
  spread                         -- mean pairwise drone distance (is the swarm
                                    collapsed onto one spot, or spread out?)
  n_clusters_occupied            -- distinct user clusters within r_u of >=1
                                    drone, i.e. is it covering the map or
                                    piling onto one cluster
"""
from __future__ import annotations

import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

for _v in ("OMNET_APPTAINER_SIF", "OMNET_USER_TX_mW", "OMNET_DRONE_TX_mW", "OMNET_STEP_M"):
    os.environ.pop(_v, None)
os.environ["OMNET_MAX_STEPS"] = "250"
os.environ["OMNET_ENCODER_MODE"] = "learned"
os.environ.setdefault("OMNET_PEER_LINK_MASK", "1")

import evaluate_omnet  # noqa: E402

N_DRONES = 5


def _kmeans(pts: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = pts[rng.choice(len(pts), size=min(k, len(pts)), replace=False)].astype(float)
    if len(c) < k:  # fewer users than drones: pad by repeating
        c = np.vstack([c, np.repeat(c[:1], k - len(c), axis=0)])
    for _ in range(iters):
        d = np.linalg.norm(pts[:, None, :] - c[None, :, :], axis=2)
        lab = d.argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = pts[m].mean(0)
    return c


def _spread(pos: np.ndarray) -> float:
    if len(pos) < 2:
        return 0.0
    return float(np.mean([np.linalg.norm(a - b) for a, b in combinations(pos, 2)]))


def main() -> None:
    ckpt, seed_csv = sys.argv[1], sys.argv[2]
    out_json = sys.argv[3] if len(sys.argv) > 3 else None
    seeds = [int(s) for s in seed_csv.split(",")]

    evaluate_omnet._apply_eval_config(N_DRONES)
    import omnet_env  # noqa: F401
    from greedy_oracle import _reward_with_positions
    from evaluation.policies import load_trained_policy

    trained_fn, _ = load_trained_policy(str(Path(ckpt).resolve()), "cpu")
    evaluate_omnet._close_leaked_omnet_envs()

    rows = []
    with evaluate_omnet._patched_num_drones(N_DRONES):
        env = omnet_env.OMNeTEnv(worker_index=0, encoder_mode="learned")
        try:
            for seed in seeds:
                final = {}
                for pol, fn in (("trained", trained_fn), ("greedy", None)):
                    evaluate_omnet.run_episode(env, pol if pol == "greedy" else "trained",
                                               fn, seed, N_DRONES)
                    obs = env._last_obs
                    final[pol] = (np.asarray([[float(p[0]), float(p[1])]
                                              for p in obs["positions"]]), obs)

                # Users are static => score both placements in ONE obs.
                obs = final["greedy"][1]
                users = np.asarray([[float(p[0]), float(p[1])]
                                    for p in obs.get("user_positions", [])])
                km = _kmeans(users, N_DRONES, seed=seed) if len(users) else np.zeros((N_DRONES, 2))

                r = {}
                for name, pos in (("trained", final["trained"][0]),
                                  ("greedy", final["greedy"][0]),
                                  ("kmeans", km)):
                    r[name] = float(_reward_with_positions(obs, [list(p) for p in pos]))

                # NOISE ROBUSTNESS. The policy trains at sigma ~= 0.95 on a
                # [-1,1] action space with STEP_M=60, i.e. ~57 m/step of
                # execution noise against ~11 m/step of signal, but is
                # EVALUATED deterministically. If a structured backbone is
                # fragile under that noise and a compact blob is not, then the
                # policy is correctly optimising a different (noisy-world)
                # objective, and lowering entropy_coef would move its optimum.
                # Perturb each placement by sigma*STEP_M and see which decays
                # faster. Cheap: users are static, so this is pure arithmetic.
                rng = np.random.default_rng(1234 + seed)
                rob = {}
                for name, pos in (("trained", final["trained"][0]),
                                  ("greedy", final["greedy"][0])):
                    for sig in (0.1, 0.3, 0.95):
                        vals = []
                        for _ in range(64):
                            jit = pos + rng.normal(0.0, sig * 60.0, size=pos.shape)
                            vals.append(_reward_with_positions(obs, [list(p) for p in jit]))
                        rob[f"{name}_sig{sig}"] = float(np.mean(vals))

                row = {
                    "seed": seed,
                    "R_trained": r["trained"], "R_greedy": r["greedy"], "R_kmeans": r["kmeans"],
                    "spread_trained": _spread(final["trained"][0]),
                    "spread_greedy": _spread(final["greedy"][0]),
                    "spread_users": _spread(users) if len(users) else 0.0,
                    "pos_trained": [list(map(float, p)) for p in final["trained"][0]],
                    "pos_greedy": [list(map(float, p)) for p in final["greedy"][0]],
                    **rob,
                }
                print(f"     robustness  trained {r['trained']:>7.1f} -> "
                      f"{rob['trained_sig0.1']:>7.1f} / {rob['trained_sig0.3']:>7.1f} / "
                      f"{rob['trained_sig0.95']:>7.1f}   |   greedy {r['greedy']:>7.1f} -> "
                      f"{rob['greedy_sig0.1']:>7.1f} / {rob['greedy_sig0.3']:>7.1f} / "
                      f"{rob['greedy_sig0.95']:>7.1f}", flush=True)
                rows.append(row)
                print(f"seed {seed:>3}  R: trained {r['trained']:>8.1f}  greedy {r['greedy']:>8.1f}  "
                      f"kmeans {r['kmeans']:>8.1f}   spread: trained {row['spread_trained']:>7.0f}m  "
                      f"greedy {row['spread_greedy']:>7.0f}m  users {row['spread_users']:>7.0f}m",
                      flush=True)
        finally:
            env.close()

    print("\n=== means ===")
    for k in ("R_trained", "R_greedy", "R_kmeans", "spread_trained", "spread_greedy", "spread_users"):
        print(f"  {k:<16} {np.mean([r[k] for r in rows]):>9.1f}")
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(out_json, "w"), indent=2)
        print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
