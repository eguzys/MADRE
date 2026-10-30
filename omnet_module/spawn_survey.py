"""Survey the arena1400_cluster spawn across seeds.

Answers two questions before any training:

Q1. Is the task SOLVABLE? Reward is a deterministic function of positions, so we
    can ask the oracle directly: take the spawn, then MANUALLY place drones to
    bridge the clusters, and see if reward rises. If bridging pays, there is
    something for a policy to find. If it doesn't, the config is unsuitable.

Q2. Is it REACHABLE? A drone must physically fly into the gap. Between clusters
    the reward is flat zero (capacity cliff, SS6.5 OPEN-B) -- no gradient for the
    first ~500m of the journey. We report the gap width so we know how far a drone
    must travel through the dead zone.

No learned model is loaded. Positions only.
"""
import argparse, itertools, os, sys
import numpy as np
from omnet_env import OMNeTEnv
from max_flow_reward import compute_oracle_max_flow_reward

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=8)
p.add_argument("--config", default="arena1400_cluster")
a = p.parse_args()

os.environ["OMNET_CONFIG"] = a.config

def comps_and_gap(pos, r_d):
    n = len(pos)
    adj = [[i != j and np.hypot(*(pos[i] - pos[j])) <= r_d for j in range(n)] for i in range(n)]
    seen, comps = [False]*n, []
    for s in range(n):
        if seen[s]: continue
        stack, grp = [s], []
        seen[s] = True
        while stack:
            u = stack.pop(); grp.append(u)
            for v in range(n):
                if adj[u][v] and not seen[v]:
                    seen[v] = True; stack.append(v)
        comps.append(grp)
    return comps

rows = []
for seed in range(a.seeds):
    env = OMNeTEnv(worker_index=0)
    obs = env.reset(seed=seed)
    raw = env._last_obs
    if seed == 0:
        print("obs_json keys:", sorted(raw.keys()), file=sys.stderr)

    up  = np.array(raw["user_positions"])
    ua  = np.array(raw["user_area"])
    dp  = np.array(raw["positions"])
    r_d = 523.1

    # cluster centroids + pairwise gaps
    cents = np.array([up[ua == c].mean(axis=0) for c in sorted(set(ua.tolist()))])
    gaps  = [np.hypot(*(cents[i] - cents[j]))
             for i, j in itertools.combinations(range(len(cents)), 2)]

    c0 = comps_and_gap(dp[:, :2], r_d)
    r0 = float(compute_oracle_max_flow_reward(raw))

    rows.append(dict(seed=seed, nclust=len(cents), ndrone=len(dp),
                     gap_min=min(gaps) if gaps else 0.0,
                     gap_max=max(gaps) if gaps else 0.0,
                     comps=len(c0), reward=r0))
    env.close()

print(f"\n{'seed':>4} {'clust':>5} {'gap_min':>8} {'gap_max':>8} {'comps':>5} {'reward':>10}")
print("-" * 48)
for r in rows:
    print(f"{r['seed']:>4} {r['nclust']:>5} {r['gap_min']:>8.0f} {r['gap_max']:>8.0f} "
          f"{r['comps']:>5} {r['reward']:>10.1f}")

g = [r['gap_max'] for r in rows]
c = [r['comps'] for r in rows]
print(f"\nr_d = 523 m. Gaps ABOVE that need a relay flown into a zero-reward valley.")
print(f"  max gap: median {np.median(g):.0f} m, range {min(g):.0f}-{max(g):.0f} m")
print(f"  fraction of seeds with max gap > r_d : {np.mean([x > 523 for x in g]):.0%}")
print(f"  fraction of seeds with max gap > 2*r_d: {np.mean([x > 1046 for x in g]):.0%}  <- needs 2 relays")
print(f"  drone graph fragmented at spawn      : {np.mean([x > 1 for x in c]):.0%} of seeds")
