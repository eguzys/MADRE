"""Map the reward landscape along the path a relay drone must fly.

THE QUESTION: 80% of episodes start with a fragmented backbone (spawn_survey.py).
To repair it, a drone must fly from its cluster into the gap. The reward is a decode
CLIFF -- exactly 0 below threshold, not a fade. So what does the reward look like
ALONG that path?

  cliff   : flat 0, then a jump exactly at bridging distance
            -> PPO cannot find it by exploration. Config needs geometry help.
  gradient: reward rises smoothly as the drone approaches bridging position
            -> RL can climb it. Train.
  valley  : reward DROPS on leaving the cluster, recovers only at the bridge
            -> learnable, but needs enough exploration to cross a penalty region.

Reward is a deterministic function of positions, so we can evaluate it directly at
any hypothetical drone placement. NO SIMULATION STEPS, NO TRAINING, NO LEARNED MODEL.
"""
import argparse, copy, itertools, os
import numpy as np
from omnet_env import OMNeTEnv
from max_flow_reward import compute_oracle_max_flow_reward

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, nargs="+", default=[0, 3, 5, 8])
p.add_argument("--config", default="arena1400_cluster")
p.add_argument("--steps", type=int, default=24, help="samples along the path")
a = p.parse_args()
os.environ["OMNET_CONFIG"] = a.config

R_D = 523.1

def reward_at(raw, positions):
    """Reward with drones moved to `positions` (n,2). Nothing else touched."""
    j = copy.deepcopy(raw)
    j["positions"] = [[float(x), float(y)] for x, y in positions]
    return float(compute_oracle_max_flow_reward(j))

for seed in a.seeds:
    env = OMNeTEnv(worker_index=0)
    env.reset(seed=seed)
    raw = env._last_obs
    up = np.array(raw["user_positions"])
    ua = np.array(raw["user_area"])
    dp = np.array(raw["positions"])[:, :2].astype(float)
    env.close()

    cents = np.array([up[ua == c].mean(axis=0) for c in sorted(set(ua.tolist()))])
    if len(cents) < 2:
        print(f"seed {seed}: only 1 cluster, skipping"); continue

    # the widest gap = the bridge that must be built
    (i, j), gap = max(
        (((i, j), np.hypot(*(cents[i] - cents[j])))
         for i, j in itertools.combinations(range(len(cents)), 2)),
        key=lambda t: t[1])
    A, B = cents[i], cents[j]

    # pick the drone nearest cluster A -- it is the natural relay candidate
    relay = int(np.argmin([np.hypot(*(d - A)) for d in dp]))

    print(f"\n{'='*68}")
    print(f"seed {seed}: {len(cents)} clusters | widest gap {gap:.0f} m "
          f"(r_d={R_D:.0f}) | flying drone {relay} from cluster {i} -> {j}")
    print(f"{'frac':>5} {'dist_from_A':>11} {'reward':>10} {'delta':>9}  landscape")
    print("-" * 68)

    base = reward_at(raw, dp)
    rewards = []
    for k in range(a.steps + 1):
        f = k / a.steps
        pos = dp.copy()
        pos[relay] = A + f * (B - A)          # fly straight down the gap
        r = reward_at(raw, pos)
        rewards.append(r)
        d = r - base
        bar = "#" * int(max(0, r) / max(rewards + [1]) * 34)
        print(f"{f:>5.2f} {f*gap:>11.0f} {r:>10.1f} {d:>+9.1f}  {bar}")

    rewards = np.array(rewards)
    start, best, end = rewards[0], rewards.max(), rewards[-1]
    argbest = int(np.argmax(rewards))
    # is there a dead flat region on the way out?
    mid = rewards[1:argbest] if argbest > 1 else np.array([])
    flat = np.allclose(mid, mid[0], rtol=1e-3) if len(mid) > 1 else False
    dips = (mid < start - 1.0).any() if len(mid) else False

    print(f"\n  start {start:.1f} | best {best:.1f} at frac {argbest/a.steps:.2f} "
          f"({argbest/a.steps*gap:.0f} m out) | end {end:.1f}")
    print(f"  gain from bridging: {best - start:+.1f} ({(best/start - 1)*100:+.0f}%)")
    if best <= start + 1.0:
        print("  >> NO GAIN. Bridging this gap does not pay. Config problem.")
    elif flat:
        print("  >> CLIFF: reward is FLAT on the way out, then jumps. "
              "PPO cannot find this by exploration.")
    elif dips:
        print("  >> VALLEY: reward DROPS on leaving the cluster before recovering. "
              "Learnable, but needs exploration through a penalty region.")
    else:
        print("  >> GRADIENT: reward rises on the way out. RL can climb this.")
