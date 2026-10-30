"""Sensed user set at the FROZEN arena.

Earlier numbers (0.16, 16.99) came from an earlier simulator build and a
100 m toy arena; this measures at L=1700 / mc=2 / cluster spawn.

mean ~0-1  ==> DeepSets == FFN == identity on the USER axis; the OMNeT user-
encoder work-stream is vacuous (the DRONE axis, where the thesis claim lives,
is untouched). Measure BEFORE recollecting.
"""
import numpy as np
from omnet_env import OMNeTEnv

env = OMNeTEnv(worker_index=0)                       # drone count comes from *.numDrones (frozen ini)
obs, info = env.reset(seed=0)
agents = list(env.possible_agents)
print(f"agents: {agents}")

counts = []
for step in range(1000):
    act = {a: np.zeros(2, dtype=np.float32) for a in env.agents}   # hold position
    obs, rew, term, trunc, info = env.step(act)

    oj = getattr(env, "_last_obs", None)
    if not isinstance(oj, dict) or "user_rssi_valid" not in oj:
        raise KeyError(
            f"user_rssi_valid not reachable. _last_obs type={type(oj)}, "
            f"keys={sorted(oj) if isinstance(oj, dict) else 'n/a'}"
        )
    rows = oj["user_rssi_valid"]                       # per-drone list of per-user flags
    counts.append([float(np.sum(r)) for r in rows])

    if (isinstance(term, dict) and any(term.values())) or \
       (isinstance(trunc, dict) and any(trunc.values())) or term is True or trunc is True:
        break
env.close()

c = np.asarray(counts, dtype=float).ravel()
print("=" * 62)
print(f"drone-steps       {c.size}")
print(f"mean set size     {c.mean():.3f}")
print(f"p50 / p90 / max   {np.percentile(c,50):.0f} / {np.percentile(c,90):.0f} / {c.max():.0f}")
print(f"frac empty        {(c == 0).mean():.3f}")
print("=" * 62)
print("Encoder is meaningful iff mean >> 1 and frac_empty is not ~1.0")
