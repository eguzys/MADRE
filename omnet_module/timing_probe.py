"""Storm probe: time per-step wall clock under a random policy.

The question this answers: is the slowdown REAL (network collapsing) or is it
INSTRUMENTATION (DROP_ANY printf + fflush, which scales with the drop rate)?

Run with the printf gated (default) and again with OMNET_DROP_DIAG=1.
  gated  : steps stay flat  -> it was I/O. Storm is smaller than it looked.
  gated  : steps still climb -> the network really is collapsing. Go to step 2.
"""
import argparse, os, time
import numpy as np
from omnet_env import OMNeTEnv

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=80)
p.add_argument("--block", type=int, default=10)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

log_path = os.environ.get("OMNET_LOG_STDOUT", "")
rng = np.random.default_rng(a.seed)

env = OMNeTEnv(worker_index=0)
t0 = time.perf_counter()
obs = env.reset(seed=a.seed)
print(f"reset: {time.perf_counter()-t0:.2f}s", flush=True)

n_drones = env.num_drones if hasattr(env, "num_drones") else 5
times, rewards = [], []

for i in range(a.steps):
    act = {f"drone_{d}": rng.uniform(-1.0, 1.0, size=2).tolist()
           for d in range(n_drones)}
    t = time.perf_counter()
    out = env.step(act)
    times.append(time.perf_counter() - t)
    r = out[1] if isinstance(out, tuple) else None
    if isinstance(r, dict):
        r = list(r.values())
    rewards.append(float(np.mean(r)) if r is not None else float("nan"))

    if (i + 1) % a.block == 0:
        blk = times[-a.block:]
        sz = os.path.getsize(log_path) / 1e6 if log_path and os.path.exists(log_path) else -1
        print(f"steps {i+1-a.block:>3}-{i+1:<3} | "
              f"mean {np.mean(blk):6.3f}s  max {np.max(blk):6.3f}s | "
              f"reward {np.mean(rewards[-a.block:]):9.2f} | "
              f"log {sz:8.1f} MB", flush=True)

print("\n--- VERDICT INPUT ---")
first = np.mean(times[:a.block]); last = np.mean(times[-a.block:])
print(f"first block {first:.3f}s -> last block {last:.3f}s   ratio {last/first:.2f}x")
print(f"total {np.sum(times):.1f}s for {a.steps} steps")
print(f"projected 1000-step episode: {np.mean(times)*1000/60:.1f} min (at CURRENT mean)")
print(f"projected 1000-step episode: {last*1000/60:.1f} min (at LAST-BLOCK rate)")
try:
    env.close()
except Exception:
    pass
