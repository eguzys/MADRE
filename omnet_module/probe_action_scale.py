#!/usr/bin/env python3
"""probe_action_scale.py — is the trained policy UNDER-ACTUATED?

WHY. At 1M frames the training log shows policy/drone/scale_mean ~= 0.95 (the
Gaussian sigma, on a [-1,1] action space) while policy/drone/abs_loc_mean is
only ~0.23. Eval runs DETERMINISTIC (policies.py:279), so the executed action
is the mean: the drones command ~23% of max speed. This probe re-runs a
checkpoint with that mean rescaled, on an ARBITRARY seed list, and reports the
realised |a| distribution alongside the score.

  PROBE_MODE=measure  -> no change, just record |a|      (control)
  PROBE_MODE=scale    -> a := clip(a * PROBE_SCALE, -1, 1)
  PROBE_MODE=unit     -> a := a / |a|   (direction only, full speed)

Usage:
  PROBE_MODE=unit python3 omnet_module/probe_action_scale.py <ckpt> 2,4,6,10,14,18 out.csv
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "src")))

MODE = os.environ.get("PROBE_MODE", "measure")
SCALE = float(os.environ.get("PROBE_SCALE", "1.0"))
N_DRONES = 5

# Same world binding eval_checkpoint.sh applies. Must precede `import omnet_env`.
for _v in ("OMNET_APPTAINER_SIF", "OMNET_USER_TX_mW", "OMNET_DRONE_TX_mW", "OMNET_STEP_M"):
    os.environ.pop(_v, None)
os.environ["OMNET_MAX_STEPS"] = "250"
os.environ["OMNET_ENCODER_MODE"] = "learned"
os.environ.setdefault("OMNET_PEER_LINK_MASK", "1")

import evaluate_omnet  # noqa: E402


def main() -> None:
    ckpt, seed_csv, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    seeds = [int(s) for s in seed_csv.split(",")]

    evaluate_omnet._apply_eval_config(N_DRONES)
    import omnet_env  # noqa: F401  (import-time env reads now satisfied)

    from evaluation.policies import load_trained_policy

    base_fn, cfg = load_trained_policy(str(Path(ckpt).resolve()), "cpu")
    evaluate_omnet._close_leaked_omnet_envs()

    mags: list[float] = []

    def action_fn(obs_dict, n_agents):
        acts = base_fn(obs_dict, n_agents)
        for k, v in acts.items():
            v = np.asarray(v, dtype=np.float32)
            m = float(np.linalg.norm(v))
            mags.append(m)
            if MODE == "unit":
                if m > 1e-8:
                    acts[k] = (v / m).astype(np.float32)
            elif MODE == "scale":
                acts[k] = np.clip(v * SCALE, -1.0, 1.0).astype(np.float32)
        return acts

    rows = []
    with evaluate_omnet._patched_num_drones(N_DRONES):
        env = omnet_env.OMNeTEnv(worker_index=0, encoder_mode="learned")
        try:
            for seed in seeds:
                r, s = evaluate_omnet.run_episode(env, "trained", action_fn, seed, N_DRONES)
                rows.append({"seed": seed, "episode_reward": r, "steps": s})
                print(f"  seed={seed:3d}  {r/s:9.1f}/step", flush=True)
        finally:
            env.close()

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "episode_reward", "steps"])
        w.writeheader()
        w.writerows(rows)

    a = np.asarray(mags)
    print(f"\nmode={MODE} scale={SCALE}  |a| over {a.size} commands: "
          f"mean={a.mean():.3f} median={np.median(a):.3f} "
          f"p10={np.percentile(a,10):.3f} p90={np.percentile(a,90):.3f} max={a.max():.3f}")
    print(f"mean/step = {np.mean([r['episode_reward']/r['steps'] for r in rows]):.1f}")


if __name__ == "__main__":
    main()
