#!/usr/bin/env python3
"""handover_probe.py — does the policy HOLD a good configuration it did not find?

Usage:
  python3 omnet_module/handover_probe.py <ckpt> <seed,seed,...> <switch_step> <out.csv>

THE QUESTION. convergence.py established that the whole
trained-vs-greedy deficit is the PLATEAU -- the configuration the swarm settles
into -- and none of it is transit. That leaves two possibilities with opposite
fixes, which no amount of end-of-episode measurement can separate:

  (a) DISCOVERY failure. The policy would happily sit in greedy's configuration;
      it just never explores its way there. Fix: warm start / BC / curriculum.
  (b) PREFERENCE failure. The policy actively leaves greedy's configuration for
      its own, worse one -- i.e. what it learned to maximise is not the reward
      being measured. Fix: credit assignment / reward. Exploration knobs
      (entropy, sigma floor) cannot help.

This runs greedy to convergence (greedy's t_conv is ~6 steps; use a switch_step
well past that), then hands control to the trained policy for the rest of the
episode and traces what happens to the reward.

  reward stays at greedy's level  => (a) discovery
  reward decays toward trained's  => (b) preference

The decay SHAPE matters too and is why this traces per step rather than just
reporting a final number: an immediate collapse means the policy's action at
greedy's configuration points straight downhill, while a slow drift means it is
close to indifferent there.
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

for _v in ("OMNET_APPTAINER_SIF", "OMNET_USER_TX_mW", "OMNET_DRONE_TX_mW", "OMNET_STEP_M"):
    os.environ.pop(_v, None)
os.environ["OMNET_MAX_STEPS"] = "250"
os.environ["OMNET_ENCODER_MODE"] = "learned"
os.environ.setdefault("OMNET_PEER_LINK_MASK", "1")

import evaluate_omnet  # noqa: E402

N_DRONES = 5


def main() -> None:
    ckpt, seed_csv, switch_s, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    seeds = [int(s) for s in seed_csv.split(",")]
    switch = int(switch_s)

    evaluate_omnet._apply_eval_config(N_DRONES)
    import omnet_env  # noqa: F401
    from evaluation.policies import load_trained_policy

    trained_fn, _ = load_trained_policy(str(Path(ckpt).resolve()), "cpu")
    evaluate_omnet._close_leaked_omnet_envs()

    rows = []
    with evaluate_omnet._patched_num_drones(N_DRONES):
        env = omnet_env.OMNeTEnv(worker_index=0, encoder_mode="learned")
        try:
            for seed in seeds:
                obs, _ = env.reset(seed=seed)
                step = 0
                while True:
                    pol = "greedy" if step < switch else "trained"
                    acts = evaluate_omnet._select_actions(
                        pol, trained_fn, obs, N_DRONES, env)
                    obs, _r, _t, _tr, infos = env.step(acts)
                    step += 1
                    rows.append({"seed": seed, "step": step,
                                 "reward": float(infos["drone_0"]["global_reward"]),
                                 "phase": pol})
                    if not env.agents:
                        break
                ep = [r for r in rows if r["seed"] == seed]
                pre = np.mean([r["reward"] for r in ep if r["step"] in
                               range(max(1, switch - 20), switch + 1)])
                post = np.mean([r["reward"] for r in ep if r["step"] > len(ep) * 0.75])
                print(f"seed {seed:>3}  greedy@switch {pre:>8.1f}  ->  final {post:>8.1f}  "
                      f"({100*(post-pre)/pre:>+6.1f}%)", flush=True)
        finally:
            env.close()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "step", "reward", "phase"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
