#!/usr/bin/env python3
"""
reward_variance.py -- Test 1 of the OMNeT reward checks: does the oracle
max-flow reward separate trained policies from random and greedy baselines?

Reward fn under test is omnet_module/max_flow_reward.py::
compute_oracle_max_flow_reward -- the ACTUAL training/eval reward (asserted
below against what omnet_env.py itself calls). NOT smoke_test.py's legacy
compute_weighted_maxmin_flow diagnostic.

If oracle reward does NOT separate trained/greedy/random, flat episode-return
curves are consistent with a non-discriminative reward signal (nothing to
learn a gradient from) rather than with a training or checkpoint-loading
problem. See eval/setw_allocation.py (Test 2) for the
complementary per-user allocation-shape check, which can catch learning
that a scalar mean-return comparison sums away.

Usage:
    python reward_variance.py --policy random  --num-drones 5 -n 30 --out random_n5.csv
    python reward_variance.py --policy greedy  --num-drones 5 -n 30 --out greedy_n5.csv
    python reward_variance.py --policy trained --checkpoint <path.pt> \\
        --num-drones 4 -n 30 --out trained_mlp_n4.csv

Checkpoint loading goes through eval/checkpoint_utils.py::load_and_verify_policy,
which HARD-FAILS if the manual named_parameters() weight copy didn't take
(BenchMARL's native restore_file does not restore ST/DeepSets/GAT -- see that
module's docstring), so a bad load stops the run instead of producing a number.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_OMNET_MODULE_DIR = os.path.abspath(os.path.join(_HERE, "..", "omnet_module"))
if _OMNET_MODULE_DIR not in sys.path:
    sys.path.insert(0, _OMNET_MODULE_DIR)
_SRC_DIR = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import omnet_env  # noqa: E402
from evaluate_omnet import (  # noqa: E402
    run_episode,
    _patched_num_drones,
    _close_leaked_omnet_envs,
)

# Assert on the actual training/eval reward function -- never the legacy
# compute_weighted_maxmin_flow smoke-test diagnostic (settled context).
from max_flow_reward import compute_oracle_max_flow_reward  # noqa: E402
assert omnet_env.compute_oracle_max_flow_reward is compute_oracle_max_flow_reward, (
    "omnet_env.py is not calling max_flow_reward.compute_oracle_max_flow_reward "
    "-- reward wiring drifted, do not trust this script's numbers."
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Test 1 -- oracle-reward distribution across policies "
                    "(trained/greedy/random), fixed-seed episodes.")
    p.add_argument("--policy", choices=["trained", "greedy", "random"], required=True)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="BenchMARL .pt checkpoint. Required for --policy trained.")
    p.add_argument("--num-drones", type=int, required=True, dest="num_drones")
    p.add_argument("-n", "--episodes", type=int, default=30)
    p.add_argument("--seed-start", type=int, default=0, dest="seed_start",
                   help="Episodes use seeds seed_start..seed_start+n-1. "
                        "SEED DISCIPLINE: 0-99 is the evaluation band; "
                        "100-199 is tuning-only. Straying past 99 is refused.")
    p.add_argument("--out", type=str, default=None, help="Optional per-episode CSV path.")
    p.add_argument("--device", type=str, default=None,
                   help="Compute device for --policy trained (default: cuda if available else cpu).")
    a = p.parse_args()
    if a.policy == "trained" and a.checkpoint is None:
        p.error("--policy trained requires --checkpoint")
    if a.seed_start < 0:
        p.error(f"--seed-start {a.seed_start} is negative.")
    _last = a.seed_start + a.episodes - 1
    if _last > 99:
        p.error(
            f"seeds {a.seed_start}..{_last} reach past 99 into the tuning band "
            "(100-199). Reported numbers come from seeds 0-99 only. Reduce "
            "-n or --seed-start."
        )
    return a


def _build_action_source(args):
    """Returns action_fn (None for greedy -- run_episode/_select_actions
    special-case it via env._last_obs, same convention as evaluate_omnet.py)."""
    if args.policy == "trained":
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        from checkpoint_utils import load_and_verify_policy

        ckpt_path = str(Path(args.checkpoint).resolve())
        print(f"Loading checkpoint: {ckpt_path}", flush=True)
        action_fn, task_config = load_and_verify_policy(ckpt_path, device, args.num_drones)
        print(f"Policy loaded + verified. Training-time task config: {task_config}", flush=True)

        n_closed = _close_leaked_omnet_envs()
        print(f"[load] closed {n_closed} OMNeT env(s) spawned while loading "
              "the checkpoint", flush=True)
        return action_fn

    if args.policy == "random":
        from evaluation.policies import make_random_action_fn
        return make_random_action_fn(args.num_drones)

    return None  # greedy: action_fn stays None, _select_actions special-cases it


def main():
    args = parse_args()
    action_fn = _build_action_source(args)

    rewards = []
    steps_list = []
    # checkpoint_utils.load_and_verify_policy always composes the trained
    # policy with model.encoder_mode="learned" (omnet/static_learnedenc task) --
    # it hard-fails on any other checkpoint shape. The env must therefore emit
    # the 343-wide learned-mode token (raw user block), not OMNeTEnv's
    # encoder_mode="frozen" default (26-wide, pooled 8-dim user latent), or
    # the actor's reassemble_with_learned_encoder step slices a token that's
    # 13 columns too narrow and would reshape the wrong columns rather than
    # stop on a width mismatch. greedy/random ignore this obs entirely
    # (greedy reads env._last_obs raw JSON; random never touches obs_dict),
    # so forcing "learned" here is safe for every --policy value.
    with _patched_num_drones(args.num_drones):
        env = omnet_env.OMNeTEnv(worker_index=0, encoder_mode="learned")
        try:
            for seed in range(args.seed_start, args.seed_start + args.episodes):
                reward, steps = run_episode(env, args.policy, action_fn, seed, args.num_drones)
                rewards.append(reward)
                steps_list.append(steps)
                print(f"  ep seed={seed:4d}  reward={reward:12.4f}  steps={steps:4d}", flush=True)
        finally:
            env.close()

    rewards_arr = np.array(rewards, dtype=float)
    print("\n" + "=" * 65)
    print(f"Test 1 summary -- policy={args.policy}  N={args.num_drones}"
          + (f"  ckpt={args.checkpoint}" if args.checkpoint else ""))
    print(f"  episodes: {len(rewards_arr)}")
    print(f"  mean={rewards_arr.mean():.4f}  std={rewards_arr.std():.4f}  "
          f"min={rewards_arr.min():.4f}  max={rewards_arr.max():.4f}")
    print("=" * 65)

    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["seed", "episode_reward", "steps"])
            writer.writeheader()
            for seed, r, s in zip(
                range(args.seed_start, args.seed_start + args.episodes), rewards, steps_list
            ):
                writer.writerow({"seed": seed, "episode_reward": r, "steps": s})
        print(f"Wrote {len(rewards)} episode(s) to {args.out}")


if __name__ == "__main__":
    main()
