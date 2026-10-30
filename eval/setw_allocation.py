#!/usr/bin/env python3
"""
setw_allocation.py -- Test 2 of the OMNeT reward checks: does the trained policy's per-user allocation SHAPE move toward the
oracle-optimal allocation over the late-tail checkpoints, even if mean
episode return is flat?

Rationale (per-user, not per-episode): the training/eval reward
(max_flow_reward.py::compute_oracle_max_flow_reward) is Sigma_k log1p(tau_k)
summed over commodities -- a scalar that sums away allocation FAIRNESS.
Two very different per-user allocations can produce the same scalar reward.
This script instead compares, per step, the DISTRIBUTION over users of:

  oracle_per_user[u]  -- the position-oracle LP's optimal flow tau_k for
                          user u's fixed commodity (compute_position_oracle_
                          reward's per_commodity_tau, keyed by src user),
                          i.e. the best achievable allocation GIVEN the
                          actual drone positions the trained policy chose
                          this step. Same LP the training reward already
                          solves at this exact step -- not a separate
                          "optimal positions" baseline.
  actual_per_user[u]   -- obs_json["user_goodput"][u] (bytes/sec, sim-
                          measured real delivered throughput), converted to
                          bps to match tau's units.

setW (this script's metric, NOT a percentage despite the historical name in
src/encoder_experiments/distributional_recon_omnet.py -- that script is a
DIFFERENT metric, encoder-reconstruction quality, and does not apply to
these actor-only checkpoints, which have no decoder) is the Wasserstein-1
distance between these two per-user distributions, averaged over steps and
episodes, normalized by the GLOBAL std of the pooled oracle values (mirrors
the normalization convention of the encoder-recon script: raw per-set WD
values are pooled and only the final aggregate is divided by the natural
scale -- see _run for the exact computation).

Lower setW = the policy's actual delivered allocation shape sits closer to
what the LP says is achievable for its own chosen positions. Track this
across an architecture's 3 checkpoints: shrinking = the policy is learning
allocation quality invisible to mean return; flat = it isn't.

Usage:
    python setw_allocation.py --checkpoint <path.pt> --num-drones 5 \\
        -n 10 --seed-start 0 --out setw_st_n5_ckpt2160000.csv

Checkpoint loading goes through eval/checkpoint_utils.py::load_and_verify_policy
(hard-fails on a bad weight-copy load -- see that module's docstring).
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

from scipy.stats import wasserstein_distance  # noqa: E402

import omnet_env  # noqa: E402
from evaluate_omnet import _patched_num_drones, _close_leaked_omnet_envs  # noqa: E402

# Assert on the actual training/eval reward machinery -- never the legacy
# compute_weighted_maxmin_flow smoke-test diagnostic (settled context).
from max_flow_reward import (  # noqa: E402
    compute_position_oracle_reward,
    compute_oracle_max_flow_reward,
)
assert omnet_env.compute_oracle_max_flow_reward is compute_oracle_max_flow_reward, (
    "omnet_env.py is not calling max_flow_reward.compute_oracle_max_flow_reward "
    "-- reward wiring drifted, do not trust this script's numbers."
)

_BYTES_TO_BITS = 8.0


def _oracle_per_user(obs_json: dict) -> dict:
    """{user_idx: tau_bps} -- optimal LP flow for user_idx's commodity,
    aggregated over all commodities where user_idx is the source (usually
    exactly one, per the fixed sender_pairs; the legacy all-pairs fallback
    can have more than one, so sum rather than assume uniqueness)."""
    result = compute_position_oracle_reward(obs_json)
    per_user: dict = {}
    for (src, _dst), tau in result["per_commodity_tau"].items():
        per_user[src] = per_user.get(src, 0.0) + tau
    return per_user


def _actual_per_user(obs_json: dict) -> dict:
    """{user_idx: bps} -- sim-measured actually-delivered throughput per user."""
    goodput = obs_json.get("user_goodput", [])
    return {u: float(g) * _BYTES_TO_BITS for u, g in enumerate(goodput)}


def _step_wasserstein(obs_json: dict):
    """Returns (raw_wd, oracle_vals) for one step, or None if too few users
    to form a distribution (Wasserstein needs >=1 point per side, but a
    single-point comparison is degenerate -- require >=2 users, matching
    distributional_recon_omnet.py's `if k >= 2` per-set gate)."""
    oracle = _oracle_per_user(obs_json)
    actual = _actual_per_user(obs_json)
    users = sorted(set(oracle.keys()) & set(actual.keys()))
    if len(users) < 2:
        return None
    oracle_vals = np.array([oracle[u] for u in users], dtype=float)
    actual_vals = np.array([actual[u] for u in users], dtype=float)
    raw_wd = float(wasserstein_distance(oracle_vals, actual_vals))
    return raw_wd, oracle_vals


def run_episode_allocation(env, action_fn, seed: int, num_drones: int):
    """One seeded episode: returns (list of per-step raw WD, list of
    per-step oracle-value arrays for global-std pooling, steps)."""
    obs, _infos = env.reset(seed=seed)
    per_step_wd = []
    per_step_oracle_vals = []
    steps = 0
    while True:
        actions = action_fn(obs, num_drones)
        obs, _rewards, _terms, _truncs, _infos = env.step(actions)
        steps += 1

        result = _step_wasserstein(env._last_obs)
        if result is not None:
            raw_wd, oracle_vals = result
            per_step_wd.append(raw_wd)
            per_step_oracle_vals.append(oracle_vals)

        if not env.agents:
            break
    return per_step_wd, per_step_oracle_vals, steps


def parse_args():
    p = argparse.ArgumentParser(
        description="Test 2 -- per-user allocation-shape Wasserstein distance "
                    "(oracle LP tau vs sim-actual goodput), fixed-seed episodes.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="BenchMARL .pt checkpoint (trained policy only -- this "
                        "metric is meaningless for random/greedy).")
    p.add_argument("--num-drones", type=int, required=True, dest="num_drones")
    p.add_argument("-n", "--episodes", type=int, default=30)
    p.add_argument("--seed-start", type=int, default=0, dest="seed_start",
                   help="Episodes use seeds seed_start..seed_start+n-1. "
                        "SEED DISCIPLINE: 0-99 is the evaluation band; "
                        "100-199 is tuning-only. Straying past 99 is refused.")
    p.add_argument("--out", type=str, default=None,
                   help="Optional per-episode CSV path (seed, mean_step_wd, n_steps_scored).")
    p.add_argument("--device", type=str, default=None,
                   help="Compute device (default: cuda if available else cpu).")
    a = p.parse_args()
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


def main():
    args = parse_args()

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

    all_step_wd = []          # raw per-step WD, pooled across every episode
    all_oracle_vals = []      # raw per-user oracle tau values, pooled (for global std)
    per_episode_rows = []

    with _patched_num_drones(args.num_drones):
        env = omnet_env.OMNeTEnv(worker_index=0)
        try:
            for seed in range(args.seed_start, args.seed_start + args.episodes):
                step_wd, step_oracle_vals, steps = run_episode_allocation(
                    env, action_fn, seed, args.num_drones
                )
                all_step_wd.extend(step_wd)
                all_oracle_vals.extend(step_oracle_vals)
                mean_wd = float(np.mean(step_wd)) if step_wd else float("nan")
                per_episode_rows.append(
                    {"seed": seed, "mean_step_wd": mean_wd,
                     "n_steps_scored": len(step_wd), "n_steps": steps}
                )
                print(f"  ep seed={seed:4d}  steps={steps:4d}  "
                      f"scored={len(step_wd):4d}  mean_step_wd={mean_wd:14.4f}",
                      flush=True)
        finally:
            env.close()

    if not all_step_wd:
        raise RuntimeError(
            "No steps produced a scoreable per-user distribution (every step "
            "had <2 users with both an oracle tau and a goodput value) -- "
            "cannot compute setW. This is a data/wiring problem, not a "
            "training-quality finding; investigate before reporting anything."
        )

    global_std = float(np.std(np.concatenate(all_oracle_vals)))
    mean_raw_wd = float(np.mean(all_step_wd))
    if global_std <= 0:
        setw = float("nan")
        print("\nWARNING: global std of pooled oracle tau values is 0 -- "
              "setW is undefined (every oracle allocation was identical). "
              "Reporting NaN rather than a divide-by-zero artifact.")
    else:
        setw = mean_raw_wd / global_std

    print("\n" + "=" * 65)
    print(f"Test 2 summary -- N={args.num_drones}  ckpt={args.checkpoint}")
    print(f"  episodes: {len(per_episode_rows)}   steps scored: {len(all_step_wd)}")
    print(f"  mean raw per-step WD (bps): {mean_raw_wd:.4f}")
    print(f"  global std of pooled oracle tau (bps): {global_std:.4f}")
    print(f"  setW (mean raw WD / global std, NOT a percentage): {setw:.6f}")
    print("=" * 65)

    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["seed", "mean_step_wd", "n_steps_scored", "n_steps"]
            )
            writer.writeheader()
            writer.writerows(per_episode_rows)
        print(f"Wrote {len(per_episode_rows)} episode row(s) to {args.out}")


if __name__ == "__main__":
    main()
