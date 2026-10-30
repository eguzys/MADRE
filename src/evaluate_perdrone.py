"""evaluate_perdrone.py — minimal evaluation entry point for a model
trained with `train_perdrone.py`.

Loads a saved PPO model, builds a fresh `PerDroneSingleStepEnv` with the
matching env config, and runs a few episodes. Reports mean episode
reward, mean reward per drone-step, mean network throughput, and
episode lengths.

Standalone — does NOT import from `evaluate.py`,
`evaluate_MANETAgents.py`, or `train_MANETAgent.py`.

Usage (from repo root, padre venv):
    python src/evaluate_perdrone.py \\
        --model out/perdrone/ppo_perdrone --episodes 3 --numbOfNodes 4
"""
from __future__ import annotations

import argparse
import os
import sys

from sklearn import base

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import matplotlib
matplotlib.use("Agg")  # headless backend; only needed for --render save
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

import MANET.Environments as envs
import MANET.RewardWrappers as rw

from train_perdrone import PerDroneSingleStepEnv  # the same wrapper used at training time


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a per-drone PPO model.")
    p.add_argument("--model", type=str, required=True,
                   help="Path to a saved model (without or with .zip).")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--numbOfNodes", type=int, default=4)
    p.add_argument("--numbOfJammers", type=int, default=1)
    p.add_argument("--numbOfUsers", type=int, default=10)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--state_loss_rate", type=float, default=0.0)
    p.add_argument(
        "--self_flag",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--aoi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Toggle the Age-of-Information column. Must match the "
             "trained model's setting.",
    )
    p.add_argument(
        "--history_K",
        type=int,
        default=1,
        help="Per-link history length. Must match training-time value.",
    )
    p.add_argument(
        "--mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Toggle the mask column. Must match training-time value.",
    )
    p.add_argument("--ep_length", type=int, default=1000)
    p.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--render",
        choices=["none", "plot", "save"],
        default="none",
        help="Visualisation mode: none (default), plot (live tkinter GUI), "
             "save (write PNG frames to --render_dir).",
    )
    p.add_argument(
        "--render_dir",
        type=str,
        default=None,
        help="Directory for saved frames when --render save. "
             "Defaults to ./eval_renders/<model_basename>.",
    )
    return p.parse_args()


def _save_frame(env_u, path: str) -> None:
    """Draw env state as a matplotlib figure and save to path as PNG."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    for u, v in env_u.network.getGraph().edges:
        ax.plot([u.pos[0], v.pos[0]], [u.pos[1], v.pos[1]], "k-", lw=0.5)
    up = np.asarray(env_u.user_positions)
    np_ = np.asarray(env_u.node_positions)
    jp = np.asarray(env_u.jammer_positions)
    ax.scatter(up[:, 0], up[:, 1], c="green", s=20, zorder=3, label="users")
    ax.scatter(np_[:, 0], np_[:, 1], c="blue", s=40, zorder=3, label="nodes")
    ax.scatter(jp[:, 0], jp[:, 1], c="red", s=40, zorder=3, label="jammers")
    ax.legend(loc="upper right", fontsize=6)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def build_env(args, render_mode: str = None, joint_output: bool = False) -> PerDroneSingleStepEnv:
    base = envs.Static_MANETEnv(
        numbOfNodes=args.numbOfNodes,
        numbOfJammers=args.numbOfJammers,
        numbOfUsers=args.numbOfUsers,
        seed=args.seed,
        networkConnectedAtStart=True,
        render_mode=render_mode,
    )
    base.set_ep_length(args.ep_length)

    base.set_step_jammers_start_moving(100)
    base.set_steps_till_activate_jammers(500)
    rew = rw.LinearReward(base)
    return PerDroneSingleStepEnv(
        rew,
        state_loss_rate=args.state_loss_rate,
        self_flag=args.self_flag,
        aoi=args.aoi,
        history_K=args.history_K,
        mask=args.mask,
        joint_output=joint_output,
    )


def main():
    args = parse_args()

    render = args.render
    render_dir = None
    if render == "save":
        model_base = os.path.basename(args.model).removesuffix(".zip")
        render_dir = args.render_dir or os.path.join("eval_renders", model_base)
        os.makedirs(render_dir, exist_ok=True)

    print(f"Loading model from {args.model}")
    model = PPO.load(args.model)
    saved_cfg = getattr(model, "training_config", {}) or {}
    joint_output = saved_cfg.get("joint_output", False)
    print(f"joint_output={joint_output} (from training_config)")
    drone_env = build_env(args, render_mode="plot" if render == "plot" else None, joint_output=joint_output)
    print(f"obs_space={drone_env.observation_space}")
    print(f"act_space={drone_env.action_space}")

    # Fail loudly if the env's per-drone obs shape doesn't match what
    # the saved policy was trained on. SB3 would raise on first
    # predict() anyway, but the message there is generic.
    model_shape = tuple(model.observation_space.shape)
    env_shape = tuple(drone_env.observation_space.shape)
    if model_shape != env_shape:
        raise ValueError(
            f"Observation-shape mismatch: saved policy expects "
            f"{model_shape}, current env produces {env_shape}. "
            f"Check --self_flag / --aoi / --history_K / --mask flags "
            f"against the training-time values."
        )

    ep_rewards: list[float] = []  # sum of reward signals per episode
    ep_lengths: list[int] = []    # drone-steps per episode
    ep_throughput: list[float] = []  # mean throughput across env-steps

    N = drone_env.numbOfNodes
    ep_env_rewards: list[float] = []  # sum of REAL env-step rewards (un-replicated)
    for ep in range(args.episodes):
        obs, info = drone_env.reset(seed=args.seed + ep)
        terminated = truncated = False
        ppo_reward_sum = 0.0  # PPO-view sum (each env-step's R replicated N times)
        env_reward_sum = 0.0  # real env-step reward sum
        ep_len = 0
        env_throughputs: list[float] = []
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = drone_env.step(action)
            ppo_reward_sum += reward
            ep_len += 1
            # Real env transition happens every Nth drone-step (the last
            # call of each loop, where the action buffer is dispatched).
            # The wrapper tags that call's `info` with the un-replicated
            # env-step reward.
            if ep_len % N == 0:
                env_throughputs.append(
                    float(drone_env.unwrapped_env.network.getThroughput())
                )
                env_reward_sum += float(info.get("env_step_reward", 0.0))
                if render == "save":
                    env_step = ep_len // N
                    _save_frame(
                        drone_env.unwrapped_env,
                        os.path.join(render_dir, f"ep{ep:03d}_step{env_step:05d}.png"),
                    )
        ep_rewards.append(ppo_reward_sum)
        ep_env_rewards.append(env_reward_sum)
        ep_lengths.append(ep_len)
        ep_throughput.append(
            float(np.mean(env_throughputs)) if env_throughputs else 0.0
        )
        print(
            f"  episode {ep}: env_reward={env_reward_sum:.4f}  "
            f"ppo_view_sum={ppo_reward_sum:.4f}  "
            f"drone_steps={ep_len}  env_steps={ep_len // N}  "
            f"mean_throughput={ep_throughput[-1]:.4f}"
        )

    print()
    print(f"  mean env reward          : {np.mean(ep_env_rewards):.4f}")
    print(f"  mean PPO-view reward sum : {np.mean(ep_rewards):.4f}  "
          f"(~ N x env reward; PPO replicates joint reward across drones)")
    print(f"  mean drone-steps/episode : {np.mean(ep_lengths):.1f}")
    print(f"  mean network throughput  : {np.mean(ep_throughput):.4f}")


if __name__ == "__main__":
    main()
