"""
visualize_padre_expert.py

Loads the published static FFN expert (PPO_4) and runs it with the live PADRE
GUI so you can watch what the drones actually do. Slows step rate so you can
take screenshots.

Usage (from repo root, with `padre` venv active):
    python src/visualize_padre_expert.py
    python src/visualize_padre_expert.py --episodes 1 --jammers 1 --users 10 --speed 0.3
    python src/visualize_padre_expert.py --no-jammer       # see what they do without any threat
    python src/visualize_padre_expert.py --seed 7

Args:
    --episodes   how many episodes to run before exiting (default 1)
    --jammers    fixed number of jammers (default 1; use 0 to disable)
    --users      fixed number of users (default 10; PADRE training used U(2,21))
    --speed      seconds to sleep between env steps (default 0.2 -- larger = slower)
    --seed       env seed (default 42)
    --ep-length  max steps per episode (default 1000, matching the paper)
    --no-jammer  shortcut for --jammers 0

What you should look for visually:
    1. Do the MANET nodes (blue) form a connected backbone, or do they sit in
       individual sweet spots?
    2. When users (green) cluster in two separate groups, do the nodes bridge
       between the clusters, or just sit on top of one group each?
    3. After the jammer (red) activates at step 500, do the nodes coordinate
       the recovery, or each scramble independently?
"""

import argparse
import os
import sys
import time

# Make `src` importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
from stable_baselines3 import PPO

import MANET.Environments as envs
import MANET.ObservationWrappers as ow
import MANET.RewardWrappers as rw
import Utils.training_utils as ut


EXPERT_PATH = (
    "out/best_agents/"
    "27_04_filtered-PPO_4_[0,5]_[2,21]_13k_ultra_final_Sttc_MANE_True/"
    "rl_model_10000000_steps.zip"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--jammers", type=int, default=1)
    p.add_argument("--users", type=int, default=10)
    p.add_argument("--speed", type=float, default=0.2,
                   help="seconds between steps (larger = slower)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ep-length", type=int, default=1000)
    p.add_argument("--no-jammer", action="store_true",
                   help="shortcut for --jammers 0")
    p.add_argument("--model", type=str, default=EXPERT_PATH,
                   help="path to a .zip PPO model (default: static FFN expert)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.no_jammer:
        args.jammers = 0

    print(f"Loading model: {args.model}")
    model = PPO.load(args.model)
    print(f"  obs space: {model.observation_space}")
    print(f"  act space: {model.action_space}")

    print(f"\nBuilding env: nodes=4, jammers={args.jammers}, users={args.users}, "
          f"seed={args.seed}, ep_length={args.ep_length}")

    env = ut.createAndWrapEnv(
        env_constructor=envs.Static_MANETEnv,
        render_mode="plot",            # PADRE's GUI mode (NOT 'human')
        numbOfNodes=4,                 # matches PPO_4 expert
        numbOfJammers=args.jammers,
        numbOfUsers=args.users,
        seed=args.seed,
        obs_wrapper=ow.VS_COW,         # matches expert's training config
        obs_config=None,
        rew_wrapper=rw.LinearReward,
        ep_length=args.ep_length,
        obs_type="full",               # god mode, matches training
    )

    print(f"\nRunning {args.episodes} episode(s). Step delay = {args.speed}s.")
    print("Close the GUI window (or Ctrl+C in terminal) to stop.\n")

    for ep in range(args.episodes):
        obs, info = env.reset()
        terminated = truncated = False
        step = 0
        ep_throughput = []

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_throughput.append(float(reward))

            # Periodic console heartbeat so you know what's happening
            if step % 50 == 0:
                avg = np.mean(ep_throughput[-50:]) if ep_throughput else 0.0
                print(f"  ep {ep} step {step:4d}  recent avg reward = {avg:.4f}")

            step += 1
            if args.speed > 0:
                time.sleep(args.speed)

        mean_tp = float(np.mean(ep_throughput)) if ep_throughput else 0.0
        print(f"\n  episode {ep} done after {step} steps. "
              f"mean throughput (reward) = {mean_tp:.4f}\n")

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()