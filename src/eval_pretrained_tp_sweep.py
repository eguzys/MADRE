"""Eval pretrained SB3 FFN-N4 expert at varying TRANSMITTER_PROB (active sender count).

TP is a module-level constant in MANET/Environments.py — monkey-patch before env build.
"""
import argparse, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__) or ".")

import MANET.Environments as envs  # import FIRST so we can patch the constant
from stable_baselines3 import PPO
import MANET.ObservationWrappers as ow
import MANET.RewardWrappers as rw
import Utils.training_utils as ut

EXPERT = ("out/best_agents/upload-v3.2-stable/best_agents/"
          "27_04_filtered-PPO_4_[0,5]_[2,21]_13k_ultra_final_Sttc_MANE_True/"
          "rl_model_10000000_steps.zip")

ap = argparse.ArgumentParser()
ap.add_argument("--tp", type=float, required=True, help="TRANSMITTER_PROB override")
ap.add_argument("--users", type=int, default=15)
ap.add_argument("--episodes", type=int, default=20)
ap.add_argument("--jammers", type=int, default=1)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--model", default=EXPERT)
args = ap.parse_args()

# Patch BEFORE any env construction
orig_tp = envs.Static_MANETEnv.TRANSMITTER_PROB
envs.Static_MANETEnv.TRANSMITTER_PROB = args.tp
print(f"[patch] TRANSMITTER_PROB: {orig_tp} -> {envs.Static_MANETEnv.TRANSMITTER_PROB}")

model = PPO.load(args.model)
rewards = []
for ep in range(args.episodes):
    env = ut.createAndWrapEnv(
        env_constructor=envs.Static_MANETEnv,
        render_mode=None,
        numbOfNodes=4,
        numbOfJammers=args.jammers,
        numbOfUsers=args.users,
        seed=args.seed + ep,
        obs_wrapper=ow.VS_COW,
        obs_config=None,
        rew_wrapper=rw.LinearReward,
        ep_length=1000,
        obs_type="full",
    )
    obs, _ = env.reset()
    ep_r, done = 0.0, False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(a)
        ep_r += float(r)
        done = term or trunc
    rewards.append(ep_r)
    print(f"  ep {ep:2d}: {ep_r:.1f}")

r = np.array(rewards)
print(f"\nTP={args.tp}  users={args.users}  n={len(r)}  "
      f"mean={r.mean():.1f}  std={r.std():.1f}  median={np.median(r):.1f}")
