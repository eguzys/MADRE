"""Eval pretrained SB3 FFN-N4 vs PADRE heuristic baselines at varying TRANSMITTER_PROB."""
import argparse, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__) or ".")

import MANET.Environments as envs
from stable_baselines3 import PPO
import MANET.ObservationWrappers as ow
import MANET.RewardWrappers as rw
import MANET.Agents as ag
import MANET.ConditionStrategy as cs
import Utils.training_utils as ut

EXPERT = ("out/best_agents/upload-v3.2-stable/best_agents/"
          "27_04_filtered-PPO_4_[0,5]_[2,21]_13k_ultra_final_Sttc_MANE_True/"
          "rl_model_10000000_steps.zip")

ap = argparse.ArgumentParser()
ap.add_argument("--tp", type=float, required=True)
ap.add_argument("--agent", choices=["ppo", "greedy", "random"], required=True)
ap.add_argument("--users", type=int, default=15)
ap.add_argument("--episodes", type=int, default=20)
ap.add_argument("--jammers", type=int, default=1)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

envs.Static_MANETEnv.TRANSMITTER_PROB = args.tp
print(f"[patch] TRANSMITTER_PROB -> {args.tp}  agent={args.agent}")

# PPO loaded once; heuristics constructed per-episode (they bind to env)
ppo = PPO.load(EXPERT) if args.agent == "ppo" else None

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

    if args.agent == "ppo":
        model = ppo
    elif args.agent == "greedy":
        model = ag.GreedyMaximizer(env=env, condition_strategy=cs.SINRConditionStrategy())
    elif args.agent == "random":
        model = ag.RandomAgent(env=env)

    obs, _ = env.reset()
    ep_r, done = 0.0, False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(a)
        ep_r += float(r)
        done = term or trunc
    rewards.append(ep_r)

r = np.array(rewards)
print(f"\n{args.agent.upper():6}  TP={args.tp}  users={args.users}  n={len(r)}  "
      f"mean={r.mean():.1f}  std={r.std():.1f}  median={np.median(r):.1f}")
