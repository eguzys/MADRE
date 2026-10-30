"""Collect raw per-drone user sets for PISA (inner set-encoder) pretraining.

Runs the greedy heuristic on the RAW-obs PADRE env and dumps, for every
(episode, step, drone), the user-token set produced by
RawUserSet_OW._build_user_set_for_active_drone() plus its mask.

Greedy is env/condition-driven (reads env state, not obs columns), so it
drives the raw env unchanged and produces realistic drone positions — the
right distribution to pretrain the frozen encoder on.

numbOfUsers is resampled every episode from [--min-users, --max-users]
(inclusive), so drone user-sets vary in size across the dataset. Output
stays rectangular by zero-padding every set up to --max-users; the mask
marks which slots are real (whether or not sensed) vs padding.

Output: an .npz with
  tokens : (M, W, 6) float32   -- M samples, W = --max-users, 6 feature cols
  masks  : (M, W)    float32   -- 1.0 sensed real user, 0.0 unsensed/padding
plus printed summary stats so you know what PISA will train on.

Usage:
  python collect_user_sets.py --episodes 50 --steps 200 --out user_sets.npz
  python collect_user_sets.py --episodes 50 --steps 200 --mix   # greedy+random
  python collect_user_sets.py --users 15   # back-compat: fixed count, still padded to --max-users
  python collect_user_sets.py --policy random --out user_sets_random.npz
  python collect_user_sets.py --policy trained --checkpoint /path/to/checkpoint.pt \
      --out user_sets_trained.npz   # distribution-shift check vs greedy-collected data
"""

import argparse
import numpy as np

from MANET.build import build_padre_env
from MANET.Parallel_enviroments import PADREParallelEnv


def _greedy_fn(condition="sinr"):
    from evaluation.policies import make_greedy_action_fn
    return make_greedy_action_fn(condition_name=condition)


def collect(episodes, steps, out_path, users=None, min_users=5, max_users=22,
            numb_nodes=4, seed0=0, mix=False, fading_sigma_dB=4.0,
            policy="greedy", checkpoint=None, device="cpu"):
    if policy == "trained" and checkpoint is None:
        raise ValueError("--checkpoint is required when --policy trained")

    trained_action_fn = None
    if policy == "trained":
        # Canonical config.pkl path only, same as evaluation/evaluate.py --
        # NOT the bare-.pt Hydra-compose fallback (see policies.py FEATURE_DIM
        # note: that path is cooked-obs-only and would mis-load a
        # raw-obs checkpoint).
        from evaluation.policies import load_trained_policy
        trained_action_fn, task_config = load_trained_policy(str(checkpoint), device)
        print(f"Loaded trained checkpoint: {checkpoint} "
              f"(task_config numbOfNodes={task_config.get('numbOfNodes')})")

    if users is not None and users > max_users:
        raise ValueError(
            f"--users {users} exceeds --max-users {max_users}; "
            "raise --max-users or lower --users"
        )
    # numbOfUsers accepts a fixed int (back-compat, --users) or a (low, high)
    # range that Environments.py resamples every episode via
    # rng.integers(low, high) (high EXCLUSIVE) -- so (min_users, max_users+1)
    # samples min_users..max_users inclusive.
    numb_of_users_cfg = users if users is not None else (min_users, max_users + 1)
    cfg = dict(
        numbOfNodes=numb_nodes, numbOfJammers=1, numbOfUsers=numb_of_users_cfg,
        ep_length=1000, state_loss_rate=0.3, self_flag=True, aoi=True,
        history_K=1, mask=True, obs_wrapper="raw",
        fading_sigma_dB=fading_sigma_dB,
    )

    tokens_all = []
    masks_all = []

    for ep in range(episodes):
        seed = seed0 + ep
        env = PADREParallelEnv(
            build_env_fn=lambda s=seed: build_padre_env(seed=s, **cfg)
        )
        obs_dict, _ = env.reset()
        wrapper = env._wrapper  # RawUserSet_OW

        if policy == "trained":
            action_fn = trained_action_fn
        elif policy == "random":
            from evaluation.policies import make_random_action_fn
            action_fn = make_random_action_fn(env.N)
        else:
            action_fn = _greedy_fn()
        # GreedyHeuristic.reset does env._wrapper.unwrapped_env itself,
        # so it wants the PADREParallelEnv, not the inner env. Trained/random
        # action_fns have no reset() (see policies.py), so this is skipped
        # for them -- same guard runner.py uses.
        if hasattr(action_fn, "reset"):
            action_fn.reset(env, seed)

        for t in range(steps):
            # --- dump every drone's user set at the current state ---
            n_users_ep = env._wrapper.unwrapped_env.numbOfUsers
            for di in range(env.N):
                wrapper._active_drone = di
                try:
                    ut, um = wrapper._build_user_set_for_active_drone()
                finally:
                    wrapper._active_drone = None
                # ut/um are sized to this episode's numbOfUsers (n_users_ep,
                # which may be < max_users); pad up to a rectangular width.
                padded_tok = np.zeros((max_users, 6), dtype=np.float32)
                padded_tok[:n_users_ep] = ut
                padded_mask = np.zeros((max_users,), dtype=np.float32)
                padded_mask[:n_users_ep] = um
                tokens_all.append(padded_tok)
                masks_all.append(padded_mask)

            # --- advance the env with the selected policy ---
            if policy == "greedy":
                if mix and (t % 2 == 0):
                    actions = {
                        f"drone_{i}": np.random.uniform(-1, 1, size=2).astype(np.float32)
                        for i in range(env.N)
                    }
                else:
                    obs = {f"drone_{i}": np.zeros(1, dtype=np.float32)
                           for i in range(env.N)}  # greedy ignores obs content
                    try:
                        actions = action_fn(obs, env.N)
                    except Exception:
                        # fall back to zero action if greedy interface differs
                        actions = {f"drone_{i}": np.zeros(2, dtype=np.float32)
                                   for i in range(env.N)}
            else:
                # random/trained actually consume the env's real observation.
                actions = action_fn(obs_dict, env.N)
            # ensure correct keys/shape
            actions = {f"drone_{i}": np.asarray(
                actions.get(f"drone_{i}", np.zeros(2)), dtype=np.float32
            ).reshape(2) for i in range(env.N)}

            obs_dict, _, term, trunc, _ = env.step(actions)
            # term/trunc are dicts keyed by agent; a non-empty dict is
            # always truthy, so check the VALUES, not the dict itself.
            done = (any(term.values()) if isinstance(term, dict) else bool(term)) or \
                   (any(trunc.values()) if isinstance(trunc, dict) else bool(trunc))
            if done:
                break

        if (ep + 1) % 10 == 0:
            print(f"  episode {ep+1}/{episodes} done, "
                  f"{len(tokens_all)} samples so far")

    tokens = np.stack(tokens_all, axis=0)  # (M, U, 6)
    masks = np.stack(masks_all, axis=0)    # (M, U)

    np.savez_compressed(out_path, tokens=tokens, masks=masks)

    # ---- summary stats: what PISA will train on ----
    print("\n=== collected dataset ===")
    print(f"samples (M):        {tokens.shape[0]}")
    print(f"users per set (U):  {tokens.shape[1]}")
    print(f"feature cols:       {tokens.shape[2]}")
    sensed = masks.sum(axis=1)
    print(f"sensed users/set:   mean={sensed.mean():.2f} "
          f"min={int(sensed.min())} max={int(sensed.max())}")
    print(f"fraction of non-empty sets: "
          f"{(sensed > 0).mean():.2%}")
    # per-column ranges over SENSED entries only (mask==1)
    m = masks.astype(bool)
    print("\nper-column stats over SENSED tokens "
          "[angle, power, snr, noise, in, out]:")
    for c, name in enumerate(
        ["angle", "power", "snr", "noise", "thr_in", "thr_out"]
    ):
        vals = tokens[:, :, c][m]
        print(f"  {name:7s} min={vals.min():+.3f} "
              f"max={vals.max():+.3f} mean={vals.mean():+.3f} "
              f"std={vals.std():.3f}")
    print(f"\nsaved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", type=str, default="user_sets.npz")
    ap.add_argument("--users", type=int, default=None,
                     help="fixed numbOfUsers override (back-compat); if "
                          "unset, numbOfUsers is resampled every episode "
                          "from [--min-users, --max-users]")
    ap.add_argument("--min-users", type=int, default=5,
                     help="inclusive lower bound for per-episode numbOfUsers")
    ap.add_argument("--max-users", type=int, default=22,
                     help="inclusive upper bound for per-episode numbOfUsers; "
                          "also the padded token/mask width")
    ap.add_argument("--nodes", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--mix", action="store_true",
                    help="alternate greedy with random moves for broader "
                         "position coverage (reduces distribution shift); "
                         "only applies with --policy greedy")
    ap.add_argument("--policy", choices=["greedy", "random", "trained"],
                     default="greedy",
                     help="policy driving the env while user-sets are dumped "
                          "(default: greedy)")
    ap.add_argument("--checkpoint", type=str, default=None,
                     help="BenchMARL checkpoint .pt path; required when "
                          "--policy trained")
    ap.add_argument("--device", type=str, default="cpu",
                     help="compute device for --policy trained (default: cpu)")
    args = ap.parse_args()
    if args.policy == "trained" and args.checkpoint is None:
        ap.error("--checkpoint is required when --policy trained")
    collect(args.episodes, args.steps, args.out, args.users, args.min_users,
             args.max_users, args.nodes, args.seed0, args.mix,
             policy=args.policy, checkpoint=args.checkpoint, device=args.device)


if __name__ == "__main__":
    main()
