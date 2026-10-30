"""train_perdrone.py — training entry point for `PerDrone_OW`.

Builds a `Static_MANETEnv` + `LinearReward`, attaches `PerDrone_OW`, and
exposes the result as a single-drone gym env so PPO trains one shared
policy that is called N times per underlying env step (once per drone,
on that drone's own per-drone observation).

Standalone — does NOT import from `train.py` or `train_MANETAgent.py`.
The existing training stack remains untouched.

Long-run features (added so 24-hour cluster jobs survive crashes and
can be extended without redoing all the training):
- `--checkpoint_freq INT` snapshots the model every N drone-steps via
  SB3's `CheckpointCallback`. Default 100_000. Set to 0 to disable.
- `--tensorboard_log DIR` enables TensorBoard event logging.
- `--resume PATH` continues training from an existing model
  (intermediate checkpoint OR final save). Env-construction flags
  (`history_K`, `mask`, `self_flag`, `aoi`, `numbOfNodes`,
  `architecture`, `numbOfJammers`, `numbOfUsers`, `state_loss_rate`,
  `ep_length`, `seed`) are auto-loaded from the model's embedded
  `training_config`. PPO's `num_timesteps` counter continues from
  where it left off.
- `--allow_cpu` is required to run on a CUDA-less host. ST always
  requires CUDA regardless. No interactive prompt — cluster jobs
  can't answer one.

Usage (from repo root, padre venv):

  # Fresh training run with checkpointing + TensorBoard:
    python src/train_perdrone.py \\
        --timesteps 500000 --architecture ST --history_K 10 --mask \\
        --checkpoint_freq 100000 --tensorboard_log out/tb/run1 \\
        --save out/perdrone/run1

  # Resume from the most recent intermediate checkpoint (no
  # env flags needed — they all come from the embedded config):
    python src/train_perdrone.py \\
        --resume out/perdrone/run1_checkpoints/run1_400000_steps.zip \\
        --timesteps 100000 --tensorboard_log out/tb/run1 \\
        --save out/perdrone/run1

  # Extend a fully-trained model:
    python src/train_perdrone.py \\
        --resume out/perdrone/run1 --timesteps 200000 \\
        --save out/perdrone/run1_extended

Runtime-only flags (`--numbOfJammers`, `--numbOfUsers`,
`--state_loss_rate`, `--ep_length`, `--seed`) may be overridden on
resume — a notice is printed and the new value is used. Shape-
affecting flags (`--history_K`, `--mask`, `--self_flag`, `--aoi`,
`--numbOfNodes`, `--architecture`) must match the embedded config
or the resume aborts.
"""
from __future__ import annotations

import argparse
from email.mime import base
import os
import sys

from sklearn import base

# Make MANET importable when run from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

import MANET.Environments as envs
import MANET.ObservationWrappers as ow
import MANET.RewardWrappers as rw


class PerDroneSingleStepEnv(gym.Env):
    """Bridges a `PerDrone_OW`-equipped env to a single-drone gym
    interface. Each call to `.step()` is one drone's policy call. Every
    `numbOfNodes` calls, the accumulated `(N, 2)` action vector is
    dispatched to the underlying env.

    Reward attribution (cooperative-MARL replicate-uniformly pattern):
    every drone-call in a given round returns the SAME reward — the
    reward produced by the PREVIOUS round's env-step. Concretely, when
    the env dispatch on call N-1 of round T produces R_T, that R_T is
    held and returned by every drone-call in round T+1. Round 0's
    drone-calls return 0 (no prior env-step). Action a_{T,k} is
    therefore credited with R_{T-1}, not R_T — a one-round delay — but
    every drone in the round gets the same credit, so PPO does not
    favor drone N-1 over drones 0..N-2. The total reward PPO sees over
    the rollout is the env reward overcounted by a factor of N, which
    is intentional and standard in cooperative MARL.

    Termination / truncation flags follow the same delay so `(s, a, r,
    s', done)` tuples remain consistent: when env.step on round T
    returns terminated/truncated, the flag is surfaced on round T+1's
    first drone-call (which is when PPO will auto-reset).

    The dispatched env-step reward is also attached to `info` as
    `info["env_step_reward"]` on the call where dispatch happens, so
    evaluation code can recover the un-replicated env reward.

    Observation space: `Box((N, K))` where K is the per-slot feature
    width reported by `PerDrone_OW` (features + self-flag + AoI
    columns, whichever are enabled). K is probed at construction
    rather than computed from a formula, so future PerDrone_OW
    columns flow through automatically.

    Action space: `Box((2,))` by default — `(dx, dy)` for the drone
    whose turn it is. When `joint_output=True`, the space is
    `Box((N*2,))` and drone i's effective action is
    `action[2*i:2*i+2]`; the remaining components are discarded.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env: gym.Env,
        state_loss_rate: float = 0.0,
        self_flag: bool = True,
        aoi: bool = True,
        history_K: int = 1,
        mask: bool = False,
        joint_output: bool = False,
    ):
        super().__init__()
        self.env = env
        self.unwrapped_env = env.unwrapped
        self.numbOfNodes = self.unwrapped_env.numbOfNodes
        self.self_flag = self_flag
        self.aoi = aoi
        self.history_K = history_K
        self.mask = mask
        self.joint_output = joint_output

        # PerDrone_OW reads the env via `env.unwrapped`, so wrapping over
        # a RewardWrapper is fine.
        self._per_drone_ow = ow.PerDrone_OW(
            state_loss_rate=state_loss_rate,
            env=env,
            self_flag=self_flag,
            aoi=aoi,
            history_K=history_K,
            mask=mask,
        )

        # Probe the observation tensor to get the per-drone token-set
        # shape directly from the wrapper. Output shape is (N, N, F') for
        # history_K=1 and (N, N*K, F') for history_K>1, so we derive the
        # per-drone view shape as the trailing two dims. Any future
        # PerDrone_OW column flows through automatically.
        # PPO will reset again at the start of `learn()`; PerDrone_OW
        # handles that via its own `hasEnvBeenReset` check.
        self.env.reset()
        probe_obs = self._per_drone_ow.get_observation()
        obs_shape = tuple(probe_obs.shape[1:])
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=obs_shape, dtype=np.float32
        )
        action_dim = self.numbOfNodes * 2 if joint_output else 2
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

        self._action_buffer = np.zeros(
            (self.numbOfNodes, 2), dtype=np.float32
        )
        self._active_drone = 0
        self._cached_per_drone_obs: np.ndarray | None = None  # (N, N, F)
        # Pending signals from the previous env-step, replicated across
        # every drone-call of the current round.
        self._pending_reward = 0.0
        self._pending_terminated = False
        self._pending_truncated = False
        self._pending_info: dict = {}

    def _refresh_obs(self) -> None:
        self._cached_per_drone_obs = self._per_drone_ow.get_observation()

    def _current_drone_view(self) -> np.ndarray:
        return self._cached_per_drone_obs[self._active_drone].astype(np.float32)

    def reset(self, seed=None, options=None):
        _, info = self.env.reset(seed=seed, options=options)
        self._action_buffer.fill(0.0)
        self._active_drone = 0
        self._refresh_obs()
        # Pending signals are zeroed each reset — round 0 returns no
        # reward and no terminated/truncated.
        self._pending_reward = 0.0
        self._pending_terminated = False
        self._pending_truncated = False
        self._pending_info = {}
        return self._current_drone_view(), info

    def step(self, action):
        a = np.asarray(action, dtype=np.float32)
        if self.joint_output:
            i = self._active_drone
            effective = a[2 * i : 2 * i + 2]
        else:
            effective = a
        self._action_buffer[self._active_drone] = effective

        if self._active_drone < self.numbOfNodes - 1:
            # Mid-loop drone call: replicate the most recent env-step's
            # signals (held in _pending_*). No real env step yet.
            self._active_drone += 1
            return (
                self._current_drone_view(),
                self._pending_reward,
                self._pending_terminated,
                self._pending_truncated,
                dict(self._pending_info),
            )

        # Last drone call: dispatch the buffered joint action. Return
        # the OLD pending signals (so this call is the same as its
        # round-mates), then promote the freshly dispatched signals to
        # _pending_* for the next round.
        _, dispatched_reward, dispatched_terminated, dispatched_truncated, dispatched_info = (
            self.env.step(self._action_buffer)
        )
        self._active_drone = 0
        if not (dispatched_terminated or dispatched_truncated):
            self._refresh_obs()

        out_reward = self._pending_reward
        out_terminated = self._pending_terminated
        out_truncated = self._pending_truncated
        out_info = dict(self._pending_info)
        # Tag the call where env.step actually fired with the un-
        # replicated reward, so eval can recover the real env reward.
        out_info["env_step_reward"] = float(dispatched_reward)

        self._pending_reward = float(dispatched_reward)
        self._pending_terminated = bool(dispatched_terminated)
        self._pending_truncated = bool(dispatched_truncated)
        self._pending_info = (
            dict(dispatched_info) if dispatched_info is not None else {}
        )

        return (
            self._current_drone_view(),
            out_reward,
            out_terminated,
            out_truncated,
            out_info,
        )


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on PerDrone_OW.")
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--numbOfNodes", type=int, default=4)
    p.add_argument("--numbOfJammers", type=int, default=1)
    p.add_argument("--numbOfUsers", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--state_loss_rate", type=float, default=0.0)
    p.add_argument(
        "--self_flag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Toggle the self-flag column. Default on.",
    )
    p.add_argument(
        "--aoi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Toggle the Age-of-Information column. Default on.",
    )
    p.add_argument(
        "--history_K",
        type=int,
        default=1,
        help="Number of historical observations kept per (i, j) link. "
             "K=1 (default) reproduces the single-timestep view. K>1 "
             "packs the last K env-steps as separate tokens — output "
             "shape becomes (N, N*K, token_dim).",
    )
    p.add_argument(
        "--mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append a mask column (1.0 = real observation, 0.0 = "
             "filler). Default off so output stays bit-identical to "
             "the pre-history wrapper at --history_K 1. Recommended "
             "when history_K > 1.",
    )
    p.add_argument("--ep_length", type=int, default=1000)
    p.add_argument("--n_steps", type=int, default=2048,
                   help="PPO rollout buffer size in drone-steps.")
    p.add_argument(
        "--architecture",
        type=str,
        default=None,
        choices=["MlpPolicy", "ST"],
        help="Policy architecture. MlpPolicy flattens the per-drone "
             "token tensor; ST consumes it as a set and routes the "
             "mask column to its attention layers (requires --mask "
             "for full effect). Default is MlpPolicy for a fresh "
             "run; ignored when --resume is set (the architecture is "
             "baked into the saved model).",
    )
    p.add_argument(
        "--checkpoint_freq",
        type=int,
        default=100_000,
        help="Snapshot the model every N drone-steps via SB3's "
             "CheckpointCallback. Set to 0 to disable. Default "
             "100_000 — at this rate a crashed 24h job loses at most "
             "one snapshot interval. Checkpoints go to "
             "{save}_checkpoints/ with name_prefix matching the "
             "save basename.",
    )
    p.add_argument(
        "--tensorboard_log",
        type=str,
        default=None,
        help="Directory for TensorBoard event files. Forwarded to "
             "PPO's tensorboard_log= kwarg. Default None (no TB).",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to an existing saved model (.zip, with or without "
             "extension). When provided, training continues from the "
             "loaded weights for --timesteps ADDITIONAL drone-steps. "
             "Env-construction flags are auto-loaded from the model's "
             "embedded training_config — only --timesteps (and "
             "optional runtime overrides like --numbOfJammers / "
             "--state_loss_rate) need to be re-passed.",
    )
    p.add_argument(
        "--allow_cpu",
        action="store_true",
        default=False,
        help="Explicit confirmation that running on CPU is intentional. "
             "Without this flag, the script aborts on CUDA-less hosts "
             "to prevent accidental multi-hour CPU runs. ST requires "
             "CUDA and ignores --allow_cpu.",
    )
    p.add_argument(
        "--joint_output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When on, the policy outputs (N*2,)-shaped actions and drone i "
             "uses slots [2i, 2i+1]. Mirrors god-mode FFN output shape. "
             "Incompatible with --architecture ST. Default off.",
    )
    p.add_argument("--save", type=str, default="out/perdrone/ppo_perdrone")
    return p.parse_args()


# Flags whose values change the saved policy's input shape. If the
# user tries to resume with a SHAPE_FLAG that contradicts the saved
# model's config, abort — the model literally cannot consume the new
# observation. Architecture is in this set because the policy class
# itself is baked into the saved zip.
SHAPE_FLAGS = (
    "numbOfNodes", "self_flag", "aoi", "history_K", "mask", "architecture",
    "joint_output",
)

# Flags that don't affect the obs/policy shape, only the env's
# behaviour during the rollout. Safe to override on resume; we print
# a notice so the user is aware they're departing from the saved
# config.
RUNTIME_FLAGS = (
    "numbOfJammers", "numbOfUsers", "state_loss_rate", "ep_length", "seed",
)

CONFIG_FLAGS = SHAPE_FLAGS + RUNTIME_FLAGS


def _build_training_config(args) -> dict:
    """Snapshot the env-construction flags into a plain dict so it
    can be pickled as a PPO model attribute. SB3's `save()` stores
    everything in `model.__dict__` (minus a known excluded list);
    `PPO.load()` then re-applies it via `__dict__.update(data)`, so
    a plain dict round-trips cleanly without subclassing PPO."""
    arch = args.architecture or "MlpPolicy"
    return {
        "numbOfNodes": args.numbOfNodes,
        "numbOfJammers": args.numbOfJammers,
        "numbOfUsers": args.numbOfUsers,
        "seed": args.seed,
        "state_loss_rate": args.state_loss_rate,
        "self_flag": args.self_flag,
        "aoi": args.aoi,
        "history_K": args.history_K,
        "mask": args.mask,
        "ep_length": args.ep_length,
        "architecture": arch,
        "joint_output": args.joint_output,
    }


def _explicit_flags(argv) -> set:
    """Return the set of CLI flag dests (e.g. "history_K") that were
    actually present on the command line. `argparse` fills in defaults
    silently, so this is the only way to distinguish "user said
    --numbOfNodes 4" from "user said nothing". BooleanOptionalAction
    flags register on either `--X` or `--no-X`.
    """
    bool_dests = {"self_flag", "aoi", "mask", "joint_output"}
    long_to_dest = {
        "--timesteps": "timesteps",
        "--numbOfNodes": "numbOfNodes",
        "--numbOfJammers": "numbOfJammers",
        "--numbOfUsers": "numbOfUsers",
        "--seed": "seed",
        "--state_loss_rate": "state_loss_rate",
        "--history_K": "history_K",
        "--ep_length": "ep_length",
        "--n_steps": "n_steps",
        "--architecture": "architecture",
        "--checkpoint_freq": "checkpoint_freq",
        "--tensorboard_log": "tensorboard_log",
        "--resume": "resume",
        "--save": "save",
        "--allow_cpu": "allow_cpu",
    }
    for dest in bool_dests:
        long_to_dest[f"--{dest}"] = dest
        long_to_dest[f"--no-{dest}"] = dest
    seen = set()
    for raw in argv:
        bare = raw.split("=", 1)[0]
        if bare in long_to_dest:
            seen.add(long_to_dest[bare])
    return seen


def _resolve_resumed_config(saved_cfg: dict, args, explicit: set) -> dict:
    """Merge a saved config with CLI overrides for a --resume call.

    - SHAPE_FLAGS: any explicit CLI value MUST match the saved value.
      Mismatch aborts the run with a clear error (the saved model
      cannot consume a differently-shaped observation).
    - RUNTIME_FLAGS: explicit CLI values override the saved value
      and emit a notice so the user sees what changed.

    Returns the resolved config dict that should be used to build
    the env.
    """
    resolved = dict(saved_cfg)
    for flag in SHAPE_FLAGS:
        if flag not in explicit:
            continue
        cli_value = getattr(args, flag)
        # --architecture has default=None on the CLI; resolve to the
        # saved value's shape (architecture class is baked into the
        # zip, so this is informational only).
        if flag == "architecture" and cli_value is None:
            continue
        if cli_value != saved_cfg.get(flag):
            print(
                f"ERROR: --resume requires {flag} to match the saved "
                f"model. Saved={saved_cfg.get(flag)!r}, "
                f"CLI={cli_value!r}. This is a shape-affecting flag — "
                f"the model literally cannot consume a different "
                f"observation. Either drop --{flag} from the command "
                f"line or load a different model.",
                file=sys.stderr,
            )
            sys.exit(1)
    for flag in RUNTIME_FLAGS:
        if flag not in explicit:
            continue
        cli_value = getattr(args, flag)
        if cli_value != saved_cfg.get(flag):
            print(
                f"Overriding {flag}: saved={saved_cfg.get(flag)!r}, "
                f"CLI={cli_value!r}",
            )
            resolved[flag] = cli_value
    return resolved


def build_env(args=None, cfg: dict = None) -> PerDroneSingleStepEnv:
    """Construct a PerDrone-wrapped env. Accepts either an argparse
    `args` object (fresh runs) or an explicit config dict (resume
    path, where the config came from the saved model). Exactly one
    of the two must be supplied."""
    if (args is None) == (cfg is None):
        raise ValueError("build_env: pass exactly one of args= or cfg=.")
    if args is not None:
        cfg = _build_training_config(args)
    base = envs.Static_MANETEnv(
        numbOfNodes=cfg["numbOfNodes"],
        numbOfJammers=cfg["numbOfJammers"],
        numbOfUsers=cfg["numbOfUsers"],
        seed=cfg["seed"],
        networkConnectedAtStart=True,
    )
    base.set_ep_length(cfg["ep_length"])

    base.set_step_jammers_start_moving(100)
    base.set_steps_till_activate_jammers(500)
    rew = rw.LinearReward(base)
    return PerDroneSingleStepEnv(
        rew,
        state_loss_rate=cfg["state_loss_rate"],
        self_flag=cfg["self_flag"],
        aoi=cfg["aoi"],
        history_K=cfg["history_K"],
        mask=cfg["mask"],
        joint_output=cfg.get("joint_output", False),
    )


def _resolve_policy(architecture: str, drone_env: "PerDroneSingleStepEnv"):
    """Map the --architecture flag to an SB3-compatible policy class
    and its policy_kwargs. MlpPolicy needs no kwargs; ST needs the
    drone count and the mask flag so it can split the mask column."""
    if architecture == "MlpPolicy":
        return "MlpPolicy", {}
    if architecture == "ST":
        if drone_env.joint_output:
            print(
                "ERROR: --joint_output is incompatible with --architecture ST. "
                "ST uses a per-drone action head that outputs (2,) — a (N*2,) "
                "output shape does not fit its design. Use --architecture MlpPolicy "
                "with --joint_output.",
                file=sys.stderr,
            )
            sys.exit(1)
        from architectures.ST import ST
        return ST, {
            "num_nodes": drone_env.numbOfNodes,
            "mask": drone_env.mask,
        }
    raise ValueError(f"Unknown --architecture: {architecture}")


def _confirm_cpu_or_exit(architecture: str, allow_cpu: bool) -> None:
    """Refuse to silently start a CPU training run.

    A 24-hour run that should have been on CUDA but landed on CPU
    burns a day and produces nothing useful. So if no CUDA is
    detected, we either abort (ST cannot run on CPU at all) or
    require an explicit `--allow_cpu` flag to acknowledge the
    intent. No interactive prompt — cluster jobs can't answer one.
    """
    import torch
    if torch.cuda.is_available():
        return
    if architecture == "ST":
        print(
            "ERROR: --architecture ST requires CUDA, which is not "
            "available on this machine. --allow_cpu does NOT override "
            "this. Switch to --architecture MlpPolicy or run on a "
            "CUDA-enabled host.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not allow_cpu:
        print(
            "ERROR: CUDA not available. Training on CPU is very slow "
            "(24h+ for a serious PADRE run). If this is intentional, "
            "pass --allow_cpu to confirm.",
            file=sys.stderr,
        )
        sys.exit(1)


def _build_callbacks(args):
    """Build the SB3 callback list. Returns None when no callback
    is requested so SB3's call path is identical to before."""
    if args.checkpoint_freq <= 0:
        return None
    save_basename = os.path.basename(args.save) or "ppo_perdrone"
    ckpt_dir = args.save + "_checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    print(
        f"checkpoint_freq={args.checkpoint_freq} drone-steps -> "
        f"{ckpt_dir}/{save_basename}_*.zip"
    )
    return [
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=ckpt_dir,
            name_prefix=save_basename,
        )
    ]


def main():
    args = parse_args()
    explicit = _explicit_flags(sys.argv[1:])

    if args.resume:
        # Resume path: load the model first so we can read its
        # embedded training_config and reconstruct the env without
        # re-passing every flag.
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume)
        saved_cfg = getattr(model, "training_config", None)

        if saved_cfg is None:
            print(
                "warning: resumed model has no embedded training "
                "config (saved before in-zip config was added). "
                "Falling back to CLI flags purely.",
                file=sys.stderr,
            )
            resolved_cfg = _build_training_config(args)
        else:
            resolved_cfg = _resolve_resumed_config(saved_cfg, args, explicit)

        # CUDA check (and any abort) must happen before _build_callbacks
        # so we don't leave an orphan _checkpoints/ directory behind.
        _confirm_cpu_or_exit(
            resolved_cfg.get("architecture", "MlpPolicy"), args.allow_cpu
        )
        callbacks = _build_callbacks(args)

        drone_env = build_env(cfg=resolved_cfg)

        print(f"obs_space={drone_env.observation_space}")
        print(f"act_space={drone_env.action_space}")
        print(
            f"resolved config: "
            f"numbOfNodes={resolved_cfg['numbOfNodes']}, "
            f"self_flag={resolved_cfg['self_flag']}, "
            f"aoi={resolved_cfg['aoi']}, "
            f"history_K={resolved_cfg['history_K']}, "
            f"mask={resolved_cfg['mask']}, "
            f"architecture={resolved_cfg['architecture']}, "
            f"joint_output={resolved_cfg.get('joint_output', False)}"
        )

        # SB3's set_env runs check_for_correct_spaces as a backstop.
        # Our SHAPE_FLAGS reconciliation should have caught any
        # mismatch earlier with a friendlier message.
        model.set_env(drone_env)
        if args.tensorboard_log is not None:
            model.tensorboard_log = args.tensorboard_log
        # Refresh the embedded config so the next save reflects any
        # runtime-flag overrides that were just applied.
        model.training_config = resolved_cfg
        print(
            f"Resumed; num_timesteps={model.num_timesteps}, "
            f"continuing for {args.timesteps} more drone-steps."
        )
        model.learn(
            total_timesteps=args.timesteps,
            reset_num_timesteps=False,
            callback=callbacks,
        )
    else:
        # Fresh run.
        architecture = args.architecture or "MlpPolicy"
        _confirm_cpu_or_exit(architecture, args.allow_cpu)
        callbacks = _build_callbacks(args)
        drone_env = build_env(args=args)

        print(f"obs_space={drone_env.observation_space}")
        print(f"act_space={drone_env.action_space}")
        print(
            f"numbOfNodes={drone_env.numbOfNodes}, "
            f"self_flag={drone_env.self_flag}, "
            f"aoi={drone_env.aoi}, "
            f"history_K={drone_env.history_K}, "
            f"mask={drone_env.mask}, "
            f"architecture={architecture}, "
            f"joint_output={drone_env.joint_output}"
        )

        policy_cls_or_str, policy_kwargs = _resolve_policy(
            architecture, drone_env
        )
        model = PPO(
            policy_cls_or_str,
            drone_env,
            verbose=1,
            seed=args.seed,
            n_steps=args.n_steps,
            policy_kwargs=policy_kwargs,
            tensorboard_log=args.tensorboard_log,
        )
        # Set BEFORE learn() so the very first checkpoint includes
        # the embedded config.
        model.training_config = _build_training_config(args)
        model.learn(total_timesteps=args.timesteps, callback=callbacks)

    save_dir = os.path.dirname(args.save)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    model.save(args.save)
    print(f"Saved model to {args.save}.zip (num_timesteps={model.num_timesteps})")


if __name__ == "__main__":
    main()
