"""Seeded episode loop for PADRE/OMNeT evaluation.

Handles:
  - PADREParallelEnv or OMNeTEnv (both PettingZoo-shaped) episode execution,
    chosen per task_config['_env_family'] -- see _build_env.
  - Per-step logging (reward, throughput, utilization)
  - Optional rendering via PADRE's plot GUI (PADRE only)
  - Returns list of per-step logs (one list per episode)
"""
from __future__ import annotations
import os

from typing import Callable

import numpy as np

from MANET.build import build_padre_env
from MANET.Parallel_enviroments import PADREParallelEnv
from evaluation.metrics import compute_episode_metrics


def run_episodes(
    action_fn: Callable[[dict, int], dict],
    task_config: dict,
    seeds: list[int],
    render: bool = False,
    encoder_meta: dict | None = None,
) -> list[dict]:
    """Run one episode per seed, return list of per-episode metric dicts.

    Args:
        action_fn:     callable(obs_dict, n_agents) -> action_dict
        task_config:   training-time env params from the checkpoint
        seeds:         list of integer seeds, one per episode
        render:        True to open PADRE's plot GUI for the first episode
        encoder_meta:  {'arch', 'input_width'} from
                       policies.inspect_checkpoint_encoder, or None to skip
                       the obs-width preflight (random/greedy/spsa baselines
                       have no checkpoint to check against).

    Returns:
        list of dicts, one per episode, with keys from metrics.REGISTRY
    """
    if encoder_meta is not None:
        _probe_env = _build_env(task_config)
        try:
            assert_obs_width(_probe_env, task_config, encoder_meta)
        finally:
            _probe_env.close()

    per_episode_metrics = []
    for ep_idx, seed in enumerate(seeds):
        is_render = render and ep_idx == 0
        steps = _run_single_episode(action_fn, task_config, seed, is_render)
        metrics = compute_episode_metrics(steps)
        metrics["seed"] = float(seed)
        per_episode_metrics.append(metrics)
        _print_episode_summary(ep_idx, seed, metrics)

    return per_episode_metrics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def assert_obs_width(env, task_config: dict, encoder_meta: dict) -> None:
    """Fail loud, before the episode loop, if the env this task_config builds
    doesn't feed the checkpoint's first layer the width it was trained on.

    Left unchecked this either crashes deep inside torch's matmul (cooked
    static.yaml handed to a raw-obs DeepSets/ST model) or -- worse -- silently
    runs and reports numbers for the wrong env (a degenerate task_config, e.g.
    from an unpersisted config.pkl, falling back to build_padre_env's cooked
    4-node defaults). Both must surface here as a clear error instead.

    `env` is whatever runner._build_env actually built (PADREParallelEnv or
    OMNeTEnv) -- both expose observation_space(agent) and possible_agents,
    which is all this needs, so the check is family-agnostic.
    """
    expected = encoder_meta.get("input_width")
    if expected is None:
        return  # couldn't detect a shape to check against; nothing to assert

    token_width = env.observation_space("drone_0").shape[-1]
    mask_offset = 1 if task_config.get("mask", True) else 0

    arch = encoder_meta.get("arch")
    if arch in ("deepsets", "st"):
        # DeepSets/ST strip the trailing mask column before their phi/
        # input_layer (see architectures/{deepsets,st}_model.py: raw_input_dim
        # = token_features - use_mask) -- they need it separately as the
        # attention/pooling key_padding_mask, not as a feature.
        actual = token_width - mask_offset
    else:
        # FFN has no separate masking mechanism: it flattens the whole
        # per-token vector, mask column included, across all peers. Read the
        # peer count off the env's own possible_agents rather than
        # task_config['numbOfNodes'] -- that key doesn't exist for OMNeT
        # (drone count is the module constant MAX_N), and possible_agents is
        # already sized to the same upper bound for both env families.
        n_nodes = len(env.possible_agents)
        actual = token_width * n_nodes if n_nodes else token_width

    if actual != expected:
        raise RuntimeError(
            f"Observation width mismatch: the env built from this task_config "
            f"feeds the policy input width {actual} (per-agent token={token_width}, "
            f"mask_offset={mask_offset}, arch={arch}, numbOfNodes="
            f"{task_config.get('numbOfNodes')!r}), but the checkpoint's first "
            f"layer expects {expected}. This is almost always a cooked-vs-raw "
            f"task config mismatch (a raw-obs checkpoint needs obs_wrapper="
            f"'raw'/task=padre/raw, not the cooked default) or a task_config "
            f"that didn't persist the real training params (e.g. numbOfNodes "
            f"drift). Fix the task config -- do not evaluate against a "
            f"mismatched env."
        )


def _build_env(task_config: dict, render_mode=None):
    """Build a fresh env from task_config -- OMNeTEnv or PADREParallelEnv.

    OMNeT (task_config['_env_family'] == 'omnet'): obs shape is driven
    entirely by module constants in omnet_module/omnet_env.py (DRONE_TOKEN,
    MAX_N, ...), not by task_config -- see the SEAM comment in
    benchmarl_padre/omnet_common.py's get_env_fun. So this builds a plain
    OMNeTEnv(worker_index=0) directly, bypassing BenchMARL/TorchRL entirely
    (no PettingZooWrapper/TransformedEnv needed for a manual action_fn loop)
    -- the same pattern omnet_module/evaluate_omnet.py already uses to drive
    eval episodes. render_mode is ignored: OMNeTEnv.render() is a no-op
    (has_render() is False) and there is no GUI to open.

    PADRE: builds a fresh PADREParallelEnv, overriding numbOfUsers to
    scalar 10 to match the training distribution and avoid the known
    obs-wrapper off-by-one on the default (2,20) range.

    That scalar override pins the live per-episode user count but, taken on
    its own, also destroys `numbOfUsers`'s upper bound -- the one thing
    encoder_mode=="learned" needs to size its padded user block (must always
    match the training upper bound, not however many users are live this
    episode). So the true upper bound is captured from task_config here,
    before the override, and threaded through separately as `max_users`
    (a no-op for encoder_mode=="frozen", which pools before this distinction
    matters).
    """
    if task_config.get("_env_family") == "omnet":
        from benchmarl_padre.omnet_common import OMNeTEnv
        return OMNeTEnv(worker_index=0)

    cfg = dict(task_config)
    # '_env_family' is bookkeeping added by policies.load_trained_policy, not
    # a build_padre_env parameter -- build_padre_env takes no **kwargs, so a
    # stray key here is a hard TypeError, not a harmless no-op.
    cfg.pop("_env_family", None)

    orig_numbOfUsers = cfg.get("numbOfUsers")
    if isinstance(orig_numbOfUsers, (tuple, list)) and len(orig_numbOfUsers) == 2:
        max_users = int(orig_numbOfUsers[1])
    else:
        max_users = int(orig_numbOfUsers)

    cfg["numbOfUsers"] = int(os.environ.get("PADRE_EVAL_USERS", 15))  # sweep-overridable
    cfg["max_users"] = max_users

    return PADREParallelEnv(
        build_env_fn=lambda: build_padre_env(**cfg, render_mode=render_mode)
    )


def _run_single_episode(
    action_fn: Callable,
    task_config: dict,
    seed: int,
    render: bool,
) -> list[dict]:
    """Execute one episode and return its per-step log."""
    render_mode = "plot" if render else None
    env = _build_env(task_config, render_mode=render_mode)

    try:
        obs_dict, _ = env.reset(seed=seed)
        n_agents = len(env.possible_agents)

        # Heuristics need access to the env's node list after each reset.
        if hasattr(action_fn, "reset"):
            action_fn.reset(env, seed)

        steps = []
        step_idx = 0
        done = False
        # Purpose-assert: an episode must END. ep_length=1000; 5x is
        # unreachable in a healthy run, so tripping this is a real fault,
        # not a slow world. Fail loudly rather than spin to the wall.
        _MAX_STEPS = 5000

        while not done:
            actions = action_fn(obs_dict, n_agents)
            obs_dict, rewards, terms, truncs, _ = env.step(actions)

            reward = float(list(rewards.values())[0])
            terminated = any(terms.values())
            truncated = any(truncs.values())

            entry = {
                "step": step_idx,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            }
            if task_config.get("_env_family") != "omnet":
                # Read network stats from the unwrapped PADRE base env.
                # OMNeTEnv has no equivalent -- there is no single "network"
                # object with a getThroughput()/getMaxThroughput() API, only
                # the raw obs_json fields build_drone_tokens reads (see
                # omnet_env.py). Leaving these keys out (rather than faking
                # a number) is what lets metrics.py report them as N/A
                # instead of a fabricated throughput.
                base_env = env._wrapper.unwrapped_env
                entry["throughput"] = float(base_env.network.getThroughput())
                entry["max_throughput"] = float(base_env.network.getMaxThroughput())
            # Heuristics expose last_committed; log it when present.
            committed = getattr(action_fn, "last_committed", None)
            if committed is not None:
                entry["committed"] = bool(committed)
            steps.append(entry)

            step_idx += 1
            done = terminated or truncated or (len(env.agents) == 0)
            if not done and step_idx >= _MAX_STEPS:
                raise RuntimeError(
                    f'episode seed={seed} exceeded {_MAX_STEPS} steps without '
                    f'terminating (term={terminated} trunc={truncated} '
                    f'agents={len(env.agents)}) -- env never signalled done')

    finally:
        try:
            env.close()
        except Exception:
            pass

    return steps


def _print_episode_summary(ep_idx: int, seed: int, metrics: dict) -> None:
    tp = metrics.get("mean_throughput_mbps", 0.0)
    util = metrics.get("mean_utilization", 0.0)
    rew = metrics.get("episode_reward_total", 0.0)
    ep_len = int(metrics.get("episode_length", 0))
    outcome = "done" if metrics.get("episode_outcome", 0.0) > 0.5 else "trunc"
    print(
        f"  ep {ep_idx:3d} seed={seed:4d}  "
        f"reward={rew:8.3f}  tp={tp:6.3f} Mbps  "
        f"util={util:.3f}  len={ep_len}  [{outcome}]"
    )
