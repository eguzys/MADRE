"""Policy loaders for PADRE evaluation.

Supported sources:
  trained  -- BenchMARL/MAPPO checkpoint, loaded via Experiment.reload_from_file
               when config.pkl exists (proper experiment folder), otherwise via
               Hydra-compose fallback using src/conf/ (standalone .pt files).
  random   -- uniform action-space sampling baseline.
  greedy   -- probe-greedy heuristic (GreedyHeuristic from heuristics.py).
  spsa     -- SPSA gradient-ascent heuristic (SPSAHeuristic from heuristics.py).

Heuristics implement the same action_fn(obs_dict, n_agents) -> action_dict
interface plus a reset(env, seed) method called by runner.py at episode start.
"""
from __future__ import annotations

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

# Per-token observation feature width for the COOKED obs path (mask=True,
# history_K=1). Only used to infer numbOfNodes for FFN checkpoints, whose
# first layer flattens numbOfNodes peer tokens into one vector -- FFN has no
# phi/input_layer to read a per-token width from directly (see
# inspect_checkpoint_encoder). Raw obs's per-token width isn't a fixed
# constant (it depends on encoder_mode/history_K/mask), so it's read
# dynamically off the checkpoint instead -- never hardcoded here.
FEATURE_DIM = 22


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_trained_policy(checkpoint_path: str, device: str,
                        stochastic: bool = False,
                        sigma_sink: Optional[list] = None) -> tuple:
    """Load a trained PADRE or OMNeT policy from a BenchMARL checkpoint.

    stochastic=True samples actions at the policy's learned sigma instead of
    executing the distribution mean -- see _make_trained_action_fn.

    Returns (action_fn, task_config) where:
      action_fn(obs_dict, n_agents) -> action_dict  (single-step, deterministic
                   unless stochastic=True)
      task_config: dict of training-time env params (numbOfUsers, ep_length, …),
                   plus task_config['_env_family'] in {'padre', 'omnet'} so
                   downstream code (evaluate.py's gates, runner.py's env
                   builder) knows which env family actually trained this
                   checkpoint -- see runner._build_env.

    Strategy:
      1. If config.pkl exists two levels up (proper experiment folder),
         use Experiment.reload_from_file — the canonical BenchMARL path.
         config.pkl pickles the actual TaskClass instance (OMNeTClass or
         PadreClass), so this works for BOTH families unmodified; which one
         it built is read back off `experiment.task`.
      2. Otherwise use Hydra compose from src/conf/ as a fallback for
         standalone .pt files (e.g. checkpoints copied out of experiment
         dirs). PADRE-only: OMNeT's obs shape is module-constant-driven
         (omnet_env.py), not config-driven, so there is nothing in a bare
         .pt + src/conf/task/padre/*.yaml that could reconstruct it.
    """
    ckpt = Path(checkpoint_path).resolve()
    config_pkl = ckpt.parent.parent / "config.pkl"

    if config_pkl.exists():
        policy, task_config, env_family = _load_via_reload_from_file(ckpt, device)
    else:
        policy, task_config, env_family = _load_via_hydra_compose(ckpt, device)

    if env_family == "omnet":
        # Building the Experiment above spawns real OMNeT++ subprocesses
        # (test_env + collector env(s), each ZMQ-bound) as a side effect --
        # they're never exposed to this caller. runner.py builds its own
        # fresh OMNeTEnv for the actual episode loop, so these must be
        # closed now or they leak (same cleanup omnet_module/evaluate_omnet.py
        # already does before building its own eval env).
        _close_leaked_omnet_envs()

    task_config = dict(task_config)
    task_config["_env_family"] = env_family

    policy = policy.eval().to(device)
    action_fn = _make_trained_action_fn(policy, device, stochastic=stochastic,
                                        sigma_sink=sigma_sink)
    return action_fn, task_config


def _close_leaked_omnet_envs() -> None:
    """Close every OMNeTEnv still tracked in omnet_env._live_envs.

    Every OMNeTEnv instance self-registers in that module-level WeakSet on
    construction (omnet_module/omnet_env.py's process-leak-prevention
    layer) -- walking it is the only way to reach and close envs that
    reload_from_file built but never returned. Called both after a
    successful OMNeT load and from the reload_from_file except-clause below
    (Experiment.__init__ builds envs in _setup() before the load that can
    fail) -- in the latter case the checkpoint might turn out to be PADRE's,
    which never imports omnet_env at all, so bail out instead of forcing
    the import (and the omnet_module/INET setup it requires) unconditionally.
    """
    if "omnet_env" not in sys.modules:
        return
    import omnet_env
    for env in list(omnet_env._live_envs):
        try:
            env.close()
        except Exception:
            try:
                env._kill_proc()
            except Exception:
                pass


def _sample_random_action(rng):
    """One uninformed action. $OM2G_RANDOM_DIST selects the shape.

    "uniform" (default): U(-1,1) per axis, per-axis std 1/sqrt(3) = 0.577.
    "gauss":             N(0, $OM2G_RANDOM_SIGMA) per axis, default sigma
                         0.577 so the DIFFUSION RATE is unchanged.

    Read that default carefully. A random walk's displacement is set by the
    per-step standard deviation, not by the shape of the distribution, so
    swapping uniform for a Gaussian of the SAME std leaves the baseline's
    score statistically unchanged -- it is a cosmetic change. Raising sigma
    makes random diffuse FASTER (and the C++ maxSpeed clip renormalises
    over-length draws onto the unit circle, which pushes the mean step up
    further, so it is stronger still). Lowering sigma makes random weaker by
    collapsing it toward hover, which is already measured separately. Set
    sigma deliberately and report it.
    """
    import os as _os
    import numpy as _np
    if _os.environ.get("OM2G_RANDOM_DIST", "uniform").lower() != "gauss":
        return rng.uniform(-1.0, 1.0, 2)
    sigma = float(_os.environ.get("OM2G_RANDOM_SIGMA", "0.5773502691896258"))
    return _np.clip(rng.normal(0.0, sigma, 2), -1.0, 1.0)


def make_random_action_fn(n_agents: int, seed: Optional[int] = None) -> Callable:
    """Return an action function that samples uniformly from [-1, 1]^2 per agent.

    SEEDING: the action stream is derived from the episode seed via
    .reset(seed) below (runner/evaluate_omnet call it at episode start), so
    two runs with matched seeds see the same episode AND the same action
    stream. The paired-seed protocol needs every policy under comparison to
    be reproducible.

    This baseline is a *random walk* because OMNeTEnv.step() scales the unit
    action by STEP_M. See make_hover_action_fn for the trivial-control
    (no movement) baseline.
    """
    state = {"rng": np.random.default_rng(seed)}

    def action_fn(obs_dict: dict, n_agents_: int = n_agents) -> dict:
        return {
            f"drone_{i}": _sample_random_action(state["rng"]).astype(np.float32)
            for i in range(n_agents_)
        }

    def reset(env=None, seed: Optional[int] = None) -> None:
        """Re-seed the action stream for a new episode. Signature matches
        GreedyHeuristic.reset(env, seed) so callers can invoke it uniformly."""
        state["rng"] = np.random.default_rng(seed)

    action_fn.reset = reset
    return action_fn


def make_hover_action_fn(n_agents: int) -> Callable:
    """Return an action function that always commands zero velocity.

    The trivial-control baseline: the drones hold the placement that
    spawn_staging gave them for the whole episode. Added 2026-08-07 because it
    turned out to be the baseline every previous OMNeT number was implicitly
    being compared against without anyone measuring it -- spawn_staging stages
    the swarm on the largest user cluster, so standing still is already a good
    placement, and a genuine random walk at the real step size scores WORSE
    than it (152.9 vs 247.4 reward/step measured on cfg_99). Any claim of the
    form "the policy learned to position the swarm" has to clear hover, not
    just random.
    """
    zero = np.zeros(2, dtype=np.float32)

    def action_fn(obs_dict: dict, n_agents_: int = n_agents) -> dict:
        return {f"drone_{i}": zero.copy() for i in range(n_agents_)}

    def reset(env=None, seed: Optional[int] = None) -> None:
        return None

    action_fn.reset = reset
    return action_fn


def make_greedy_action_fn(condition_name: str = "sinr", threshold=None):
    """Return a GreedyHeuristic instance (callable + reset protocol).

    The returned object is used directly as action_fn by runner.py:
      action_fn(obs_dict, n_agents) -> action_dict
      action_fn.reset(env, seed)         -- called at episode start
      action_fn.last_committed           -- bool, read by metrics
    """
    from evaluation.heuristics import GreedyHeuristic
    return GreedyHeuristic(condition_name=condition_name, threshold=threshold)


def make_spsa_action_fn(
    condition_name: str = "sinr",
    c: float = 1.0,
    a: float = 1.0,
    threshold=None,
):
    """Return an SPSAHeuristic instance (callable + reset protocol).

    See make_greedy_action_fn for the interface contract.
    c: perturbation size in action units (default 1.0)
    a: gradient ascent gain (default 1.0)
    """
    from evaluation.heuristics import SPSAHeuristic
    return SPSAHeuristic(condition_name=condition_name, c=c, a=a, threshold=threshold)


def inspect_checkpoint_encoder(checkpoint_path: str) -> dict:
    """Detect the encoder architecture and expected input width straight off
    a checkpoint's policy_state_dict, independent of which load path
    (reload_from_file or hydra-compose) is used to build the model.

    Used by evaluate.py as a fail-loud preflight -- see runner.assert_obs_width
    -- to catch a live env whose per-agent observation doesn't match what the
    checkpoint's first layer expects (e.g. cooked-vs-raw task config, or a
    config.pkl that didn't persist the real env params) BEFORE it reaches
    torch's matmul as a bare shape-mismatch crash.

    Returns {'arch': 'deepsets'|'st'|'other', 'input_width': int|None} where
    input_width is:
      deepsets/st -- per-token width, EXCLUDING the trailing mask column
                     (matches phi.0.weight / input_layer.weight dim 1).
      other (FFN) -- flat first-layer width (numbOfNodes * per-token width).
      None        -- state dict didn't match any known shape (can't check).
    """
    psd = torch.load(str(checkpoint_path), map_location="cpu")["collector"]["policy_state_dict"]

    # "_user_encoder." ends with "_encoder.", so a plain substring test would
    # match the learned user encoder's own phi (e.g. `_user_encoder.phi.0.weight`)
    # and report an ST checkpoint as deepsets. Exclude the user encoder before
    # detecting the body architecture.
    body = [k for k in psd if "_user_encoder." not in k]
    has_learned_user_encoder = len(body) != len(psd)
    is_deepsets = any("_encoder.phi." in k for k in body)
    is_st = (not is_deepsets) and any("_encoder.input_layer." in k for k in body)

    if has_learned_user_encoder:
        # A learned encoder breaks the env-width -> first-layer-width identity
        # that runner.assert_obs_width checks: the per-agent observation is
        # decomposed into per-user tokens, pooled by the user encoder, and
        # REASSEMBLED into a narrower per-peer token before it ever reaches the
        # body's first layer (architectures/user_encoder.py:
        # reassemble_with_learned_encoder). 149-wide obs -> 20x6 user tokens ->
        # 8-dim summary -> 16-wide peer token is not a width the preflight can
        # derive from observation_space alone, so return None ("can't check")
        # rather than a number that is guaranteed to mismatch.
        #
        # A mismatched env still fails loudly a moment later, because the
        # reassembly reshape is itself strict (a shape error at that line).
        arch = "deepsets" if is_deepsets else ("st" if is_st else "other")
        return {"arch": arch, "input_width": None, "learned_encoder": True}

    if is_deepsets:
        w = next((v for k, v in psd.items() if k.endswith("_encoder.phi.0.weight")), None)
        return {"arch": "deepsets", "input_width": int(w.shape[1]) if w is not None else None}
    if is_st:
        w = next((v for k, v in psd.items() if k.endswith("_encoder.input_layer.weight")), None)
        return {"arch": "st", "input_width": int(w.shape[1]) if w is not None else None}

    w = psd.get("module.0.module.0.module.0.mlp.params.0.weight")
    return {"arch": "other", "input_width": int(w.shape[1]) if w is not None else None}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sigma_from_td(td_out):
    """Best-effort mean policy sigma out of a forward pass, or None.

    [2026-08-13] Realised sigma is NOT logged anywhere -- not in the BenchMARL
    metrics json, not in the offline wandb summary. Every sigma number in the
    thesis so far is the INIT (`scale_mapping=biased_softplus_{b}`), and sigma
    DRIFTS during training (0.3 -> 0.61 measured), so arms are currently ordered
    on the x-axis by a value none of them actually had. Plotting the curve
    against init would be wrong. This reads the real thing off the policy.

    The key name depends on the torchrl version and how the distribution params
    were wired, so search rather than assume; return None if nothing matches
    and let the caller say "unavailable" instead of inventing a number.
    """
    import torch as _t
    for container in (td_out.get("drone", None), td_out):
        if container is None:
            continue
        try:
            keys = list(container.keys())
        except Exception:  # noqa: BLE001
            continue
        for k in keys:
            if not isinstance(k, str):
                continue
            if k.lower() in ("scale", "sigma", "loc_scale", "action_scale"):
                try:
                    v = container[k]
                    if isinstance(v, _t.Tensor) and v.numel():
                        return float(v.detach().abs().mean().cpu())
                except Exception:  # noqa: BLE001
                    pass
    return None


def _make_trained_action_fn(policy, device: str, stochastic: bool = False,
                            sigma_sink: Optional[list] = None) -> Callable:
    # DETERMINISTIC executes the distribution mean; RANDOM samples at the
    # policy's own learned sigma. The default eval is deterministic, but the
    # policy is TRAINED under sigma~=0.95 on a +-1 action space, so the two
    # differ a lot -- `stochastic=True` scores the policy in the world it was
    # actually optimised for. See omnetFix8 section 10 (train/eval sigma mismatch).
    expl = ExplorationType.RANDOM if stochastic else ExplorationType.DETERMINISTIC

    def action_fn(obs_dict: dict, n_agents: int) -> dict:
        obs = np.stack([obs_dict[f"drone_{i}"] for i in range(n_agents)], axis=0)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        td = TensorDict(
            {"drone": TensorDict({"observation": obs_t}, batch_size=[1])},
            batch_size=[1],
        )
        with torch.no_grad(), set_exploration_type(expl):
            td_out = policy(td)
        if sigma_sink is not None:
            s = _sigma_from_td(td_out)
            if s is not None:
                sigma_sink.append(s)
        actions = td_out["drone"]["action"].squeeze(0).cpu().numpy()
        return {f"drone_{i}": actions[i] for i in range(n_agents)}

    return action_fn


def _load_via_reload_from_file(ckpt: Path, device: str) -> tuple:
    """Use Experiment.reload_from_file — requires config.pkl at ckpt/../../config.pkl.

    Returns (policy, task_config, env_family) where env_family is
    'omnet'|'padre' -- see load_trained_policy's docstring.
    """
    # Ensure custom task and algo are importable before unpickling config.pkl
    _ensure_benchmarl_registry()

    from benchmarl.experiment import Experiment

    tmpdir = tempfile.mkdtemp(prefix="padre_eval_")
    try:
        try:
            experiment = Experiment.reload_from_file(
                str(ckpt),
                experiment_patch={
                    "loggers": [],
                    "restore_map_location": device,
                    "sampling_device": device,
                    "train_device": device,
                    "buffer_device": device,
                    "on_policy_n_envs_per_worker": 1,
                    "evaluation_episodes": 1,
                    "save_folder": tmpdir,
                    "create_json": False,
                    "checkpoint_at_end": False,
                    # OMNeT envs spawn real subprocesses (ZMQ + apptainer/bare
                    # spawn); parallel_collection=True would fork/pickle them
                    # across worker processes. Force serial collection so the
                    # only processes reload_from_file spawns are the ones
                    # omnet_env.py's _live_envs tracks and can be closed.
                    "parallel_collection": False,
                },
            )
        except (RuntimeError, ValueError, KeyError) as e:
            # Experiment.__init__ builds the env/model from whatever module
            # constants (omnet_env.py) / task yaml (PADRE) are installed
            # RIGHT NOW, THEN force-loads this checkpoint's old tensors into
            # it (BenchMARL's own collector.load_state_dict, called before
            # reload_from_file ever returns control here) -- any drift since
            # training is a hard shape-mismatch crash deep in
            # tensordict._td._set_str, reported as a bare "size of tensor a
            # (25) must match ... (17)" plus a multi-hundred-line TensorDict
            # dump, with no indication of WHERE the mismatch came from.
            # Surface it as what it actually is instead: this checkpoint
            # predates a constants change and cannot be reloaded as-is (a
            # checkpoint from an earlier constants version).
            _close_leaked_omnet_envs()  # _setup() runs before the load that just failed
            raise RuntimeError(
                f"reload_from_file failed to load checkpoint weights for "
                f"{ckpt}: {type(e).__name__}: {e}\n"
                f"This is almost always the checkpoint's training-time obs "
                f"width no longer matching the CURRENTLY-INSTALLED env "
                f"constants it's being reloaded against (omnet_env.py module "
                f"constants for OMNeT checkpoints, src/conf/task/padre/*.yaml "
                f"for PADRE ones) -- reload_from_file rebuilds the env/model "
                f"from what's on disk NOW, then force-loads this checkpoint's "
                f"old tensors into it. Not a bug in this checkpoint; it "
                f"predates a constants change and cannot be reloaded as-is."
            ) from e
        policy = experiment.algorithm.get_policy_for_loss("drone")
        task_config = dict(experiment.task.config)
        # experiment.task is the actual TaskClass instance pickled into
        # config.pkl (Experiment._save_experiment pickles self.task, not a
        # name) -- its concrete type tells OMNeT-trained checkpoints from
        # PADRE ones directly, with no reliance on task_config carrying any
        # env-identifying field (it never does for OMNeT; see module docstring).
        _task_cls = type(experiment.task)
        env_family = (
            "omnet"
            if _task_cls.__name__ == "OMNeTClass"
            and _task_cls.__module__ == "benchmarl_padre.omnet_common"
            else "padre"
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return policy, task_config, env_family



def _load_model_config(cfg_model):
    """Load model config, handling custom models BenchMARL doesn't know about."""
    from benchmarl.hydra_config import load_model_config_from_hydra
    name = cfg_model.get("name", "") if hasattr(cfg_model, "get") else str(cfg_model)
    if name == "padre_st":
        from architectures.st_model import STConfig
        return STConfig(
            latent_dim=cfg_model.get("latent_dim", 64),
            n_heads=cfg_model.get("n_heads", 4),
            n_layers=cfg_model.get("n_layers", 2),
            dim_feedforward=cfg_model.get("dim_feedforward", 256),
            dropout=cfg_model.get("dropout", 0.1),
            use_mask=cfg_model.get("use_mask", False),
        )
    return load_model_config_from_hydra(cfg_model)

def _load_via_hydra_compose(ckpt: Path, device: str) -> tuple:
    """Build experiment from src/conf/ using Hydra compose (standalone checkpoint)."""
    _ensure_benchmarl_registry()

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from benchmarl.hydra_config import (
        load_algorithm_config_from_hydra,
        load_experiment_config_from_hydra,
        load_model_config_from_hydra,
        load_task_config_from_hydra,
    )
    from benchmarl.experiment import Experiment

    src_dir = Path(__file__).parent.parent
    conf_dir = str((src_dir / "conf").resolve())

    GlobalHydra.instance().clear()
    tmpdir = tempfile.mkdtemp(prefix="padre_eval_")

    # Pick the task config that matches the checkpoint's observation width.
    # raw-obs models train with obs_wrapper=raw; cooked static.yaml (mask=True)
    # produces a 22-dim per-drone token, well below _RAW_W. Read the width off
    # the encoder input layer so we compose the SAME obs family the model was
    # trained on -- a fixed padre/static config would give a raw model a
    # cooked env and a phi matmul shape error. The exact raw width still isn't
    # guaranteed to match this composed env (encoder_mode/history_K/mask can
    # differ per run and aren't recoverable from a bare .pt); the shape check
    # right before the weight-copy loop below catches that residual case.
    _wsd = torch.load(str(ckpt), map_location="cpu")["collector"]["policy_state_dict"]
    _phi_w = next((v for k, v in _wsd.items()
                   if k.endswith("_encoder.phi.0.weight") or k.endswith("_encoder.input_proj.weight")), None)
    _obs_w = int(_phi_w.shape[1]) if (_phi_w is not None and _phi_w.ndim == 2) else None
    _RAW_W = 25
    _task = "padre/raw" if _obs_w == _RAW_W else "padre/static"
    print(f"[compose] checkpoint obs width={_obs_w} -> task={_task}")
    try:
        with initialize_config_dir(config_dir=conf_dir, version_base="1.2"):
            cfg = compose(
                config_name="config",
                overrides=[
                    f"task={_task}",
                    "algorithm=padre_mappo_norm",
                    "experiment.loggers=[]",
                    "experiment.on_policy_n_envs_per_worker=1",
                    "experiment.evaluation_episodes=1",
                    "experiment.create_json=false",
                    "experiment.checkpoint_at_end=false",
                    f"experiment.sampling_device={device}",
                    f"experiment.train_device={device}",
                    f"experiment.buffer_device={device}",
                    f"experiment.save_folder={tmpdir}",
                ],
            )

            algorithm_config = load_algorithm_config_from_hydra(cfg.algorithm)
            experiment_config = load_experiment_config_from_hydra(cfg.experiment)
            task_config_obj = load_task_config_from_hydra(cfg.task, "padre/static")
            # Detect model type from checkpoint weights, instantiate directly
            _psd_keys = list(torch.load(str(ckpt), map_location="cpu")["collector"]["policy_state_dict"].keys())
            _is_deepsets = any("_encoder.phi." in k for k in _psd_keys)
            _is_st = (not _is_deepsets) and any("_encoder" in k for k in _psd_keys)
            if _is_deepsets:
                from architectures.deepsets_model import DeepSetsConfig
                model_config = DeepSetsConfig(latent_dim=64, phi_hidden=128,
                                              rho_hidden=128, dropout=0.1,
                                              use_mask=True)
            elif _is_st:
                from architectures.st_model import STConfig
                model_config = STConfig(latent_dim=64, n_heads=4, n_layers=2,
                                        dim_feedforward=256, dropout=0.1, use_mask=False)
            else:
                model_config = _load_model_config(cfg.model)
            # Override numbOfNodes from the checkpoint so the model is built at
            # the checkpoint's true N regardless of static.yaml. Only FFN encodes
            # N in its layer width; ST/DeepSets are N-agnostic. Kills the race.
            if not _is_deepsets and not _is_st:
                _ffn_w = torch.load(str(ckpt), map_location="cpu")["collector"]["policy_state_dict"].get("module.0.module.0.module.0.mlp.params.0.weight")
                if _ffn_w is not None and _ffn_w.ndim == 2:
                    if _ffn_w.shape[1] % FEATURE_DIM == 0:
                        task_config_obj.config["numbOfNodes"] = _ffn_w.shape[1] // FEATURE_DIM
                    else:
                        raise ValueError(
                            f"FFN first-layer width {_ffn_w.shape[1]} not divisible by "
                            f"cooked FEATURE_DIM={FEATURE_DIM}. FFN's per-token width isn't "
                            f"detected from the checkpoint the way DeepSets/ST's is (no "
                            f"phi/input_layer to read), so this path assumes cooked "
                            f"(task=padre/static) obs for FFN -- an FFN checkpoint trained "
                            f"with obs_wrapper=raw will hit this. Eval it from a full run "
                            f"dir via config.pkl instead, where the real task_config "
                            f"(including obs_wrapper) is known."
                        )

            critic_model_config = load_model_config_from_hydra(cfg.critic_model)

            # Build fresh experiment (no restore_file) — BenchMARL's
            # collector.load_state_dict breaks on non-MLP tensordict params.
            # Load policy weights manually instead.
            _loaded = torch.load(str(ckpt), map_location=device)
            _psd = _loaded["collector"]["policy_state_dict"]

            experiment = Experiment(
                task=task_config_obj,
                algorithm_config=algorithm_config,
                model_config=model_config,
                critic_model_config=critic_model_config,
                seed=cfg.seed,
                config=experiment_config,
            )
            policy = experiment.algorithm.get_policy_for_loss("drone")
            # Copy weights directly into parameters, bypassing tensordict's
            # load_state_dict which breaks on plain tensor dicts.
            _prefix = "module.0.module.0.module.0."
            _st_sd = {k[len(_prefix):]: v for k, v in _psd.items() if k.startswith(_prefix)}
            _target = policy.module[0].module[0]

            # Fail loud, BEFORE the copy loop, if the model this task config
            # builds doesn't match the checkpoint's own shapes anywhere --
            # otherwise the first offending param.copy_ crashes with a bare
            # tensor-size RuntimeError deep in torch, with no hint that the
            # root cause is an env/task config that doesn't reproduce this
            # checkpoint's exact training-time obs (encoder_mode/history_K/
            # mask/etc, which src/conf/task/padre/*.yaml defaults may not
            # match for a standalone checkpoint with no config.pkl).
            _shape_mismatches = [
                (name, tuple(param.shape), tuple(_st_sd[name].shape))
                for name, param in _target.named_parameters()
                if name in _st_sd and tuple(_st_sd[name].shape) != tuple(param.shape)
            ]
            if _shape_mismatches:
                _n, _built, _ckpt = _shape_mismatches[0]
                raise ValueError(
                    f"Shape mismatch loading checkpoint weights into the model "
                    f"built from task={_task}: {len(_shape_mismatches)} "
                    f"parameter(s) differ, e.g. {_n!r}: model built {_built} vs "
                    f"checkpoint {_ckpt}. src/conf/task/padre/"
                    f"{_task.rsplit('/', 1)[-1]}.yaml's defaults don't "
                    f"reproduce this checkpoint's training-time obs config. "
                    f"Eval it from a full run dir via config.pkl instead, "
                    f"where the real params are known."
                )

            with torch.no_grad():
                for name, param in _target.named_parameters():
                    if name in _st_sd:
                        param.copy_(_st_sd[name])
                    else:
                        raise KeyError(f"Missing key in checkpoint: {name}")
                for name, buf in _target.named_buffers():
                    if name in _st_sd:
                        buf.copy_(_st_sd[name])

        task_config = dict(experiment.task.config)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        GlobalHydra.instance().clear()

    return policy, task_config, "padre"


def _ensure_benchmarl_registry():
    """Register PADRE task and NormalizedMappo into BenchMARL's registries.

    Safe to call multiple times — guard checks prevent duplicate registration.
    """
    from benchmarl.environments import task_config_registry
    from benchmarl.algorithms import algorithm_config_registry
    from hydra.core.config_store import ConfigStore

    from benchmarl_padre import PadreTask
    from benchmarl_padre.normalized_mappo import NormalizedMappoConfig

    if "padre/static" not in task_config_registry:
        task_config_registry["padre/static"] = PadreTask.STATIC
    if "padre_mappo_norm" not in algorithm_config_registry:
        algorithm_config_registry["padre_mappo_norm"] = NormalizedMappoConfig
        ConfigStore.instance().store(
            name="padre_mappo_norm_config",
            group="algorithm",
            node=NormalizedMappoConfig,
        )
