"""checkpoint_utils.py -- OMNeT checkpoint load + load-verification helper.

Loads OMNeT MARL checkpoints (ST/DeepSets/MLP x N) and verifies the load.

WHY THIS ISN'T evaluation.policies.load_trained_policy: that function's
Hydra-compose fallback (used whenever config.pkl isn't beside the
checkpoint -- true for every checkpoint under eval/diag_checkpoints/) is
PADRE-only: it hardcodes `task=padre/static` and PADRE's per-token
FEATURE_DIM=22. Two things are different for OMNeT:
  1. The task is a separate BenchMARL registration, `omnet/static` ->
     OMNeTTask.STATIC (see benchmarl_padre/omnet_common.py; registered
     lazily by src/run.py since OMNeT needs INET/apptainer to import).
  2. OMNeT's per-drone token width is 26 (omnet_env.DRONE_TOKEN), not
     PADRE's 22, AND drone count N is controlled by the omnet_env.MAX_N
     module constant (patched via evaluate_omnet._patched_num_drones),
     NOT by a task-config key the way PADRE's numbOfNodes override works.
     Composing task=padre/static for an OMNeT checkpoint silently builds
     the wrong environment's spec entirely.

The manual named_parameters() weight-copy (BenchMARL's native
restore_file does not restore ST/DeepSets/GAT) is otherwise identical to
evaluation.policies._load_via_hydra_compose -- same source prefix
"module.0.module.0.module.0.", same target policy.module[0].module[0].
This module re-verifies that copy actually landed: a policy that "loads"
without error but never moved from its random init is indistinguishable,
from the outside, from a policy that trained to a flat return.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_OMNET_MODULE_DIR = str(_HERE.parent / "omnet_module")
_SRC_DIR = str(_HERE.parent / "src")
for _p in (_OMNET_MODULE_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Source prefix BenchMARL's checkpoint stores the actor under; target is
# policy.module[0].module[0] -- same remap evaluation/policies.py uses.
_PREFIX = "module.0.module.0.module.0."


def _ensure_omnet_registry() -> None:
    """Register omnet/static + padre_mappo_norm + custom models, idempotently."""
    from benchmarl.environments import task_config_registry
    from benchmarl.algorithms import algorithm_config_registry
    from benchmarl.models import model_config_registry
    from hydra.core.config_store import ConfigStore

    from benchmarl_padre.omnet_common import OMNeTTask
    from benchmarl_padre.normalized_mappo import NormalizedMappoConfig
    from architectures.st_model import STConfig
    from architectures.deepsets_model import DeepSetsConfig

    if "omnet/static" not in task_config_registry:
        task_config_registry["omnet/static"] = OMNeTTask.STATIC
    if "padre_mappo_norm" not in algorithm_config_registry:
        algorithm_config_registry["padre_mappo_norm"] = NormalizedMappoConfig
        ConfigStore.instance().store(
            name="padre_mappo_norm_config", group="algorithm",
            node=NormalizedMappoConfig,
        )
    if "padre_st" not in model_config_registry:
        model_config_registry["padre_st"] = STConfig
        ConfigStore.instance().store(
            name="padre_st_config", group="model", node=STConfig
        )
    if "padre_deepsets" not in model_config_registry:
        model_config_registry["padre_deepsets"] = DeepSetsConfig
        ConfigStore.instance().store(
            name="padre_deepsets_config", group="model", node=DeepSetsConfig
        )


def _detect_arch(psd: dict) -> str:
    """Detect architecture from checkpoint key patterns (same heuristic as
    evaluation/policies.py's PADRE-side loader)."""
    keys = list(psd.keys())
    # Detect off the BODY encoder, not the learned user-encoder. In learned
    # mode every arch carries a `_user_encoder.phi.*` submodule, so the old
    # `"_encoder.phi." in k` test matched `_user_encoder.phi.` as a substring
    # and misclassified ST/FFN learned checkpoints as deepsets. Exclude the
    # user-encoder keys first, then key off body-specific params:
    #   ST body      -> `_encoder.attention.`   (transformer layers)
    #   DeepSets body-> `_encoder.phi.`/`.rho.`  (set-pooling MLP)
    #   FFN/MLP      -> neither
    body = [k for k in keys if "_user_encoder." not in k]
    if any("_encoder.attention." in k for k in body):
        return "st"
    if any(("_encoder.phi." in k) or ("_encoder.rho." in k) for k in body):
        return "deepsets"
    if any("_encoder" in k for k in body):
        return "st"
    return "mlp"


def _build_model_config(arch: str, cfg_model):
    """Build the model config object for `arch`, reading fields from the
    Hydra-composed cfg.model (model=padre_{arch}) so use_mask/latent_dim/etc
    match what training actually used -- not a hardcoded guess."""
    if arch == "st":
        from architectures.st_model import STConfig
        return STConfig(
            latent_dim=cfg_model.get("latent_dim", 64),
            n_heads=cfg_model.get("n_heads", 4),
            n_layers=cfg_model.get("n_layers", 2),
            dim_feedforward=cfg_model.get("dim_feedforward", 256),
            dropout=cfg_model.get("dropout", 0.1),
            use_mask=cfg_model.get("use_mask", True),
            encoder_mode=cfg_model.get("encoder_mode", "frozen"),
            max_users=cfg_model.get("max_users", 20),
            obs_layout=cfg_model.get("obs_layout", "padre"),
        )
    if arch == "deepsets":
        from architectures.deepsets_model import DeepSetsConfig
        return DeepSetsConfig(
            latent_dim=cfg_model.get("latent_dim", 64),
            phi_hidden=cfg_model.get("phi_hidden", 128),
            rho_hidden=cfg_model.get("rho_hidden", 128),
            dropout=cfg_model.get("dropout", 0.1),
            use_mask=cfg_model.get("use_mask", True),
            encoder_mode=cfg_model.get("encoder_mode", "frozen"),
            max_users=cfg_model.get("max_users", 20),
            obs_layout=cfg_model.get("obs_layout", "padre"),
        )
    if arch in ("mlp", "ffn"):
        # FFN learned-encoder checkpoints detect as "mlp" but need the repo's
        # FFNConfig (which accepts encoder_mode/max_users/obs_layout), NOT
        # BenchMARL's stock MlpConfig (rejects encoder_mode -> TypeError).
        from architectures.ffn_model import FFNConfig
        return FFNConfig(
            num_cells=list(cfg_model.get("num_cells", [256, 256])),
            encoder_mode=cfg_model.get("encoder_mode", "frozen"),
            max_users=cfg_model.get("max_users", 20),
            obs_layout=cfg_model.get("obs_layout", "padre"),
        )
    from benchmarl.hydra_config import load_model_config_from_hydra
    return load_model_config_from_hydra(cfg_model)


def load_and_verify_policy(checkpoint_path: str, device: str, num_drones: int):
    """Load an OMNeT BenchMARL checkpoint (ST/DeepSets/MLP) and hard-verify
    the manual named_parameters() weight copy actually landed.

    Returns (action_fn, task_config). action_fn(obs_dict, n_agents) ->
    action_dict, matching evaluation.policies' contract.

    Raises RuntimeError if the live policy's parameters don't match the raw
    checkpoint tensors -- a silent bad load is worse than no result, so this
    never downgrades to a warning.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from benchmarl.hydra_config import (
        load_algorithm_config_from_hydra,
        load_experiment_config_from_hydra,
        load_task_config_from_hydra,
        load_model_config_from_hydra,
    )
    from benchmarl.experiment import Experiment
    from evaluation.policies import _make_trained_action_fn
    from evaluate_omnet import _patched_num_drones, _close_leaked_omnet_envs

    _ensure_omnet_registry()

    ckpt = Path(checkpoint_path).resolve()
    raw = torch.load(str(ckpt), map_location="cpu")
    psd = raw["collector"]["policy_state_dict"]
    arch = _detect_arch(psd)

    conf_dir = str((Path(_SRC_DIR) / "conf").resolve())

    GlobalHydra.instance().clear()
    tmpdir = tempfile.mkdtemp(prefix="omnet_eval_")
    try:
        with initialize_config_dir(config_dir=conf_dir, version_base="1.2"):
            cfg = compose(
                config_name="config",
                overrides=[
                    "task=omnet/static_learnedenc",
                    "algorithm=padre_mappo_norm",
                    f"model=padre_{arch}",
                    "++model.encoder_mode=learned",
                    "++model.obs_layout=omnet",
                    "++model.max_users=25",
                    "experiment=omnet",
                    "experiment.loggers=[]",
                    "experiment.on_policy_n_envs_per_worker=1",
                    "experiment.evaluation_episodes=1",
                    "experiment.create_json=false",
                    "experiment.checkpoint_at_end=false",
                    "experiment.parallel_collection=false",
                    f"experiment.sampling_device={device}",
                    f"experiment.train_device={device}",
                    f"experiment.buffer_device={device}",
                    f"experiment.save_folder={tmpdir}",
                ],
            )

            algorithm_config = load_algorithm_config_from_hydra(cfg.algorithm)
            experiment_config = load_experiment_config_from_hydra(cfg.experiment)
            task_config_obj = load_task_config_from_hydra(cfg.task, "omnet/static")
            model_config = _build_model_config(arch, cfg.model)
            critic_model_config = load_model_config_from_hydra(cfg.critic_model)

            # omnet_env.MAX_N (module constant) decides N for the env(s) this
            # Experiment() call spawns -- must match the checkpoint's N or the
            # observation/action specs (and therefore the model's built shapes)
            # silently disagree with the trained weights.
            with _patched_num_drones(num_drones):
                experiment = Experiment(
                    task=task_config_obj,
                    algorithm_config=algorithm_config,
                    model_config=model_config,
                    critic_model_config=critic_model_config,
                    seed=cfg.seed,
                    config=experiment_config,
                )
                n_closed = _close_leaked_omnet_envs()
                print(f"[load] closed {n_closed} OMNeT env(s) spawned while "
                      f"building the Experiment for load", flush=True)

            policy = experiment.algorithm.get_policy_for_loss("drone")

            # Copy weights directly into parameters, bypassing tensordict's
            # load_state_dict (breaks on plain tensor dicts) -- BenchMARL's
            # native restore_file does not restore ST/DeepSets/GAT.
            stripped = {
                k[len(_PREFIX):]: v for k, v in psd.items() if k.startswith(_PREFIX)
            }
            if not stripped:
                raise RuntimeError(
                    f"no checkpoint keys start with prefix {_PREFIX!r} in "
                    f"{ckpt} -- prefix drifted or this isn't the expected "
                    f"checkpoint format. Raw keys (first 5): "
                    f"{list(psd.keys())[:5]}"
                )
            target = policy.module[0].module[0]
            with torch.no_grad():
                for name, param in target.named_parameters():
                    if name in stripped:
                        param.copy_(stripped[name])
                    else:
                        raise KeyError(f"Missing key in checkpoint: {name}")
                for name, buf in target.named_buffers():
                    if name in stripped:
                        buf.copy_(stripped[name])

            task_config = dict(experiment.task.config)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        GlobalHydra.instance().clear()

    policy = policy.eval().to(device)

    # ── Load verification: pick a deterministic parameter, compare live vs
    # raw checkpoint. HARD-FAIL on mismatch -- a flat result from a silently
    # bad load is worse than no result at all. ──────────────────────────────
    # Exclude TensorDict metadata entries (__batch_size, __device, ...) --
    # these sort last alphabetically (ASCII '_' > digits) but aren't real
    # tensors (torch.Size / None), so picking one would crash the check
    # itself rather than verify anything.
    _tensor_keys = [k for k in stripped if isinstance(stripped[k], torch.Tensor)]
    if not _tensor_keys:
        raise RuntimeError(
            f"load-verify: no real tensor keys found in stripped checkpoint "
            f"params for {ckpt} -- only metadata keys: {list(stripped.keys())}"
        )
    check_name = sorted(_tensor_keys)[-1]
    expected = stripped[check_name].detach().cpu()
    target = policy.module[0].module[0]
    live_params = dict(target.named_parameters())
    live_buffers = dict(target.named_buffers())
    if check_name in live_params:
        actual = live_params[check_name].detach().cpu()
    elif check_name in live_buffers:
        actual = live_buffers[check_name].detach().cpu()
    else:
        raise RuntimeError(
            f"load-verify: parameter {check_name!r} present in checkpoint "
            f"but absent from the live policy module ({ckpt}). Refusing to "
            f"report a result from an unverifiable load."
        )

    print(f"[load-verify] arch={arch} N={num_drones}  {ckpt}")
    print(f"  checking parameter: {check_name}  shape={tuple(expected.shape)}")
    print(f"  checkpoint (first 5, flat): {expected.flatten()[:5].tolist()}")
    print(f"  live policy (first 5, flat): {actual.flatten()[:5].tolist()}")

    if not torch.allclose(actual, expected, atol=1e-6, rtol=1e-5):
        raise RuntimeError(
            f"LOAD VERIFICATION FAILED for {ckpt}\n"
            f"  parameter: {check_name}\n"
            f"  checkpoint (first 10): {expected.flatten()[:10].tolist()}\n"
            f"  live policy (first 10): {actual.flatten()[:10].tolist()}\n"
            "The weight copy did not take -- this policy is NOT the trained "
            "checkpoint. A flat/no-learning result from here would be a "
            "silent-bad-load artifact, not a real training finding. STOP."
        )
    print(f"  [OK] live policy matches checkpoint for {check_name}\n")

    action_fn = _make_trained_action_fn(policy, device)
    return action_fn, task_config
