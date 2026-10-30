"""Hydra entry point for training PADRE with BenchMARL.

Run with:
    python run.py task=padre/static algorithm=mappo

This mirrors BenchMARL's own run.py, but adds one line that registers the
PADRE task into BenchMARL's registry at import time -- so we never edit the
installed library. Our configs live in ./conf; BenchMARL's bundled configs
are pulled in via the hydra.searchpath block in conf/config.yaml.
"""

import faulthandler
import os
import sys

import hydra
import torch

# Fatal-signal handler ONLY. enable() costs nothing and prints a Python
# traceback on SIGSEGV/SIGABRT, which makes native-level hangs and crashes
# (e.g. in the ZMQ bridge) diagnosable. It stays unconditional.
#
# Do not use faulthandler.dump_traceback_later(): that periodic timer
# inspects every thread on a schedule, which races C-level mutation during
# the MAPPO optimizer update and can segfault a long run without a message.
faulthandler.enable()

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from hydra.core.config_store import ConfigStore
from benchmarl.algorithms import algorithm_config_registry
from benchmarl.environments import task_config_registry
from benchmarl.models import model_config_registry
from benchmarl.hydra_config import load_experiment_from_hydra

from benchmarl_padre import PadreTask
from benchmarl_padre.normalized_mappo import NormalizedMappoConfig
from benchmarl_padre.omnet_checkpoint import resume_experiment
from benchmarl_padre.policy_stats_callback import PolicyStatsCallback
from architectures.st_model import STConfig
from architectures.gat_model import GATConfig
from architectures.deepsets_model import DeepSetsConfig
from architectures.ffn_model import FFNConfig

# Register custom task and algorithm into BenchMARL's phone books.
task_config_registry["padre/static"] = PadreTask.STATIC
task_config_registry["padre/static_n5"] = PadreTask.STATIC
task_config_registry["padre/static_n3"] = PadreTask.STATIC
task_config_registry["padre/static_n2"] = PadreTask.STATIC
task_config_registry["padre/static_n4"] = PadreTask.STATIC
task_config_registry["padre/raw"] = PadreTask.STATIC
task_config_registry["padre/raw_learnenc"] = PadreTask.STATIC
# OMNeT is registered LAZILY. omnet_common imports omnet_env, which raises at
# module level if INET is absent -- so an unconditional import here makes a
# PADRE-only machine (the student cluster has no OMNeT install) unable to run
# run.py AT ALL, for a task it was never going to use. Registration failure is
# only fatal if you actually asked for an omnet task, and Hydra says exactly
# that ("Could not find 'omnet/static'") if you do.
try:
    from benchmarl_padre.omnet_common import OMNeTTask
    task_config_registry["omnet/static"] = OMNeTTask.STATIC
    # Same TaskClass, different task yaml -- the frozen-vs-learned user-set
    # encoder is selected by conf/task/omnet/static_learnedenc.yaml's
    # encoder_mode field, not by a distinct enum member. Mirrors the existing
    # padre/static_n5 / _n3 / _n2 / _n4 precedent above, which likewise maps
    # several yamls onto one enum.
    task_config_registry["omnet/static_learnedenc"] = OMNeTTask.STATIC
except Exception as _omnet_exc:  # noqa: BLE001
    _OMNET_IMPORT_ERROR = _omnet_exc
algorithm_config_registry["padre_mappo_norm"] = NormalizedMappoConfig
ConfigStore.instance().store(
    name="padre_mappo_norm_config", group="algorithm", node=NormalizedMappoConfig
)

# Register custom model (Set Transformer) so Hydra can compose `model=padre_st`.
model_config_registry["padre_st"] = STConfig
model_config_registry["padre_gat"] = GATConfig
model_config_registry["padre_deepsets"] = DeepSetsConfig
model_config_registry["padre_ffn"] = FFNConfig
ConfigStore.instance().store(
    name="padre_st_config", group="model", node=STConfig
)
ConfigStore.instance().store(
    name="padre_gat_config", group="model", node=GATConfig
)
ConfigStore.instance().store(
    name="padre_deepsets_config", group="model", node=DeepSetsConfig
)
# FFNConfig has a Type[nn.Module] field (activation_class), same as stock
# BenchMARL's own MlpConfig -- OmegaConf's structured-config validation
# (which ConfigStore.store triggers) rejects that annotation, so FFNConfig
# is registered in model_config_registry only, same as MlpConfig.


@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_experiment(cfg: DictConfig) -> None:
    task_name = HydraConfig.get().runtime.choices.task
    algorithm_name = HydraConfig.get().runtime.choices.algorithm

    print(f"\nAlgorithm: {algorithm_name}, Task: {task_name}")
    print("\nLoaded config:\n")
    print(OmegaConf.to_yaml(cfg))

    # Refuse CPU training — a silent CPU fallback wastes hours.
    # Devices come from conf/experiment/padre.yaml; this only VERIFIES them.
    #
    # $OMNET_ALLOW_CPU=1 is a DELIBERATE opt-out, not a workaround. On the OM2G
    # archipelago track the measured split per PPO iteration is collection
    # 2902 s vs training 7.8 s: 99.7% of wall-clock is OMNeT + the reward LP,
    # both CPU-bound, so the GPU runs at a 0.2% duty cycle while environments
    # starve for cores. There, cpu.normal with 20 cores is ~2.5-3x FASTER than
    # a GPU node with 8. Setting this asserts "I measured the split".
    _allow_cpu = os.environ.get("OMNET_ALLOW_CPU", "") == "1"
    if _allow_cpu:
        print("=== OMNET_ALLOW_CPU=1: CPU training ENABLED DELIBERATELY "
              "(collection-bound workload; GPU duty cycle was 0.2%) ===",
              flush=True)
    else:
        assert torch.cuda.is_available(), (
            "CUDA not available — refusing CPU training. "
            "Set OMNET_ALLOW_CPU=1 only if you have MEASURED that the run is "
            "collection-bound."
        )
        for _dev in ("sampling_device", "train_device", "buffer_device"):
            assert str(cfg.experiment[_dev]).lower() == "cuda", (
                f"experiment.{_dev}={cfg.experiment[_dev]} is not cuda — refusing CPU. "
                f"Set it to cuda in conf/experiment/padre.yaml."
            )

    # ── ENCODER-MODE COHERENCE GATE (omnet track) ────────────────────────────
    #
    # encoder_mode exists in TWO places and they mean different halves of one
    # decision:
    #   task.encoder_mode  -> what the ENV emits per peer token. "frozen" puts
    #                         an 8-dim pooled user latent at cols 5:13
    #                         (DRONE_TOKEN=28); "learned" puts the raw padded
    #                         user block there instead (345 cols total).
    #   model.encoder_mode -> whether the MODEL owns a trainable set encoder to
    #                         pool that raw block back down.
    #
    # Mixing them is silent and plausible, which is the dangerous part: with
    # task=learned + model=frozen every shape still lines up (widths are read
    # off the env spec), so training runs to completion -- but the raw user
    # block is consumed as ordinary flat features with NO set encoder and no
    # permutation invariance over users, which is not the experiment anyone
    # meant to run. Same failure shape as the entropy_coef drift the launcher
    # guards against. Fail here instead.
    _task_enc = OmegaConf.select(cfg, "task.encoder_mode", default=None)
    if _task_enc is not None:
        _model_enc = OmegaConf.select(cfg, "model.encoder_mode", default=None)
        if _model_enc is None:
            raise SystemExit(
                f"FATAL: task.encoder_mode={_task_enc!r} but model={cfg.model.get('name', '?')!r} "
                "has no encoder_mode field at all (stock BenchMARL Mlp cannot own a "
                "trainable user-set encoder). Use model=padre_st / padre_deepsets / "
                "padre_ffn for the omnet track."
            )
        if str(_model_enc) != str(_task_enc):
            raise SystemExit(
                f"FATAL: encoder_mode mismatch -- task.encoder_mode={_task_enc!r} "
                f"but model.encoder_mode={_model_enc!r}. The env would emit one token "
                "layout and the model would interpret a different one. Both must be "
                f"{_task_enc!r} (pass model.encoder_mode={_task_enc})."
            )
        if str(_task_enc) == "learned":
            # The learned path slices the raw block by (obs_layout, max_users);
            # the PADRE defaults (own_state=4, token=6, max_users=20) would
            # mis-slice an omnet token silently.
            _layout = OmegaConf.select(cfg, "model.obs_layout", default=None)
            _mu = OmegaConf.select(cfg, "model.max_users", default=None)
            # The required value is whatever omnet_env.MAX_USERS_CAP resolves to,
            # NOT a literal: $OMNET_MAX_USERS_CAP moves both sides together (the
            # OM2G track runs 40). Reading the same env var the env reads keeps
            # this a real lockstep check.
            _cap = int(os.environ.get("OMNET_MAX_USERS_CAP", "25"))
            # OM2G: the env emits the flat, arm-sized layout, and the model
            # MUST parse it as 'om2g' -- the 'omnet' row geometry against a
            # flat vector would slice the wrong columns. The two tracks REQUIRE
            # different layouts, checked both ways so neither can run with the
            # other's.
            _want_layout = "om2g" if os.environ.get("OM2G_TRACK") == "1" else "omnet"
            if str(_layout) != _want_layout or int(_mu) != _cap:
                raise SystemExit(
                    f"FATAL: this learned-encoder run needs model.obs_layout="
                    f"{_want_layout} (OM2G_TRACK="
                    f"{os.environ.get('OM2G_TRACK')!r}) and "
                    f"model.max_users={_cap} (omnet_env.MAX_USERS_CAP, from "
                    f"$OMNET_MAX_USERS_CAP), got obs_layout="
                    f"{_layout!r}, max_users={_mu!r}. A layout mismatch would "
                    "slice the observation at the wrong offsets and train on "
                    "garbage."
                )

    # Resume-from-checkpoint: `+resume_from=<experiment>/checkpoints/checkpoint_<N>.pt`.
    # Bypasses load_experiment_from_hydra entirely -- the resumed task/model/
    # algorithm config comes from that experiment's own config.pkl, not from
    # this invocation's cfg. cfg.task/model/algorithm/experiment.* are only
    # used to satisfy Hydra's mandatory defaults (task=???, algorithm=???)
    # and are otherwise IGNORED once resuming, by design: max_n_frames in
    # particular must stay the ORIGINAL total, not whatever this invocation's
    # CLI happens to pass. See benchmarl_padre/omnet_checkpoint.py.
    resume_from = OmegaConf.select(cfg, "resume_from", default=None)
    resume_max_n_frames = OmegaConf.select(cfg, "resume_max_n_frames", default=None)
    # +resume_lr=<float>: config.pkl's lr is otherwise what the resumed
    # optimizer runs at, whatever experiment.lr says (2026-09-10).
    resume_lr = OmegaConf.select(cfg, "resume_lr", default=None)
    if resume_from is not None:
        experiment = resume_experiment(
            resume_from, max_n_frames=resume_max_n_frames,
            lr=(float(resume_lr) if resume_lr is not None else None),
        )
    else:
        # PolicyStatsCallback logs policy/<group>/scale_mean|scale_min|
        # abs_loc_mean each collection, so a policy whose sigma stays pinned at
        # its initialisation value of 1.0 is visible in the logs. The callback
        # is fully guarded and can never raise into the training loop.
        experiment = load_experiment_from_hydra(
            cfg, task_name=task_name, callbacks=[PolicyStatsCallback()]
        )
    experiment.run()


if __name__ == "__main__":
    hydra_experiment()
