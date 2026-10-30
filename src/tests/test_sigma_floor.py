"""Offline unit tests for the scale_lb sigma floor (NormalizedMappo).

The OMNeT simulator cannot run on this laptop (no INET / no .sif image), so
every test here is a pure-unit test: no simulator, no GPU, no network I/O.
The one test that builds a real Experiment (test_normalize_advantage_and_
scale_lb_applied_on_a_real_policy) uses PadreTask.STATIC, a lightweight
synthetic-users task with no OMNeT dependency, on CPU, for 1 iteration.

Covers:
  1. NormalParamExtractor(scale_lb=0.1) clamps sigma below 0.1 and passes
     sigma above 0.1 through unchanged.
  2. NormalizedMappoConfig accepts scale_lb; NormalizedMappo stores it and
     does not leak the kwarg into Mappo/Algorithm.__init__.
  3. NormalizedMappo.get_policy_for_loss raises RuntimeError when the built
     policy contains zero NormalParamExtractor instances (loud failure,
     not a silent no-op).
  4. normalize_advantage=True is still set on the live loss module (no
     regression of the pre-existing NormalizedMappo behaviour).
  5. Every conf/algorithm/padre_mappo_norm*.yaml parses with plain
     yaml.safe_load and contains a scale_lb key.
"""
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn
import yaml
from tensordict.nn.distributions import NormalParamExtractor

from benchmarl.algorithms.common import Algorithm
from benchmarl.algorithms.mappo import Mappo
from benchmarl_padre.normalized_mappo import (
    DEFAULT_SCALE_LB,
    NormalizedMappo,
    NormalizedMappoConfig,
)

HERE = Path(__file__).parent.parent
CONF_ALGORITHM_DIR = HERE / "conf" / "algorithm"
NORM_YAML = CONF_ALGORITHM_DIR / "padre_mappo_norm.yaml"
STATIC_TASK_YAML = HERE / "conf" / "task" / "padre" / "static.yaml"


# ---------------------------------------------------------------------------
# 1. NormalParamExtractor scale_lb clamping (library-native floor mechanism)
# ---------------------------------------------------------------------------


def test_scale_lb_clamps_low_sigma_to_the_floor():
    extractor = NormalParamExtractor(scale_mapping="biased_softplus_1.0", scale_lb=0.1)
    # biased_softplus_1.0 maps pre-activation -> softplus(x + bias) with
    # bias chosen so that f(0.0) == 1.0. A very negative pre-activation
    # drives the raw softplus output towards 0, i.e. far below the 0.1
    # floor, without the clamp.
    very_negative = torch.full((8,), -50.0)
    loc_input = torch.zeros((8,))
    tensor_in = torch.stack([loc_input, very_negative], dim=-1)  # chunk(2,-1) -> loc, scale-pre
    loc, scale = extractor(tensor_in)
    assert torch.all(scale == 0.1), f"expected scale clamped to exactly 0.1, got {scale}"


def test_scale_lb_passes_through_high_sigma_unchanged():
    extractor_floored = NormalParamExtractor(scale_mapping="biased_softplus_1.0", scale_lb=0.1)
    extractor_default = NormalParamExtractor(scale_mapping="biased_softplus_1.0", scale_lb=1e-4)
    # Zero pre-activation -> biased_softplus_1.0 outputs sigma == 1.0, well
    # above the 0.1 floor, so both extractors must agree exactly.
    tensor_in = torch.zeros((8, 2))
    _, scale_floored = extractor_floored(tensor_in)
    _, scale_default = extractor_default(tensor_in)
    assert torch.allclose(scale_floored, scale_default), (
        "scale_lb=0.1 altered a sigma that was already above the floor -- "
        "the clamp must be a no-op for sigma > scale_lb"
    )
    assert torch.allclose(scale_floored, torch.ones_like(scale_floored)), (
        f"expected sigma==1.0 at zero pre-activation, got {scale_floored}"
    )


# ---------------------------------------------------------------------------
# 2. NormalizedMappoConfig / NormalizedMappo plumbing
# ---------------------------------------------------------------------------


def test_normalized_mappo_config_has_scale_lb_field():
    cfg = NormalizedMappoConfig(
        share_param_critic=True,
        clip_epsilon=0.2,
        entropy_coef=0.03,
        critic_coef=1.0,
        loss_critic_type="l2",
        lmbda=0.9,
        scale_mapping="biased_softplus_1.0",
        use_tanh_normal=True,
        minibatch_advantage=False,
        scale_lb=0.1,
    )
    assert cfg.scale_lb == 0.1
    assert cfg.associated_class() is NormalizedMappo


def test_normalized_mappo_stores_scale_lb_and_does_not_leak_kwarg_to_mappo():
    """NormalizedMappo.__init__ must pop scale_lb before calling
    Mappo.__init__/Algorithm.__init__, so the base classes never see it."""
    received_kwargs = {}
    original_mappo_init = Mappo.__init__

    def spy_init(self, **kwargs):
        received_kwargs.update(kwargs)
        # Skip Algorithm.__init__ (it needs a real `experiment` and specs);
        # we only care that `scale_lb` never reaches this point.
        self.share_param_critic = kwargs.get("share_param_critic")

    with patch.object(Mappo, "__init__", spy_init):
        instance = NormalizedMappo(
            share_param_critic=True,
            clip_epsilon=0.2,
            entropy_coef=0.03,
            critic_coef=1.0,
            loss_critic_type="l2",
            lmbda=0.9,
            scale_mapping="biased_softplus_1.0",
            use_tanh_normal=True,
            minibatch_advantage=False,
            scale_lb=0.1,
        )

    assert instance.scale_lb == 0.1, "NormalizedMappo did not store scale_lb on self"
    assert "scale_lb" not in received_kwargs, (
        f"scale_lb leaked into Mappo.__init__ kwargs: {received_kwargs.keys()}"
    )
    assert original_mappo_init is Mappo.__init__  # patch.object restored cleanly


# ---------------------------------------------------------------------------
# 3. Loud failure when zero NormalParamExtractor instances are found
# ---------------------------------------------------------------------------


def test_get_policy_for_loss_raises_if_no_extractor_found():
    # Build an uninitialized instance (bypassing __init__, which needs a
    # real Experiment/action-spec) and set only what get_policy_for_loss
    # touches: self.scale_lb.
    instance = NormalizedMappo.__new__(NormalizedMappo)
    instance.scale_lb = DEFAULT_SCALE_LB

    dummy_policy = nn.Sequential(nn.Linear(4, 4), nn.ReLU())  # no NormalParamExtractor anywhere

    with patch.object(Algorithm, "get_policy_for_loss", return_value=dummy_policy):
        with pytest.raises(RuntimeError, match="NormalParamExtractor"):
            instance.get_policy_for_loss("drone")


def test_get_policy_for_loss_applies_floor_when_extractor_present():
    instance = NormalizedMappo.__new__(NormalizedMappo)
    instance.scale_lb = 0.37

    extractor = NormalParamExtractor(scale_mapping="biased_softplus_1.0", scale_lb=1e-4)
    dummy_policy = nn.Sequential(extractor)

    with patch.object(Algorithm, "get_policy_for_loss", return_value=dummy_policy):
        returned = instance.get_policy_for_loss("drone")

    assert returned is dummy_policy
    assert extractor.scale_lb == 0.37, "get_policy_for_loss did not overwrite scale_lb"


# ---------------------------------------------------------------------------
# 4 & 5. Real (CPU, no simulator) Experiment: normalize_advantage regression
#         guard + end-to-end scale_lb wiring through the real yaml + policy.
# ---------------------------------------------------------------------------


def _build_cpu_experiment(max_n_iters=1, save_folder=None):
    from benchmarl_padre import PadreTask
    from benchmarl.experiment import Experiment, ExperimentConfig
    from benchmarl.models.mlp import MlpConfig

    task = PadreTask.STATIC.get_from_yaml(str(STATIC_TASK_YAML))
    algorithm_config = NormalizedMappoConfig.get_from_yaml(str(NORM_YAML))
    model_config = MlpConfig.get_from_yaml()
    model_config.num_feature_dims = 2
    critic_model_config = MlpConfig.get_from_yaml()

    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.loggers = []
    experiment_config.create_json = False
    experiment_config.train_device = "cpu"
    experiment_config.sampling_device = "cpu"
    experiment_config.buffer_device = "cpu"
    experiment_config.max_n_iters = max_n_iters
    experiment_config.max_n_frames = None
    # Experiment() creates <save_folder>/<run_name>/ with a config.pkl at
    # construction time. Without an explicit save_folder that lands in the CWD
    # (i.e. src/), so every pytest run left an untracked
    # normalizedmappo_static_mlp__*/ directory behind in the repo. Point it at
    # the test's tmp dir instead.
    if save_folder is not None:
        experiment_config.save_folder = str(save_folder)

    return Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )


def test_normalize_advantage_and_scale_lb_applied_on_a_real_policy(tmp_path):
    experiment = _build_cpu_experiment(max_n_iters=1, save_folder=tmp_path)

    # Regression guard: don't lose the pre-existing normalize_advantage=True
    # behaviour while adding scale_lb.
    groups = list(experiment.losses.keys())
    assert groups, "no agent groups found in experiment"
    for group in groups:
        assert experiment.losses[group].normalize_advantage is True, (
            f"normalize_advantage regressed to False for group {group!r}"
        )

    # End-to-end scale_lb wiring: the real policy built from
    # padre_mappo_norm.yaml (scale_lb: 0.1) must have that value applied to
    # every NormalParamExtractor found in its module tree, and there must be
    # at least one (continuous PADRE action space).
    for group in groups:
        policy = experiment.algorithm.get_policy_for_loss(group)
        extractors = [m for m in policy.modules() if isinstance(m, NormalParamExtractor)]
        assert extractors, f"no NormalParamExtractor found for group {group!r}"
        for e in extractors:
            assert e.scale_lb == 0.1, (
                f"expected scale_lb==0.1 (from padre_mappo_norm.yaml) on "
                f"group {group!r}'s extractor, got {e.scale_lb}"
            )


# ---------------------------------------------------------------------------
# 5. Every padre_mappo_norm*.yaml parses and declares scale_lb
# ---------------------------------------------------------------------------


def test_every_norm_yaml_declares_scale_lb():
    norm_yamls = sorted(CONF_ALGORITHM_DIR.glob("padre_mappo_norm*.yaml"))
    assert norm_yamls, f"no padre_mappo_norm*.yaml files found under {CONF_ALGORITHM_DIR}"
    for path in norm_yamls:
        with open(path) as f:
            parsed = yaml.safe_load(f)
        assert isinstance(parsed, dict), f"{path} did not parse to a dict"
        assert "scale_lb" in parsed, f"{path} is missing the scale_lb field"
        assert isinstance(parsed["scale_lb"], (int, float)), (
            f"{path}: scale_lb is {parsed['scale_lb']!r}, expected a number"
        )


if __name__ == "__main__":
    print("=== Sigma-floor (scale_lb) verification ===\n")
    test_scale_lb_clamps_low_sigma_to_the_floor()
    print("[PASS] scale_lb clamps low sigma to the floor")
    test_scale_lb_passes_through_high_sigma_unchanged()
    print("[PASS] scale_lb passes through high sigma unchanged")
    test_normalized_mappo_config_has_scale_lb_field()
    print("[PASS] NormalizedMappoConfig has scale_lb field")
    test_normalized_mappo_stores_scale_lb_and_does_not_leak_kwarg_to_mappo()
    print("[PASS] NormalizedMappo stores scale_lb, does not leak kwarg to Mappo")
    test_get_policy_for_loss_raises_if_no_extractor_found()
    print("[PASS] get_policy_for_loss raises RuntimeError when 0 extractors found")
    test_get_policy_for_loss_applies_floor_when_extractor_present()
    print("[PASS] get_policy_for_loss applies scale_lb to found extractors")
    test_normalize_advantage_and_scale_lb_applied_on_a_real_policy()
    print("[PASS] real CPU experiment: normalize_advantage True + scale_lb wired end-to-end")
    test_every_norm_yaml_declares_scale_lb()
    print("[PASS] every padre_mappo_norm*.yaml declares scale_lb")
    print("\n=== All checks passed ===")
