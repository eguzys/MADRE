"""Verify NormalizedMappo BenchMARL port — CPU, no Hydra, no GPU assertion.

Checks:
  1. experiment.losses["drone"].normalize_advantage is True (live loss module)
  2. MappoConfig.associated_class() == Mappo (original unchanged)
  3. 5 training iterations with NormalizedMappoConfig complete without exception
"""
import sys
import os
from pathlib import Path

HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from benchmarl_padre import PadreTask
from benchmarl_padre.normalized_mappo import NormalizedMappo, NormalizedMappoConfig
from benchmarl.algorithms import MappoConfig
from benchmarl.algorithms.mappo import Mappo
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

STATIC_YAML = HERE / "conf" / "task" / "padre" / "static.yaml"
NORM_YAML = HERE / "conf" / "algorithm" / "padre_mappo_norm.yaml"


def banner(msg):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def build_experiment(max_n_iters=1):
    task = PadreTask.STATIC.get_from_yaml(str(STATIC_YAML))
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

    return Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )


def check1_live_loss_normalize_advantage():
    banner("CHECK 1: normalize_advantage is True in live loss module")
    experiment = build_experiment(max_n_iters=1)

    groups = list(experiment.losses.keys())
    print(f"  Agent groups: {groups}")
    for group in groups:
        flag = experiment.losses[group].normalize_advantage
        print(f"  losses[{group!r}].normalize_advantage = {flag!r}")
        assert flag is True, (
            f"FAIL: expected True, got {flag!r} for group {group!r}"
        )
    print("[PASS] normalize_advantage is True in all agent-group loss modules")
    return experiment


def check2_original_mappo_unchanged():
    banner("CHECK 2: original MappoConfig is unchanged")
    assert MappoConfig.associated_class() is Mappo, (
        f"FAIL: MappoConfig.associated_class() returned {MappoConfig.associated_class()}, expected Mappo"
    )
    assert NormalizedMappoConfig.associated_class() is NormalizedMappo, (
        f"FAIL: NormalizedMappoConfig.associated_class() returned wrong class"
    )
    print(f"  MappoConfig.associated_class() == Mappo ✓")
    print(f"  NormalizedMappoConfig.associated_class() == NormalizedMappo ✓")
    print("[PASS] Original Mappo/MappoConfig untouched")


def check3_five_training_iterations():
    banner("CHECK 3: 5 training iterations without exception")
    experiment = build_experiment(max_n_iters=5)
    experiment.run()
    print(f"  n_iters_performed = {experiment.n_iters_performed}")
    assert experiment.n_iters_performed >= 5, (
        f"FAIL: expected >=5 iters, got {experiment.n_iters_performed}"
    )
    print("[PASS] 5 training iterations completed without exception")


if __name__ == "__main__":
    print("=== NormalizedMappo Verification ===\n")
    check2_original_mappo_unchanged()
    check1_live_loss_normalize_advantage()
    check3_five_training_iterations()
    print("\n=== All checks passed ===")
