"""Local smoke test for the Phase-3 reward scale (replaces the removed
PopArt value normalizer -- see src/benchmarl_padre/normalized_mappo.py and
src/benchmarl_padre/omnet_common.py's REWARD_SCALE / RewardScaling wiring).

Rationale: ~1000-step OMNeT episodes with per-step
reward ~125 produce GAE value targets O(1e4-1e5). Instead of PopArt's
running-mean/std machinery, a single constant multiply (REWARD_SCALE=1e-4)
applied to the reward at collection time (upstream of GAE) brings those
same value targets down to O(1-10) directly, with no critic-side code at
all. This test asserts that landing zone explicitly and fails loud if the
scale ever stops doing its job (e.g. someone changes REWARD_SCALE without
re-checking the regime it needs to cover).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tensordict import TensorDict
from torchrl.envs.transforms import RewardScaling

from benchmarl_padre.omnet_common import REWARD_SCALE

# Documented regime (normalized_mappo.py's module docstring): per-step
# reward ~125 over ~1000-step episodes -> raw GAE value targets O(1e4-1e5).
_RAW_TARGET_LOW = 1e4
_RAW_TARGET_HIGH = 1e5


def test_reward_scale_is_the_expected_constant():
    assert REWARD_SCALE == 1e-4, (
        f"REWARD_SCALE changed to {REWARD_SCALE} -- re-derive against the "
        f"documented ~1e4-1e5 raw value-target regime before changing this "
        f"test's expectation"
    )
    print(f"[PASS] REWARD_SCALE == {REWARD_SCALE}")


def test_scaled_raw_regime_lands_in_1_to_10():
    """The exact transform get_env_fun wires (nested ('drone','reward') key),
    applied to synthetic returns spanning the documented raw regime."""
    torch.manual_seed(0)
    n = 4096
    # log-uniform over [1e4, 1e5] so both ends of the documented regime are
    # covered, not just the geometric middle.
    log_low, log_high = torch.log10(torch.tensor(_RAW_TARGET_LOW)), torch.log10(torch.tensor(_RAW_TARGET_HIGH))
    raw = 10 ** (torch.rand(n) * (log_high - log_low) + log_low)

    transform = RewardScaling(
        loc=0.0, scale=REWARD_SCALE,
        in_keys=[("drone", "reward")], out_keys=[("drone", "reward")],
    )
    td = TensorDict({("drone", "reward"): raw}, batch_size=[n])
    scaled = transform._call(td).get(("drone", "reward"))

    assert torch.isfinite(scaled).all(), "scaled reward has NaN/Inf"
    lo, hi = scaled.min().item(), scaled.max().item()
    assert lo >= 1.0 and hi <= 10.0, (
        f"scaled targets landed at [{lo:.4g}, {hi:.4g}], expected within "
        f"[1, 10] for the documented raw regime [{_RAW_TARGET_LOW:.0e}, "
        f"{_RAW_TARGET_HIGH:.0e}] -- REWARD_SCALE is not doing its job, "
        "fix it before training on this regime"
    )
    print(f"[PASS] raw [{_RAW_TARGET_LOW:.0e}, {_RAW_TARGET_HIGH:.0e}] "
          f"-> scaled [{lo:.4g}, {hi:.4g}] (target: within [1, 10])")


def test_scale_is_a_pure_multiply_not_an_affine_shift():
    """loc must stay 0 -- this is meant to be dimensional conditioning, not
    a mean-centering trick (that was PopArt's job, deliberately removed)."""
    transform = RewardScaling(
        loc=0.0, scale=REWARD_SCALE,
        in_keys=[("drone", "reward")], out_keys=[("drone", "reward")],
    )
    assert transform.loc.item() == 0.0, "reward scale must be a pure multiply (loc=0)"
    print("[PASS] RewardScaling has loc=0.0 (pure multiply, no shift)")


if __name__ == "__main__":
    print("=== Phase 3: reward scale verification ===\n")
    test_reward_scale_is_the_expected_constant()
    test_scaled_raw_regime_lands_in_1_to_10()
    test_scale_is_a_pure_multiply_not_an_affine_shift()
    print("\n=== All checks passed ===")
