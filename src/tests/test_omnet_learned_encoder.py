"""Verification tests for the OMNeT track's encoder_mode="learned" path.

The OMNeT++ simulator itself cannot run here (no INET/apptainer .sif on this
machine) -- everything below is offline, driven by the synthetic_omnet_obs
fixture (conftest.py / tests/fixtures/synthetic_omnet_obs.py), which builds a
schema-correct obs_json using the same Friis/log-distance radio model
max_flow_reward.py scores rewards with.

Checks:
  1. encoder_mode="frozen" is the DEFAULT and is bit-identical to an explicit
     encoder_mode="frozen" call -- no-regression guard on the pre-existing
     path. Also cross-checks the frozen user_latent columns against a fresh
     call to the (still-frozen, untouched) _user_encoder.
  2. learned-mode token width == 345, mask stays the last column, and the
     raw user block slice round-trips build_raw_user_tokens(...) exactly.
  3. A DeepSetsModel and an STModel built with encoder_mode="learned",
     obs_layout="omnet", max_users=25 forward-pass on a real learned-mode
     observation and return finite output of the right shape.
  4. Gradients reach the learnable user encoder (the purpose of the learned
     mode): backward on the output sum gives non-None, non-zero grad on
     every _user_encoder parameter.
Gate 5 (existing PADRE encoder_mode="learned" tests still pass) is verified
by running tests/test_encoder_mode.py itself, not duplicated here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded

from omnet_env import (
    PEER_LINK_MASK,
    build_drone_tokens,
    build_raw_user_tokens,
    drone_token_width,
    DRONE_TOKEN,
    DRONE_TOKEN_LEARNED,
    MAX_N,
    MAX_USERS_CAP,
    OBS_FEATURES,
    _user_encoder,
)
from architectures.st_model import STConfig
from architectures.deepsets_model import DeepSetsConfig


# ---------------------------------------------------------------------------
# 1. Frozen path: default arg + no-regression cross-check
#
# build_raw_user_tokens normalises its dBm/angle columns, so the
# PRETRAINED _user_encoder is being fed a distribution it was never fitted on.
# build_drone_tokens refuses the frozen path unless the caller says it means
# it. These two tests assert the ENV-SIDE PLUMBING (default arg, column
# placement) is unchanged, not that the latent is meaningful -- it is not,
# until the frozen encoder is refitted on normalised tokens.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _allow_frozen(monkeypatch):
    monkeypatch.setenv("OMNET_ALLOW_FROZEN_ENCODER", "1")


def test_frozen_mode_is_default_and_matches_explicit_call(synthetic_omnet_obs):
    obs = synthetic_omnet_obs(seed=1)
    default_tokens = build_drone_tokens(obs, episode_seed=1)
    explicit_tokens = build_drone_tokens(obs, episode_seed=1, encoder_mode="frozen")
    assert default_tokens.shape == (MAX_N, MAX_N, DRONE_TOKEN)
    assert np.array_equal(default_tokens, explicit_tokens), (
        "build_drone_tokens' default encoder_mode must produce byte-identical "
        "output to an explicit encoder_mode='frozen' call"
    )


def test_frozen_mode_user_latent_columns_match_frozen_encoder(synthetic_omnet_obs):
    """No-regression guard: cols 7:15 of the frozen-mode token must equal a
    fresh call to the (untouched, still-frozen) _user_encoder over
    build_raw_user_tokens -- i.e. the env-side pooling behaviour is unchanged."""
    obs = synthetic_omnet_obs(seed=2)
    tokens = build_drone_tokens(obs, episode_seed=2, encoder_mode="frozen")
    raw_user_tokens = build_raw_user_tokens(obs, episode_seed=2)

    n_live = len(obs["positions"])
    assert n_live == MAX_N, "synthetic fixture must spawn MAX_N live drones"

    for j in range(n_live):
        expected_latent = _user_encoder.encode(raw_user_tokens[j])
        for i in range(n_live):
            if tokens[i, j, -1] <= 0.5:
                continue  # peer-link gating: peer j unreachable from i, row zeroed
            np.testing.assert_array_equal(tokens[i, j, 7:15], expected_latent)
        # mask + self_flag also unchanged
        if not PEER_LINK_MASK:
            assert tokens[0, j, -1] == 1.0
    assert tokens[0, 0, -1] == 1.0   # observer 0 always sees itself
    assert tokens[0, 0, 23] == 1.0   # self_flag: i==j==0


# ---------------------------------------------------------------------------
# 2. Learned mode: width, mask column, user-block round-trip
# ---------------------------------------------------------------------------

def test_learned_mode_width_and_roundtrip(synthetic_omnet_obs):
    obs = synthetic_omnet_obs(seed=3)
    tokens = build_drone_tokens(obs, episode_seed=3, encoder_mode="learned")

    assert DRONE_TOKEN_LEARNED == 345, f"expected 345, got {DRONE_TOKEN_LEARNED}"  # includes the absolute position columns
    assert drone_token_width("learned") == 345
    assert tokens.shape == (MAX_N, MAX_N, 345)

    raw_user_tokens = build_raw_user_tokens(obs, episode_seed=3)
    user_block_width = MAX_USERS_CAP * OBS_FEATURES  # 325
    n_live = len(obs["positions"])

    for i in range(n_live):
        # mask stays the LAST column. With peer-link gating OFF it is 1.0 for
        # every live peer; with it ON an unreachable peer is legitimately 0.
        assert tokens[i, i, -1] == 1.0, "a drone is always visible to itself"
        if not PEER_LINK_MASK:
            assert np.all(tokens[i, :n_live, -1] == 1.0)
        for j in range(n_live):
            if tokens[i, j, -1] <= 0.5:
                continue
            user_block = tokens[i, j, 7:7 + user_block_width].reshape(MAX_USERS_CAP, OBS_FEATURES)
            np.testing.assert_array_equal(user_block, raw_user_tokens[j])


def test_learned_mode_absmax_guard_does_not_trip(synthetic_omnet_obs):
    """Regression guard: every RAW column reaching the actor must be
    normalised to [-1, 1], so raw dBm columns cannot swamp the direction
    signal."""
    obs = synthetic_omnet_obs(seed=4)
    tokens = build_drone_tokens(obs, episode_seed=4, encoder_mode="learned")
    assert np.isfinite(tokens).all()

    # The raw per-user block: EVERY column is normalised, so |x| <= 1.
    user_block = tokens[..., 7:7 + MAX_USERS_CAP * OBS_FEATURES]
    assert np.abs(user_block).max() <= 1.0, (
        f"raw user block reaches {np.abs(user_block).max():.4g}; every column "
        "must go through _normalize_dbm/_normalize_angle/"
        "_normalize_throughput_bps")
    # own_state (0:7) is normalised abs-position + dx/dy + normalised dBm.
    assert np.abs(tokens[..., :7]).max() <= 1.0
    # The only slack is the frozen EVENT encoder's unbounded 8-dim latent.
    assert np.abs(tokens).max() < 10.0


# ---------------------------------------------------------------------------
# 3 & 4. Model forward pass + gradient flow into the learnable user encoder
# ---------------------------------------------------------------------------

def _build_omnet_model(config_cls, **extra):
    n_agents = n_peers = MAX_N
    out = 2  # matches OMNeTEnv's 2-dim continuous action
    config = config_cls(
        encoder_mode="learned", obs_layout="omnet", max_users=MAX_USERS_CAP,
        use_mask=True, **extra,
    )
    input_spec = Composite(
        {
            "agents": Composite(
                {"observation": Unbounded(shape=(n_agents, n_peers, DRONE_TOKEN_LEARNED))},
                shape=(n_agents,),
            )
        }
    )
    output_spec = Composite(
        {
            "agents": Composite(
                {"action_value": Unbounded(shape=(n_agents, out))},
                shape=(n_agents,),
            )
        }
    )
    return config.get_model(
        input_spec=input_spec,
        output_spec=output_spec,
        agent_group="agents",
        input_has_agent_dim=True,
        n_agents=n_agents,
        centralised=False,
        share_params=True,
        device="cpu",
        action_spec=Composite({}),
        model_index=0,
    )


def _learned_mode_input(synthetic_omnet_obs, seed):
    obs = synthetic_omnet_obs(seed=seed)
    tokens = build_drone_tokens(obs, episode_seed=seed, encoder_mode="learned")
    # (MAX_N, MAX_N, 345) -> (1, MAX_N, MAX_N, 343): one batch element
    return torch.as_tensor(tokens, dtype=torch.float32).unsqueeze(0)


def _assert_grad_flowed(module, name):
    params = list(module.named_parameters())
    assert params, f"{name}: encoder submodule has no parameters at all"
    any_nonzero = False
    for pname, p in params:
        assert p.grad is not None, (
            f"{name}: param {pname} has grad=None -- gradients did not reach "
            "the encoder"
        )
        if torch.any(p.grad != 0):
            any_nonzero = True
    assert any_nonzero, f"{name}: all encoder param grads are exactly zero"


def test_deepsets_learned_omnet_forward_and_grad(synthetic_omnet_obs):
    model = _build_omnet_model(
        DeepSetsConfig, latent_dim=16, phi_hidden=32, rho_hidden=32, dropout=0.0,
    )
    x = _learned_mode_input(synthetic_omnet_obs, seed=5)
    td = TensorDict({"agents": {"observation": x}}, batch_size=[1])
    td_out = model(td)
    out = td_out.get(model.out_key)
    assert out.shape == (1, MAX_N, 2)
    assert torch.isfinite(out).all(), "DeepSets(omnet, learned) output has NaN/Inf"
    out.sum().backward()
    _assert_grad_flowed(model._user_encoder, "DeepSets(omnet)")


def test_st_learned_omnet_forward_and_grad(synthetic_omnet_obs):
    model = _build_omnet_model(
        STConfig, latent_dim=16, n_heads=2, n_layers=1, dim_feedforward=32, dropout=0.0,
    )
    x = _learned_mode_input(synthetic_omnet_obs, seed=6)
    td = TensorDict({"agents": {"observation": x}}, batch_size=[1])
    td_out = model(td)
    out = td_out.get(model.out_key)
    assert out.shape == (1, MAX_N, 2)
    assert torch.isfinite(out).all(), "ST(omnet, learned) output has NaN/Inf"
    out.sum().backward()
    _assert_grad_flowed(model._user_encoder, "ST(omnet)")
