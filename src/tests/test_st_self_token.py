"""Self-token path — the actor must have an un-pooled path to its OWN token.

The per-peer token carries a `self_flag`, but pooling the peer set with a PMA
whose single CONSTANT learned seed attends uniformly (measured 0.1946 on the
own token vs 0.2014 on the others, against 1/5 = 0.2000) behaves like a mean
pool. Under a mean pool

    pool_i = (1/N)[f(t_i, flag=1) - f(t_i, flag=0)] + (1/N) sum_j f(t_j, flag=0)

so identity reached the action only as a 1/N-amplitude perturbation on top of a
term identical for every agent. Measured without the un-pooled path: moving
the self bit between tokens changed the action by 6.7% of its temporal
variation, and all five drones acted as one (cosine 0.956-0.996 between mean
action directions).

These tests pin the property, not the mechanism: WHO the observer is must
materially change what it does.
"""
from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from architectures.st_model import STConfig
from architectures.user_encoder import OBS_LAYOUTS
from omnet_env import MAX_N, build_drone_tokens

# synthetic_omnet_obs is a conftest.py fixture — injected, not imported.
from test_omnet_learned_encoder import _build_omnet_model, _learned_mode_input

# self_flag_col is in REASSEMBLED coordinates (after the 325-wide raw user
# block is replaced by an 8-wide pooled summary). Tests that poke the model's
# RAW input must shift by that difference — 23 -> 340.
SELF_COL = OBS_LAYOUTS["omnet"]["self_flag_col"]
RAW_SELF_COL = SELF_COL + (25 * 13 - 8)

_KW = dict(latent_dim=16, n_heads=2, n_layers=1, dim_feedforward=32, dropout=0.0)


def _model(**extra):
    torch.manual_seed(0)
    return _build_omnet_model(STConfig, **_KW, **extra)


def _act(model, x):
    td = TensorDict({"agents": {"observation": x}}, batch_size=[x.shape[0]])
    with torch.no_grad():
        return model(td).get(model.out_key)


# ---------------------------------------------------------------------------
# 1. The layout constant is in lockstep with the env
# ---------------------------------------------------------------------------

def test_self_flag_col_matches_the_env(synthetic_omnet_obs):
    """OBS_LAYOUTS' self_flag_col must point at the column build_drone_tokens
    actually sets, in REASSEMBLED coordinates. The reassembly replaces the
    325-wide raw user block with an 8-wide summary, so the raw index shifts by
    (325 - 8); this checks the two agree rather than trusting the arithmetic."""
    obs = synthetic_omnet_obs(seed=3)
    tok = build_drone_tokens(obs, episode_seed=3, encoder_mode="learned")
    for i in range(MAX_N):
        hot = tok[i, :, RAW_SELF_COL]
        assert hot[i] == pytest.approx(1.0), (
            f"observer {i}'s own row is not flagged at raw col {RAW_SELF_COL}")
        assert hot.sum() == pytest.approx(1.0), (
            f"self flag is not one-hot for observer {i}: {hot}")


# ---------------------------------------------------------------------------
# 2. The identity actually reaches the action
# ---------------------------------------------------------------------------

def test_moving_the_self_flag_changes_the_action(synthetic_omnet_obs):
    """Hold the token SET fixed and move only which token is flagged 'me'.

    Without the un-pooled path this probe returned 6.7%. With the
    un-pooled path the output must move by a size comparable to the action's
    own spread, not a rounding error.
    """
    model = _model()
    x = _learned_mode_input(synthetic_omnet_obs, seed=6)
    base = x[:, 0:1]                      # observer 0's token set, (1,1,P,W)

    outs = []
    for slot in range(MAX_N):
        v = base.clone()
        v[..., RAW_SELF_COL] = 0.0
        v[..., slot, RAW_SELF_COL] = 1.0
        outs.append(_act(model, v.expand(-1, MAX_N, -1, -1).contiguous())[0, 0])
    outs = torch.stack(outs)              # (MAX_N, 2)

    spread = outs.std(dim=0).mean()
    scale = outs.abs().mean()
    assert spread > 0.05 * scale, (
        f"moving the self flag barely moved the action (std {spread:.5f} vs "
        f"|action| {scale:.5f}) — the observer's identity is not reaching the "
        f"action head; the self-token path has regressed")


def test_agents_do_not_all_emit_the_same_action(synthetic_omnet_obs):
    """The rigid-body signature: with a shared-parameter actor, different
    observers must still produce different actions."""
    model = _model()
    x = _learned_mode_input(synthetic_omnet_obs, seed=6)
    out = _act(model, x)[0]               # (MAX_N, 2)
    across_agents = out.std(dim=0).mean()
    assert across_agents > 1e-3 * out.abs().mean(), (
        f"all {MAX_N} agents emitted essentially the same action "
        f"(std {across_agents:.6f}) — the swarm is a rigid body")


# ---------------------------------------------------------------------------
# 3. What the self-token path must NOT break
# ---------------------------------------------------------------------------

def test_still_permutation_invariant_over_peers(synthetic_omnet_obs):
    """Gathering by argmax(self_flag) follows the flag under a permutation, so
    reordering the peer axis must leave the action unchanged."""
    model = _model()
    x = _learned_mode_input(synthetic_omnet_obs, seed=6)
    perm = torch.tensor([2, 0, 4, 1, 3])
    out_a = _act(model, x)
    out_b = _act(model, x[:, :, perm])
    torch.testing.assert_close(out_a, out_b, rtol=1e-4, atol=1e-5)


def test_ablation_switch_restores_the_old_architecture(synthetic_omnet_obs):
    off = _model(use_self_token=False)
    assert off._own_proj is None
    assert off._output_proj.in_features == off.latent_dim
    on = _model()
    assert on._own_proj is not None
    assert on._output_proj.in_features == 2 * on.latent_dim
    out = _act(off, _learned_mode_input(synthetic_omnet_obs, seed=6))
    assert torch.isfinite(out).all()


def test_padre_layout_leaves_the_path_disabled():
    """The padre track has no self column; the path must be a no-op there."""
    assert OBS_LAYOUTS["padre"]["self_flag_col"] is None


def test_gradients_reach_the_own_token_projection(synthetic_omnet_obs):
    model = _model()
    x = _learned_mode_input(synthetic_omnet_obs, seed=6)
    td = TensorDict({"agents": {"observation": x}}, batch_size=[1])
    model(td).get(model.out_key).sum().backward()
    g = model._own_proj.weight.grad
    assert g is not None and torch.any(g != 0), (
        "no gradient reached _own_proj — the un-pooled path is detached")
