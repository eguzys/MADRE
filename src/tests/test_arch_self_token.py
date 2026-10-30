"""Self-token path for the NON-ST architectures — deepsets and ffn must also
have an un-pooled path to their OWN token.

Sibling to `test_st_self_token.py`, which pins the same property for ST.
Measured on 1.002M-frame runs WITHOUT the bypass, evaluated at n=20 on the
OMNeT track:

    deepsets  -0.70   |  ffn  -0.53   |  ST (with bypass)  +0.68
                         of the hover->greedy gap; hover = 0, random = -0.26.

Both were far below hover — i.e. worse than not moving at all — with the same
signature: start at the spawn value, decay to ~0 by step 60 (16/20 and 12/20
seeds ended below half their step-1 reward). The observation is the same for
all three architectures (token_width=345, self_flag at col 23 reassembled), and
neither network learned to weight the flag: measured |W[:,23]| / mean|W[:,other]|
in the first layer that ingests the token was 1.06 (deepsets) and 0.97 (ffn) —
and 0.92 for ST, which is LOWER. So the flag's weight is not what separates
them; the ROUTING is.

These tests pin the property, not the mechanism: WHO the observer is must
materially change what it does.
"""
from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded

from architectures.deepsets_model import DeepSetsConfig
from architectures.ffn_model import FFNConfig
from architectures.user_encoder import OBS_LAYOUTS
from omnet_env import MAX_N, MAX_USERS_CAP, DRONE_TOKEN_LEARNED, build_drone_tokens

# synthetic_omnet_obs is a conftest.py fixture — injected, not imported.
from test_omnet_learned_encoder import _learned_mode_input

SELF_COL = OBS_LAYOUTS["omnet"]["self_flag_col"]
# self_flag_col is in REASSEMBLED coordinates (after the 325-wide raw user
# block is replaced by an 8-wide pooled summary). Probes that poke the model's
# RAW input must shift by that difference — 23 -> 340.
RAW_SELF_COL = SELF_COL + (MAX_USERS_CAP * 13 - 8)


def _build(config_cls, **extra):
    """Local builder: test_omnet_learned_encoder._build_omnet_model always
    passes use_mask=True, which FFNConfig does not accept (it never strips a
    mask column — the peer axis is flattened whole)."""
    torch.manual_seed(0)
    n_agents = n_peers = MAX_N
    config = config_cls(
        encoder_mode="learned", obs_layout="omnet", max_users=MAX_USERS_CAP,
        **extra,
    )
    input_spec = Composite(
        {"agents": Composite(
            {"observation": Unbounded(shape=(n_agents, n_peers, DRONE_TOKEN_LEARNED))},
            shape=(n_agents,))}
    )
    output_spec = Composite(
        {"agents": Composite({"action_value": Unbounded(shape=(n_agents, 2))},
                             shape=(n_agents,))}
    )
    return config.get_model(
        input_spec=input_spec, output_spec=output_spec, agent_group="agents",
        input_has_agent_dim=True, n_agents=n_agents, centralised=False,
        share_params=True, device="cpu", action_spec=Composite({}), model_index=0,
    )


def _ds(**extra):
    return _build(DeepSetsConfig, latent_dim=16, phi_hidden=32, rho_hidden=32,
                  dropout=0.0, use_mask=True, **extra)


def _ffn(**extra):
    return _build(FFNConfig, num_cells=[32, 32], **extra)


ARCHS = [("deepsets", _ds), ("ffn", _ffn)]


def _act(model, x):
    td = TensorDict({"agents": {"observation": x}}, batch_size=[x.shape[0]])
    with torch.no_grad():
        return model(td).get(model.out_key)


# ---------------------------------------------------------------------------
# 1. The identity actually reaches the action
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,build", ARCHS)
def test_moving_the_self_flag_changes_the_action(name, build, synthetic_omnet_obs):
    """Hold the token SET fixed and move only which token is flagged 'me'.

    Without the un-pooled path this probe returned 6.7%. With the un-pooled
    path the output must move by a size comparable to the action's own spread,
    not a rounding error."""
    model = build()
    x = _learned_mode_input(synthetic_omnet_obs, seed=6)
    base = x[:, :1]                       # (1, 1, MAX_N, W) — one token set
    outs = []
    for i in range(MAX_N):
        v = base.clone()
        v[..., RAW_SELF_COL] = 0.0
        v[..., i, RAW_SELF_COL] = 1.0
        outs.append(_act(model, v.expand(-1, MAX_N, -1, -1).contiguous())[0, 0])
    outs = torch.stack(outs)              # (MAX_N, 2)

    spread = outs.std(dim=0).mean()
    scale = outs.abs().mean()
    assert spread > 0.05 * scale, (
        f"{name}: moving the self flag barely moved the action (std "
        f"{spread:.5f} vs |action| {scale:.5f}) — the observer's identity is "
        f"not reaching the action head; the self-token path has regressed")


@pytest.mark.parametrize("name,build", ARCHS)
def test_agents_do_not_all_emit_the_same_action(name, build, synthetic_omnet_obs):
    """The rigid-body signature: with a shared-parameter actor, different
    observers must still produce different actions."""
    model = build()
    out = _act(model, _learned_mode_input(synthetic_omnet_obs, seed=6))[0]
    across = out.std(dim=0).mean()
    assert across > 1e-3 * out.abs().mean(), (
        f"{name}: all {MAX_N} agents emitted essentially the same action "
        f"(std {across:.6f}) — the swarm is a rigid body")


# ---------------------------------------------------------------------------
# 2. The ablation switch still reproduces the old architecture
# ---------------------------------------------------------------------------

def test_deepsets_ablation_switch(synthetic_omnet_obs):
    off = _ds(use_self_token=False)
    assert off._own_proj is None
    assert off._output_proj.in_features == off.latent_dim
    on = _ds()
    assert on._own_proj is not None
    assert on._output_proj.in_features == 2 * on.latent_dim
    assert torch.isfinite(_act(off, _learned_mode_input(synthetic_omnet_obs, seed=6))).all()


def test_ffn_ablation_switch(synthetic_omnet_obs):
    off = _ffn(use_self_token=False)
    on = _ffn()
    # The FFN has no _output_proj — the widening shows up on the MLP's input.
    assert on.input_features == off.input_features + DRONE_TOKEN_LEARNED - (MAX_USERS_CAP * 13 - 8)
    assert torch.isfinite(_act(off, _learned_mode_input(synthetic_omnet_obs, seed=6))).all()


# ---------------------------------------------------------------------------
# 3. What the self-token path must NOT break
# ---------------------------------------------------------------------------

def test_deepsets_still_permutation_invariant_over_peers(synthetic_omnet_obs):
    """Gathering by argmax(self_flag) follows the flag under a permutation, so
    reordering the peer axis must leave the action unchanged. Deepsets only —
    the FFN flattens the peer axis and is permutation-SENSITIVE by design,
    which is the whole point of the ST/deepsets/ffn comparison."""
    model = _ds()
    x = _learned_mode_input(synthetic_omnet_obs, seed=6)
    perm = torch.tensor([2, 0, 4, 1, 3])
    torch.testing.assert_close(_act(model, x), _act(model, x[:, :, perm]),
                               rtol=1e-4, atol=1e-5)
