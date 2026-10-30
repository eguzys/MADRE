"""Verification tests for encoder_mode="learned" (gates B and C from the
raw-obs learnable-encoder task) — CPU only, no Experiment.

Gate A (frozen path byte-identical) is verified separately against real
checkpoints + an env-level observation dump (see conversation record / PR
description), not here — this file covers the parts that only exist in
learned mode: grad-flow into each architecture's own encoder submodule,
and NaN-safety with an empty (fully-masked) user set under backprop.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded

from architectures.st_model import STConfig
from architectures.deepsets_model import DeepSetsConfig
from architectures.ffn_model import FFNConfig
from architectures.user_encoder import OWN_STATE_WIDTH, USER_SLOT_WIDTH


MAX_USERS = 5
TAIL_WIDTH = 3  # stand-in for drone_pair(2)+self_flag(1), composition doesn't matter
TOKEN_FEATURES = OWN_STATE_WIDTH + MAX_USERS * USER_SLOT_WIDTH + TAIL_WIDTH  # 4+35+3=42


def build_model(config_cls, **extra):
    n_agents, n_peers, out = 4, 4, 4
    config = config_cls(encoder_mode="learned", max_users=MAX_USERS, **extra)
    input_spec = Composite(
        {
            "agents": Composite(
                {"observation": Unbounded(shape=(n_agents, n_peers, TOKEN_FEATURES))},
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


def make_input(batch=3, n_agents=4, n_peers=4, all_masked_peer=None):
    """Build a synthetic (batch, n_agents, n_peers, TOKEN_FEATURES) input.

    all_masked_peer: if given, that peer index's user-slot mask columns are
    forced to all-zero (an empty sensed-user set) for every agent/batch --
    gate C's required case.
    """
    x = torch.randn(batch, n_agents, n_peers, TOKEN_FEATURES)
    # zero out the mask columns within each user slot, then set explicitly
    user_block = x[..., OWN_STATE_WIDTH: OWN_STATE_WIDTH + MAX_USERS * USER_SLOT_WIDTH]
    user_block = user_block.reshape(batch, n_agents, n_peers, MAX_USERS, USER_SLOT_WIDTH)
    mask = (torch.rand(batch, n_agents, n_peers, MAX_USERS) > 0.3).float()
    if all_masked_peer is not None:
        mask[:, :, all_masked_peer, :] = 0.0
    user_block[..., -1] = mask
    x[..., OWN_STATE_WIDTH: OWN_STATE_WIDTH + MAX_USERS * USER_SLOT_WIDTH] = (
        user_block.reshape(batch, n_agents, n_peers, -1)
    )
    return x


def assert_grad_flowed(module, name):
    params = list(module.named_parameters())
    assert params, f"{name}: encoder submodule has no parameters at all"
    any_nonzero = False
    for pname, p in params:
        assert p.grad is not None, f"{name}: param {pname} has grad=None -- gradients did not reach the encoder"
        if torch.any(p.grad != 0):
            any_nonzero = True
    assert any_nonzero, f"{name}: all encoder param grads are exactly zero -- gradients did not actually flow"


def run_forward_backward(model, x):
    td = TensorDict({"agents": {"observation": x}}, batch_size=[x.shape[0]])
    td_out = model(td)
    out = td_out.get(model.out_key)
    assert torch.isfinite(out).all(), "model output contains NaN/Inf"
    loss = out.sum()
    loss.backward()
    return out


# ---------------------------------------------------------------------------
# Gate B: grad-flow assert (positive case)
# ---------------------------------------------------------------------------

def test_grad_flow_deepsets():
    model = build_model(DeepSetsConfig, latent_dim=16, phi_hidden=32, rho_hidden=32,
                         dropout=0.0, use_mask=True)
    x = make_input()
    run_forward_backward(model, x)
    assert_grad_flowed(model._user_encoder, "DeepSets")
    print("[PASS] DeepSets learned-mode grad flow")


def test_grad_flow_st():
    model = build_model(STConfig, latent_dim=16, n_heads=2, n_layers=1,
                         dim_feedforward=32, dropout=0.0, use_mask=True)
    x = make_input()
    run_forward_backward(model, x)
    assert_grad_flowed(model._user_encoder, "ST")
    print("[PASS] ST learned-mode grad flow")


def test_grad_flow_ffn():
    model = build_model(FFNConfig, num_cells=[16, 16])
    # FFN's yaml/dataclass has no use_mask field -- its raw input has no
    # trailing peer-mask column, unlike DeepSets/ST.
    x = make_input()
    run_forward_backward(model, x)
    assert_grad_flowed(model._user_encoder, "FFN")
    print("[PASS] FFN learned-mode grad flow")


# ---------------------------------------------------------------------------
# Gate B: demonstrate the assert actually catches a severed-gradient case
# ---------------------------------------------------------------------------

def test_grad_flow_assert_catches_severed_gradients():
    model = build_model(DeepSetsConfig, latent_dim=16, phi_hidden=32, rho_hidden=32,
                         dropout=0.0, use_mask=True)
    x = make_input()
    # Sever the encoder from autograd on purpose (simulates the silent-
    # failure mode the gate exists to catch: a learned-mode config that
    # LOOKS wired but doesn't actually train the encoder).
    for p in model._user_encoder.parameters():
        p.requires_grad_(False)
    run_forward_backward(model, x)
    try:
        assert_grad_flowed(model._user_encoder, "DeepSets(severed)")
    except AssertionError as e:
        print(f"[PASS] assert_grad_flowed correctly raised on severed grads: {e}")
        return
    raise RuntimeError("assert_grad_flowed FAILED to catch severed gradients -- gate B is not load-bearing")


# ---------------------------------------------------------------------------
# Gate C: empty-set / NaN guard under backprop
# ---------------------------------------------------------------------------

def test_empty_set_no_nan_deepsets():
    model = build_model(DeepSetsConfig, latent_dim=16, phi_hidden=32, rho_hidden=32,
                         dropout=0.0, use_mask=True)
    x = make_input(all_masked_peer=1)
    out = run_forward_backward(model, x)
    assert torch.isfinite(out).all(), "output has NaN/Inf with an empty-set drone"
    for pname, p in model._user_encoder.named_parameters():
        assert torch.isfinite(p.grad).all(), f"grad NaN/Inf in {pname} with an empty-set drone"
    print("[PASS] DeepSets empty-set (all-masked) drone: no NaN in output or grads")


def test_empty_set_no_nan_st():
    model = build_model(STConfig, latent_dim=16, n_heads=2, n_layers=1,
                         dim_feedforward=32, dropout=0.0, use_mask=True)
    x = make_input(all_masked_peer=1)
    out = run_forward_backward(model, x)
    assert torch.isfinite(out).all(), "output has NaN/Inf with an empty-set drone"
    for pname, p in model._user_encoder.named_parameters():
        assert torch.isfinite(p.grad).all(), f"grad NaN/Inf in {pname} with an empty-set drone"
    print("[PASS] ST empty-set (all-masked) drone: no NaN in output or grads")


def test_empty_set_no_nan_ffn():
    model = build_model(FFNConfig, num_cells=[16, 16])
    x = make_input(all_masked_peer=1)
    out = run_forward_backward(model, x)
    assert torch.isfinite(out).all(), "output has NaN/Inf with an empty-set drone"
    for pname, p in model._user_encoder.named_parameters():
        assert torch.isfinite(p.grad).all(), f"grad NaN/Inf in {pname} with an empty-set drone"
    print("[PASS] FFN empty-set (all-masked) drone: no NaN in output or grads")


if __name__ == "__main__":
    print("=== encoder_mode=learned verification (gates B, C) ===\n")
    test_grad_flow_deepsets()
    test_grad_flow_st()
    test_grad_flow_ffn()
    test_grad_flow_assert_catches_severed_gradients()
    test_empty_set_no_nan_deepsets()
    test_empty_set_no_nan_st()
    test_empty_set_no_nan_ffn()
    print("\n=== All checks done ===")
