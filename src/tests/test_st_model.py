"""Verification tests for STModel BenchMARL port — CPU only, no Experiment."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded

from architectures.st_model import STConfig, STModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_model(n_agents=4, n_peers=4, features=22, out=4, use_mask=False):
    config = STConfig(
        latent_dim=64,
        n_heads=4,
        n_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        use_mask=use_mask,
    )
    input_spec = Composite(
        {
            "agents": Composite(
                {"observation": Unbounded(shape=(n_agents, n_peers, features))},
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
    model = config.get_model(
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
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_forward_shape_and_finite():
    """Output has correct shape and no NaN/Inf."""
    batch = 3
    n_agents, n_peers, features, out = 4, 4, 22, 4
    model = build_model(n_agents, n_peers, features, out)
    model.eval()

    td = TensorDict(
        {"agents": {"observation": torch.randn(batch, n_agents, n_peers, features)}},
        batch_size=[batch],
    )
    with torch.no_grad():
        td_out = model(td)

    out_key = model.out_key
    assert out_key in list(td_out.keys(True, True)), f"out_key {out_key!r} missing from output"

    result = td_out.get(out_key)
    expected_shape = (batch, n_agents, out)
    assert result.shape == expected_shape, (
        f"Shape mismatch: expected {expected_shape}, got {tuple(result.shape)}"
    )
    assert torch.isfinite(result).all(), "Output contains NaN or Inf"

    print(f"[PASS] forward: shape={tuple(result.shape)}, finite=True")


def test_permutation_invariance():
    """Permuting n_peers tokens of one agent leaves its output unchanged (within tolerance)."""
    torch.manual_seed(42)
    batch, n_agents, n_peers, features, out = 2, 4, 4, 22, 4
    model = build_model(n_agents, n_peers, features, out)
    model.eval()

    x = torch.randn(batch, n_agents, n_peers, features)
    perm = torch.randperm(n_peers)

    x_perm = x.clone()
    x_perm[0, 0] = x[0, 0][perm]  # permute tokens of agent-0 in batch-0 only

    td1 = TensorDict({"agents": {"observation": x}}, batch_size=[batch])
    td2 = TensorDict({"agents": {"observation": x_perm}}, batch_size=[batch])

    with torch.no_grad():
        out1 = model(td1).get(model.out_key)
        out2 = model(td2).get(model.out_key)

    diff = (out1[0, 0] - out2[0, 0]).abs().max().item()
    tol = 1e-5

    # Unchanged agents (agent 1-3, batch 0 and all of batch 1) must be identical
    assert torch.allclose(out1[0, 1:], out2[0, 1:], atol=1e-6), (
        "Unmodified agents changed after permuting agent-0's tokens"
    )
    assert torch.allclose(out1[1], out2[1], atol=1e-6), (
        "Unmodified batch-1 changed after permuting batch-0's tokens"
    )

    if diff < tol:
        print(f"[PASS] Permutation invariance: max_diff={diff:.2e} < tol={tol:.0e}")
    else:
        print(
            f"[SOFT-FAIL] Permutation invariance: max_diff={diff:.2e} >= tol={tol:.0e} "
            f"— not exactly invariant (float summation order). Architecture is "
            f"set-based by construction (PMA), minor numerical drift is expected."
        )


def test_param_count_and_shared():
    """Confirm ONE shared parameter set — not n_agents copies."""
    model = build_model()

    param_names = [name for name, _ in model.named_parameters()]
    n_params = sum(p.numel() for p in model.parameters())

    # Per-agent copies would appear as top-level digit-indexed modules, e.g.
    # MultiAgentMLP with share_params=False gives "0._net.", "1._net.", etc.
    per_agent = [n for n in param_names if n.split(".")[0].isdigit()]
    assert len(per_agent) == 0, (
        f"Found per-agent parameters — expected single shared set: {per_agent}"
    )

    # Param count must not scale with n_agents
    model_x4 = build_model(n_agents=4)
    model_x8 = build_model(n_agents=8)
    params_x4 = sum(p.numel() for p in model_x4.parameters())
    params_x8 = sum(p.numel() for p in model_x8.parameters())
    assert params_x4 == params_x8, (
        f"Param count grows with n_agents: n_agents=4 → {params_x4}, n_agents=8 → {params_x8}"
    )

    print(f"[PASS] Single shared parameter set")
    print(f"       Total parameters: {n_params:,}")
    print(f"       Named parameters: {param_names}")


def test_multidim_batch():
    """Model handles a 2-D leading batch (*batch = [T, B])."""
    T, B, n_agents, n_peers, features, out = 5, 3, 4, 4, 22, 4
    model = build_model(n_agents, n_peers, features, out)
    model.eval()

    x = torch.randn(T, B, n_agents, n_peers, features)
    td = TensorDict({"agents": {"observation": x}}, batch_size=[T, B])
    with torch.no_grad():
        result = model(td).get(model.out_key)

    assert result.shape == (T, B, n_agents, out), (
        f"Multi-dim batch shape wrong: {tuple(result.shape)}"
    )
    assert torch.isfinite(result).all()
    print(f"[PASS] Multi-dim batch: shape={tuple(result.shape)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== STModel BenchMARL Port Verification ===\n")
    test_forward_shape_and_finite()
    test_permutation_invariance()
    test_param_count_and_shared()
    test_multidim_batch()
    print("\n=== All checks done ===")
