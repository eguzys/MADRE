#!/usr/bin/env python3
"""Content-gating verification: OBSERVE, per arm, that a drone delivers to
its peers exactly the L-float summary it paid to send.

Runs WITHOUT the OMNeT binary: a synthetic obs_json drives the real
OM2GEnv._compress_and_gate / build_drone_tokens code paths, and the real
model-side splitter parses the result. One interpreter per configuration
(module-level env-var reads), so this script is an orchestrator when called
with no args and a single check when called with a mode:

    python om2g_compressfix_check.py             # run the whole matrix
    python om2g_compressfix_check.py compressed 2
    python om2g_compressfix_check.py uncompressed
    ...

Checks:
  A  compressed L=2 / L=32: peer slot widths 22/52; the delivered user
     summary EQUALS the codec output on the sender's raw block (to the
     float); staleness serves the OLD snapshot; undelivered rows all-zero;
     own raw block delivered intact; L=2 vs L=32 delivered content differs.
  B  uncompressed: the delivered peer user block equals the sender's raw
     680-float padded block VERBATIM (this arm delivers the raw block it
     pays for).
  C  nocontrol: peer rows all-zero; a delivered beacon RAISES.
  D  fail-loudly: codec out_dim != L refuses to construct; a model
     configured for a different L refuses to split the env's obs.
  E  STModel forward at L=2/L=32 (synthetic specs): builds, runs, output
     shape correct — and its input width comes from the arm-sized layout.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

N_DRONES = 4          # keep the synthetic world small but multi-peer
N_USERS = 11


def base_env(arm, latent=None, mode="frozen"):
    e = dict(os.environ)
    e.update({
        "OM2G_TRACK": "1",
        "OM2G_ARM": arm,
        "OM2G_USER_PARTNER_TAG": "1",
        "OMNET_MAX_USERS_CAP": "40",
        "OMNET_NUM_DRONES": str(N_DRONES),
        "OM2G_USERS": "40",
        # import-time guard skip: this check never launches the simulator,
        # and with the apptainer wrapper flagged, INET lives in-container.
        "OMNET_APPTAINER_SIF": "/dev/null",
    })
    e.pop("OM2G_CODEC_PATH", None)
    e.pop("OM2G_LATENT_DIM", None)
    e.pop("OM2G_CODEC_MODE", None)
    e.pop("OMNET_USER_ENC_OUT_DIM", None)
    if arm == "compressed":
        e["OM2G_LATENT_DIM"] = str(latent)
        if mode in ("e2e", "e2e_rt"):
            e["OM2G_CODEC_MODE"] = mode
        else:
            e["OM2G_CODEC_PATH"] = os.path.join(
                HERE, f"codec_check_o{latent}.pt")
    return e


def make_codec(latent, path):
    """Seeded random-weight codec in the trainer's on-disk format.
    Mechanism check only — REAL runs use reconstruction-trained codecs."""
    import torch
    import torch.nn as nn
    torch.manual_seed(1234 + latent)
    phi = nn.Sequential(nn.Linear(17, 64), nn.ReLU(),
                        nn.Linear(64, 64), nn.ReLU())
    to_latent = nn.Linear(64, latent)
    sd = {f"phi.{k}": v for k, v in phi.state_dict().items()}
    sd.update({f"to_latent.{k}": v for k, v in to_latent.state_dict().items()})
    torch.save(sd, path)
    with open(path.replace(".pt", "_cfg.json"), "w") as fh:
        json.dump({"hidden": 64, "out_dim": latent, "dim": 17,
                   "latent_scale": 1.0,
                   "note": "RANDOM-INIT mechanism-check codec, NOT trained"},
                  fh)


# ─────────────────────────────────────────────────────────────────────────────
# The single-configuration check (runs in its own interpreter)
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_obs(step, rng, age, valid):
    """A complete-enough obs_json for build_drone_tokens + the gate."""
    pos = [[200.0 + 150.0 * i, 300.0 + 90.0 * i] for i in range(N_DRONES)]
    upos = [[100.0 + 80.0 * u, 900.0 - 60.0 * u] for u in range(N_USERS)]
    return {
        "step": step,
        "positions": pos,
        "user_positions": upos,
        "noise_floor": [-96.6 + 0.1 * i for i in range(N_DRONES)],
        "interference": [-95.0] * N_DRONES,
        "user_to_drone": [u % N_DRONES for u in range(N_USERS)],
        "user_goodput": [1000.0 * (u + 1) for u in range(N_USERS)],
        "user_rssi": [[-80.0 - u - 0.31 * i - 0.11 * step
                       for u in range(N_USERS)]
                      for i in range(N_DRONES)],
        "user_rssi_valid": [[1 if (u + i) % 3 != 0 else 0
                             for u in range(N_USERS)]
                            for i in range(N_DRONES)],
        "sender_pairs": [[u, (u + 1) % N_USERS] for u in range(0, N_USERS, 2)],
        "peer_rssi": [[-70.0 - abs(i - j) for j in range(N_DRONES)]
                      for i in range(N_DRONES)],
        "peer_rssi_valid": [[1 if i != j else 0 for j in range(N_DRONES)]
                            for i in range(N_DRONES)],
        "drone_fwd": [10.0 * i for i in range(N_DRONES)],
        "channel_busy": [0.1 * (i + 1) for i in range(N_DRONES)],
        "time_since_rx": [0.5 * i for i in range(N_DRONES)],
        "beacon_rx_age": age,
        "beacon_rx_valid": valid,
        "arena_width": 1200.0,
        "arena_height": 1200.0,
        "max_steps": 250.0,
        "throughput": [0.0] * N_DRONES,
        "ctrl_bytes_tx": [0.0] * N_DRONES,
        "ctrl_bytes_rx": [0.0] * N_DRONES,
    }


def run_one(arm, latent_arg):
    import numpy as np
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(ROOT, "omnet_module"))
    sys.path.insert(0, os.path.join(ROOT, "src"))

    import om2g_track as T
    from om2g_env import (OM2GEnv, om2g_arm_latent, om2g_obs_width,
                          om2g_peer_slot_width)
    import omnet_env as OE

    mode = os.environ.get("OM2G_CODEC_MODE", "frozen")
    L = om2g_arm_latent(arm)
    if arm == "compressed":
        assert L == int(latent_arg), (L, latent_arg)
    Wp = om2g_peer_slot_width(L)
    U = OE.MAX_USERS_CAP * OE.OBS_FEATURES
    assert OE.MAX_N == N_DRONES, (OE.MAX_N, N_DRONES)
    assert U == 40 * 17 == 680, U

    # Construct the env WITHOUT launching a simulator: __new__ + the real
    # __init__ tail is too fragile, so run the REAL __init__ of the parent
    # only for what the gate needs. We call the actual OM2GEnv.__init__?
    # It would try to reach ZMQ/binary via OMNeTEnv.__init__ -- instead
    # instantiate bare and copy the EXACT init logic under test by calling
    # the class's own code: we re-run the arm/latent/codec block by
    # constructing via __new__ and invoking the real methods it uses.
    env = object.__new__(OM2GEnv)
    env._arm = arm
    env._latent = L
    env._codec = None
    env._codec_mode = mode
    if arm == "compressed" and mode == "frozen":
        from om2g_env import _OM2GUserCodec
        env._codec = _OM2GUserCodec(os.environ["OM2G_CODEC_PATH"], L)
    env._slot_w = Wp
    env._raw_w = U
    env._inbox_w = N_DRONES * U if mode == "e2e" else 0
    env._obs_w = om2g_obs_width(L) + env._inbox_w
    env._msg_history = {}
    env._raw_history = {}
    env._telemetry = []
    env._telemetry_csv = ""
    env._episode_index = 0
    env._encoder_mode = "learned"
    env._episode_counter = 7

    def build(step, age, valid):
        obs_json = synthetic_obs(step, None, age, valid)
        tokens = OE.build_drone_tokens(obs_json, env._episode_counter,
                                       "learned")
        return env._compress_and_gate(tokens, obs_json), obs_json, tokens

    never = [[-1.0] * N_DRONES for _ in range(N_DRONES)]
    none_valid = [[0.0] * N_DRONES for _ in range(N_DRONES)]

    # step 0: nothing delivered yet anywhere
    obs0, _, tokens0 = build(0, never, none_valid)
    assert obs0.shape == (N_DRONES, env._obs_w), obs0.shape

    # own raw block delivered intact, every arm
    for i in range(N_DRONES):
        np.testing.assert_array_equal(
            obs0[i, :U], tokens0[i, i, OE.OWN_STATE_W:OE.OWN_STATE_W + U],
            err_msg="own raw user block not delivered intact")

    peers0 = obs0[:, U:U + N_DRONES * Wp].reshape(N_DRONES, N_DRONES, Wp)
    for i in range(N_DRONES):
        for j in range(N_DRONES):
            if i == j:
                assert peers0[i, i, 15 + L] == 1.0, "self flag missing"
            else:
                assert np.all(peers0[i, j] == 0.0), (
                    f"undelivered peer row {i},{j} not blank")

    if arm == "nocontrol":
        # C: any delivered beacon must raise
        age = [[1.0 if i != j else -1.0 for j in range(N_DRONES)]
               for i in range(N_DRONES)]
        val = [[1.0 if i != j else 0.0 for j in range(N_DRONES)]
               for i in range(N_DRONES)]
        try:
            build(1, age, val)
        except RuntimeError as e:
            assert "nocontrol" in str(e)
            print(f"PASS nocontrol: peers blank; delivered beacon raised "
                  f"loudly; obs width {env._obs_w} (slot {Wp}, L=0)")
            return
        raise AssertionError("nocontrol accepted a delivered beacon SILENTLY")

    # steps 1..3: build history, then deliver with mixed ages
    for s in (1, 2):
        build(s, never, none_valid)

    # at step 3: i=0 hears j=1 fresh (age dt), j=2 stale (age 3*dt), j=3 never
    dt = T.OM2G_RL_STEP_S
    age = [[-1.0] * N_DRONES for _ in range(N_DRONES)]
    val = [[0.0] * N_DRONES for _ in range(N_DRONES)]
    age[0][1] = 0.0
    val[0][1] = 1.0        # rx this step -> snapshot @ step 3
    age[0][2] = 3.0 * dt
    val[0][2] = 1.0        # snapshot @ step 0
    obs3, _, tokens3 = build(3, age, val)
    peers3 = obs3[:, U:U + N_DRONES * Wp].reshape(N_DRONES, N_DRONES, Wp)

    # the delivered summary must equal the codec applied to the SENDER's raw
    # block at the snapshot step -- exactly, to the float
    def expect_summary(sender_tokens, j):
        blk = sender_tokens[j, j, OE.OWN_STATE_W:OE.OWN_STATE_W + U]
        if arm == "compressed" and mode == "e2e":
            # slot stays empty in e2e; E fills it model-side from the inbox
            return np.zeros(L, dtype=np.float32)
        if arm == "compressed":
            return env._codec.encode(blk.reshape(40, 17))
        return blk

    def expect_inbox(sender_tokens, j):
        return sender_tokens[j, j, OE.OWN_STATE_W:OE.OWN_STATE_W + U]

    got_fresh = peers3[0, 1, 7:7 + L]
    want_fresh = expect_summary(tokens3, 1)
    np.testing.assert_array_equal(got_fresh, want_fresh,
                                  err_msg="fresh delivery != codec(sender raw @ rx step)")

    got_stale = peers3[0, 2, 7:7 + L]
    want_stale = expect_summary(tokens0, 2)
    np.testing.assert_array_equal(got_stale, want_stale,
                                  err_msg="stale delivery != codec(sender raw @ OLD step)")
    # and staleness is real: the stale snapshot differs from today's
    _stale_probe = (expect_inbox if mode == "e2e" else expect_summary)
    assert not np.array_equal(_stale_probe(tokens0, 2),
                              _stale_probe(tokens3, 2)), (
        "synthetic world did not change between steps; staleness untestable")

    assert np.all(peers3[0, 3] == 0.0), "never-heard peer not blank"
    assert np.all(peers3[1, 0] == 0.0), "delivery direction leaked"

    if mode == "e2e":
        # The inbox: the ONLY carrier of peer user content in e2e -- must be
        # the sender's raw block at the delivered rx step, exactly, and only
        # where a message was delivered.
        inbox3 = obs3[:, U + N_DRONES * Wp:].reshape(N_DRONES, N_DRONES, U)
        np.testing.assert_array_equal(
            inbox3[0, 1], expect_inbox(tokens3, 1),
            err_msg="e2e fresh inbox != sender raw @ rx step")
        np.testing.assert_array_equal(
            inbox3[0, 2], expect_inbox(tokens0, 2),
            err_msg="e2e stale inbox != sender raw @ OLD step")
        assert np.all(inbox3[0, 3] == 0.0), "undelivered inbox not blank"
        assert np.all(inbox3[1, 0] == 0.0), "inbox delivery direction leaked"
        np.testing.assert_array_equal(
            inbox3[2, 2], expect_inbox(tokens3, 2),
            err_msg="e2e own-row inbox != own current raw")
        # slot summary columns must be EMPTY everywhere in e2e
        assert np.all(peers3[:, :, 7:7 + L] == 0.0), (
            "e2e slot summary columns not empty -- env leaked content")

    # geometry-channel pricing
    gb = T.OM2G_GEO_BYTES
    assert T.om2g_message_bytes(arm, L) == gb + 4 * (12 + L), (
        T.om2g_message_bytes(arm, L), gb, L)
    if gb != 8:
        # delivered peer position must be the QUANTIZED sender position --
        # exactly what the wire paid for, nothing finer
        true_pos = tokens3[1, 1, 0:2]
        got_pos = peers3[0, 1, 0:2]
        levels = float((1 << (gb * 4)) - 1) if gb > 0 else None
        if gb == 0:
            assert np.all(got_pos == 0.0), "GEO=0 leaked a position"
        else:
            err = np.abs(got_pos - true_pos).max()
            assert err <= 2.0 / levels + 1e-6, (
                f"quantized pos error {err} > grid step (gb={gb})")
            grid = np.round((got_pos + 1) / 2 * levels)
            np.testing.assert_allclose(
                (grid / levels) * 2 - 1, got_pos, atol=1e-6,
                err_msg="delivered position is not on the paid-for grid")
        # own row keeps the EXACT own position (local knowledge is free)
        np.testing.assert_array_equal(
            peers3[2, 2, 0:2], tokens3[2, 2, 0:2],
            err_msg="own-row position was quantized -- it must stay exact")
        print(f"PASS geo gb={gb}: delivered pos on the {gb * 4}-bit grid, "
              f"own pos exact, charge {T.om2g_message_bytes(arm, L)} B")

    # delivered floats total == charged floats
    msg = env._msg_history[3][1]
    assert msg.shape == (14 + L,), msg.shape
    assert T.om2g_message_floats(arm, L) == 14 + L

    # the raw block must NOT be recoverable from a compressed slot: the slot
    # simply has no columns for it
    if arm == "compressed":
        assert Wp == 20 + L < 700, Wp

    # model-side split agrees
    from architectures.user_encoder import om2g_flat_layout, split_om2g_obs
    import torch
    lo = om2g_flat_layout()
    assert lo["flat_dim"] == env._obs_w, (lo["flat_dim"], env._obs_w)
    rt, rm, pr, ibt, ibm = split_om2g_obs(torch.as_tensor(obs3), lo)
    if mode == "e2e":
        assert ibt is not None and ibt.shape[-3:] == (N_DRONES, 40, 17)
        np.testing.assert_array_equal(
            ibt.numpy().reshape(N_DRONES, N_DRONES, U),
            obs3[:, U + N_DRONES * Wp:].reshape(N_DRONES, N_DRONES, U),
            err_msg="model-side inbox split != env inbox section")
    else:
        assert ibt is None and ibm is None
    np.testing.assert_array_equal(pr.numpy().reshape(N_DRONES, N_DRONES, Wp),
                                  peers3)
    np.testing.assert_array_equal(
        rt.numpy().reshape(N_DRONES, 40, 17)[0].reshape(-1), obs3[0, :U])

    print(f"PASS {arm} L={L} mode={mode}: slot {Wp}, obs {env._obs_w}, "
          f"delivered {14 + L} floats == charged {T.om2g_message_bytes(arm, L)} B"
          f"; fresh/stale/never + own-raw + model split all exact")
    # emit a fingerprint so the orchestrator can prove L=2 and L=32 differ
    print("SUMMARY_FP", arm, L, ",".join(f"{v:.6f}" for v in got_fresh[:8]))


# ─────────────────────────────────────────────────────────────────────────────
# Model forward check (separate mode: needs benchmarl)
# ─────────────────────────────────────────────────────────────────────────────

def _build_model(arch, lo):
    import torch.nn as nn
    from architectures.st_model import STConfig
    from architectures.deepsets_model import DeepSetsConfig
    from architectures.ffn_model import FFNConfig
    from architectures.gat_model import GATConfig
    from torchrl.data import Composite, Unbounded

    n_agents = lo["max_n"]
    D = lo["flat_dim"]
    input_spec = Composite(
        {"agents": Composite(
            {"observation": Unbounded(shape=(n_agents, D))},
            shape=(n_agents,))})
    output_spec = Composite(
        {"agents": Composite(
            {"param": Unbounded(shape=(n_agents, 4))},
            shape=(n_agents,))})
    cfg = dict(
        st=lambda: STConfig(latent_dim=32, n_heads=4, n_layers=1,
                            dim_feedforward=64, dropout=0.0, use_mask=True,
                            encoder_mode="learned", max_users=lo["max_users"],
                            obs_layout="om2g"),
        deepsets=lambda: DeepSetsConfig(
            latent_dim=32, phi_hidden=32, rho_hidden=32, dropout=0.0,
            use_mask=True, encoder_mode="learned", max_users=lo["max_users"],
            obs_layout="om2g"),
        ffn=lambda: FFNConfig(num_cells=[64, 64], activation_class=nn.Tanh,
                              encoder_mode="learned",
                              max_users=lo["max_users"], obs_layout="om2g"),
        gat=lambda: GATConfig(latent_dim=32, n_heads=4, n_layers=1,
                              encoder_mode="learned",
                              max_users=lo["max_users"], obs_layout="om2g"),
    )[arch]()
    model = cfg.get_model(
        input_spec=input_spec, output_spec=output_spec, agent_group="agents",
        input_has_agent_dim=True, n_agents=n_agents, centralised=False,
        share_params=True, device="cpu", action_spec=Composite({}),
        model_index=0)
    return model, D, n_agents


def run_model(arm, latent_arg):
    import torch
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(ROOT, "omnet_module"))
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from architectures.user_encoder import om2g_flat_layout

    lo = om2g_flat_layout()
    arch = sys.argv[4] if len(sys.argv) > 4 else "st"
    model, D, n_agents = _build_model(arch, lo)
    from tensordict import TensorDict
    x = _synthetic_flat_obs(lo, torch)
    assert x.shape[-1] == D, (x.shape, D)
    td = TensorDict({"agents": {"observation": x}}, batch_size=[3])
    out = model._forward(td)
    y = out.get(("agents", "param"))
    assert y.shape == (3, n_agents, 4), y.shape
    assert torch.isfinite(y).all()
    if lo["codec_mode"] in ("e2e", "e2e_rt"):
        assert getattr(model, "_peer_msg_encoder", None) is not None, (
            "e2e mode but the model built no peer message encoder E")
    n_par = sum(p.numel() for p in model.parameters())
    print(f"PASS model {arch} {arm} L={lo['latent']} mode={lo['codec_mode']}: "
          f"forward on flat obs {D} -> {tuple(y.shape)}, params={n_par}")


def _synthetic_flat_obs(lo, torch):
    """Random flat obs with plausible masks/self-flags/present bits."""
    torch.manual_seed(7)
    n = lo["max_n"]
    L = lo["latent"]
    Wp = lo["w_peer"]
    raw = torch.randn(3, n, lo["raw_w"]).abs() * 0.1
    slots = torch.randn(3, n, n, Wp).abs() * 0.1
    slots[..., Wp - 1] = 1.0
    slots[..., 15 + L] = 0.0
    for i in range(n):
        slots[:, i, i, 15 + L] = 1.0
    parts = [raw, slots.reshape(3, n, n * Wp)]
    if lo["inbox_w"]:
        inbox = torch.randn(3, n, n, lo["max_users"],
                            lo["user_feat"]).abs() * 0.1
        inbox[..., lo["user_mask_col"]] = (inbox[..., 0] > 0.05).float()
        parts.append(inbox.reshape(3, n, lo["inbox_w"]))
    return torch.cat(parts, dim=-1)


def run_hook(arch):
    """C1 — THE funnel invariant (e2e): the actor's output may depend on the
    stale-raw inbox ONLY through the shared message encoder E.

    Pass 1: forward on obs A, capturing E's output via a forward hook.
    Pass 2: randomize the ENTIRE inbox section, pin E's output to the
            captured values via the hook, forward again.
    Any path from the inbox to the action that bypasses E makes the outputs
    differ; they must be bit-identical.
    """
    import torch
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(ROOT, "omnet_module"))
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from architectures.user_encoder import om2g_flat_layout
    lo = om2g_flat_layout()
    assert lo["codec_mode"] in ("e2e", "e2e_rt"), (
        "hook test requires OM2G_CODEC_MODE=e2e or e2e_rt")
    model, D, n_agents = _build_model(arch, lo)
    model.eval()
    from tensordict import TensorDict

    x_a = _synthetic_flat_obs(lo, torch)
    captured = []
    h = model._peer_msg_encoder.register_forward_hook(
        lambda m, i, o: captured.append(o.detach().clone()))
    with torch.no_grad():
        out_a = model._forward(TensorDict(
            {"agents": {"observation": x_a}}, batch_size=[3])).get(
                ("agents", "param")).clone()
    h.remove()
    assert len(captured) == 1, f"E called {len(captured)} times, expected 1"

    peer_end = lo["raw_w"] + lo["max_n"] * lo["w_peer"]
    x_b = x_a.clone()
    torch.manual_seed(999)
    x_b[..., peer_end:] = torch.randn_like(x_b[..., peer_end:]) * 17.3
    h = model._peer_msg_encoder.register_forward_hook(
        lambda m, i, o: captured[0])
    with torch.no_grad():
        out_b = model._forward(TensorDict(
            {"agents": {"observation": x_b}}, batch_size=[3])).get(
                ("agents", "param")).clone()
    h.remove()
    if not torch.equal(out_a, out_b):
        raise AssertionError(
            f"FUNNEL VIOLATION ({arch}): actor output changed when the "
            f"inbox was randomized with E's output pinned -- a path from "
            f"raw peer blocks to the action bypasses the message encoder. "
            f"max|diff|={float((out_a - out_b).abs().max()):.3e}")
    print(f"PASS hook-invariant {arch}: inbox reaches the action ONLY "
          f"through E (outputs bit-identical under inbox randomization)")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def orchestrate():
    py = sys.executable
    # codecs for the compressed cells
    for L in (2, 32):
        make_codec(L, os.path.join(HERE, f"codec_check_o{L}.pt"))

    fps = {}
    for arm, L in (("compressed", 2), ("compressed", 32),
                   ("uncompressed", None), ("nocontrol", None)):
        r = subprocess.run([py, __file__, "one", arm, str(L)],
                           env=base_env(arm, L), capture_output=True,
                           text=True)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            sys.stderr.write(r.stderr)
            raise SystemExit(f"FAILED: {arm} L={L}")
        for line in r.stdout.splitlines():
            if line.startswith("SUMMARY_FP"):
                fps[(arm, line.split()[2])] = line.split()[3]

    # L=2 vs L=32: widths AND content differ
    assert fps[("compressed", "2")].count(",") == 1     # 2 floats
    assert fps[("compressed", "32")].count(",") == 7    # first 8 of 32
    assert fps[("compressed", "2")] != fps[("compressed", "32")]
    print("PASS matrix: L=2 delivers 2 floats, L=32 delivers 32; contents "
          "differ; uncompressed delivers the verbatim raw block; nocontrol "
          "delivers nothing and trips on a delivered beacon")

    # GEO split-pricing probe: quantized geometry channel (frozen + e2e)
    for mode, gb in (("frozen", 2), ("e2e", 2), ("e2e", 0)):
        e = base_env("compressed", 8, mode=mode)
        e["OM2G_GEO_BYTES"] = str(gb)
        r = subprocess.run([py, __file__, "one", "compressed", "8"], env=e,
                           capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            sys.stderr.write(r.stderr)
            raise SystemExit(f"FAILED: geo gb={gb} mode={mode}")

    # D: fail-loudly — L=32 env var against an out_dim=2 codec must refuse
    e = base_env("compressed", 32)
    e["OM2G_CODEC_PATH"] = os.path.join(HERE, "codec_check_o2.pt")
    r = subprocess.run([py, __file__, "one", "compressed", "32"], env=e,
                       capture_output=True, text=True)
    assert r.returncode != 0 and "out_dim" in (r.stdout + r.stderr), (
        "codec/latent mismatch did NOT fail loudly")
    print("PASS fail-loudly: out_dim=2 codec under OM2G_LATENT_DIM=32 refused")

    # D2: model configured for the wrong L must refuse the env's obs
    r = subprocess.run([py, __file__, "splitmismatch"], env=base_env(
        "compressed", 2), capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit("FAILED: splitmismatch")

    # E: model forwards -- all four architectures, both L extremes, plus
    # uncompressed on the set archs (GAT stays compressed-only, as before).
    for arch in ("st", "deepsets", "ffn", "gat"):
        for arm, L in (("compressed", 2), ("compressed", 32),
                       ("uncompressed", None)):
            r = subprocess.run([py, __file__, "model", arm, str(L), arch],
                               env=base_env(arm, L),
                               capture_output=True, text=True)
            sys.stdout.write(r.stdout)
            if r.returncode != 0:
                sys.stderr.write(r.stderr)
                raise SystemExit(f"FAILED: model {arch} {arm} L={L}")

    # ── e2e mode: env content + model forwards + C1 hook ──
    for L in (2, 32):
        r = subprocess.run([py, __file__, "one", "compressed", str(L)],
                           env=base_env("compressed", L, mode="e2e"),
                           capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            sys.stderr.write(r.stderr)
            raise SystemExit(f"FAILED: e2e env L={L}")
    for arch in ("st", "deepsets", "ffn", "gat"):
        for cmd, tag in (("model", "e2e model"), ("hook", "C1 hook"),
                         ("grad", "E grad-flow")):
            args = ([py, __file__, "model", "compressed", "8", arch]
                    if cmd == "model" else [py, __file__, cmd, arch])
            r = subprocess.run(args, env=base_env("compressed", 8,
                                                  mode="e2e"),
                               capture_output=True, text=True)
            sys.stdout.write(r.stdout)
            if r.returncode != 0:
                sys.stderr.write(r.stderr)
                raise SystemExit(f"FAILED: {tag} {arch}")
    print("=== COMPRESS-FIX CHECK: ALL PASS ===")


def run_grad(arch):
    """e2e: RL loss gradient must actually reach E's parameters -- an
    accidentally-detached inbox would never train E, and the arm would
    become a random-codec arm."""
    import torch
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(ROOT, "omnet_module"))
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from architectures.user_encoder import om2g_flat_layout
    from tensordict import TensorDict
    lo = om2g_flat_layout()
    model, D, n_agents = _build_model(arch, lo)
    model.train()
    x = _synthetic_flat_obs(lo, torch)
    out = model._forward(TensorDict(
        {"agents": {"observation": x}}, batch_size=[3])).get(
            ("agents", "param"))
    out.sum().backward()
    g = sum(float(p.grad.abs().sum()) for p in
            model._peer_msg_encoder.parameters() if p.grad is not None)
    n_with = sum(1 for p in model._peer_msg_encoder.parameters()
                 if p.grad is not None and float(p.grad.abs().sum()) > 0)
    n_tot = sum(1 for _ in model._peer_msg_encoder.parameters())
    assert g > 0 and n_with == n_tot, (
        f"E gradient missing: |grad| sum={g}, {n_with}/{n_tot} params with "
        f"nonzero grad -- the inbox path is detached somewhere.")
    print(f"PASS grad-flow {arch}: RL loss reaches all {n_tot} E params "
          f"(|grad| sum {g:.3e})")


def run_splitmismatch():
    import torch
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from architectures.user_encoder import om2g_flat_layout, split_om2g_obs
    lo = om2g_flat_layout()          # built at L=2
    wrong = torch.zeros(1, lo["flat_dim"] + 30 * 4)   # an L=32 env's width
    try:
        split_om2g_obs(wrong, lo)
    except RuntimeError as e:
        assert "arm=" in str(e)
        print("PASS fail-loudly: model at L=2 refused an L=32-wide obs "
              "with both sides' numbers in the message")
        return
    raise AssertionError("split accepted a mismatched width SILENTLY")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        orchestrate()
    elif sys.argv[1] == "one":
        run_one(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "model":
        run_model(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "hook":
        run_hook(sys.argv[2])
    elif sys.argv[1] == "grad":
        run_grad(sys.argv[2])
    elif sys.argv[1] == "splitmismatch":
        run_splitmismatch()
    else:
        raise SystemExit(f"unknown mode {sys.argv[1]!r}")
