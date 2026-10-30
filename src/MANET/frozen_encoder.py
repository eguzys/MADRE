"""Frozen DeepSets set-encoder. Drop-in replacement for PISAStub.

Same call contract: numpy (numbOfUsers, 6) tokens + (numbOfUsers,) mask
-> numpy (out_dim,) latent. Permutation-invariant (sum-pool),
mask-respecting (padded elements zeroed before sum). Runs the DeepSetsAE
encoder path only (phi -> masked sum -> to_latent); decoder weights unused.
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn

# Per-user token width for the PADRE raw-obs track (angle, power, snr,
# noise-context, throughput-in, throughput-out — see
# observation.py:_build_user_set_for_active_drone). Shared by the frozen
# and learned encoders and by architectures/user_encoder.py's slicing math.
USER_TOKEN_DIM = 6


class FrozenDeepSetsEncoder:
    def __init__(self, weights_path, dim=6, hidden=64, out_dim=8):
        self.out_dim = out_dim
        self._phi = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self._to_latent = nn.Linear(hidden, out_dim)

        sd = torch.load(weights_path, map_location="cpu")
        self._phi[0].weight.data.copy_(sd["phi.0.weight"])
        self._phi[0].bias.data.copy_(sd["phi.0.bias"])
        self._phi[2].weight.data.copy_(sd["phi.2.weight"])
        self._phi[2].bias.data.copy_(sd["phi.2.bias"])
        self._to_latent.weight.data.copy_(sd["to_latent.weight"])
        self._to_latent.bias.data.copy_(sd["to_latent.bias"])

        self._phi.eval()
        self._to_latent.eval()
        for p in list(self._phi.parameters()) + list(self._to_latent.parameters()):
            p.requires_grad_(False)

    def __call__(self, user_tokens, user_mask):
        # user_tokens: (U, 6) numpy, user_mask: (U,) numpy -> (out_dim,) numpy
        with torch.no_grad():
            x = torch.as_tensor(user_tokens, dtype=torch.float32)      # (U, 6)
            m = torch.as_tensor(user_mask, dtype=torch.float32).unsqueeze(-1)  # (U, 1)
            h = self._phi(x) * m                 # zero padded elements
            z = self._to_latent(h.sum(0))        # perm-invariant sum -> (out_dim,)
        return z.numpy().astype(np.float32)


class FrozenOMNeTUserEncoder:
    """OMNeT user-set encoder: 13-col token -> 8-dim latent, h128.

    Same encoder-half math as FrozenDeepSetsEncoder (phi -> masked sum ->
    to_latent), but sized for the OMNeT 13-col token and the
    h128_o8 frozen-arena winner. Mask is derived from the `present` column
    (col 12): sensed users have present=1, padding/unsensed rows are all-zero
    (present=0), so they are zeroed before the sum -- identical to training,
    where the mask zeroed padding before phi.sum(1).

    encode(tokens) takes ONLY the (U, 13) token array (no separate mask arg),
    matching the OMNeTUserEncoderStub contract it replaces.
    """
    def __init__(self, weights_path, dim=13, hidden=128, out_dim=8, present_col=12):
        # Like the event encoder, this one was trained on raw-scale inputs
        # (dBm, radians), so its output is large: p99.9 67.57, absmax 109.55
        # over all 1.5M sets of omnet_sets_merged.npz (21.1 peak over a
        # 150-step random drive). The latent is divided by a fitted constant
        # (100.0, headroom above the training set) stored in the cfg. There is
        # no default: an unscaled latent must not reach the actor token.
        _cfg_path = str(weights_path).replace(".pt", "_cfg.json")
        with open(_cfg_path) as _f:
            _c = json.load(_f)
        if "latent_scale" not in _c:
            raise RuntimeError(
                f"{_cfg_path} has no 'latent_scale'. Refusing to default to 1.0 "
                "-- an unscaled latent must not reach the actor token.")
        self._latent_scale = float(_c["latent_scale"])
        if not (self._latent_scale > 0 and np.isfinite(self._latent_scale)):
            raise RuntimeError(
                f"latent_scale={self._latent_scale!r}; must be finite and > 0.")
        self.out_dim = out_dim
        self._present_col = present_col
        self._phi = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self._to_latent = nn.Linear(hidden, out_dim)
        sd = torch.load(weights_path, map_location="cpu")
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        self._phi[0].weight.data.copy_(sd["phi.0.weight"])
        self._phi[0].bias.data.copy_(sd["phi.0.bias"])
        self._phi[2].weight.data.copy_(sd["phi.2.weight"])
        self._phi[2].bias.data.copy_(sd["phi.2.bias"])
        self._to_latent.weight.data.copy_(sd["to_latent.weight"])
        self._to_latent.bias.data.copy_(sd["to_latent.bias"])
        self._phi.eval(); self._to_latent.eval()
        for p in list(self._phi.parameters()) + list(self._to_latent.parameters()):
            p.requires_grad_(False)

    def encode(self, user_tokens):
        # user_tokens: (U, 13) numpy -> (out_dim,) numpy
        with torch.no_grad():
            x = torch.as_tensor(user_tokens, dtype=torch.float32)          # (U,13)
            m = x[:, self._present_col].unsqueeze(-1)                        # (U,1)
            h = self._phi(x) * m                                            # zero padding
            z = self._to_latent(h.sum(0)) / self._latent_scale              # (out_dim,)
        return z.numpy().astype(np.float32)


class FrozenOMNeTEventEncoder:
    """Frozen DeepSets AE encoder over the OMNeT per-event token set.

    Same phi -> masked-sum -> to_latent shape as FrozenOMNeTUserEncoder, but
    sized for the 12-col event token and the h64_o8 configuration selected
    by the encoder sweep. Mask comes from the `present` column (col 11).

    Trained on the CUT set (receptions + real-failure drops, NOT_ADDRESSED_TO_US
    excluded, cap 64). The live obs path MUST apply the same cut before calling
    encode() -- see omnet_env.cut_event_tokens(). Feeding the raw firehose here
    is a silent distribution break: 87% of raw events are not-for-us and the
    encoder has never seen one.
    """
    def __init__(self, weights_path, dim=12, hidden=64, out_dim=8, present_col=11):
        self.out_dim = out_dim
        self._present_col = present_col
        # This encoder was trained on raw-scale inputs (size in bytes 21-1500,
        # snir in dB), so its output is byte-scale too: p99.9 110.25, absmax
        # 275.73 over all 150150 rows of the cut set. An unscaled latent cannot
        # go into an actor token. The divisor is FITTED on the encoder's own training
        # distribution (not a principled scale like arena width, which the
        # simulator owns) and therefore travels with the weights, in the cfg.
        # It NEVER defaults: a missing key means a cfg predating the
        # measurement, and a silent 1.0 puts 276 back in front of the actor.
        cfg_path = str(weights_path).replace(".pt", "_cfg.json")
        with open(cfg_path) as _f:
            _cfg = json.load(_f)
        if "latent_scale" not in _cfg:
            raise RuntimeError(
                f"{cfg_path} has no 'latent_scale'. Refusing to default to 1.0 "
                "-- an unscaled event latent is what produced bug 20.")
        self._latent_scale = float(_cfg["latent_scale"])
        if not (self._latent_scale > 0 and np.isfinite(self._latent_scale)):
            raise RuntimeError(
                f"latent_scale={self._latent_scale!r}; must be finite and > 0.")
        self._phi = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self._to_latent = nn.Linear(hidden, out_dim)
        sd = torch.load(weights_path, map_location="cpu")
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        self._phi[0].weight.data.copy_(sd["phi.0.weight"])
        self._phi[0].bias.data.copy_(sd["phi.0.bias"])
        self._phi[2].weight.data.copy_(sd["phi.2.weight"])
        self._phi[2].bias.data.copy_(sd["phi.2.bias"])
        self._to_latent.weight.data.copy_(sd["to_latent.weight"])
        self._to_latent.bias.data.copy_(sd["to_latent.bias"])
        self._phi.eval(); self._to_latent.eval()
        for q in list(self._phi.parameters()) + list(self._to_latent.parameters()):
            q.requires_grad_(False)

    def encode(self, event_tokens):
        # event_tokens: (E, 12) numpy -> (out_dim,) numpy
        with torch.no_grad():
            x = torch.as_tensor(event_tokens, dtype=torch.float32)
            m = x[:, self._present_col].unsqueeze(-1)
            h = self._phi(x) * m
            z = self._to_latent(h.sum(0)) / self._latent_scale
        return z.numpy().astype(np.float32)


class LearnableUserSetEncoder(nn.Module):
    """Trainable, from-scratch counterpart to FrozenDeepSetsEncoder.

    Same phi -> masked-sum -> to_latent architecture, but a real nn.Module
    with its own trainable parameters instead of frozen weights loaded from
    a .pt file. Meant to be instantiated ONCE PER ARCHITECTURE (FFN/ST/
    DeepSets each construct their own instance — see architectures/
    user_encoder.py), so its parameters register with that architecture's
    optimizer and receive gradients from the RL loss (encoder_mode=
    "learned"). Never warm-started from the frozen .pt.

    Batched call contract (differs from FrozenDeepSetsEncoder's single-drone
    numpy contract, since this runs inside a BenchMARL model's forward pass):
      tokens: (*, U, dim) float tensor
      mask:   (*, U) float tensor, 1.0 = real user, 0.0 = padded/absent
      -> (*, out_dim) float tensor
    Sum-pool (not mean), so an all-masked-out set naturally yields
    to_latent(0) = to_latent.bias — no 0/0 divide, mirroring the frozen
    encoder's empty-set behaviour.
    """

    def __init__(self, dim: int = USER_TOKEN_DIM, hidden: int = 64, out_dim: int = 8):
        super().__init__()
        self.out_dim = out_dim
        self.phi = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.to_latent = nn.Linear(hidden, out_dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.phi(tokens) * mask.unsqueeze(-1)
        return self.to_latent(h.sum(dim=-2))


class PassthroughUserSetEncoder(nn.Module):
    """The OM2G track's NO-COMPRESSOR baseline: no encoder at all.

    ╔══════════════════════════════════════════════════════════════════════╗
    ║  OM2G TRACK (IEEE 802.15.4).  Not used by the OMNET track.           ║
    ╚══════════════════════════════════════════════════════════════════════╝

    Drop-in for LearnableUserSetEncoder -- same (tokens, mask) -> (*, out_dim)
    contract, so reassemble_with_learned_encoder and every downstream Level-2
    aggregator are untouched -- but it POOLS NOTHING.  It masks the padded
    slots to zero and flattens, so the "message" is literally the raw per-user
    state padded to max_users, in slot order:

        out_dim = max_users * dim   (25 * 13 = 325 on the omnet layout)

    Two properties matter and both are deliberate:

      * It has ZERO parameters.  The baseline is the ABSENCE of a learned
        compressor, not a differently-shaped one.  Any capacity it needs to
        make sense of 325 numbers has to come from the Level-2 aggregator,
        which is exactly the cost the thesis is measuring.

      * It is NOT permutation-invariant.  Slot order is preserved, so this is
        also the position-dependent baseline -- which is what "pad to max
        users and send that" means in practice.

    Masking (rather than passing the padded rows through) matters: the token's
    own presence bit is column `mask_col`, and an unmasked padded slot would
    feed a padding row into the aggregator as a user, which the compressed arm's masked
    sum-pool ignores.  Masking keeps the two arms comparable.
    """

    def __init__(self, dim: int = USER_TOKEN_DIM, hidden: int = 64,
                 max_users: int = 25, out_dim: int | None = None):
        super().__init__()
        self.dim = dim
        self.max_users = max_users
        self.out_dim = max_users * dim
        if out_dim is not None and out_dim != self.out_dim:
            raise ValueError(
                f"PassthroughUserSetEncoder out_dim is structural: "
                f"max_users({max_users}) * dim({dim}) = {self.out_dim}, but "
                f"{out_dim} was requested. Set $OMNET_USER_ENC_OUT_DIM={self.out_dim} "
                "so every width downstream (reconstructed_input_dim, "
                "self_flag_col, the checkpoint) agrees.")

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (tokens * mask.unsqueeze(-1)).flatten(start_dim=-2)


def make_user_set_encoder(*, dim: int, hidden: int, out_dim: int,
                          max_users: int, kind: str | None = None) -> nn.Module:
    """Pick the user-set encoder for the current arm.

    kind=None reads $OM2G_ENCODER, which defaults to "learned" -- so every
    existing OMNET caller gets LearnableUserSetEncoder exactly as before and
    no OMNET number can move.

      "learned"     -> LearnableUserSetEncoder  (OM2G_COMPRESSED arm, D=8)
      "passthrough" -> PassthroughUserSetEncoder (OM2G_UNCOMPRESSED arm, D=325)
    """
    kind = kind or os.environ.get("OM2G_ENCODER", "learned")
    if kind == "learned":
        return LearnableUserSetEncoder(dim=dim, hidden=hidden, out_dim=out_dim)
    if kind == "passthrough":
        return PassthroughUserSetEncoder(dim=dim, hidden=hidden,
                                         max_users=max_users, out_dim=out_dim)
    raise ValueError(
        f"OM2G_ENCODER must be 'learned' or 'passthrough', got {kind!r}")
