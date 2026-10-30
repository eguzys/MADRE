"""Shared learned-mode user-set encoder plumbing for the PADRE raw-obs track.

Sits between MANET/observation.py (env side: emits a padded raw per-user
token block per drone when encoder_mode=="learned", instead of an already-
pooled summary) and each of the three BenchMARL model files (FFN/ST/
DeepSets), which own a *private* LearnableUserSetEncoder instance and use
the helpers below to slice their input, run their own encoder over the raw
user block, and reassemble -- so the reassembled per-peer token has the
exact same width/layout as encoder_mode=="frozen" (own_state | pooled
summary | tail), and everything downstream (the arch's existing Level-2
aggregator: sum-pool / attention / flatten+MLP) is untouched.

OWN_STATE_WIDTH must match RawUserSet_OW's raw-only own-state width
(position=2 + receivedSignal=1 + interference=1 -- see observation.py's
raw_config). It's a fixed structural fact of this track's observation
design, not derived at runtime, because a BenchMARL model has no other way
to know where the user-token block starts within its flattened per-peer
feature vector. If observation.py's raw_config ever changes which/how many
own-state columns it emits, this constant must change with it.
"""
from __future__ import annotations

import os

import torch
from torch import nn

from MANET.frozen_encoder import LearnableUserSetEncoder, USER_TOKEN_DIM

OWN_STATE_WIDTH = 4
USER_SLOT_WIDTH = USER_TOKEN_DIM + 1  # 6 token cols + 1 mask col

# Learned encoder's pooled-output width. Matches FrozenDeepSetsEncoder's
# out_dim=8 so the Level-2 aggregator's input shape is identical between
# encoder_mode "frozen" and "learned" -- a fair architecture comparison
# (same downstream capacity), not a derived necessity.
#
# Overridable via $OMNET_USER_ENC_OUT_DIM for the compression
# sweep. This is THE bottleneck knob: omnet's per-peer token carries 25 users
# x 13 = 325 raw numbers that are masked-SUMMED into this many before any
# architecture sees them (frozen_encoder.LearnableUserSetEncoder.forward).
# At 8 that is 40:1 compression, and the 8 was inherited from the frozen
# encoder, never chosen. Sweeping it tests the actual hypothesis: a
# permutation-invariant Level-2 aggregator (ST, deepsets) should degrade more
# gracefully as this narrows than a position-dependent one (FFN).
#
# ⚠️ Changing this changes PARAMETER SHAPES and the reassembled token width,
# so a checkpoint trained at another value cannot be loaded.
# It also moves `self_flag_col` below (the own-token bypass), which is
# why that index is written as an expression in this constant and must never
# be hard-coded back to 23.
# ⚠️ EVAL MUST EXPORT THE SAME VALUE. This is a module constant read at import
# time; evaluate_omnet.py rebuilds the model from config.pkl against whatever
# is installed NOW, so scoring an out_dim=4 checkpoint without the env var set
# fails with a bare tensor-shape mismatch. eval_arm.sh / eval_n100.sh carry it.
_env_out_dim = os.environ.get("OMNET_USER_ENC_OUT_DIM")
if _env_out_dim is None:
    LEARNED_USER_ENCODER_OUT_DIM = 8
else:
    try:
        LEARNED_USER_ENCODER_OUT_DIM = int(_env_out_dim)
    except ValueError:
        raise RuntimeError(
            f"OMNET_USER_ENC_OUT_DIM={_env_out_dim!r} is not an integer."
        )
    # Upper bound 2048. It is a typo guard, not a
    # structural limit, and the OM2G track's no-compressor arm legitimately
    # exceeds 512: PassthroughUserSetEncoder's width is
    # max_users * user_token_dim, which is 40 * 13 = 520 at the 40-user
    # population. 2048 still catches a fat-fingered exponent.
    if not (1 <= LEARNED_USER_ENCODER_OUT_DIM <= 2048):
        raise RuntimeError(
            f"OMNET_USER_ENC_OUT_DIM={LEARNED_USER_ENCODER_OUT_DIM} outside "
            "the sane range 1..2048."
        )
LEARNED_USER_ENCODER_HIDDEN = 64

# ── Layout table (raw-obs track -> per-peer column geometry) ────────────────
#
# The two raw-obs tracks disagree on where the user-token block starts and
# how its padded slots are masked:
#   - PADRE ("padre", the default): own_state is 4 wide; each user slot is
#     the USER_TOKEN_DIM(6)-wide token PLUS a separate TRAILING mask column
#     (USER_SLOT_WIDTH = 7). This is the original, pre-obs_layout geometry --
#     every function below must stay byte-identical for this key.
#   - OMNeT ("omnet"): own_state is 7 wide (see omnet_env.py's DRONE_TOKEN
#     column-layout comment: x_abs,y_abs,dx,dy,rssi,rssi_valid,nf). The two
#     ABSOLUTE normalised position columns come first so the actor has an
#     absolute frame, not only differences. This value
#     MUST equal omnet_env.OWN_STATE_W; the model side cannot import it.
#     Each user slot is
#     the OBS_FEATURES(13)-wide raw token AND NOTHING ELSE -- the token
#     already carries its own presence bit at col 12 (build_raw_user_tokens'
#     `present` column), so user_slot_width == user_token_dim there (no
#     extra trailing column to append).
# Adding a new raw-obs track later means adding one more entry here, not
# touching the "padre" one.
#
# `self_flag_col` is the index, in the REASSEMBLED token
# [own_state | pooled user summary | tail], of the column that is 1.0 on the
# observer's own row and 0.0 elsewhere. STModel uses it to give the actor an
# un-pooled path to its own token; None disables that path for the track.
#   omnet: own_state(7) + pooled user summary(8) + event_latent(8) = 23, which
#   is the same index in BOTH encoder modes -- frozen's token is
#   own_state(7) + user_latent(8) + event_latent(8) + self_flag + 3 (+mask) and
#   learned's reassembly restores exactly that geometry (see
#   reconstructed_input_dim). Keep in lockstep with omnet_env.build_drone_tokens'
#   `row[event_end] = 1.0 if j == i else 0.0`.
#   padre: None -- that track has no self column in its per-peer token.
OBS_LAYOUTS = {
    "padre": dict(
        own_state_width=OWN_STATE_WIDTH,
        user_token_dim=USER_TOKEN_DIM,
        user_slot_width=USER_SLOT_WIDTH,
        mask_in_token=False,  # mask is the slot's own last column
        mask_col=None,
        self_flag_col=None,
    ),
    "omnet": dict(
        own_state_width=7,  # == omnet_env.OWN_STATE_W; keep in lockstep
        # $OM2G_USER_PARTNER_TAG widens the per-user token 13 -> 17 env-side
        # (omnet_env.OBS_FEATURES, mirrored in om2g_track.OM2G_OBS_FEATURES).
        # THIS TABLE MUST FOLLOW: if the env emits 17-wide user tokens while
        # this says 13, `reassemble_with_learned_encoder` would slice the
        # user block at the wrong width. The exact-tail assert in
        # reassemble_with_learned_encoder guards this.
        user_token_dim=(17 if os.environ.get("OM2G_USER_PARTNER_TAG") else 13),
        user_slot_width=(17 if os.environ.get("OM2G_USER_PARTNER_TAG") else 13),
        mask_in_token=True,  # mask is WITHIN the token (col 12, `present`)
        mask_col=12,
        self_flag_col=7 + LEARNED_USER_ENCODER_OUT_DIM + 8,  # = 23
    ),
    # OM2G flat layout: NOT a per-peer row geometry
    # at all -- the obs is one flat vector, sized by the ARM, parsed by
    # om2g_flat_layout()/split_om2g_obs() below. This entry exists so model
    # configs can name it; the row-geometry helpers refuse it loudly.
    "om2g": dict(
        own_state_width=None, user_token_dim=None, user_slot_width=None,
        mask_in_token=None, mask_col=None, self_flag_col=None,
        flat=True,
    ),
}


# ── OM2G flat layout ────────────────────────────────────────────────────────
#
# The OM2G env does not emit the OMNET track's (MAX_N, W) full-width token
# matrix. Its per-agent observation is ONE flat vector,
#
#     [ own raw user block  max_users*F ][ MAX_N peer slots, each 20+L wide ]
#
# where the peer slot carries EXACTLY the arm's wire message (14+L floats)
# plus 6 observer-local columns -- see om2g/om2g_env.py's column table. The
# raw block never crosses a radio (own sensing), so it stays full-width in
# every arm; the peer slots are as big as the information that was paid for:
#     compressed    L = $OM2G_LATENT_DIM          (frozen codec output)
#     uncompressed  L = max_users * F             (identity codec)
#     nocontrol     L = 0                         (no peer content exists)
#
# Slot columns (mirror om2g_env.py -- the env asserts its side, this module
# asserts this side, and a drift crashes in split_om2g_obs, never re-slices):
#   0 x_abs, 1 y_abs, 2 dx, 3 dy, 4 rssi, 5 rssi_valid, 6 nf,
#   7:7+L user summary, 7+L:15+L event latent, 15+L SELF FLAG,
#   16+L fwd, 17+L busy, 18+L t_rx, 19+L MASK (last, as everywhere).
#
# There is deliberately NO model-side peer user encoder in this layout: the
# summary was computed sender-side before transmission, so the receiver only
# sees what was sent. The model's only user encoder is the LOCAL one over the
# observer's own raw block.

def om2g_flat_layout() -> dict:
    """Recompute the env's arm-sized layout from the same env vars.

    Model processes cannot import om2g_env (torchrl worker layering), so the
    geometry is re-derived from the identical sources and cross-checked at
    runtime by split_om2g_obs's width assert -- a disagreement crashes with
    both sides' numbers in the message instead of re-slicing.
    """
    if os.environ.get("OM2G_TRACK") != "1":
        raise RuntimeError(
            "obs_layout='om2g' requires OM2G_TRACK=1 -- this layout exists "
            "only for the OM2G track; an OMNET run must use 'omnet'.")
    arm = os.environ.get("OM2G_ARM")
    if arm not in ("compressed", "uncompressed", "nocontrol"):
        raise RuntimeError(
            f"obs_layout='om2g' needs $OM2G_ARM in compressed/uncompressed/"
            f"nocontrol to size the peer slots, got {arm!r}. om2g_config."
            f"om2g_env_vars() exports it; a hand launch must too.")
    feat = 17 if os.environ.get("OM2G_USER_PARTNER_TAG") else 13
    max_users_env = os.environ.get("OMNET_MAX_USERS_CAP")
    if max_users_env is None:
        raise RuntimeError(
            "obs_layout='om2g' needs $OMNET_MAX_USERS_CAP (om2g runs set it "
            "to om2g_track.OM2G_MAX_USERS_CAP). Refusing to default to the "
            "OMNET track's 25.")
    max_users = int(max_users_env)
    max_n_env = os.environ.get("OMNET_NUM_DRONES")
    if max_n_env is None:
        raise RuntimeError(
            "obs_layout='om2g' needs $OMNET_NUM_DRONES (the drone-slot "
            "count omnet_env.MAX_N is sized from; om2g_config."
            "om2g_env_vars()/config_knobs export it).")
    max_n = int(max_n_env)
    if arm == "compressed":
        latent = int(os.environ.get("OM2G_LATENT_DIM", "8"))
    elif arm == "uncompressed":
        latent = max_users * feat
    else:
        latent = 0
    w_peer = 20 + latent
    raw_w = max_users * feat
    # Codec mode: "frozen" (default) -- the L-summary was
    # computed sender-side in the env and sits in the slot. "e2e" -- the slot
    # summary columns arrive ZERO and a DISTINCT trailing inbox section
    # carries, per peer, the sender's raw block from the delivered message's
    # rx step; the model computes the L floats itself with a shared trainable
    # encoder E (see om2g_fill_peer_summaries). Wire charge identical.
    codec_mode = os.environ.get("OM2G_CODEC_MODE", "frozen")
    # hand_* (content ablation): sender-side deterministic
    # codecs; model-side they are exactly the frozen layout (no inbox).
    # e2e_rt (round trip): the SENDER and the WIRE are
    # byte-for-byte "e2e" -- zeroed summary columns, the raw block in the
    # inbox, beaconBytes still 4L+56 -- and the RECEIVER expands the L wire
    # floats back to the constant width $OM2G_RT_WIDTH before any aggregator
    # sees them. L then varies the INFORMATION and not the network shape.
    if codec_mode not in ("frozen", "e2e", "e2e_rt", "hand_users",
                          "hand_agg", "hand_none"):
        raise RuntimeError(
            f"OM2G_CODEC_MODE must be 'frozen', 'e2e', 'e2e_rt' or "
            f"hand_users/hand_agg/hand_none, got {codec_mode!r}")
    if codec_mode in ("e2e", "e2e_rt") and arm != "compressed":
        raise RuntimeError(
            f"OM2G_CODEC_MODE={codec_mode} is only defined for "
            f"arm=compressed, got arm={arm!r}.")
    inbox_w = max_n * raw_w if codec_mode in ("e2e", "e2e_rt") else 0
    # RECEIVER-SIDE width. `latent` is what the WIRE pays for; every model-
    # facing size (peer-token width, self-flag index, ST/GAT/DeepSets
    # input_dim) reads token_summary_w instead. In every mode but e2e_rt the
    # two are equal, so nothing outside e2e_rt can move.
    if codec_mode == "e2e_rt":
        rt_width = int(os.environ.get("OM2G_RT_WIDTH", "128"))
        rt_hidden = int(os.environ.get("OM2G_RT_HIDDEN", "256"))
        if not (1 <= rt_width <= 2048):
            raise RuntimeError(
                f"OM2G_RT_WIDTH={rt_width} outside the sane range 1..2048.")
        if not (1 <= rt_hidden <= 4096):
            raise RuntimeError(
                f"OM2G_RT_HIDDEN={rt_hidden} outside the sane range 1..4096.")
        if latent > rt_width:
            raise RuntimeError(
                f"OM2G_LATENT_DIM={latent} exceeds OM2G_RT_WIDTH={rt_width}. "
                f"The round-trip codec masks E's {rt_width}-wide output down "
                f"to the L floats the radio is charged 4L+56 bytes for, so L "
                f"must fit inside it. Raise OM2G_RT_WIDTH for the WHOLE "
                f"family or the ladder stops being shape-constant.")
        token_summary_w = rt_width
    else:
        rt_width = rt_hidden = 0
        token_summary_w = latent
    return dict(
        arm=arm, latent=latent, user_feat=feat, max_users=max_users,
        max_n=max_n, raw_w=raw_w, w_peer=w_peer,
        user_mask_col=12,               # `present` inside the raw user token
        # MODEL-FACING indices. In e2e_rt the peer token the aggregator sees
        # is 20 + rt_width wide (the decoder's output replaces the L summary
        # columns); everywhere else token_summary_w == latent and these are
        # the historical values to the digit.
        self_flag_col=15 + token_summary_w,
        w_peer_model=20 + token_summary_w,
        token_summary_w=token_summary_w,
        rt_width=rt_width, rt_hidden=rt_hidden,
        # WIRE/OBSERVATION geometry. Untouched by e2e_rt: this is what the
        # env actually emits and what split_om2g_obs slices.
        slot_mask_col=w_peer - 1,       # last slot column, as everywhere
        codec_mode=codec_mode,
        inbox_w=inbox_w,
        flat_dim=raw_w + max_n * w_peer + inbox_w,
    )


def split_om2g_obs(x: torch.Tensor, lo: dict):
    """Flat om2g obs -> its sections, as a 5-tuple.

    x: (..., flat_dim). Returns
      raw_tokens:   (..., max_users, user_feat)      own sensing
      raw_mask:     (..., max_users)                 its `present` column
      peers:        (..., max_n, w_peer)             arm-sized peer slots
      inbox_tokens: (..., max_n, max_users, user_feat) | None   e2e only
      inbox_mask:   (..., max_n, max_users) | None              e2e only
    Width-asserted: a flat_dim mismatch is the env and the model disagreeing
    about the arm/latent/user population, and MUST be fatal here so the
    block is never re-sliced at the wrong width.
    """
    if x.shape[-1] != lo["flat_dim"]:
        raise RuntimeError(
            f"om2g obs is {x.shape[-1]} wide; model expects {lo['flat_dim']} "
            f"= raw {lo['raw_w']} + {lo['max_n']} slots x {lo['w_peer']} "
            f"(arm={lo['arm']}, L={lo['latent']}, users={lo['max_users']}x"
            f"{lo['user_feat']}). Env and model are configured for different "
            f"arms/latents/populations -- check OM2G_ARM, OM2G_LATENT_DIM, "
            f"OMNET_MAX_USERS_CAP, OMNET_NUM_DRONES, OM2G_USER_PARTNER_TAG.")
    raw = x[..., :lo["raw_w"]]
    raw_tokens = raw.reshape(*raw.shape[:-1], lo["max_users"], lo["user_feat"])
    raw_mask = raw_tokens[..., lo["user_mask_col"]]
    peer_end = lo["raw_w"] + lo["max_n"] * lo["w_peer"]
    peers = x[..., lo["raw_w"]:peer_end].reshape(
        *x.shape[:-1], lo["max_n"], lo["w_peer"])
    inbox_tokens = inbox_mask = None
    if lo["inbox_w"]:
        inbox_tokens = x[..., peer_end:].reshape(
            *x.shape[:-1], lo["max_n"], lo["max_users"], lo["user_feat"])
        inbox_mask = inbox_tokens[..., lo["user_mask_col"]]
    return raw_tokens, raw_mask, peers, inbox_tokens, inbox_mask


def om2g_fill_peer_summaries(peers, inbox_tokens, inbox_mask, encoder, lo):
    """e2e mode: z = E(inbox) -> written into every slot's L-summary columns.

    THE funnel: this is the only path from a peer's raw block to anything a
    policy head can read, and it is exactly lo["latent"] floats wide -- the
    width the wire was charged for. Gated by each slot's live-mask so a
    blanked (undelivered) row stays all-zero even for a mask-ignoring
    architecture (FFN flattens everything). The hook invariant test in
    om2g_compressfix_check.py proves at runtime that the actor's output
    depends on the inbox ONLY through `encoder`.
    """
    L = lo["latent"]
    W = lo["token_summary_w"]           # == L except in e2e_rt
    z = encoder(inbox_tokens, inbox_mask)                # (..., max_n, W)
    if z.shape[-1] != W:
        raise RuntimeError(
            f"message encoder emitted {z.shape[-1]} floats, the peer token's "
            f"summary block is {W} wide (codec_mode={lo['codec_mode']}, wire "
            f"L={L}) -- encoder out_dim out of lockstep.")
    gate = peers[..., lo["slot_mask_col"]:lo["slot_mask_col"] + 1]
    z = z * gate
    # Slice widths: the OBSERVATION slot is 20+L (7 head | L summary | 8
    # event | self,fwd,busy,trx | mask), so peers[..., 7+L:] is always the
    # 13-column tail and the rebuilt token is 7 + W + 13 = 20 + W wide. The
    # self flag lands at 15 + W and the mask stays last -- which is why
    # self_flag_col above is 15 + token_summary_w.
    return torch.cat([peers[..., :7], z, peers[..., 7 + L:]], dim=-1)



class OM2GRoundTripCodec(nn.Module):
    """OM2G_CODEC_MODE=e2e_rt: compress to L on the wire, DECOMPRESS on
    arrival to a width that does not depend on L.

        raw inbox block --E--> R^W --mask_L--> L live floats --D--> R^W

    * E is exactly the e2e message encoder except that its output width is
      the CONSTANT rt_width W, not L.
    * mask_L is a non-trainable 0/1 buffer keeping coordinates [0, L). It is
      the entire bottleneck: coordinates >= L are multiplied by zero, so
      neither information nor gradient passes them. What survives has exactly
      L degrees of freedom -- the L floats the radio was charged 4L+56 bytes
      for (om2g_config.py:165; that path never reads OM2G_CODEC_MODE).
    * D is a fixed-shape MLP W -> H -> H -> W. Its input is the ZERO-PADDED
      wire vector, which is precisely what a real receiver holds.

    Every parameter shape here is independent of L. Consequences worth
    stating because they are what makes the ladder clean:
      (i)   the model's parameter count is EXACTLY constant across the ladder
            -- capacity does not vary with L at all;
      (ii)  at a fixed seed, the L=8 and L=128 runs start from bit-identical
            weights (nothing L-dependent consumes the RNG), so the ladder is
            a paired design;
      (iii) no optimizer state or checkpoint shape depends on L.
    E.to_latent's rows >= L are dead: multiplied by 0 in the forward, so
    weight decay or any other optimizer effect on them cannot reach the
    policy no matter what the trainer does.

    Gating: the caller (om2g_fill_peer_summaries) zeroes the whole summary
    block for a blanked/undelivered peer AFTER this module runs, which is
    what keeps D's bias from leaking a phantom peer into a mask-ignoring
    aggregator. Do not move that gate before D.
    """

    def __init__(self, *, dim: int, hidden: int, max_users: int,
                 latent: int, rt_width: int, rt_hidden: int):
        super().__init__()
        if not 1 <= latent <= rt_width:
            raise ValueError(
                f"round-trip codec needs 1 <= L <= rt_width, got L={latent}, "
                f"rt_width={rt_width}")
        self.latent = int(latent)
        self.rt_width = int(rt_width)
        self.rt_hidden = int(rt_hidden)
        self.out_dim = int(rt_width)          # what the peer token receives
        self.bottleneck_dim = int(latent)     # what the WIRE pays for
        self.max_users = int(max_users)
        self.encoder = LearnableUserSetEncoder(
            dim=dim, hidden=hidden, out_dim=rt_width)
        _m = torch.zeros(rt_width)
        _m[:latent] = 1.0
        self.register_buffer("wire_mask", _m, persistent=True)
        self.decoder = nn.Sequential(
            nn.Linear(rt_width, rt_hidden), nn.ReLU(),
            nn.Linear(rt_hidden, rt_hidden), nn.ReLU(),
            nn.Linear(rt_hidden, rt_width),
        )
        # Last wire vector, for probes/telemetry and for the self-test.
        # OFF by default: a training minibatch makes this (B, n, n, W), which
        # is ~200 MB at B=6000, and it would be held between forwards. Set
        # $OM2G_RT_KEEP_BOTTLENECK=1 to keep it (always detached -- holding
        # the autograd graph across steps would leak outright).
        self.bottleneck = None
        self._keep_bottleneck = bool(
            int(os.environ.get("OM2G_RT_KEEP_BOTTLENECK", "0")))
        # Value-assert the bottleneck for the first N forwards of each
        # process (0 disables, a negative number checks forever). A shape
        # assert cannot catch a leak; this one reads the actual floats.
        self._checks_left = int(os.environ.get("OM2G_RT_ASSERT_STEPS", "64"))

    def forward(self, tokens: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        z = self.encoder(tokens, mask) * self.wire_mask
        if self._checks_left != 0:
            if self._checks_left > 0:
                self._checks_left -= 1
            tail = z[..., self.latent:]
            if tail.numel() and float(tail.detach().abs().max()) != 0.0:
                raise RuntimeError(
                    f"e2e_rt bottleneck leaked past coordinate L="
                    f"{self.latent}: max |z[L:]| = "
                    f"{float(tail.detach().abs().max()):.3e}. More than the "
                    f"wire was charged for reached the decoder.")
        if self._keep_bottleneck:
            self.bottleneck = z.detach()
        return self.decoder(z)


def make_om2g_peer_msg_encoder(lo: dict) -> nn.Module:
    """THE model-side message encoder, for the codec modes that have one.

      e2e     -> LearnableUserSetEncoder, out_dim = L   (unchanged behaviour)
      e2e_rt  -> OM2GRoundTripCodec: E to a constant width, masked to L, D
                 back out to that same constant width

    Both are a single nn.Module whose forward is (tokens, mask) -> summary,
    so the C1 hook-invariant test (om2g_compressfix_check.py run_hook) keeps
    working unchanged: hooking this module still intercepts the ONLY path
    from the inbox to the action.
    """
    from MANET.frozen_encoder import make_user_set_encoder
    if lo["codec_mode"] == "e2e_rt":
        return OM2GRoundTripCodec(
            dim=lo["user_feat"],
            hidden=LEARNED_USER_ENCODER_HIDDEN,
            max_users=lo["max_users"],
            latent=lo["latent"],
            rt_width=lo["rt_width"],
            rt_hidden=lo["rt_hidden"],
        )
    if lo["codec_mode"] != "e2e":
        raise RuntimeError(
            f"make_om2g_peer_msg_encoder called for codec_mode="
            f"{lo['codec_mode']!r}, which has no model-side encoder.")
    return make_user_set_encoder(
        dim=lo["user_feat"],
        hidden=LEARNED_USER_ENCODER_HIDDEN,
        out_dim=lo["latent"],
        max_users=lo["max_users"],
        kind="learned",
    )


def _layout(obs_layout: str) -> dict:
    if obs_layout not in OBS_LAYOUTS:
        raise ValueError(
            f"obs_layout must be one of {sorted(OBS_LAYOUTS)}, got {obs_layout!r}"
        )
    p = OBS_LAYOUTS[obs_layout]
    if p.get("flat"):
        raise ValueError(
            f"obs_layout={obs_layout!r} is the flat OM2G layout; it has no "
            f"per-peer row geometry. Use om2g_flat_layout()/split_om2g_obs() "
            f"-- reassemble_with_learned_encoder/reconstructed_input_dim do "
            f"not apply (the peer summary is computed sender-side).")
    return p


def reconstructed_input_dim(
    raw_input_dim: int, max_users: int, out_dim: int, obs_layout: str = "padre"
) -> int:
    """Level-2 aggregator input width AFTER learned-mode reassembly.

    raw_input_dim: the per-peer token width the model actually receives in
        learned mode, EXCLUDING the trailing peer-mask column (i.e. after
        stripping `use_mask`'s last column, if any) -- own_state +
        max_users*user_slot_width + tail.
    obs_layout: selects the own_state/user-slot column geometry (see
        OBS_LAYOUTS above). Default "padre" reproduces the original,
        obs_layout-less formula byte-for-byte.
    Returns own_state + the encoder's pooled out_dim + tail, i.e. the same
    width the model would see in encoder_mode=="frozen" with an out_dim-wide
    pooled summary in place of the user block.
    """
    p = _layout(obs_layout)
    user_block_width = max_users * p["user_slot_width"]
    tail_width = raw_input_dim - p["own_state_width"] - user_block_width
    assert tail_width >= 0, (
        f"max_users={max_users} implies a user block of {user_block_width} "
        f"cols, leaving {tail_width} for the tail from a {raw_input_dim}-col "
        f"token (after stripping the peer-mask column, if any) -- check the "
        f"task's numbOfUsers upper bound matches this model's max_users "
        f"(obs_layout={obs_layout!r})."
    )
    # EXACT tail for the OMNeT track. `tail_width >= 0` above is necessary but
    # not sufficient: a 17-wide user token parsed against a 13-wide slot table
    # leaves tail_width=172 instead of 12, which would route raw user tokens
    # around the compressor. A width mismatch must be fatal here, not absorbed
    # by the tail.
    if obs_layout == "omnet":
        expected_tail = 12  # event latent 8 + self flag + fwd + busy + t_rx
        assert tail_width == expected_tail, (
            f"omnet tail_width={tail_width}, expected {expected_tail}. The "
            f"per-user token width and this model's user_slot_width="
            f"{p['user_slot_width']} are out of lockstep (raw_input_dim="
            f"{raw_input_dim}, max_users={max_users}). If $OM2G_USER_PARTNER_TAG "
            f"is set env-side it must also be set for the model process."
        )
    return p["own_state_width"] + out_dim + tail_width


def reassemble_with_learned_encoder(
    x: torch.Tensor,
    user_encoder: LearnableUserSetEncoder,
    max_users: int,
    obs_layout: str = "padre",
) -> torch.Tensor:
    """Slice x's [own_state | padded user block | tail] layout (last dim),
    replace the padded user block with the trainable encoder's pooled
    summary, and reassemble -- so the result matches encoder_mode=="frozen"
    layout exactly ([own_state | pooled summary | tail]).

    x: (..., raw_input_dim) with raw_input_dim = own_state_width +
       max_users * user_slot_width + tail_width (widths from OBS_LAYOUTS[
       obs_layout]). Any number of leading dims; only the last dim is
       touched, so this can run before or after folding batch/agent/peer
       dims.
    obs_layout: "padre" (default, byte-identical to the pre-obs_layout
       behaviour) | "omnet" -- see OBS_LAYOUTS above.
    """
    p = _layout(obs_layout)
    own_state_width = p["own_state_width"]
    user_token_dim = p["user_token_dim"]
    user_slot_width = p["user_slot_width"]
    user_block_width = max_users * user_slot_width

    own_state = x[..., :own_state_width]
    user_flat = x[..., own_state_width : own_state_width + user_block_width]
    tail = x[..., own_state_width + user_block_width :]

    user_slots = user_flat.reshape(*user_flat.shape[:-1], max_users, user_slot_width)
    user_tokens = user_slots[..., :user_token_dim]
    if p["mask_in_token"]:
        # OMNeT: the token carries its own presence bit -- no separate
        # trailing mask column exists (user_slot_width == user_token_dim).
        user_mask = user_slots[..., p["mask_col"]]
    else:
        # PADRE: a dedicated trailing mask column after the token.
        user_mask = user_slots[..., user_token_dim]

    user_summary = user_encoder(user_tokens, user_mask)  # (..., out_dim)
    return torch.cat([own_state, user_summary, tail], dim=-1)
