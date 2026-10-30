"""BenchMARL Set Transformer actor model for PADRE.

Ports the reviewed-good STEncoder + PMA from architectures/ST.py into
the BenchMARL Model/ModelConfig interface so it can be selected alongside
the MLP baseline in the FFN/ST/GNN comparison.

Architecture preserved verbatim from ST.py:
  - pre-LN TransformerEncoderLayer (norm_first=True)
  - latent_dim = 64
  - PMA pooling with one learned seed
  - 1.0=real / <0.5=pad-out mask convention

Only the SB3 wrapper and CUDA-required guard are absent (not needed here).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict import TensorDictBase
from benchmarl.models.common import Model, ModelConfig

from MANET.frozen_encoder import (LearnableUserSetEncoder,
                                  make_user_set_encoder)
import os

from architectures.user_encoder import (
    reassemble_with_learned_encoder,
    reconstructed_input_dim,
    LEARNED_USER_ENCODER_OUT_DIM,
    LEARNED_USER_ENCODER_HIDDEN,
    OBS_LAYOUTS,
    om2g_flat_layout,
    split_om2g_obs,
    om2g_fill_peer_summaries,
    make_om2g_peer_msg_encoder,
)

# ── Un-pooled OWN user block ───────────────────────────────────────────────
# The 40:1 user sum-pool models the COMMUNICATION constraint: what a drone can
# put into an 802.15.4 state message for its peers. A drone's OWN user
# observations are readings from its own antenna that are never transmitted
# and cost no airtime, so they need not go through that pool.
#
# So: peers still get LEARNED_USER_ENCODER_OUT_DIM (the message, still charged
# at its on-air size), while the observer additionally gets a wider encoding of
# ITS OWN raw user block, concatenated to the action head beside the own-token
# path.
#
# Opt-in: existing checkpoints load unchanged when this is off, and turning it
# on changes parameter shapes, so checkpoints of the two settings are not
# interchangeable.
SELF_USERS_UNPOOLED = os.environ.get("OM2G_SELF_USERS_UNPOOLED", "0") == "1"
SELF_USER_OUT_DIM = int(os.environ.get("OM2G_SELF_USER_OUT_DIM", "64"))


class PMA(nn.Module):
    """Pooling-by-Multihead-Attention — verbatim from architectures/ST.py.

    A learned seed attends to the encoder output to produce a fixed-size
    representation. Forwards the key_padding_mask so masked tokens don't
    contribute to the pooled vector.
    """

    def __init__(self, d_model: int, num_heads: int, num_seeds: int = 1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, num_seeds, d_model))
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        batch_size = x.shape[0]
        seeds = self.seeds.expand(batch_size, -1, -1)
        pooled, _ = self.attention(seeds, x, x, key_padding_mask=key_padding_mask)
        return self.norm(pooled + seeds).squeeze(1)


class _STEncoderCore(nn.Module):
    """inputLayer → pre-LN TransformerEncoder → PMA.

    Stripped of SB3 and CUDA dependencies; architecture is otherwise
    identical to STEncoder in ST.py.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,  # pre-LN, same as STEncoder
        )
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pooling = PMA(latent_dim, num_heads=n_heads)

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        # x: (B, n_tokens, input_dim) → (B, latent_dim)
        x = self.input_layer(x)
        x = self.attention(x, src_key_padding_mask=key_padding_mask)
        return self.pooling(x, key_padding_mask=key_padding_mask)


class STModel(Model):
    """BenchMARL actor backed by the Set Transformer encoder.

    Input:  (*batch, n_agents, n_peers, token_features)
    Output: (*batch, n_agents, output_features)

    share_params=True only — n_agents is folded into the batch dimension so
    a single shared encoder processes all agents in one forward pass.
    """

    def __init__(self, **kwargs):
        self.latent_dim: int = kwargs.pop("latent_dim")
        self.n_heads: int = kwargs.pop("n_heads")
        self.n_layers: int = kwargs.pop("n_layers")
        self.dim_feedforward: int = kwargs.pop("dim_feedforward")
        self.dropout: float = kwargs.pop("dropout")
        self.use_mask: bool = kwargs.pop("use_mask")
        self.encoder_mode: str = kwargs.pop("encoder_mode", "frozen")
        self.max_users: int = kwargs.pop("max_users", 20)
        self.obs_layout: str = kwargs.pop("obs_layout", "padre")
        self.use_self_token: bool = kwargs.pop("use_self_token", True)
        assert self.encoder_mode in ("frozen", "learned"), (
            f"encoder_mode must be 'frozen' or 'learned', got {self.encoder_mode!r}"
        )
        assert self.obs_layout in OBS_LAYOUTS, (
            f"obs_layout must be one of {sorted(OBS_LAYOUTS)}, got {self.obs_layout!r}"
        )

        super().__init__(
            input_spec=kwargs.pop("input_spec"),
            output_spec=kwargs.pop("output_spec"),
            agent_group=kwargs.pop("agent_group"),
            input_has_agent_dim=kwargs.pop("input_has_agent_dim"),
            n_agents=kwargs.pop("n_agents"),
            centralised=kwargs.pop("centralised"),
            share_params=kwargs.pop("share_params"),
            device=kwargs.pop("device"),
            action_spec=kwargs.pop("action_spec"),
            model_index=kwargs.pop("model_index"),
            is_critic=kwargs.pop("is_critic"),
        )

        # input_leaf_spec.shape = (n_agents, n_peers, token_features)
        token_features: int = self.input_leaf_spec.shape[-1]
        raw_input_dim = token_features - (1 if self.use_mask else 0)

        # output_leaf_spec.shape = (n_agents, output_features)
        self.output_features: int = self.output_leaf_spec.shape[-1]

        self._om2g = None
        self._peer_msg_encoder = None
        if self.obs_layout == "om2g":
            # OM2G flat layout: peer slots arrive
            # ALREADY compressed to the arm's L (sender-side codec in the
            # env), so there is NO model-side peer user encoder. The only
            # user encoder here is the LOCAL one over the observer's own raw
            # block (built in the self-users section below, always on).
            self._om2g = om2g_flat_layout()
            if not self.use_mask:
                # Not configurable on this layout: every peer slot carries
                # its live-mask as its own last column, and ignoring it would
                # feed all-zero phantom peers to the pool. Forced, and
                # announced with a print.
                print(f"[{type(self).__name__}] obs_layout=om2g forces "
                      f"use_mask=True (was False)", flush=True)
                self.use_mask = True
            if token_features != self._om2g["flat_dim"]:
                raise RuntimeError(
                    f"STModel: obs spec is {token_features} wide, om2g "
                    f"layout says {self._om2g['flat_dim']} "
                    f"(arm={self._om2g['arm']}, L={self._om2g['latent']}). "
                    f"Env and model disagree -- check OM2G_ARM/"
                    f"OM2G_LATENT_DIM/OMNET_MAX_USERS_CAP/OMNET_NUM_DRONES.")
            self._user_encoder = None
            if self._om2g["codec_mode"] in ("e2e", "e2e_rt"):
                # THE message encoder E (shared across drones by
                # parameter sharing): raw inbox block -> the L floats
                # the wire was charged for. kind='learned' PINNED --
                # e2e exists only for the compressed arm.
                # e2e_rt wraps E with a fixed-width DECODER D so the token
                # below is the same shape at every L (round-trip arm).
                # The wire is still exactly L floats: see
                # OM2GRoundTripCodec's mask.
                self._peer_msg_encoder = make_om2g_peer_msg_encoder(self._om2g)
            # w_peer_model, NOT w_peer: in e2e_rt the decoder makes
            # this width CONSTANT in L (19 + OM2G_RT_WIDTH). In
            # every other mode it is the historical 19 + L.
            input_dim = self._om2g["w_peer_model"] - 1
        elif self.encoder_mode == "learned":
            # OM2G: the factory picks the arm's encoder from $OM2G_ENCODER.
            # Unset (every OMNET run) returns LearnableUserSetEncoder with
            # exactly these arguments, so this is a no-op for that track.
            #   "learned"     -> compressor arm      (D = 8)
            #   "passthrough" -> no-compressor arm   (D = max_users * 13 = 325)
            self._user_encoder = make_user_set_encoder(
                dim=OBS_LAYOUTS[self.obs_layout]["user_token_dim"],
                hidden=LEARNED_USER_ENCODER_HIDDEN,
                out_dim=LEARNED_USER_ENCODER_OUT_DIM,
                max_users=self.max_users,
            )
            input_dim = reconstructed_input_dim(
                raw_input_dim, self.max_users, LEARNED_USER_ENCODER_OUT_DIM,
                obs_layout=self.obs_layout,
            )
        else:
            self._user_encoder = None       # frozen: no encoder submodule
            input_dim = raw_input_dim

        self._encoder = _STEncoderCore(
            input_dim=input_dim,
            latent_dim=self.latent_dim,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )

        # Un-pooled path to the observer's OWN token. See _forward.
        # Disabled (self_flag_col None) for tracks whose per-peer token has no
        # self column, and by use_self_token=False for A/B ablation.
        if self._om2g is not None:
            self._self_flag_col = self._om2g["self_flag_col"]   # 15 + L
        else:
            self._self_flag_col = OBS_LAYOUTS[self.obs_layout]["self_flag_col"]
        self._self_token_active = bool(self.use_self_token
                                       and self._self_flag_col is not None)
        if self._self_token_active:
            if not 0 <= self._self_flag_col < input_dim:
                raise ValueError(
                    f"obs_layout={self.obs_layout!r} puts self_flag_col at "
                    f"{self._self_flag_col}, outside the {input_dim}-wide "
                    f"reassembled token — OBS_LAYOUTS is out of lockstep with "
                    f"omnet_env.build_drone_tokens.")
            self._own_proj = nn.Linear(input_dim, self.latent_dim)
            head_in = 2 * self.latent_dim
        else:
            self._own_proj = None
            head_in = self.latent_dim

        # Un-pooled OWN user block -- see SELF_USERS_UNPOOLED above. On the
        # om2g flat layout this path is STRUCTURAL, not opt-in: the own raw
        # block rides in the flat section precisely so the drone's knowledge
        # of its OWN users never depends on the arm's wire latent L. On the
        # omnet layout it requires the learned encoder path (that is where
        # the raw block is still present) and the self column.
        if self._om2g is not None:
            self._self_users_active = True
        else:
            self._self_users_active = bool(
                SELF_USERS_UNPOOLED and self.encoder_mode == "learned"
                and self._self_token_active)
        if self._self_users_active:
            # kind="learned" is PINNED, not left to $OM2G_ENCODER. The factory
            # otherwise returns PassthroughUserSetEncoder on the uncompressed
            # arm, which ignores out_dim and emits max_users*13=520 --
            # changing this path's width and the head's input size depending on
            # which arm is training. The self path is a local encoder, not the
            # transmitted message, so it does not follow the arm.
            self._self_user_encoder = make_user_set_encoder(
                dim=(self._om2g["user_feat"] if self._om2g is not None
                     else OBS_LAYOUTS[self.obs_layout]["user_token_dim"]),
                hidden=LEARNED_USER_ENCODER_HIDDEN,
                out_dim=SELF_USER_OUT_DIM,
                max_users=self.max_users,
                kind="learned",
            )
            self._self_user_proj = nn.Linear(SELF_USER_OUT_DIM, self.latent_dim)
            head_in += self.latent_dim
        else:
            self._self_user_encoder = None
            self._self_user_proj = None

        # Print which regime this model is in. $OM2G_SELF_USERS_UNPOOLED
        # is read at import time and appears in NO hydra config or checkpoint,
        # so this line is how a run's own log distinguishes the two
        # architectures. Printed once per model build.
        if not self.is_critic:
            _peer_dim = (self._om2g["latent"] if self._om2g is not None
                         else LEARNED_USER_ENCODER_OUT_DIM)
            _src = ""
            if self._om2g is not None:
                if self._om2g["codec_mode"] == "e2e_rt":
                    _src = (f" (model-side E, e2e_rt: wire L="
                            f"{self._om2g['latent']} floats -> D -> token "
                            f"summary {self._om2g['token_summary_w']}, "
                            f"aggregator input_dim="
                            f"{self._om2g['w_peer_model'] - 1})")
                elif self._om2g["codec_mode"] == "e2e":
                    _src = " (model-side E, e2e)"
                else:
                    _src = " (sender-side codec)"
            print(f"[st_model] layout={self.obs_layout}"
                  f"  self_users_unpooled="
                  f"{'ON' if self._self_users_active else 'off'}"
                  f"  peer_user_dim={_peer_dim}"
                  f"{_src}"
                  f"  self_user_dim={SELF_USER_OUT_DIM if self._self_users_active else 0}"
                  f"  head_in={head_in}", flush=True)
        self._output_proj = nn.Linear(head_in, self.output_features)

        # Small-init the POLICY head (loc rows only). A default-init linear
        # gives |loc| ~ 0.13, which a velocity action space integrates into
        # ~1.4 km of drift over a 250-step episode, which parks drones at the
        # walls. Standard PPO practice is gain ~0.01 on the policy head.
        # Scale rows are left alone: sigma has three interacting knobs here.
        # Pin sigma_0: with default init of the SCALE rows of _output_proj,
        # sigma_0 depends on the seed. Shrinking them makes
        # biased_softplus emit its nominal sigma for every seed. Init-only.
        _scale_init = float(os.environ.get("OM2G_ACTOR_SCALE_INIT", "0") or 0)
        if _scale_init > 0.0 and not self.is_critic:
            _h = self.output_features // 2
            if self.output_features % 2 == 0 and _h > 0:
                with torch.no_grad():
                    self._output_proj.weight[_h:].mul_(_scale_init)
                    if self._output_proj.bias is not None:
                        # 0.0 -> sigma_0 ~ 1.0; 0.44 -> ~1.3 (the value the one
                        # seed that ever escaped happened to draw).
                        _sb = float(os.environ.get("OM2G_ACTOR_SCALE_BIAS", "0") or 0)
                        self._output_proj.bias[_h:].fill_(_sb)
                print(f"[st_model] actor SCALE-head pinned, gain={_scale_init} "
                      f"(rows {_h}:{self.output_features}) bias={_sb} "
                      f"-- sigma_0 seed-independent", flush=True)

        _init_gain = float(os.environ.get("OM2G_ACTOR_INIT_GAIN", "0") or 0)
        if _init_gain > 0.0 and not self.is_critic:
            _half = self.output_features // 2
            if self.output_features % 2 == 0 and _half > 0:
                with torch.no_grad():
                    self._output_proj.weight[:_half].mul_(_init_gain)
                    if self._output_proj.bias is not None:
                        self._output_proj.bias[:_half].zero_()
                print(f"[st_model] actor loc-head small-init gain={_init_gain} "
                      f"(rows 0:{_half} of {self.output_features})", flush=True)
            else:
                print(f"[st_model] OM2G_ACTOR_INIT_GAIN set but output_features="
                      f"{self.output_features} is not an even loc/scale split "
                      f"-- NOT applied", flush=True)

        if self.device is not None:
            self._encoder = self._encoder.to(self.device)
            self._output_proj = self._output_proj.to(self.device)
            if self._own_proj is not None:
                self._own_proj = self._own_proj.to(self.device)
            if self._user_encoder is not None:
                self._user_encoder = self._user_encoder.to(self.device)
            if self._peer_msg_encoder is not None:
                self._peer_msg_encoder = self._peer_msg_encoder.to(self.device)
            if self._self_users_active:
                self._self_user_encoder = self._self_user_encoder.to(self.device)
                self._self_user_proj = self._self_user_proj.to(self.device)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        x = tensordict.get(self.in_key)
        # omnet/padre: x is (*batch, n_agents, n_peers, token_features).
        # om2g: x is (*batch, n_agents, flat_dim) -- split it into the own
        # raw user block and the arm-sized peer-slot matrix FIRST, so the
        # code below sees the same 4-D shape either way.

        own_user_summary = None
        if self._om2g is not None:
            raw_toks, raw_mask, x, inbox_toks, inbox_mask = \
                split_om2g_obs(x, self._om2g)
            if inbox_toks is not None:   # e2e: fill slots via E
                x = om2g_fill_peer_summaries(
                    x, inbox_toks, inbox_mask,
                    self._peer_msg_encoder, self._om2g)
            # LOCAL encoding of the drone's OWN raw sensing (never on the
            # wire, arm-independent). Peer slots arrive pre-compressed.
            own_user_summary = self._self_user_encoder(raw_toks, raw_mask)
            raw_body = None
        elif self.encoder_mode == "learned":
            # Slice out the padded raw user-token block, pool it with our
            # own trainable encoder, and reassemble -- everything below
            # this is then byte-identical to encoder_mode=="frozen".
            if self.use_mask:
                peer_mask_col = x[..., -1:]
                body = x[..., :-1]
            else:
                peer_mask_col = None
                body = x
            raw_body = body if self._self_users_active else None
            body = reassemble_with_learned_encoder(
                body, self._user_encoder, self.max_users, obs_layout=self.obs_layout
            )
            x = torch.cat([body, peer_mask_col], dim=-1) if peer_mask_col is not None else body
        else:
            raw_body = None

        leading = x.shape[:-3]
        n_agents = x.shape[-3]
        n_peers = x.shape[-2]
        token_features = x.shape[-1]

        # Fold all leading dims + n_agents into one batch dim
        x = x.reshape(-1, n_peers, token_features)  # (B*n_agents, n_peers, token_features)
        if raw_body is not None:
            # Same fold, so a row index into x is the same row in raw_body.
            raw_body = raw_body.reshape(-1, n_peers, raw_body.shape[-1])

        if self.use_mask:
            # 1.0=real, <0.5=pad-out — same convention as STEncoder
            mask_col = x[..., -1]
            x = x[..., :-1]
            key_padding_mask = mask_col < 0.5
            # Detect fully-masked rows (absent agents). All peers padded → NaN
            # from softmax(-inf,...,-inf). Standard workaround (pytorch#24816):
            # unmask one position to avoid NaN, then zero the output post-hoc.
            fully_masked = key_padding_mask.all(dim=-1)  # (B*n_agents,)
            if fully_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[fully_masked, 0] = False
        else:
            key_padding_mask = None
            fully_masked = None

        latent = self._encoder(x, key_padding_mask=key_padding_mask)  # (B*n_agents, latent_dim)

        if self._self_token_active:
            # Hand the action head the observer's OWN token directly,
            # bypassing the pool.
            #
            # WHY. The peer tokens are pooled by PMA, which attends with ONE
            # CONSTANT learned seed. When that attention is near uniform
            # (measured: 0.1946 on the own token vs 0.2014 on the
            # other four, against 1/5 = 0.2000), under a mean pool
            #   pool_i = (1/N)[f(t_i,flag=1) - f(t_i,flag=0)]
            #            + (1/N) sum_j f(t_j,flag=0)
            # and the second term is IDENTICAL for every drone. The self_flag
            # already in the token therefore reached the action only as a
            # 1/N-amplitude perturbation on top of an N-times-larger
            # common-mode term, recoverable only as a difference. Attention
            # could not break the tie either: 332 of the 345 raw columns are
            # byte-identical across observers, so the keys are near-identical.
            #
            # The flag is one-hot over the peer axis by construction
            # (omnet_env.build_drone_tokens: row[event_end] = 1.0 iff j == i,
            # and the j == i row is never link-masked), so argmax selects it.
            # A fully-masked agent has an all-zero token block and argmax
            # returns 0; `out` is zeroed for those rows below regardless.
            self_idx = x[..., self._self_flag_col].argmax(dim=-1)      # (B*n_agents,)
            rows = torch.arange(x.shape[0], device=x.device)
            own_tok = x[rows, self_idx]
            latent = torch.cat([latent, self._own_proj(own_tok)], dim=-1)

            if raw_body is not None:
                # The observer's OWN raw user block, never pooled. Peer tokens
                # above are untouched and still carry only the transmitted
                # LEARNED_USER_ENCODER_OUT_DIM summary, so message size and
                # airtime are unchanged. See SELF_USERS_UNPOOLED.
                p = OBS_LAYOUTS[self.obs_layout]
                osw, usw = p["own_state_width"], p["user_slot_width"]
                blk = raw_body[rows, self_idx][..., osw:osw + self.max_users * usw]
                slots = blk.reshape(blk.shape[0], self.max_users, usw)
                toks = slots[..., :p["user_token_dim"]]
                mask = (slots[..., p["mask_col"]] if p["mask_in_token"]
                        else slots[..., p["user_token_dim"]])
                latent = torch.cat(
                    [latent, self._self_user_proj(
                        self._self_user_encoder(toks, mask))], dim=-1)

        if own_user_summary is not None:
            # om2g: the own-raw-block path, folded to the same row order as
            # the peer fold above ((*leading, n_agents) -> rows).
            s = own_user_summary.reshape(-1, own_user_summary.shape[-1])
            latent = torch.cat([latent, self._self_user_proj(s)], dim=-1)

        out = self._output_proj(latent)                                 # (B*n_agents, output_features)
        if fully_masked is not None and fully_masked.any():
            out = torch.where(fully_masked.unsqueeze(-1), torch.zeros_like(out), out)
        out = out.reshape(*leading, n_agents, self.output_features)

        tensordict.set(self.out_key, out)
        return tensordict


@dataclass
class STConfig(ModelConfig):
    """Dataclass config for :class:`STModel`."""

    latent_dim: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    use_mask: bool = False
    # "frozen" (default): per-peer tokens already carry a pooled user
    # summary (byte-identical to pre-encoder_mode behaviour). "learned":
    # tokens instead carry a padded raw user-token block (max_users wide),
    # pooled here by this model's OWN trainable encoder submodule.
    encoder_mode: str = "frozen"
    max_users: int = 20  # only consulted when encoder_mode=="learned"
    # "padre" (default, byte-identical): own_state=4, 6-col user token + a
    # trailing mask column. "omnet": own_state=5, 13-col user token with its
    # presence bit AT col 12 (no extra mask column) -- see
    # architectures/user_encoder.py's OBS_LAYOUTS. Only consulted when
    # encoder_mode=="learned".
    obs_layout: str = "padre"
    # Concatenate the observer's OWN token to the pooled peer vector
    # before the action head, so identity does not have to survive a
    # permutation-invariant pool at 1/N amplitude. No effect on tracks whose
    # OBS_LAYOUTS entry has self_flag_col=None (currently "padre"). Set False
    # for the architecture without the own-token path (for an A/B).
    use_self_token: bool = True

    @staticmethod
    def associated_class():
        return STModel
