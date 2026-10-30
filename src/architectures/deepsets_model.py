"""
Deep Sets model for variable-N MANET MAPPO actor.

Sibling to st_model.py. Same BenchMARL Model contract, same input shape,
same absent-agent handling. Differs in the aggregation:
  ST        -> attention over peers (every token attends to every other)
  DeepSets  -> shared MLP per token, sum-pool over peers

Story: ST = permutation-invariance with attention. DeepSets = permutation-
invariance without attention. The pair isolates whether attention is doing
real work beyond the invariance itself.

Reference: Zaheer et al. 2017, "Deep Sets" (NeurIPS).

Token format (must match PerDrone_OW):
  cols [...]  = node features
  col  [-1]   = mask col, 1.0 = real peer, <0.5 = padded
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type

import torch
import torch.nn as nn
from tensordict import TensorDictBase

from benchmarl.models.common import Model, ModelConfig

from MANET.frozen_encoder import (LearnableUserSetEncoder,
                                  make_user_set_encoder)
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

# Local (never-transmitted) encoder width for the observer's OWN raw user
# block on the om2g flat layout -- same knob as st_model's SELF_USER_OUT_DIM.
import os as _os
OM2G_SELF_USER_OUT_DIM = int(_os.environ.get("OM2G_SELF_USER_OUT_DIM", "64"))


class _DeepSetsCore(nn.Module):
    """phi: shared per-token MLP. rho: post-sum MLP.

    Forward returns the rho-transformed sum-pooled representation, with
    padded tokens excluded from the sum via the mask.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        phi_hidden: int,
        rho_hidden: int,
        dropout: float,
    ):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, phi_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(phi_hidden, latent_dim),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(latent_dim, rho_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rho_hidden, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, n_tokens, input_dim)
        embeds = self.phi(x)  # (B, n_tokens, latent_dim)
        if key_padding_mask is not None:
            # 1.0 in mask col = real, <0.5 = padded. key_padding_mask is True
            # for padded slots (transformer convention). Zero them before sum.
            keep = (~key_padding_mask).unsqueeze(-1).float()
            embeds = embeds * keep
        pooled = embeds.sum(dim=1)  # (B, latent_dim)
        return self.rho(pooled)


class DeepSetsModel(Model):
    """BenchMARL actor backed by a Deep Sets encoder.

    Input:  (*batch, n_agents, n_peers, token_features)
    Output: (*batch, n_agents, output_features)

    share_params=True only — n_agents is folded into the batch dimension so
    the shared phi/rho networks process all agents in one forward pass.
    """

    def __init__(self, **kwargs):
        self.latent_dim: int = kwargs.pop("latent_dim")
        self.phi_hidden: int = kwargs.pop("phi_hidden")
        self.rho_hidden: int = kwargs.pop("rho_hidden")
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

        token_features: int = self.input_leaf_spec.shape[-1]
        raw_input_dim = token_features - (1 if self.use_mask else 0)
        self.output_features: int = self.output_leaf_spec.shape[-1]

        self._om2g = None
        self._peer_msg_encoder = None
        self._self_user_encoder = None
        self._self_user_proj = None
        if self.obs_layout == "om2g":
            # OM2G flat layout: peer slots arrive
            # ALREADY compressed to the arm's L (sender-side codec in the
            # env); no model-side peer user encoder exists. The only user
            # encoder is the LOCAL one over the observer's own raw block.
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
                    f"DeepSetsModel: obs spec is {token_features} wide, om2g "
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
            # w_peer_model: constant in L under e2e_rt. See
            # user_encoder.om2g_flat_layout.
            input_dim = self._om2g["w_peer_model"] - 1
            self._self_user_encoder = make_user_set_encoder(
                dim=self._om2g["user_feat"],
                hidden=LEARNED_USER_ENCODER_HIDDEN,
                out_dim=OM2G_SELF_USER_OUT_DIM,
                max_users=self.max_users,
                kind="learned",
            )
            self._self_user_proj = nn.Linear(
                OM2G_SELF_USER_OUT_DIM, self.latent_dim)
        elif self.encoder_mode == "learned":
            # Own, from-scratch user-set encoder -- its params register
            # with THIS model (and therefore this arch's optimizer), so
            # gradients from the RL loss reach it. Never warm-started.
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

        self._encoder = _DeepSetsCore(
            input_dim=input_dim,
            latent_dim=self.latent_dim,
            phi_hidden=self.phi_hidden,
            rho_hidden=self.rho_hidden,
            dropout=self.dropout,
        )
        # Un-pooled path to the observer's OWN token (as in st_model.py).
        # Deepsets needs this at least as much as ST: with a sum-pool,
        #   pool_i = [phi(t_i,flag=1) - phi(t_i,flag=0)] + sum_j phi(t_j,flag=0)
        # where the second term is IDENTICAL for every drone, so the self_flag
        # reaches the action head only as a 1/N-amplitude perturbation. ST has
        # attention as a partial fallback; deepsets' phi is per-token
        # independent, so it has none.
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
        if self._om2g is not None:
            head_in += self.latent_dim      # + the own-raw-user-block path
        self._output_proj = nn.Linear(head_in, self.output_features)

        if self.device is not None:
            self._encoder = self._encoder.to(self.device)
            self._output_proj = self._output_proj.to(self.device)
            if self._own_proj is not None:
                self._own_proj = self._own_proj.to(self.device)
            if self._user_encoder is not None:
                self._user_encoder = self._user_encoder.to(self.device)
            if self._peer_msg_encoder is not None:
                self._peer_msg_encoder = self._peer_msg_encoder.to(self.device)
            if self._self_user_encoder is not None:
                self._self_user_encoder = self._self_user_encoder.to(self.device)
                self._self_user_proj = self._self_user_proj.to(self.device)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        x = tensordict.get(self.in_key)
        # omnet/padre: x is (*batch, n_agents, n_peers, token_features).
        # om2g: x is (*batch, n_agents, flat_dim) -- split first, see
        # user_encoder.split_om2g_obs.

        own_user_summary = None
        if self._om2g is not None:
            raw_toks, raw_mask, x, inbox_toks, inbox_mask = \
                split_om2g_obs(x, self._om2g)
            if inbox_toks is not None:   # e2e: fill slots via E
                x = om2g_fill_peer_summaries(
                    x, inbox_toks, inbox_mask,
                    self._peer_msg_encoder, self._om2g)
            own_user_summary = self._self_user_encoder(raw_toks, raw_mask)
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
            body = reassemble_with_learned_encoder(
                body, self._user_encoder, self.max_users, obs_layout=self.obs_layout
            )
            x = torch.cat([body, peer_mask_col], dim=-1) if peer_mask_col is not None else body

        leading = x.shape[:-3]
        n_agents = x.shape[-3]
        n_peers = x.shape[-2]
        token_features = x.shape[-1]

        # Fold leading dims + n_agents into one batch dim, mirroring ST.
        x = x.reshape(-1, n_peers, token_features)  # (B*n_agents, n_peers, token_features)

        if self.use_mask:
            mask_col = x[..., -1]
            x = x[..., :-1]
            key_padding_mask = mask_col < 0.5  # True = padded
            fully_masked = key_padding_mask.all(dim=-1)  # (B*n_agents,)
        else:
            key_padding_mask = None
            fully_masked = None

        latent = self._encoder(x, key_padding_mask=key_padding_mask)  # (B*n_agents, latent)

        if self._self_token_active:
            # Hand the action head the observer's OWN token directly,
            # bypassing the sum-pool. The flag is one-hot over the peer axis by
            # construction (omnet_env.build_drone_tokens: row[event_end] = 1.0
            # iff j == i, and the j == i row is never link-masked), so argmax
            # selects it. A fully-masked agent has an all-zero token block and
            # argmax returns 0; `out` is zeroed for those rows below regardless.
            self_idx = x[..., self._self_flag_col].argmax(dim=-1)      # (B*n_agents,)
            own_tok = x[torch.arange(x.shape[0], device=x.device), self_idx]
            latent = torch.cat([latent, self._own_proj(own_tok)], dim=-1)

        if own_user_summary is not None:
            # om2g: the own-raw-block path, folded to the same row order.
            s_ = own_user_summary.reshape(-1, own_user_summary.shape[-1])
            latent = torch.cat([latent, self._self_user_proj(s_)], dim=-1)

        out = self._output_proj(latent)                                # (B*n_agents, output_features)

        # Zero output for viewing agents that are themselves absent under the
        # variable-N padding convention, so no gradient flows through them.
        if fully_masked is not None and fully_masked.any():
            out = torch.where(fully_masked.unsqueeze(-1), torch.zeros_like(out), out)

        out = out.reshape(*leading, n_agents, self.output_features)
        tensordict.set(self.out_key, out)
        return tensordict


@dataclass
class DeepSetsConfig(ModelConfig):
    """Dataclass config for :class:`DeepSetsModel`."""

    latent_dim: int = 64
    phi_hidden: int = 128
    rho_hidden: int = 128
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
    # Concatenate the observer's OWN token to the sum-pooled peer
    # vector before the action head, so identity does not have to survive a
    # permutation-invariant pool at 1/N amplitude. No effect on tracks whose
    # OBS_LAYOUTS entry has self_flag_col=None (currently "padre"). Set False
    # for the architecture without the own-token path (for an A/B).
    #
    # Default True to match STConfig, so the architectures are compared
    # with the same own-token path.
    use_self_token: bool = True

    @staticmethod
    def associated_class() -> Type[Model]:
        return DeepSetsModel