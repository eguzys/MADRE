"""Custom BenchMARL FFN model for PADRE, with an optional per-architecture
learnable user-set encoder (encoder_mode=="learned").

BenchMARL's own stock `Mlp`/`MlpConfig` (see conf/model/padre_mlp.yaml)
has no hook for a submodule that pre-processes part of its input, so the
raw-obs track's FFN gets this thin custom Model instead, built directly on
torchrl's MultiAgentMLP (the same building block stock Mlp uses). In
encoder_mode=="frozen" this flattens the whole (n_peers, token_features)
grid per agent exactly like stock Mlp's num_feature_dims=2 -- the drone
(peer) axis is still flattened, unchanged; only encoder_mode=="learned"
additionally slices/re-encodes/reassembles the per-peer user-token block
first (see architectures/user_encoder.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Type

import torch
import torch.nn as nn
from tensordict import TensorDictBase
from torchrl.modules import MultiAgentMLP

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


class FFNModel(Model):
    """BenchMARL actor backed by flatten + MultiAgentMLP.

    Input:  (*batch, n_agents, n_peers, token_features)
    Output: (*batch, n_agents, output_features)
    """

    def __init__(self, **kwargs):
        self.num_cells: Sequence[int] = list(kwargs.pop("num_cells"))
        self.activation_class: Type[nn.Module] = kwargs.pop("activation_class")
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

        # omnet/padre: input_leaf_spec.shape = (n_agents, n_peers,
        # token_features). om2g: (n_agents, flat_dim) -- one flat vector,
        # split by split_om2g_obs in _forward.
        n_peers: int = self.input_leaf_spec.shape[-2]
        token_features: int = self.input_leaf_spec.shape[-1]
        self.output_features: int = self.output_leaf_spec.shape[-1]

        self._om2g = None
        self._peer_msg_encoder = None
        if self.obs_layout == "om2g":
            # OM2G flat layout: peer slots arrive
            # ALREADY compressed to the arm's L (sender-side codec in the
            # env). FFN is the flatten architecture, so its own-raw-block
            # path is idiomatic: the raw block is appended FLAT to the MLP
            # input, no set encoder -- same spirit as
            # PassthroughUserSetEncoder ("pad to max users and send that").
            self._om2g = om2g_flat_layout()
            if token_features != self._om2g["flat_dim"]:
                raise RuntimeError(
                    f"FFNModel: obs spec is {token_features} wide, om2g "
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
            n_peers = self._om2g["max_n"]
            # w_peer_model: constant in L under e2e_rt. See
            # user_encoder.om2g_flat_layout.
            per_peer_dim = self._om2g["w_peer_model"]
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
            per_peer_dim = reconstructed_input_dim(
                token_features, self.max_users, LEARNED_USER_ENCODER_OUT_DIM,
                obs_layout=self.obs_layout,
            )
        else:  # frozen
            per_peer_dim = token_features

        # Own-token path (as in st_model.py). The FFN already sees every
        # peer token, so this looks redundant -- it is not. The peer axis is in
        # a FIXED absolute order that is the SAME for every observer, and with
        # share_params=True all drones push a near-identical vector through one
        # network: 332 of the 345 raw columns are byte-identical across
        # observers. Which slot is "me" is therefore recoverable ONLY by reading
        # the self_flag. Appending the own token at a fixed, privileged
        # position removes the need to locate it at all.
        if self._om2g is not None:
            self._self_flag_col = self._om2g["self_flag_col"]   # 15 + L
        else:
            self._self_flag_col = OBS_LAYOUTS[self.obs_layout]["self_flag_col"]
        self._self_token_active = bool(self.use_self_token
                                       and self._self_flag_col is not None)
        if self._self_token_active and not 0 <= self._self_flag_col < per_peer_dim:
            raise ValueError(
                f"obs_layout={self.obs_layout!r} puts self_flag_col at "
                f"{self._self_flag_col}, outside the {per_peer_dim}-wide "
                f"per-peer token — OBS_LAYOUTS is out of lockstep with "
                f"omnet_env.build_drone_tokens.")

        self.input_features = n_peers * per_peer_dim
        if self._self_token_active:
            self.input_features += per_peer_dim
        if self._om2g is not None:
            self.input_features += self._om2g["raw_w"]   # own raw user block

        self.mlp = MultiAgentMLP(
            n_agent_inputs=self.input_features,
            n_agent_outputs=self.output_features,
            n_agents=self.n_agents,
            centralized=self.centralised,
            share_params=self.share_params,
            device=self.device,
            num_cells=self.num_cells,
            activation_class=self.activation_class,
        )

        if (self.device is not None and self.encoder_mode == "learned"
                and self._user_encoder is not None):
            self._user_encoder = self._user_encoder.to(self.device)
        if self.device is not None and self._peer_msg_encoder is not None:
            self._peer_msg_encoder = self._peer_msg_encoder.to(self.device)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        x = tensordict.get(self.in_key)
        # omnet/padre: x is (*batch, n_agents, n_peers, token_features).
        # om2g: (*batch, n_agents, flat_dim) -- split first.

        own_raw_flat = None
        if self._om2g is not None:
            raw_toks, raw_mask, x, inbox_toks, inbox_mask = \
                split_om2g_obs(x, self._om2g)
            if inbox_toks is not None:   # e2e: fill slots via E
                x = om2g_fill_peer_summaries(
                    x, inbox_toks, inbox_mask,
                    self._peer_msg_encoder, self._om2g)
            # Zero the padded slots (their present bit is 0) then flatten --
            # the same masking PassthroughUserSetEncoder applies, so a
            # padded row can never feed the MLP.
            own_raw_flat = (raw_toks * raw_mask.unsqueeze(-1)).flatten(
                start_dim=-2)                    # (*batch, n_agents, raw_w)
        elif self.encoder_mode == "learned":
            # Slice out the padded raw user-token block, pool it with our
            # own trainable encoder, and reassemble -- everything below
            # this is then shaped exactly like encoder_mode=="frozen".
            x = reassemble_with_learned_encoder(
                x, self._user_encoder, self.max_users, obs_layout=self.obs_layout
            )

        input = torch.flatten(x, start_dim=-2)  # (*batch, n_agents, n_peers*per_peer_dim)

        if self._self_token_active:
            # Append the observer's OWN token at a fixed position. The
            # flag is one-hot over the peer axis by construction
            # (omnet_env.build_drone_tokens: row[event_end] = 1.0 iff j == i),
            # so argmax selects it. x here is (*batch, n_agents, n_peers, dim),
            # NOT folded into one batch dim as in ST/DeepSets, so select along
            # the peer axis with take_along_dim to stay shape-agnostic.
            self_idx = x[..., self._self_flag_col].argmax(dim=-1)   # (*batch, n_agents)
            own_tok = torch.take_along_dim(
                x, self_idx[..., None, None], dim=-2).squeeze(-2)   # (*batch, n_agents, dim)
            input = torch.cat([input, own_tok], dim=-1)

        if own_raw_flat is not None:
            input = torch.cat([input, own_raw_flat], dim=-1)

        res = self.mlp.forward(input)
        if not self.output_has_agent_dim:
            # Centralised + shared params: the multi-agent dim was
            # expanded internally; drop it without loss of data.
            res = res[..., 0, :]

        tensordict.set(self.out_key, res)
        return tensordict


@dataclass
class FFNConfig(ModelConfig):
    """Dataclass config for :class:`FFNModel`."""

    num_cells: List[int] = field(default_factory=lambda: [256, 256])
    activation_class: Type[nn.Module] = nn.Tanh
    # "frozen" (default): per-peer tokens already carry a pooled user
    # summary (functionally identical to stock Mlp / padre_mlp.yaml, just
    # flattened + fed through MultiAgentMLP the same way). "learned":
    # tokens instead carry a padded raw user-token block (max_users wide),
    # pooled here by this model's OWN trainable encoder submodule -- the
    # drone/peer axis is still flattened afterwards, same as before.
    encoder_mode: str = "frozen"
    max_users: int = 20  # only consulted when encoder_mode=="learned"
    # "padre" (default, byte-identical): own_state=4, 6-col user token + a
    # trailing mask column. "omnet": own_state=5, 13-col user token with its
    # presence bit AT col 12 (no extra mask column) -- see
    # architectures/user_encoder.py's OBS_LAYOUTS. Only consulted when
    # encoder_mode=="learned".
    obs_layout: str = "padre"
    # Append the observer's OWN token to the flattened peer vector
    # before the MLP. No effect on tracks whose OBS_LAYOUTS entry has
    # self_flag_col=None (currently "padre"). Set False for the
    # architecture without the own-token path (for an A/B).
    #
    # Default True to MATCH STConfig -- see the comment in _forward for why an
    # architecture that already sees every token still needs this.
    use_self_token: bool = True

    @staticmethod
    def associated_class():
        return FFNModel
