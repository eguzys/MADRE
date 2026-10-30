"""
GAT model for variable-N MANET MAPPO actor.

Sibling to st_model.py. Same BenchMARL Model contract, same input shape,
same absent-agent handling. Differs only in attention mask:
  ST  -> key_padding_mask (full set, every peer attends to every peer)
  GAT -> adjacency mask   (drones only attend to peers within transmission range)

Story: ST = attention over the set. GAT = attention over the graph.
Same primitive, different structural prior.

Token format (must match PerDrone_OW):
  cols [0, 1] = normalized position (x, y) in [-1, 1]
  cols [...]  = other node features
  col  [-1]   = mask col, 1.0 = real peer, <0.5 = padded
"""

import math
import os
from dataclasses import dataclass
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDictBase

from benchmarl.models.common import Model, ModelConfig

from architectures.user_encoder import (
    OBS_LAYOUTS,
    om2g_flat_layout,
    split_om2g_obs,
    om2g_fill_peer_summaries,
    make_om2g_peer_msg_encoder,
    reassemble_with_learned_encoder,
    reconstructed_input_dim,
    LEARNED_USER_ENCODER_OUT_DIM,
    LEARNED_USER_ENCODER_HIDDEN,
)
from MANET.frozen_encoder import (LearnableUserSetEncoder,
                                  make_user_set_encoder)


# ----------------------------------------------------------------------
# Single multi-head GAT layer. Velickovic et al. 2018, with Pre-LN.
# ----------------------------------------------------------------------
class _GATLayer(nn.Module):
    """
    h_i' = concat_h [ sum_j alpha_ij^h W^h h_j ]
    alpha_ij = softmax_j( LeakyReLU(a^T [W h_i || W h_j]) )  masked by adjacency
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0, f"dim={dim} must be divisible by n_heads={n_heads}"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dim = dim

        self.norm = nn.LayerNorm(dim)
        self.W = nn.Linear(dim, dim, bias=False)
        # Decomposed attention vector: a = [a_src ; a_dst], applied per head.
        self.attn_src = nn.Parameter(torch.empty(n_heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(n_heads, self.head_dim))
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attn_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(0))

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        h:   (B, N, dim)        node features
        adj: (B, N, N) bool     adj[b, i, j] = edge j -> i exists
        out: (B, N, dim)
        """
        B, N, _ = h.shape
        h_norm = self.norm(h)
        Wh = self.W(h_norm).view(B, N, self.n_heads, self.head_dim)  # (B, N, H, Dh)

        src_score = (Wh * self.attn_src).sum(-1)  # (B, N, H)
        dst_score = (Wh * self.attn_dst).sum(-1)  # (B, N, H)
        # e[b, i, j, h] = src_score[b, i, h] + dst_score[b, j, h]
        e = self.leaky(src_score.unsqueeze(2) + dst_score.unsqueeze(1))  # (B, N, N, H)

        # Mask out non-edges before softmax over j (source-of-message dim).
        mask = adj.unsqueeze(-1)  # (B, N, N, 1)
        e = e.masked_fill(~mask, float("-inf"))

        alpha = F.softmax(e, dim=2)
        # If row i has no incoming edges at all, softmax(-inf,...) -> NaN. Zero it.
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.dropout(alpha)

        # h_i' per head = sum_j alpha_ij * Wh_j
        out = torch.einsum("bijh,bjhd->bihd", alpha, Wh)
        return out.reshape(B, N, self.dim)


# ----------------------------------------------------------------------
# BenchMARL Model wrapper
# ----------------------------------------------------------------------
class GATModel(Model):
    """
    Actor-side GAT. Per-drone output, not set-level (no PMA).

    Input :  (*batch, n_agents, n_peers, token_dim) — same as STModel
    Output:  (*batch, n_agents, action_dim)

    For viewing agent i, builds graph over n_peers using positions (token cols
    0,1) and a distance threshold rho_norm. Padded peers (mask col < 0.5) get
    no edges. Self-loops added for real peers. Then runs L stacked GAT layers
    with residual + Pre-LN, reads out the viewing agent's own row, projects to
    action_dim. Padded viewing agents get zero output (gradient-inert).
    """

    def __init__(
        self,
        latent_dim: int,
        n_heads: int,
        n_layers: int,
        rho_norm: float,
        dropout: float,
        **kwargs,
    ):
        # Pop before super(): Model.__init__ rejects unknown kwargs.
        self.obs_layout: str = kwargs.pop("obs_layout", "padre")
        self.use_self_token: bool = kwargs.pop("use_self_token", True)
        self.encoder_mode: str = kwargs.pop("encoder_mode", "frozen")
        self.max_users: int = kwargs.pop("max_users", 20)
        assert self.encoder_mode in ("frozen", "learned"), (
            f"encoder_mode must be 'frozen' or 'learned', got {self.encoder_mode!r}"
        )
        assert self.obs_layout in OBS_LAYOUTS, (
            f"obs_layout must be one of {sorted(OBS_LAYOUTS)}, got {self.obs_layout!r}"
        )

        super().__init__(**kwargs)

        if not self.input_has_agent_dim:
            raise ValueError("GATModel is actor-only (input_has_agent_dim must be True)")
        if self.centralised:
            raise ValueError("GATModel does not support centralised mode")
        if not self.share_params:
            raise ValueError("GATModel requires share_params=True")

        # Read shapes from BenchMARL specs (never hardcode).
        in_leaf = list(self.input_spec.values(True, True))[0]
        out_leaf = list(self.output_spec.values(True, True))[0]
        self.token_dim = in_leaf.shape[-1]
        raw_feat_dim = self.token_dim - 1             # last col is mask
        self.action_dim = out_leaf.shape[-1]

        # Learned-mode user-set encoder, as in deepsets_model.py, so the raw
        # per-user block is pooled before input_proj like in the other
        # architectures.
        # Its params register with THIS model, so the RL loss reaches them;
        # never warm-started, exactly like the other three.
        self._om2g = None
        self._peer_msg_encoder = None
        self._self_user_encoder = None
        self._self_user_proj = None
        if self.obs_layout == "om2g":
            # OM2G flat layout: peer slots arrive
            # ALREADY compressed to the arm's L (sender-side codec in the
            # env); no model-side peer user encoder. The LOCAL encoder over
            # the observer's own raw block is added below.
            self._om2g = om2g_flat_layout()
            if self.token_dim != self._om2g["flat_dim"]:
                raise RuntimeError(
                    f"GATModel: obs spec is {self.token_dim} wide, om2g "
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
            self.feat_dim = self._om2g["w_peer_model"] - 1
        elif self.encoder_mode == "learned":
            self._user_encoder = LearnableUserSetEncoder(
                dim=OBS_LAYOUTS[self.obs_layout]["user_token_dim"],
                hidden=LEARNED_USER_ENCODER_HIDDEN,
                out_dim=LEARNED_USER_ENCODER_OUT_DIM,
            )
            # After reassembly the token is own_state | pooled(out_dim) | tail,
            # which is what every layer below sees -- so feat_dim must be the
            # RECONSTRUCTED width, not the raw one.
            self.feat_dim = reconstructed_input_dim(
                raw_feat_dim, self.max_users, LEARNED_USER_ENCODER_OUT_DIM,
                obs_layout=self.obs_layout,
            )
        else:
            self._user_encoder = None
            self.feat_dim = raw_feat_dim

        self.latent_dim = latent_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.rho_norm = rho_norm

        self.input_proj = nn.Linear(self.feat_dim, latent_dim)
        self.gat_layers = nn.ModuleList([
            _GATLayer(latent_dim, n_heads, dropout=dropout) for _ in range(n_layers)
        ])
        self.output_norm = nn.LayerNorm(latent_dim)

        # Own-token bypass. GAT is the interesting case: it ALREADY reads
        # out its own node (`self_emb` below) rather than a uniform pool, but
        # that embedding has been through n_layers of message passing and so is
        # mixed with its neighbours. This path hands the head the observer's RAW
        # token as well, which is what ST/deepsets/ffn do. Whether GAT needs
        # it, given it already has an own-centric readout, is the A/B this
        # flag exists to run.
        if self._om2g is not None:
            self._self_flag_col = self._om2g["self_flag_col"]   # 15 + L
        else:
            self._self_flag_col = OBS_LAYOUTS[self.obs_layout]["self_flag_col"]
        self._self_token_active = bool(self.use_self_token
                                       and self._self_flag_col is not None)
        if self._self_token_active and self._om2g is not None:
            # om2g: the slot layout is structural (self flag at 15+L computed
            # from the SAME env vars the env used), so the omnet-only
            # raw-block heuristic below (feat_dim > 64 -> raise) must not
            # fire -- the uncompressed arm's slot is legitimately wide.
            if not 0 <= self._self_flag_col < self.feat_dim:
                raise ValueError(
                    f"om2g self_flag_col {self._self_flag_col} outside the "
                    f"{self.feat_dim}-wide slot -- layout drift.")
            self._own_proj = nn.Linear(self.feat_dim, latent_dim)
            head_in = 2 * latent_dim
        elif self._self_token_active:
            # GAT has a user encoder (above), so on the OMNeT
            # track self.feat_dim is the reassembled width (27 at out_dim=8) and
            # self_flag_col=23 lands on the real flag. This guard
            # fires if a RAW user block reaches this
            # point without having been encoded — i.e. encoder_mode=="frozen"
            # while the env is emitting raw per-user tokens. In that case
            # column 23 is a user field, not the self flag, and the bypass
            # would read the wrong column. Raise instead.
            if self.feat_dim > 64:
                raise RuntimeError(
                    f"GATModel feature block is {self.feat_dim} wide, which is "
                    f"a RAW per-user block, but encoder_mode="
                    f"{self.encoder_mode!r}. obs_layout={self.obs_layout!r} "
                    f"places self_flag_col at {self._self_flag_col}, an index "
                    f"into the REASSEMBLED token — on a raw block that column "
                    f"is a user field. Set encoder_mode='learned' so the user "
                    f"block is pooled before it reaches the own-token bypass.")
            if not 0 <= self._self_flag_col < self.feat_dim:
                raise ValueError(
                    f"obs_layout={self.obs_layout!r} puts self_flag_col at "
                    f"{self._self_flag_col}, outside the {self.feat_dim}-wide "
                    f"feature block — OBS_LAYOUTS is out of lockstep with "
                    f"omnet_env.build_drone_tokens.")
            self._own_proj = nn.Linear(self.feat_dim, latent_dim)
            head_in = 2 * latent_dim
        else:
            self._own_proj = None
            head_in = latent_dim
        if self._om2g is not None:
            # LOCAL encoder over the observer's own raw user block (never on
            # the wire, arm-independent), like st/deepsets.
            _self_out = int(os.environ.get("OM2G_SELF_USER_OUT_DIM", "64"))
            self._self_user_encoder = make_user_set_encoder(
                dim=self._om2g["user_feat"],
                hidden=LEARNED_USER_ENCODER_HIDDEN,
                out_dim=_self_out,
                max_users=self.max_users,
                kind="learned",
            )
            self._self_user_proj = nn.Linear(_self_out, latent_dim)
            head_in += latent_dim
        self.output_proj = nn.Linear(head_in, self.action_dim)

        if self.device is not None:
            self.input_proj = self.input_proj.to(self.device)
            self.gat_layers = self.gat_layers.to(self.device)
            self.output_norm = self.output_norm.to(self.device)
            self.output_proj = self.output_proj.to(self.device)
            if self._own_proj is not None:
                self._own_proj = self._own_proj.to(self.device)
            if self._user_encoder is not None:
                self._user_encoder = self._user_encoder.to(self.device)
            if self._self_user_encoder is not None:
                self._self_user_encoder = self._self_user_encoder.to(self.device)
                self._self_user_proj = self._self_user_proj.to(self.device)
            if self._peer_msg_encoder is not None:
                self._peer_msg_encoder = self._peer_msg_encoder.to(self.device)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        x = tensordict.get(self.in_key)
        # omnet/padre: x is (*batch, n_agents, n_peers, token_dim).
        # om2g: (*batch, n_agents, flat_dim) -- split first.
        own_user_summary = None
        if self._om2g is not None:
            raw_toks, raw_mask, x, inbox_toks, inbox_mask = \
                split_om2g_obs(x, self._om2g)
            if inbox_toks is not None:   # e2e: fill slots via E
                x = om2g_fill_peer_summaries(
                    x, inbox_toks, inbox_mask,
                    self._peer_msg_encoder, self._om2g)
            own_user_summary = self._self_user_encoder(raw_toks, raw_mask)
            *batch, n_agents, n_peers, token_dim = x.shape
        else:
            *batch, n_agents, n_peers, token_dim = x.shape
            assert token_dim == self.token_dim, f"token_dim mismatch {token_dim} vs {self.token_dim}"
        assert n_agents == n_peers, "PerDrone expects n_agents == n_peers"

        if self._om2g is not None:
            pass                       # peer slots arrive pre-compressed
        elif self.encoder_mode == "learned":
            # Slice out the padded raw user-token block, pool it with our own
            # trainable encoder, and reassemble -- everything below this is
            # then byte-identical to encoder_mode=="frozen". GAT always carries
            # a trailing peer-mask column, so it is split off and re-appended
            # around the reassembly (deepsets does the same under use_mask).
            peer_mask_col = x[..., -1:]
            body = reassemble_with_learned_encoder(
                x[..., :-1], self._user_encoder, self.max_users,
                obs_layout=self.obs_layout,
            )
            x = torch.cat([body, peer_mask_col], dim=-1)
            token_dim = x.shape[-1]

        # Flatten batch + agent dims so each row is one viewing agent.
        B = math.prod(batch) if batch else 1
        x_flat = x.reshape(B * n_agents, n_peers, token_dim)

        feats = x_flat[..., :-1]                  # (BA, N, feat_dim)
        mask_col = x_flat[..., -1]                # (BA, N)
        real = mask_col >= 0.5                    # bool, True = real peer

        # --- Build adjacency from token positions (cols 0, 1) ---
        pos = feats[..., :2]                                          # (BA, N, 2)
        dist = (pos.unsqueeze(2) - pos.unsqueeze(1)).norm(dim=-1)     # (BA, N, N)
        within = dist < self.rho_norm
        real_pair = real.unsqueeze(2) & real.unsqueeze(1)             # both endpoints real
        adj = within & real_pair
        # Self-loops for real peers (standard GAT). Padded rows stay isolated.
        eye = torch.eye(n_peers, dtype=torch.bool, device=x.device).unsqueeze(0)
        adj = adj | (eye & real.unsqueeze(2))

        # --- Encode + L layers of message passing with residual ---
        h = self.input_proj(feats)
        for layer in self.gat_layers:
            h = h + layer(h, adj)
        h = self.output_norm(h)

        # --- Self-node readout ---
        # Viewing agent k (flat index) has self_flag eye structure: its "own" row
        # in the n_peers dim is index (k % n_agents).
        agent_idx = torch.arange(n_agents, device=x.device).repeat(B)     # (BA,)
        flat_idx = torch.arange(B * n_agents, device=x.device)
        self_emb = h[flat_idx, agent_idx]                                  # (BA, latent)

        if self._self_token_active:
            # Select the own row by the self FLAG rather than by position, so
            # this agrees with ST/deepsets even if the row ordering ever changes.
            # The flag is one-hot over the peer axis by construction
            # (omnet_env.build_drone_tokens sets row[event_end]=1.0 iff j == i).
            self_idx = feats[..., self._self_flag_col].argmax(dim=-1)      # (BA,)
            own_tok = feats[flat_idx, self_idx]                            # (BA, feat_dim)
            self_emb = torch.cat([self_emb, self._own_proj(own_tok)], dim=-1)

        if own_user_summary is not None:
            # om2g: the own-raw-block path, folded to the same row order.
            s_ = own_user_summary.reshape(-1, own_user_summary.shape[-1])
            self_emb = torch.cat([self_emb, self._self_user_proj(s_)], dim=-1)

        action = self.output_proj(self_emb)                                # (BA, action_dim)

        # --- Zero output for viewing agents that are themselves padded ---
        viewing_real = real[flat_idx, agent_idx]                           # (BA,) bool
        action = action * viewing_real.unsqueeze(-1).float()

        action = action.reshape(*batch, n_agents, self.action_dim)
        tensordict.set(self.out_key, action)
        return tensordict


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass
class GATConfig(ModelConfig):
    latent_dim: int = 64
    n_heads: int = 4
    n_layers: int = 2
    # rho_norm = 2 * rho_world / playground_size (positions are min-max normalized
    # to [-1, 1] in token, so a normalized distance d_norm corresponds to a world
    # distance d_world = d_norm * playground_size / 2). Default below assumes
    # playground_size=100 and rho_world~30. CHECK ENV CONFIG.
    rho_norm: float = 0.6
    dropout: float = 0.1
    obs_layout: str = "padre"
    # Own-token bypass; inert on layouts whose self_flag_col is None
    # (currently "padre"). Set False to run the ablation arm.
    use_self_token: bool = True
    # "frozen": the env hands over an already-pooled user summary.
    # "learned": the env hands over the raw per-user block and this model pools
    # it with its own encoder (required on the OMNeT track). Mirrors the same
    # two fields on STConfig / DeepSetsConfig / FFNConfig.
    encoder_mode: str = "frozen"
    max_users: int = 20  # only consulted when encoder_mode=="learned"

    @staticmethod
    def associated_class() -> Type[Model]:
        return GATModel