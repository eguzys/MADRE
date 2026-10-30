"""
Set Transformer architecture for variable-size MANET observations.

This module implements a Set Transformer-based policy for handling
variable network sizes in the PADRE anti-jamming system. It now
consumes the token-set output of `PerDrone_OW`, where each observation
is a 2-D tensor of `num_tokens x token_dim` per drone:
  - num_tokens = N (history_K=1) or N*K (history_K>1)
  - token_dim  = features + self_flag + aoi + (mask if mask=True)

When `mask=True`, the last column of every token is a 1/0 receive
flag; ST splits it off as a PyTorch key_padding_mask so the
attention skips filler tokens.

Author: Emilis
"""

from stable_baselines3.common.policies import ActorCriticPolicy
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    raise RuntimeError("CUDA not available - Do you want to run it for weeks? ")


class STEncoder(torch.nn.Module):
    """Set Transformer feature extractor for the PerDrone_OW token format.

    Expects a 2-D Box observation space `(num_tokens, token_dim)`. The
    flat input that SB3 hands forward(...) is reshaped back to
    `(batch, num_tokens, token_dim)` and (optionally) split into the
    feature block plus a key-padding mask before the linear projection.
    """

    def __init__(self, observation_space, action_space, num_nodes, mask: bool = False):
        """Initialise the encoder.

        Args:
            observation_space: gym Box space; must be 2-D so we can
                read (num_tokens, token_dim) directly from `.shape`.
            action_space: gym action space; only its `.shape[0]` is
                used to size the latent dims indirectly via SB3.
            num_nodes (int): number of MANET nodes (kept on the
                encoder for downstream callers and logging; the actual
                token count is `num_tokens`).
            mask (bool, optional): When True, the last column of every
                token is treated as a 0/1 receive flag. Split off and
                converted to a PyTorch key_padding_mask
                (`True = pad-out`) for the encoder + pooler. Defaults
                to False.
        """
        super().__init__()

        if len(observation_space.shape) != 2:
            raise ValueError(
                "STEncoder requires a 2-D observation space "
                "(num_tokens, token_dim). Got shape "
                f"{observation_space.shape}. PerDrone_OW emits this "
                "format directly; legacy 1-D wrappers like VS_COW are "
                "not supported."
            )

        self.num_nodes = num_nodes
        self.num_tokens, self.token_dim = observation_space.shape
        self.has_mask = bool(mask)
        self.input_dim = self.token_dim - (1 if self.has_mask else 0)
        self.action_space = action_space.shape[0]

        self.transformerDim = 64
        self.latent_dim_pi = 64  # Actor features are 64-dim
        self.latent_dim_vf = 64  # Critic features are 64-dim
        nheads = 4

        self.inputLayer = torch.nn.Linear(self.input_dim, self.transformerDim)
        self.attentionLayer = torch.nn.TransformerEncoderLayer(
            d_model=self.transformerDim,
            nhead=nheads,
            dim_feedforward=256,
            dropout=0.1,
            activation='relu',
            batch_first=True,
            norm_first=True,
        )
        self.attention = torch.nn.TransformerEncoder(
            encoder_layer=self.attentionLayer,
            num_layers=2,
        )
        self.actorOutput = torch.nn.Linear(self.transformerDim, self.latent_dim_pi)
        self.criticOutput = torch.nn.Linear(self.transformerDim, self.latent_dim_vf)
        self.pooling = PMA(self.transformerDim, num_heads=nheads)

    def __str__(self):
        return "ST"

    def transformToElement(self, obs):
        """Reshape flat obs back to tokens and split off the mask column.

        Returns:
            elements (Tensor): (batch, num_tokens, input_dim)
            key_padding_mask (Tensor | None): (batch, num_tokens) bool,
                True = pad-out for `nn.MultiheadAttention`. None when
                `has_mask=False`.
        """
        batch_size = obs.shape[0]
        elements = obs.reshape(batch_size, self.num_tokens, self.token_dim)
        if self.has_mask:
            mask_col = elements[..., -1]
            elements = elements[..., :-1]
            key_padding_mask = mask_col < 0.5
        else:
            key_padding_mask = None
        return elements, key_padding_mask

    def _assert_at_least_one_real(self, key_padding_mask):
        """Sanity check: every row must keep at least one un-padded
        token. PerDrone_OW guarantees `mask[i, i, 0] = True` after
        reset, so this should never fire — it's a one-line guardrail
        against future wrapper regressions.
        """
        if key_padding_mask is None:
            return
        all_padded = key_padding_mask.all(dim=-1)
        if torch.any(all_padded):
            raise RuntimeError(
                "Encountered a drone row with every token masked out — "
                "MultiheadAttention would return NaN. PerDrone_OW must "
                "keep at least one real token per row (e.g. the "
                "self-diagonal at k=0)."
            )

    def forward(self, obs):
        return self.forward_actor(obs), self.forward_critic(obs)

    def forward_actor(self, features):
        elements, kpm = self.transformToElement(features)
        self._assert_at_least_one_real(kpm)
        elements = self.inputLayer(elements)
        output = self.attention(elements, src_key_padding_mask=kpm)
        pooled = self.pooling(output, key_padding_mask=kpm)
        return self.actorOutput(pooled)

    def forward_critic(self, features):
        elements, kpm = self.transformToElement(features)
        self._assert_at_least_one_real(kpm)
        elements = self.inputLayer(elements)
        output = self.attention(elements, src_key_padding_mask=kpm)
        pooled = self.pooling(output, key_padding_mask=kpm)
        return self.criticOutput(pooled)


class ST(ActorCriticPolicy):
    """Set Transformer policy for PADRE."""

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        num_nodes=5,
        mask: bool = False,
        **kwargs,
    ):
        # Save SB3-specific attrs BEFORE super().__init__() because
        # `_build_mlp_extractor` runs inside the parent constructor.
        self.num_nodes = num_nodes
        self.mask = mask
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = STEncoder(
            self.observation_space,
            self.action_space,
            num_nodes=self.num_nodes,
            mask=self.mask,
        )


class PMA(torch.nn.Module):
    """Pooling-by-Multihead-Attention. A learned seed attends to the
    encoder output to produce a fixed-size representation. Forwards
    the wrapper-provided key_padding_mask so masked-out tokens don't
    contribute to the pooled vector.
    """

    def __init__(self, d_model, num_heads, num_seeds=1):
        super().__init__()
        self.seeds = torch.nn.Parameter(torch.randn(1, num_seeds, d_model))
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        batch_size = x.shape[0]
        seeds = self.seeds.expand(batch_size, -1, -1)
        pooled, _ = self.attention(
            seeds, x, x, key_padding_mask=key_padding_mask
        )
        return self.norm(pooled + seeds).squeeze(1)
