"""Per-drone RAW USER SET builder (observer-relative) + placeholder PISA glue.

RawUserSet_OW extends PerDrone_OW with:
  - a raw-only own-state node config (position + receivedSignal +
    interference, everything else off), forced locally in this subclass
    without touching ObservationWrappers.DEFAULT_CONFIG.
  - `_build_user_set_for_active_drone`, a method that builds an
    intermediate, per-drone tensor over users — one token per user, from
    the observing drone's perspective.
  - PISAStub, a placeholder pooler standing in for a future trained
    set-encoder over that per-user tensor. It is glued onto each drone's
    raw own-state row as 10 extra feature columns.
"""

import numpy as np
import networkx as nx

from MANET.ObservationWrappers import (
    PerDrone_OW,
    normalize_and_transform,
    NODE_KEYS,
    USER_KEYS,
    FIX_BUGS,
    INITIAL_AOI,
)

from MANET.frozen_encoder import FrozenDeepSetsEncoder, USER_TOKEN_DIM


class PISAStub:
    """Placeholder pooler standing in for a future trained PISA set-encoder.

    Permutation-invariant and mask-respecting by construction (it ignores
    its inputs and returns zeros), so it is a safe drop-in for the real
    pooler later without changing any shape/width bookkeeping.
    """

    def __init__(self, out_dim: int = 10):
        self.out_dim = out_dim

    def __call__(self, user_tokens: np.ndarray, user_mask: np.ndarray) -> np.ndarray:
        # user_tokens: (numbOfUsers, 6), user_mask: (numbOfUsers,)
        # PLACEHOLDER: return zeros. Real PISA swaps in here later.
        return np.zeros(self.out_dim, dtype=np.float32)


class RawUserSet_OW(PerDrone_OW):
    """PerDrone_OW subclass that:
      1. Forces its own-state node features to raw-only (position,
         receivedSignal, interference — no cooked/capacity features).
      2. Builds a raw, per-user token set for the currently active drone
         (see `_build_user_set_for_active_drone`).
      3. Appends a 10-value PISA placeholder summary (per drone) onto
         that drone's raw own-state row before it is written into the
         parent's (i, j, k) cache.

    encoder_mode ("frozen" | "learned") picks how the per-drone user set
    is compressed into that fixed-width summary:
      frozen  -- pool it here with the frozen FrozenDeepSetsEncoder
                 (unchanged code path, byte-identical to pre-existing
                 behaviour).
      learned -- do NOT pool here. Emit the padded raw per-user tokens +
                 mask instead; the pooling submodule lives inside each
                 BenchMARL model (architectures/user_encoder.py), trained
                 end-to-end from that architecture's RL loss.
    """

    def __init__(
        self,
        state_loss_rate: float,
        env,
        config: dict = None,
        self_flag: bool = True,
        aoi: bool = True,
        history_K: int = 1,
        mask: bool = False,
        fading_sigma_dB: float = 4.0,
        encoder_mode: str = "frozen",
        max_users: int = None,
    ):
        assert encoder_mode in ("frozen", "learned"), (
            f"encoder_mode must be 'frozen' or 'learned', got {encoder_mode!r}"
        )
        self.encoder_mode = encoder_mode

        # Must exist before super().__init__ runs: PerDrone_OW.__init__
        # calls self._compute_n_features_per_node() (overridden below),
        # which reads self._user_block_width.
        if encoder_mode == "frozen":
            import os
            _enc_path = os.path.join(os.path.dirname(__file__),
                                     "encoders", "deepsets_h64_o8.pt")
            self._pisa = FrozenDeepSetsEncoder(
                _enc_path, dim=6, hidden=64, out_dim=8,
            )
            self._user_block_width = self._pisa.out_dim
        else:
            # learned: no pooling here. Pad each drone's (n_users_this_
            # episode, 6) raw tokens + mask up to a fixed `_max_users` so
            # the per-node feature width stays constant across episodes
            # despite numbOfUsers varying per-episode -- same reason
            # `_max_numbOfNodes` exists for the node axis (see
            # PerDrone_OW.__init__). `env` here is the pre-super().__init__
            # constructor arg, not yet `self.unwrapped_env`.
            #
            # `max_users`, when given, wins outright: it's how a caller
            # (evaluation/runner.py) supplies the true training upper bound
            # after it has collapsed `numbOfUsersRange` to a scalar to pin
            # the live per-episode count -- that scalar collapse destroys
            # the range this block would otherwise read the bound from, so
            # the caller must hand the bound over explicitly instead.
            if max_users is not None:
                self._max_users = int(max_users)
            else:
                _users_range = getattr(env.unwrapped, "numbOfUsersRange", None)
                if isinstance(_users_range, (tuple, list)) and len(_users_range) == 2:
                    self._max_users = int(_users_range[1])
                else:
                    self._max_users = int(env.unwrapped.numbOfUsers)
            self._user_block_width = self._max_users * (USER_TOKEN_DIM + 1)

        # Force raw-only own-state, ignoring whatever `config` the caller
        # passed. Built locally — DEFAULT_CONFIG in ObservationWrappers.py
        # is untouched.
        raw_config = {
            "nodes": {key: False for key in NODE_KEYS},
            "users": {key: False for key in USER_KEYS},
            "capacity_matrix": False,
        }
        raw_config["nodes"]["position"] = True
        raw_config["nodes"]["receivedSignal"] = True
        raw_config["nodes"]["interference"] = True

        super().__init__(
            state_loss_rate,
            env,
            config=raw_config,
            self_flag=self_flag,
            aoi=aoi,
            history_K=history_K,
            mask=mask,
        )
        self.snrMinMaxTuple = (-10, 40)  # dB
        # Per-user throughput spans ~2k..6M bits (3+ orders of magnitude),
        # so log-transform then normalize (like power/interference).
        # Range measured empirically over greedy episodes: ~33..67 dB.
        self.throughputUserMinMaxTuple = (30, 68)  # 10*log10(bits)

        # Obs-read sensor fading: multiplicative log-normal on per-source
        # power at obs-build ONLY. The sim, capacities, and reward never
        # see it (reward is a deterministic function of positions), so the
        # banked results stay byte-identical. sigma matches the OMNeT++
        # shadowing (4 dB). Set fading_sigma_dB=0.0 to disable (ablation).
        self.fading_sigma_dB = float(fading_sigma_dB)
        self._fading_rng = np.random.default_rng()

    def _build_user_set_for_active_drone(self) -> tuple[np.ndarray, np.ndarray]:
        """Builds the raw user-set tokens for `self._active_drone`.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                user_tokens: shape (numbOfUsers, 6), rows sorted by user.id.
                    Columns: [angle, received_power, snr, noise_context,
                    throughput_in, throughput_out], each normalized to
                    [-1, 1]. Unsensed users get -1.0 in the power/snr
                    columns (matches the existing "not sensed" pad value).
                user_mask: shape (numbOfUsers,), 1.0 if the active drone
                    sensed that user this step (has a receivedPower_per_source
                    entry for it), else 0.0.
        """
        drone = self.unwrapped_env.nodes[self._active_drone]
        users = sorted(self.unwrapped_env.users, key=lambda u: u.id)
        n_users = self.unwrapped_env.numbOfUsers
        epsilon = 1e-9
        to_dB = lambda x: 10 * np.log10(x + epsilon)

        user_tokens = np.zeros((n_users, 6), dtype=np.float32)
        user_mask = np.zeros((n_users,), dtype=np.float32)

        # Column 0: angle to user, world-frame, observer-relative.
        # Defined for every row regardless of whether the user was sensed.
        deltas = np.array(
            [[u.pos[0] - drone.pos[0], u.pos[1] - drone.pos[1]] for u in users]
        )
        angles = np.arctan2(deltas[:, 1], deltas[:, 0]).reshape(-1, 1)
        user_tokens[:, 0] = normalize_and_transform(
            angles, (-np.pi, np.pi)
        ).reshape(-1)

        # Column 3: noise-context — drone's own total interference floor,
        # replicated into every row (it is not per-user).
        noise_mW = drone.interferenceReceived_mW
        noise_norm = normalize_and_transform(
            np.array([[noise_mW]]), self.interferenceMinMaxTuple, to_dB
        )[0, 0]
        user_tokens[:, 3] = noise_norm
        noise_dB = to_dB(noise_mW)

        # Columns 4/5: per-USER throughput in/out through THIS drone, on
        # that user's (source-attributed) commodity. Source-attributed:
        # a user's flow is the traffic it SENT, so senders light up and
        # receivers/jammer stay dark — role emerges from behaviour.
        # per_commodity_flow[user][node_u][node_v] = flow of user's
        # traffic on edge u->v. Filled per-user inside the loop below.
        per_commodity = self.unwrapped_env.network.per_commodity_flow

        # Columns 1/2 + mask: per-user, depends on whether the active drone
        # sensed that user this step.
        for k, user in enumerate(users):
            power_mW = drone.receivedPower_per_source.get(user.id, None)
            if power_mW is None:
                user_mask[k] = 0.0
                user_tokens[k, 1] = -1.0
                user_tokens[k, 2] = -1.0
                continue

            # Sensor fading: multiplicative log-normal (sigma in dB) applied
            # to the raw power at read. Feeds both the power column and the
            # SNR below (chipset sees faded power). Obs-only; reward untouched.
            if self.fading_sigma_dB > 0.0:
                _fade_dB = self._fading_rng.normal(0.0, self.fading_sigma_dB)
                power_mW = power_mW * (10.0 ** (_fade_dB / 10.0))

            user_mask[k] = 1.0
            user_tokens[k, 1] = normalize_and_transform(
                np.array([[power_mW]]), self.signalMinMaxTuple, to_dB
            )[0, 0]

            snr_dB = to_dB(power_mW) - noise_dB
            user_tokens[k, 2] = normalize_and_transform(
                np.array([[snr_dB]]), self.snrMinMaxTuple
            )[0, 0]

            # Columns 4/5: this user's traffic through THIS drone.
            #   egress  = flow of user's commodity LEAVING this drone
            #   ingress = flow of user's commodity ARRIVING at this drone
            user_edges = per_commodity.get(user, {})
            egress_bits = sum(user_edges.get(drone, {}).values())
            ingress_bits = sum(
                nbrs.get(drone, 0) for nbrs in user_edges.values()
            )
            user_tokens[k, 4] = normalize_and_transform(
                np.array([[ingress_bits]]), self.throughputUserMinMaxTuple, to_dB
            )[0, 0]
            user_tokens[k, 5] = normalize_and_transform(
                np.array([[egress_bits]]), self.throughputUserMinMaxTuple, to_dB
            )[0, 0]

        return user_tokens, user_mask

    def _compute_drone_pair_throughput(self, i: int) -> np.ndarray:
        """Drone-to-drone relay throughput, from observer drone `i`'s
        perspective. OBSERVER-DEPENDENT: unlike the PISA block, row j's
        value differs depending on which drone is doing the observing,
        so this must be computed per-observer (not cached/shared).

        Row j = [flow i->j, flow j->i], summed over every commodity's
        `per_commodity_flow[c][node_u][node_v]` entry and normalized
        exactly like columns 4/5 of `_build_user_set_for_active_drone`
        (throughputUserMinMaxTuple + 10*log10 dB transform). Row i
        (self) is left at [0, 0] pre-normalization.

        Returns:
            np.ndarray: shape (numbOfNodes, 2).
        """
        epsilon = 1e-9
        to_dB = lambda x: 10 * np.log10(x + epsilon)
        per_commodity = self.unwrapped_env.network.per_commodity_flow

        pair = np.zeros((self.numbOfNodes, 2), dtype=np.float32)
        node_i = self.unwrapped_env.nodes[i]

        for j in range(self.numbOfNodes):
            if j == i:
                continue
            node_j = self.unwrapped_env.nodes[j]
            to_j_bits = sum(
                flows.get(node_i, {}).get(node_j, 0)
                for flows in per_commodity.values()
            )
            from_j_bits = sum(
                flows.get(node_j, {}).get(node_i, 0)
                for flows in per_commodity.values()
            )
            pair[j, 0] = normalize_and_transform(
                np.array([[to_j_bits]]), self.throughputUserMinMaxTuple, to_dB
            )[0, 0]
            pair[j, 1] = normalize_and_transform(
                np.array([[from_j_bits]]), self.throughputUserMinMaxTuple, to_dB
            )[0, 0]

        return pair

    def _compute_n_features_per_node(self) -> int:
        """Raw own-state width (4: pos=2, signal=1, interference=1) plus
        the user-summary block width (8 pooled if encoder_mode=="frozen",
        else max_users*(6+1) padded-raw if "learned") plus the drone-pair
        relay throughput width (2). Everything downstream (cache sizing,
        `_emit_observation`, `get_observation_size`) is derived from
        this by the parent, so these propagate automatically."""
        return super()._compute_n_features_per_node() + self._user_block_width + 2

    def _pad_user_set(self, user_tokens: np.ndarray, user_mask: np.ndarray) -> np.ndarray:
        """encoder_mode=="learned" only. Pad this drone's (n_users_this_
        episode, 6) raw tokens + mask up to `_max_users` and flatten to a
        fixed-width row: [tok_0(6), mask_0, tok_1(6), mask_1, ...].
        n_users_this_episode <= _max_users always (numbOfUsersRange upper
        bound); padded rows are all-zero with mask=0 -- identical
        treatment to an unsensed user."""
        n_users = user_tokens.shape[0]
        padded_tokens = np.zeros((self._max_users, USER_TOKEN_DIM), dtype=np.float32)
        padded_mask = np.zeros((self._max_users,), dtype=np.float32)
        padded_tokens[:n_users] = user_tokens
        padded_mask[:n_users] = user_mask
        return np.hstack([padded_tokens, padded_mask[:, None]]).reshape(-1)

    def _compute_user_summary_block(self) -> np.ndarray:
        """(numbOfNodes, _user_block_width) block: row k is node k's own
        user-set summary, built from k's own sensing. This is a property
        of node k alone (not of whichever drone i is asking), so it is
        computed once per `get_observation` call and reused for every
        observer i below. Pooled (frozen) or padded-raw (learned)
        depending on encoder_mode -- see `_pad_user_set`.
        """
        block = np.zeros((self.numbOfNodes, self._user_block_width), dtype=np.float32)
        for k in range(self.numbOfNodes):
            self._active_drone = k
            try:
                user_tokens, user_mask = self._build_user_set_for_active_drone()
            finally:
                self._active_drone = None
            if self.encoder_mode == "frozen":
                block[k, :] = self._pisa(user_tokens, user_mask)
            else:
                block[k, :] = self._pad_user_set(user_tokens, user_mask)
        return block

    def _compute_default_features(self) -> np.ndarray:
        """Widen the parent's raw default block with a zero user-summary
        block and a zero drone-pair-throughput block, so never-reached
        peers get zeros for both — same treatment as their zeroed
        position/signal/interference."""
        raw_default = super()._compute_default_features()
        pisa_default = np.zeros(
            (raw_default.shape[0], self._user_block_width), dtype=raw_default.dtype
        )
        pair_default = np.zeros((raw_default.shape[0], 2), dtype=raw_default.dtype)
        return np.hstack([raw_default, pisa_default, pair_default])

    def get_observation(self) -> np.ndarray:
        """Copy of `PerDrone_OW.get_observation` with only the
        feature-assembly lines changed: the raw own-state block from
        `get_selected_node_observations()` is hstacked with the per-node
        user-summary block (see `_compute_user_summary_block`) before
        being written into the cache. All cache/history/aoi/mask/graph
        logic below is
        byte-identical to the parent.
        """
        self.updateComponents()
        self._wire_to_env()

        # Reseed fading per step (reproducible within an episode).
        _step = int(getattr(self.unwrapped_env, 'ep_step', 0))
        self._fading_rng = np.random.default_rng(seed=1000 + _step)

        live_capacity = self.unwrapped_env.network.getCapacityMatrix()

        if self._needs_cache_init:
            default_features = self._compute_default_features()  # (N_real, n_features)
            N_real = default_features.shape[0]
            max_N = self._max_numbOfNodes
            if N_real < max_N:
                pad = np.zeros((max_N - N_real, self._n_features), dtype=default_features.dtype)
                default_features = np.concatenate([default_features, pad], axis=0)
            self.cache[:, :, :, : self._n_features] = default_features[
                np.newaxis, :, np.newaxis, :
            ]
            self._needs_cache_init = False

        graph = self.unwrapped_env.network.getGraph()

        # Per-node user-summary block, shared across every observer i's
        # cache write this step (see `_compute_user_summary_block` docstring).
        pisa_block = self._compute_user_summary_block()

        assert FIX_BUGS, "drone-pair throughput requires FIX_BUGS"

        K = self.history_K

        for i in range(self.numbOfNodes):
            curr_node = self.unwrapped_env.nodes[i]

            ancestors: list = sorted(
                nx.ancestors(graph, curr_node), key=lambda x: x.id
            )
            if self.state_loss_rate > 0 and ancestors:
                rng_mask = self._rngs[i].binomial(
                    1, 1 - self.state_loss_rate, len(ancestors)
                )
                reached = [a for a, k in zip(ancestors, rng_mask) if k]
            else:
                reached = list(ancestors)
            reached.append(curr_node)  # self always reached

            reached_node_indices = sorted(
                {
                    self.unwrapped_env.node_to_idx[c]
                    for c in reached
                    if c in self.unwrapped_env.node_to_idx
                }
            )
            reached_user_indices = sorted(
                {
                    self.unwrapped_env.user_to_idx[c]
                    for c in reached
                    if c in self.unwrapped_env.user_to_idx
                }
            )

            if FIX_BUGS:
                self._update_per_drone_masked_capacity(
                    i, live_capacity, reached_node_indices, reached_user_indices
                )
                self._active_drone = i
                try:
                    features_i = np.hstack(
                        [
                            self.get_selected_node_observations(),
                            pisa_block,
                            self._compute_drone_pair_throughput(i),
                        ]
                    )
                finally:
                    self._active_drone = None
            else:
                features_i = shared_features

            if self.aoi:
                self._aoi_counter[i, :, :] += 1.0

            if reached_node_indices:
                idx = np.asarray(reached_node_indices, dtype=np.int64)
                if K > 1:
                    self.cache[i, idx, 1:, : self._n_features] = (
                        self.cache[i, idx, :-1, : self._n_features]
                    )
                    self._mask_buffer[i, idx, 1:] = self._mask_buffer[i, idx, :-1]
                    if self.aoi:
                        self._aoi_counter[i, idx, 1:] = self._aoi_counter[i, idx, :-1]
                self.cache[i, idx, 0, : self._n_features] = features_i[idx, :]
                self._mask_buffer[i, idx, 0] = True
                if self.aoi:
                    self._aoi_counter[i, idx, 0] = 0.0

        if self.aoi:
            self.cache[:, :, :, self._aoi_col] = np.minimum(
                self._aoi_counter / INITIAL_AOI, 1.0
            )

        return self._emit_observation()
