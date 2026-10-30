"""OM2GEnv — the OM2G track's environment.

╔══════════════════════════════════════════════════════════════════════════╗
║  OM2G TRACK (IEEE 802.15.4 narrowband, 250 kbps).                       ║
║  NOT the OMNET track (802.11g).  Numbers are NEVER comparable.           ║
║  See code/om2g/OM2G_README.md.                                           ║
╚══════════════════════════════════════════════════════════════════════════╝

Subclasses omnet_module/omnet_env.OMNeTEnv and changes exactly three things.

1. DELIVERY GATING (half the point of the track).
   On the OMNET track, observer i's row for peer j is filled in from the
   simulator's global state -- peer j's position and sensed-user set are handed
   over for free, whether or not any packet carrying them ever crossed the
   medium.  That makes "compression relieves network load" unprovable: nothing
   is on the wire, so nothing can cost anything.

   Here, drone j's state reaches observer i ONLY through the UDP multicast
   beacon that DroneBeaconApp actually transmits, whose payload size is the
   arm's real message size, over the same 802.15.4 medium that carries the user
   voice traffic.  RLBridge records when each (i, j) message was successfully
   received and reassembled; this class serves observer i the SNAPSHOT of peer
   j's message from that moment, and blanks the row entirely when no message
   has ever arrived.  A message lost to a collision, to range, or to one
   missing IPv4 fragment is genuinely absent from the observation.

2. CONTENT GATING (the other half).
   Delivery gating alone is not enough: what a peer receives must also be
   exactly what the radio is charged for (4*L+56 bytes), not the sender's
   full-width drone token.  The compressor therefore runs on the SENDER
   side, before delivery.

   The observation itself is sized by the arm.  What a drone broadcasts
   about itself is exactly OM2G_MSG_FIXED_FLOATS + L floats, where the
   L-float user summary is computed at SEND time, sender-side, by a frozen
   codec (see _OM2GUserCodec below), and that message -- nothing more -- is
   what a peer's token slot can ever contain:

     arm=compressed    L = OM2G_LATENT_DIM      (codec REQUIRED, from
                                                 $OM2G_CODEC_PATH)
     arm=uncompressed  L = max_users * features (identity codec: the raw
                                                 padded user block, verbatim
                                                 -- behaviour unchanged)
     arm=nocontrol     L = 0                    (no message content exists;
                                                 any delivered beacon raises)

   Per-agent observation (1-D, replaces the OMNET track's (MAX_N, W) token
   matrix; see om2g_obs_width()):

     [ own raw user block  max_users*features ]  local sensing, free, always
     [ peer slot 0 .. MAX_N-1, each 20+L wide ]  see om2g_peer_slot()

   The drone's OWN raw sensing never crosses a radio, so it stays raw and
   full-width regardless of the arm -- the L sweep varies ONLY what peers
   can know, never what a drone knows about its own users.

3. TELEMETRY.  Per-episode control-plane vs data-plane accounting is appended
   to $OM2G_TELEMETRY_CSV, so the network-load claim is backed by numbers from
   the run that produced the policy, not by a separate probe.

Reward, action scaling, seeding and the ZMQ protocol are inherited unchanged.
The MODEL side of this layout is architectures/user_encoder.py's
om2g_flat_layout()/split_om2g_obs(); the two are kept in lockstep by width
asserts on both sides that fail loudly, not by convention.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "omnet_module"))

import om2g_track as T

T.require_om2g("om2g_env.py")

from omnet_env import (  # noqa: E402
    MAX_N, MAX_USERS_CAP, OBS_FEATURES, OWN_STATE_W, OMNeTEnv,
    _peer_offset_scale,
)

# Staleness cap.  A peer whose last message is older than this is treated as
# unheard-from: the row is blanked and masked out.  15 sync intervals is
# deliberately generous -- the aim is to model "the message did not arrive",
# not to impose a second, tighter freshness policy on top of the radio.  The
# uncompressed arm's measured mean message age is already an order of
# magnitude above the compressed arm's, so this cap is not what separates them.
OM2G_MAX_MSG_AGE_S = float(os.environ.get("OM2G_MAX_MSG_AGE_S", 15.0))


def om2g_message_columns(token_width: int):
    """LEGACY. Split a full-width drone token's columns into
    (message, observer-local).  The env does not deliver full-width tokens
    -- see the module docstring's CONTENT GATING section and om2g_peer_slot()
    -- but probes/tests import this, so it is kept for reading older dumps.
    """
    if token_width < OWN_STATE_W + 5 + 1:
        raise ValueError(f"token_width={token_width} is too small to be an "
                         "OMNeT drone token")
    msg = [0, 1, 6] + list(range(OWN_STATE_W, token_width - 5)) + \
          [token_width - 4, token_width - 3, token_width - 2]
    return np.asarray(msg, dtype=np.intp)


# ── The arm-sized observation layout ────────────────────────────────────────
#
# The wire message a drone broadcasts about itself is
#     [x_abs, y_abs, nf, user_summary(L), event_latent(8), fwd, busy, t_rx]
# = OM2G_MSG_FIXED_FLOATS(14) + L floats = om2g_track.om2g_message_floats(),
# charged on air as 4*L + 56 bytes.  A peer's token slot is that message plus
# the 6 observer-LOCAL columns (offsets derived from the stale position, the
# observer's own RSSI measurement of the peer, the self flag and the live
# mask):
#
#   slot col            width  source
#   [0] x_abs, [1] y_abs   2   MESSAGE (stale)
#   [2] dx,    [3] dy      2   local (recomputed: stale peer pos, CURRENT own)
#   [4] rssi,  [5] valid   2   local (observer i's own radio)
#   [6] nf                 1   MESSAGE
#   [7 : 7+L] user summary L   MESSAGE  -- THE arm knob
#   [7+L : 15+L] event     8   MESSAGE
#   [15+L] self flag       1   local
#   [16+L] fwd             1   MESSAGE
#   [17+L] busy            1   MESSAGE
#   [18+L] t_rx            1   MESSAGE
#   [19+L] mask            1   local
#
# so the slot is 20 + L wide: the token is exactly as big as the information
# it carries.  L per arm: compressed = $OM2G_LATENT_DIM, uncompressed =
# max_users * features (identity codec), nocontrol = 0.

def om2g_arm_latent(arm: str) -> int:
    """The delivered per-peer user-summary width, per arm.

    Single source of truth for the ENV side. The model side recomputes it
    from the same env vars in architectures/user_encoder.om2g_flat_layout()
    and the two are collision-checked by width asserts on both sides.
    """
    if arm == "compressed":
        return int(os.environ.get("OM2G_LATENT_DIM", "8"))
    if arm == "uncompressed":
        return T.OM2G_MAX_USERS_CAP * T.OM2G_OBS_FEATURES
    if arm == "nocontrol":
        return 0
    raise ValueError(f"unknown OM2G arm {arm!r}")


def om2g_peer_slot_width(latent: int) -> int:
    return 20 + latent


def om2g_obs_width(latent: int) -> int:
    """Per-agent observation width: own raw block + MAX_N peer slots."""
    return MAX_USERS_CAP * OBS_FEATURES + MAX_N * om2g_peer_slot_width(latent)


def _quantize_geo(xy: np.ndarray) -> np.ndarray:
    """Quantize the transmitted position pair to what $OM2G_GEO_BYTES pays
    for: OM2G_GEO_BYTES*4 bits per normalised [-1,1] coordinate. 8 (default,
    two float32s) is a no-op (message byte-identical to the float layout).
    0 transmits no position at all (zeros). The DELIVERED content is the
    quantized value, so charge == content."""
    gb = T.OM2G_GEO_BYTES
    if gb >= 8:
        return xy
    if gb == 0:
        return np.zeros_like(xy)
    levels = float((1 << (gb * 4)) - 1)
    q = np.round((np.clip(xy, -1.0, 1.0) + 1.0) * 0.5 * levels) / levels
    return (q * 2.0 - 1.0).astype(np.float32)


class _OM2GUserCodec:
    """The sender-side user-set compressor: raw (U, F) block -> (L,) floats.

    This IS the codec both ends of the radio link know, in the ordinary
    communications sense: a FROZEN DeepSets encoder (phi -> masked sum ->
    to_latent) trained OFFLINE by reconstruction on collected sensed-user
    sets (encoder_experiments/train_frozen_omnet_encoder.py), loaded from
    $OM2G_CODEC_PATH.  It runs in the env, at SEND time, because that is
    where a real drone would run it -- and because the env worker lives in
    its own ParallelEnv process, where the model's live weights physically
    do not exist.

    WHY a frozen codec in frozen mode: with the encoder in the receiver, the
    raw block would have to be delivered so the encoder could run on it.
    Gradients do not cross radios; a compressor whose weights are co-trained
    with the receiving policy via backprop THROUGH the delivered message is
    only implementable by delivering more than the message.  (The e2e codec
    modes instead share a deterministic encoder across drones; see the codec
    mode notes in __init__.)
    """

    def __init__(self, path: str, latent: int):
        import hashlib
        import json as _json
        from frozen_encoder import FrozenOMNeTUserEncoder  # src/MANET, on path
        cfg_path = str(path).replace(".pt", "_cfg.json")
        try:
            with open(cfg_path) as fh:
                cfg = _json.load(fh)
        except OSError as e:
            raise RuntimeError(
                f"OM2G codec config {cfg_path} unreadable ({e}). Every codec "
                ".pt ships with the _cfg.json the trainer wrote; do not "
                "hand-build one.") from e
        if int(cfg.get("out_dim", -1)) != latent:
            raise RuntimeError(
                f"OM2G codec {path} has out_dim={cfg.get('out_dim')!r} but the "
                f"arm's latent is {latent} ($OM2G_LATENT_DIM). The wire charge "
                f"is 4*{latent}+56 B; delivering a different width would "
                f"charge one size and deliver another. Train "
                f"a codec at --output {latent} or fix the env var.")
        import torch as _torch
        _sd = _torch.load(path, map_location="cpu")
        _sd = _sd.get("state_dict", _sd) if isinstance(_sd, dict) else _sd
        _codec_dim = int(_sd["phi.0.weight"].shape[1])
        if _codec_dim != OBS_FEATURES:
            raise RuntimeError(
                f"OM2G codec {path} was trained on {_codec_dim}-wide user "
                f"tokens but this run's OBS_FEATURES={OBS_FEATURES} "
                f"($OM2G_USER_PARTNER_TAG "
                f"{'set' if os.environ.get('OM2G_USER_PARTNER_TAG') else 'unset'}). "
                f"A codec is bound to the token schema it was trained on -- "
                f"collect and train one under the SAME partner-tag setting.")
        self._enc = FrozenOMNeTUserEncoder(
            path, dim=_codec_dim,
            hidden=int(cfg["hidden"]), out_dim=latent)
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        self.sha = h.hexdigest()[:16]
        self.out_dim = latent
        self.path = os.path.abspath(path)

    def encode(self, user_block: np.ndarray) -> np.ndarray:
        z = self._enc.encode(user_block)
        assert z.shape == (self.out_dim,), (
            f"codec emitted {z.shape}, expected ({self.out_dim},)")
        return z


HAND_CODEC_MODES = ("hand_users", "hand_agg", "hand_none")


class _OM2GHandCodec:
    """Deterministic, parameter-free SENDER-side codecs for the content
    ablation: the same L floats on the wire, the same charge, a different
    MEANING.

      hand_users  the L//2 strongest sensed users as (bearing/pi, rssi)
                  pairs -- a per-user sketch, no aggregation
      hand_agg    a cluster-level digest (counts, mean bearing vector,
                  rssi stats, throughput, interference, partner-tag mass),
                  zero-padded to L
      hand_none   L zeros: only the fixed 14 floats (position, noise
                  floor, event latent, fwd/busy/trx) carry information,
                  and the L floats of airtime are still paid for

    Input is the raw (MAX_USERS_CAP, OBS_FEATURES) block; `present` is
    column 12 (see omnet_env's column layout). Same .encode/.sha/.out_dim
    interface as _OM2GUserCodec so the rest of the env is untouched.
    """

    def __init__(self, kind: str, latent: int):
        if kind not in HAND_CODEC_MODES:
            raise ValueError(f"hand codec kind must be one of "
                             f"{HAND_CODEC_MODES}, got {kind!r}")
        self.kind = kind
        self.out_dim = int(latent)
        self.sha = kind
        self.path = kind

    @staticmethod
    def _rssi_norm(r):
        # -100 dBm (decode floor) -> 0, -60 dBm -> 1
        return np.clip((np.asarray(r, dtype=np.float32) + 100.0) / 40.0,
                       0.0, 1.0)

    def encode(self, user_block: np.ndarray) -> np.ndarray:
        L = self.out_dim
        z = np.zeros(L, dtype=np.float32)
        blk = np.asarray(user_block, dtype=np.float32)
        assert blk.ndim == 2 and blk.shape[1] >= 13, blk.shape
        present = blk[:, 12] > 0.5
        if self.kind == "hand_none" or not present.any():
            return z
        rows = blk[present]
        if self.kind == "hand_users":
            k = L // 2
            order = np.argsort(-rows[:, 1])[:k]          # strongest rssi first
            sel = rows[order]
            n = sel.shape[0]
            z[0:2 * n:2] = np.clip(sel[:, 0] / np.pi, -1.0, 1.0)
            z[1:2 * n:2] = self._rssi_norm(sel[:, 1])
            return z
        # hand_agg: a digest of the sensed set
        cap = float(blk.shape[0])
        n = rows.shape[0]
        rn = self._rssi_norm(rows[:, 1])
        feats = [
            n / cap,                                    # sensed count
            float(rows[:, 6].sum()) / cap,              # served count
            float(np.cos(rows[:, 0]).mean()),           # mean bearing vector
            float(np.sin(rows[:, 0]).mean()),
            float(rn.mean()), float(rn.max()),          # rssi stats
            float(rows[:, 5].mean()),                   # throughput (already [-1,1])
            float(np.clip((rows[:, 3].mean() + 100.0) / 40.0, 0.0, 1.0)),  # interference
        ]
        if blk.shape[1] >= 17:                          # partner-tag mass
            feats.extend(float(v) for v in rows[:, 13:17].mean(axis=0))
        feats = feats[:L]
        z[:len(feats)] = np.asarray(feats, dtype=np.float32)
        return z


class OM2GEnv(OMNeTEnv):
    """OMNeTEnv with the inter-agent state message forced onto the radio."""

    metadata = {"render_modes": [], "name": "om2g_parallel_v0"}

    def __init__(self, *, worker_index: int, encoder_mode: str = "learned",
                 arm: str | None = None):
        T.require_om2g("OM2GEnv")
        if encoder_mode != "learned":
            raise ValueError(
                "OM2G requires encoder_mode='learned': the raw per-user block "
                "must reach the model, which then applies the ARM'S encoder "
                "(LearnableUserSetEncoder for the compressed arm, "
                "PassthroughUserSetEncoder for the uncompressed one). "
                f"Got {encoder_mode!r}.")
        super().__init__(worker_index=worker_index, encoder_mode=encoder_mode)

        # Opt-in "smart"/learned send-gate: the policy itself gets a 3rd
        # action dimension (in [-1,1], same convention as dx/dy) deciding
        # whether THIS drone sends its beacon this step (>0 = send). Default
        # off leaves self._act_space at the inherited (2,) shape, so every
        # existing OM2G *and* OMNET-track instance is byte-identical --
        # this only changes anything for an env explicitly constructed with
        # OM2G_LEARNED_GATE=1. See omnet_env.step()'s shape-conditioned branch.
        self._learned_gate = os.environ.get("OM2G_LEARNED_GATE", "0") == "1"
        if self._learned_gate:
            from gymnasium import spaces as _spaces
            self._act_space = _spaces.Box(
                low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self._arm = arm or os.environ.get("OM2G_ARM", "compressed")
        # "nocontrol" (never-send floor) is model-shape-identical to
        # "compressed" -- same learned encoder, same 8-dim latent -- it just
        # never receives a real beacon (DroneBeaconApp/om2g.ini's
        # OM2G_NOCONTROL config), so peer tokens are always empty via the
        # ordinary staleness gate below. It is NOT a valid input to
        # om2g_track.om2g_message_bytes()/om2g_frames_per_message(), which
        # only know compressed/uncompressed -- handled below.
        if self._arm not in ("compressed", "uncompressed", "nocontrol"):
            raise ValueError(f"OM2G_ARM must be 'compressed', 'uncompressed' "
                             f"or 'nocontrol', got {self._arm!r}")

        # ── Arm-sized layout ──────────────────────────────────────────────
        # Widths are asserted, not assumed: every width is computed at two
        # sites (env and model), and the asserts make them agree.
        if MAX_USERS_CAP != T.OM2G_MAX_USERS_CAP:
            raise RuntimeError(
                f"omnet_env.MAX_USERS_CAP={MAX_USERS_CAP} != "
                f"om2g_track.OM2G_MAX_USERS_CAP={T.OM2G_MAX_USERS_CAP}. "
                f"Export OMNET_MAX_USERS_CAP={T.OM2G_MAX_USERS_CAP} (om2g_"
                f"config.om2g_env_vars() does this; a hand launch must too).")
        if OBS_FEATURES != T.OM2G_OBS_FEATURES:
            raise RuntimeError(
                f"omnet_env.OBS_FEATURES={OBS_FEATURES} != om2g_track."
                f"OM2G_OBS_FEATURES={T.OM2G_OBS_FEATURES}: the two sides of "
                f"the $OM2G_USER_PARTNER_TAG gate disagree.")
        self._latent = om2g_arm_latent(self._arm)
        if self._arm in ("compressed", "uncompressed"):
            _wire = T.om2g_message_bytes(self._arm, self._latent)
            _expect = T.OM2G_GEO_BYTES + 4 * (12 + self._latent)
            if _wire != _expect:
                raise RuntimeError(
                    f"wire bytes {_wire} != GEO({T.OM2G_GEO_BYTES}) + "
                    f"4*(12+{self._latent}) = {_expect}: om2g_track's "
                    f"message accounting and this env's delivered content "
                    f"have drifted apart -- refusing to charge one size and "
                    f"deliver another.")
            _enc_env = os.environ.get("OMNET_USER_ENC_OUT_DIM")
            if _enc_env is not None and int(_enc_env) != self._latent:
                raise RuntimeError(
                    f"$OMNET_USER_ENC_OUT_DIM={_enc_env} but arm="
                    f"{self._arm} delivers latent={self._latent} "
                    f"($OM2G_LATENT_DIM). om2g_config exports the two in "
                    f"lockstep; a disagreement means a half-configured run.")
        # ── Codec mode ────────────────────────────────────────────────────
        # frozen (default): the L-float summary is computed at SEND time by
        #   the frozen offline codec below. Nothing but the message exists.
        # e2e: the summary is computed MODEL-side by a trainable encoder E
        #   shared across drones -- equivalent to sender-side compression
        #   because E is deterministic and parameter-shared. The env then
        #   delivers, alongside the
        #   arm-sized peer slots (whose L-summary columns stay ZERO), a
        #   DISTINCT trailing "inbox" section carrying the sender's raw user
        #   block from the delivered message's rx step. The inbox is the
        #   training-process representation of the L floats E would have
        #   transmitted; the WIRE is still charged 4L+56 B exactly as in
        #   frozen mode, and the model-side funnel (inbox -> E -> L floats)
        #   is enforced by the hook invariant test in
        #   om2g_compressfix_check.py, not by convention.
        self._codec_mode = os.environ.get("OM2G_CODEC_MODE", "frozen")
        # e2e_rt (round trip) is, on the SENDER side and ON THE
        # WIRE, EXACTLY e2e: the summary columns go out zeroed, the raw block
        # rides in the inbox, and beaconBytes is still
        # om2g_message_bytes(arm, OM2G_LATENT_DIM) = 4L+56 -- om2g_config.py
        # :165 goes through _wire_latent -> ARM_LATENT_DIM -> $OM2G_LATENT_DIM
        # and reads $OM2G_CODEC_MODE nowhere. The round trip is purely
        # receiver-side, inside the model. Collapse the alias HERE, before any
        # env-side branch, so not one byte of environment behaviour can differ
        # between the two families; _rt_decode survives only for the banner.
        self._rt_decode = (self._codec_mode == "e2e_rt")
        if self._rt_decode:
            self._codec_mode = "e2e"
        if self._codec_mode not in ("frozen", "e2e") + HAND_CODEC_MODES:
            raise ValueError(
                f"OM2G_CODEC_MODE must be 'frozen', 'e2e' or one of "
                f"{HAND_CODEC_MODES}, got {self._codec_mode!r}")
        if self._codec_mode in HAND_CODEC_MODES and self._arm != "compressed":
            raise RuntimeError(
                f"OM2G_CODEC_MODE={self._codec_mode} is only defined for "
                f"arm=compressed, got arm={self._arm!r}.")
        if self._codec_mode == "e2e" and self._arm != "compressed":
            raise RuntimeError(
                f"OM2G_CODEC_MODE=e2e is only defined for arm=compressed "
                f"(uncompressed is an identity codec, nocontrol sends "
                f"nothing); got arm={self._arm!r}. Unset OM2G_CODEC_MODE.")
        self._codec = None
        if self._arm == "compressed" and self._codec_mode in HAND_CODEC_MODES:
            self._codec = _OM2GHandCodec(self._codec_mode, self._latent)
        if self._arm == "compressed" and self._codec_mode == "frozen":
            _codec_path = os.environ.get("OM2G_CODEC_PATH")
            if not _codec_path:
                raise RuntimeError(
                    "arm=compressed but $OM2G_CODEC_PATH is unset. In frozen "
                    "codec mode the L-float user summary is computed at SEND "
                    "time by a frozen offline codec. Train one:\n"
                    "  python om2g/om2g_collect_codec_data.py --episodes 40 "
                    "--out codec_data/om2g_sets.npz\n"
                    "  python src/encoder_experiments/train_frozen_omnet_"
                    "encoder.py --npz ... --output "
                    f"{self._latent} --out codec_o{self._latent}.pt\n"
                    "then export OM2G_CODEC_PATH=codec_o"
                    f"{self._latent}.pt")
            self._codec = _OM2GUserCodec(_codec_path, self._latent)

        self._slot_w = om2g_peer_slot_width(self._latent)
        self._raw_w = MAX_USERS_CAP * OBS_FEATURES
        self._inbox_w = (MAX_N * self._raw_w
                         if self._codec_mode == "e2e" else 0)
        self._obs_w = om2g_obs_width(self._latent) + self._inbox_w
        # Replace the inherited (MAX_N, token_width) matrix space with the
        # arm-sized flat vector. OMNET-track envs never reach this line.
        from gymnasium import spaces as _sp
        self._obs_space = _sp.Box(low=-np.inf, high=np.inf,
                                  shape=(self._obs_w,), dtype=np.float32)

        # step -> (MAX_N, 14 + L) snapshot of the exact wire message each
        # drone broadcast at that step. Bounded and cleared every reset.
        self._msg_history: dict[int, np.ndarray] = {}
        # e2e only: step -> (MAX_N, raw_w) snapshot of each drone's raw user
        # block, the input E would have compressed at send time. Same
        # bounding/clearing as _msg_history.
        self._raw_history: dict[int, np.ndarray] = {}
        self._telemetry: list[dict] = []
        self._telemetry_csv = os.environ.get("OM2G_TELEMETRY_CSV", "")
        self._episode_index = 0

        if worker_index == 0:
            if self._arm == "nocontrol":
                msg_desc = ("0 content floats delivered (4 B sender-ID "
                            "beacon; a delivered beacon RAISES)")
            else:
                n_frames = T.om2g_frames_per_message(self._arm, self._latent)
                _geo = ("" if T.OM2G_GEO_BYTES == 8
                        else f" geo={T.OM2G_GEO_BYTES}B/pair")
                msg_desc = (f"{T.om2g_message_bytes(self._arm, self._latent)}"
                            f" B ({n_frames} x 802.15.4 frame), "
                            f"{14 + self._latent} floats delivered{_geo}")
            print(f"[OM2GEnv] TRACK=OM2G arm={self._arm} L={self._latent} "
                  f"msg={msg_desc} "
                  f"slot={self._slot_w} obs={self._obs_w} "
                  f"codec={self._codec.sha if self._codec else ((('e2e_rt' if self._rt_decode else 'e2e') + '(model-side E, inbox +' + str(self._inbox_w) + ' cols)') if self._codec_mode == 'e2e' else ('identity' if self._arm == 'uncompressed' else 'none'))} "
                  f"max_msg_age={OM2G_MAX_MSG_AGE_S}s "
                  f"beacon_interval={T.OM2G_SYNC_INTERVAL_S}s "
                  f"rl_step={T.OM2G_RL_STEP_S}s",
                  flush=True)

    # ── delivery gating ──────────────────────────────────────────────────────

    def _build_obs(self, obs_json) -> np.ndarray:
        tokens = super()._build_obs(obs_json)
        return self._compress_and_gate(tokens, obs_json)

    def _wire_message(self, own_token: np.ndarray, U: int) -> np.ndarray:
        """drone j's own full-width token row -> the 14+L floats it puts on
        the air. THE compression step: the raw user block enters, the arm's
        L-float summary leaves, and nothing else of the block survives."""
        L = self._latent
        if self._arm == "compressed":
            if self._codec_mode == "e2e":
                # The L floats are computed MODEL-side by the shared E; the
                # slot's summary columns stay zero and the raw block rides in
                # the inbox section instead. The wire charge is unchanged.
                summary = np.zeros(L, dtype=np.float32)
            else:
                summary = self._codec.encode(
                    own_token[OWN_STATE_W:OWN_STATE_W + U]
                    .reshape(MAX_USERS_CAP, OBS_FEATURES))
        elif self._arm == "uncompressed":
            # Identity codec: the raw padded block, verbatim -- the
            # no-compressor baseline pays for all of it.
            summary = own_token[OWN_STATE_W:OWN_STATE_W + U]
        else:                       # nocontrol: no content exists to send
            summary = np.zeros(0, dtype=np.float32)
        msg = np.concatenate([
            _quantize_geo(own_token[0:2]),               # x_abs, y_abs (paid
                                                         # at OM2G_GEO_BYTES)
            own_token[6:7],                              # nf
            summary,                                     # L floats
            own_token[OWN_STATE_W + U:OWN_STATE_W + U + 8],   # event latent
            own_token[OWN_STATE_W + U + 9:OWN_STATE_W + U + 12],  # fwd,busy,trx
        ]).astype(np.float32)
        # Delivered payload width == the arm's charged width. A mismatch
        # crashes rather than producing a number.
        expected = 14 + L
        assert msg.shape == (expected,), (
            f"wire message is {msg.shape[0]} floats, charged as {expected} "
            f"(arm={self._arm}, L={L}) -- content/charge mismatch.")
        return msg

    def _compress_and_gate(self, tokens: np.ndarray, obs_json) -> np.ndarray:
        """Full-width base tokens (MAX_N, MAX_N, 20+U) -> the arm-sized flat
        observation (MAX_N, raw_w + MAX_N*(20+L)). See the module docstring.
        """
        W = tokens.shape[-1]
        U = MAX_USERS_CAP * OBS_FEATURES
        assert W == 20 + U, (
            f"base drone token is {W} wide, expected 20+{U}: "
            f"omnet_env.build_drone_tokens' layout moved and this slicer "
            f"must move with it (own 7 | users {U} | event 8 | self,fwd,"
            f"busy,trx | mask).")
        L = self._latent
        Wp = self._slot_w
        step = int(obs_json.get("step", 0))
        positions = obs_json.get("positions", [])
        n_live = len(positions)
        arena_w = float(obs_json.get("arena_width", T.OM2G_ARENA_M))
        arena_h = float(obs_json.get("arena_height", T.OM2G_ARENA_M))

        # 1. SENDER-SIDE COMPRESSION: what each drone puts on the air now.
        #    tokens[j, j] is drone j's own state; _wire_message reduces it to
        #    the exact charged payload before it ever enters the history, so
        #    nothing wider than the message CAN be delivered later.
        snap = np.zeros((MAX_N, 14 + L), dtype=np.float32)
        for j in range(min(MAX_N, n_live)):
            snap[j] = self._wire_message(tokens[j, j], U)
        self._msg_history[step] = snap
        if self._codec_mode == "e2e":
            self._raw_history[step] = np.array(
                tokens[np.arange(MAX_N), np.arange(MAX_N),
                       OWN_STATE_W:OWN_STATE_W + U], copy=True)
        if len(self._msg_history) > 400:                 # bound the memory
            for old in sorted(self._msg_history)[:100]:
                del self._msg_history[old]
                self._raw_history.pop(old, None)

        age = np.asarray(obs_json.get("beacon_rx_age", []), dtype=float)
        valid = np.asarray(obs_json.get("beacon_rx_valid", []), dtype=float)
        if age.size == 0 or valid.size == 0:
            # Old simulator binary: the beacon bookkeeping is not in the obs
            # JSON at all. Fail loudly rather than train an OM2G arm on
            # OMNET-style free peer information.
            raise RuntimeError(
                "obs JSON carries no 'beacon_rx_age'/'beacon_rx_valid'. The "
                "simulator binary predates the OM2G instrumentation; rebuild "
                "code/omnet_smoke (make MODE=release). Refusing to run with "
                "ungated peer state.")

        # ── multi-hop control gossip (OM2G_FORWARD_MODE) ──────────────────
        # H=1 (default) is the original behaviour: a drone knows only the peers
        # whose beacon it received DIRECTLY. H>=2 lets a beacon reach i via a
        # relay k, using the ages the radio actually measured; the relayed age
        # is age[i][k] + age[k][j], so relayed rows are older and the existing
        # staleness gate still applies to them. Direct always beats relayed.
        # Two settings only, because "2 hops" is an arbitrary cut that no real
        # relay protocol makes: a beacon either stays local or it floods until
        # the connected component is covered.
        #   OM2G_FORWARD_MODE=none  (default) -- direct neighbours only
        #   OM2G_FORWARD_MODE=flood           -- iterate to a FIXED POINT
        _mode = os.environ.get("OM2G_FORWARD_MODE", "none").strip().lower()
        # Rows that arrived via a RELAY, not a direct beacon. Under
        # OM2G_FORWARD_PAYLOAD=digest (the default) a relay pays airtime for
        # only 2 floats (x, y) per forwarded record -- so that is ALL a
        # relayed row may deliver. payload=full charges base floats per
        # record and delivers the full message, so delivered content always
        # equals charged content.
        relayed = np.zeros_like(np.asarray(valid, dtype=bool)) \
            if getattr(valid, "ndim", 0) == 2 else None
        _digest_only = (os.environ.get("OM2G_FORWARD_PAYLOAD", "digest")
                        != "full")
        if _mode == "flood" and valid.ndim == 2 and valid.shape[0] == valid.shape[1]:
            n_h = valid.shape[0]
            for _ in range(n_h):          # a fixed point is reached in < n_h passes
                v_new = valid.copy()
                a_new = age.copy()
                for i in range(n_h):
                    for j in range(n_h):
                        if i == j or valid[i, j] > 0.0:
                            continue          # direct link wins, never override
                        best = -1.0
                        for k in range(n_h):
                            if k == i or k == j:
                                continue
                            if valid[i, k] <= 0.0 or valid[k, j] <= 0.0:
                                continue
                            if age[i, k] < 0.0 or age[k, j] < 0.0:
                                continue
                            tot = age[i, k] + age[k, j]
                            if best < 0.0 or tot < best:
                                best = tot
                        if best >= 0.0:
                            v_new[i, j] = 1.0
                            a_new[i, j] = best
                            relayed[i, j] = True
                if np.array_equal(v_new, valid) and np.array_equal(a_new, age):
                    break                 # nothing new propagated -- component covered
                valid, age = v_new, a_new

        peer_scale = _peer_offset_scale()
        # RL step clock, NOT the beacon period: beacon_rx_age is in
        # seconds and one RL step is 1 s regardless of how often the
        # beacon fires.
        dt = T.OM2G_RL_STEP_S
        n_stale = n_blank = n_fresh = 0

        # 2. THE ARM-SIZED PEER MATRIX. Rows default to all-zero (= blanked
        #    AND masked): a peer is invisible until its message arrives.
        out = np.zeros((MAX_N, MAX_N, Wp), dtype=np.float32)
        inbox = (np.zeros((MAX_N, MAX_N, self._raw_w), dtype=np.float32)
                 if self._codec_mode == "e2e" else None)

        def _scatter_msg(row: np.ndarray, msg: np.ndarray) -> None:
            """wire message (14+L floats) -> its slot columns."""
            row[0:2] = msg[0:2]                    # x_abs, y_abs
            row[6] = msg[2]                        # nf
            row[7:15 + L] = msg[3:11 + L]          # user summary + event
            row[16 + L:19 + L] = msg[11 + L:14 + L]  # fwd, busy, t_rx

        for i in range(min(MAX_N, n_live)):
            ix, iy = positions[i]
            # Own row: a drone knows itself, for free. The user-summary slot
            # carries its own outgoing message content -- i.e. what its peers
            # would see -- while its full raw sensing rides in the flat block
            # appended below.
            _scatter_msg(out[i, i], snap[i])
            out[i, i, 0:2] = tokens[i, i, 0:2]   # own position: exact --
            # a drone knows where IT is; only the transmitted copy is
            # quantized to what OM2G_GEO_BYTES pays for.
            if inbox is not None:
                # own row: E runs on the drone's own current sensing
                inbox[i, i] = tokens[i, i, OWN_STATE_W:OWN_STATE_W + U]
            out[i, i, 4:6] = 0.0                   # rssi of self: meaningless
            out[i, i, 15 + L] = 1.0                # self flag
            out[i, i, 19 + L] = tokens[i, i, W - 1]  # live mask

            for j in range(min(MAX_N, n_live)):
                if i == j:
                    continue
                a = age[i][j] if i < len(age) and j < len(age[i]) else -1.0
                v = valid[i][j] if i < len(valid) and j < len(valid[i]) else 0.0
                if v <= 0.0 or a < 0.0 or a > OM2G_MAX_MSG_AGE_S:
                    n_blank += 1                   # row stays all-zero
                    continue
                if self._arm == "nocontrol":
                    # om2g.ini's OM2G_NOCONTROL never fires a beacon inside an
                    # episode. A delivered beacon here means beaconInterval
                    # was overridden and the run is not the never-communicate
                    # floor.
                    raise RuntimeError(
                        f"arm=nocontrol but the radio delivered a beacon "
                        f"({i}<-{j}, age={a:.2f}s, step={step}). "
                        f"beaconInterval is being overridden somewhere "
                        f"(check om2g_config's extra args vs om2g.ini).")
                rx_step = int(round(step - a / dt))
                if rx_step not in self._msg_history:
                    known = [s for s in self._msg_history if s <= rx_step]
                    if not known:
                        n_blank += 1               # row stays all-zero
                        continue
                    rx_step = max(known)
                msg = self._msg_history[rx_step][j]
                assert msg.shape == (14 + L,), (
                    f"history snapshot for drone {j}@{rx_step} is "
                    f"{msg.shape[0]} floats, wire message is {14 + L} "
                    f"(arm={self._arm}) -- delivered != charged.")
                _scatter_msg(out[i, j], msg)
                if inbox is not None:
                    _rh = self._raw_history.get(rx_step)
                    assert _rh is not None, (
                        f"e2e: raw history missing for rx_step={rx_step} "
                        f"while msg history has it -- the two snapshots "
                        f"drifted apart.")
                    inbox[i, j] = _rh[j]
                if (relayed is not None and _digest_only
                        and i < relayed.shape[0] and j < relayed.shape[1]
                        and relayed[i, j]):
                    # Digest-relayed: the relay paid for (x, y) ONLY. Keep
                    # the position, blank every other message column --
                    # delivered content == charged content, always.
                    keep_xy = out[i, j, 0:2].copy()
                    out[i, j, :] = 0.0
                    out[i, j, 0:2] = keep_xy
                    if inbox is not None:
                        inbox[i, j] = 0.0   # digest paid for (x, y) only
                # Local columns, observer-computed: offsets from the STALE
                # peer position and the CURRENT own position; the observer's
                # own live RSSI measurement of peer j; the live mask.
                jx = (out[i, j, 0] + 1.0) * 0.5 * arena_w
                jy = (out[i, j, 1] + 1.0) * 0.5 * arena_h
                out[i, j, 2] = np.clip((jx - ix) / peer_scale, -1.0, 1.0)
                out[i, j, 3] = np.clip((jy - iy) / peer_scale, -1.0, 1.0)
                out[i, j, 4:6] = tokens[i, j, 4:6]
                out[i, j, 19 + L] = tokens[i, j, W - 1]
                if rx_step == step:
                    n_fresh += 1
                else:
                    n_stale += 1

        self._record_telemetry(obs_json, n_fresh, n_stale, n_blank)

        # 3. FLAT ASSEMBLY: [own raw user block | MAX_N arm-sized peer slots].
        raw = tokens[np.arange(MAX_N), np.arange(MAX_N),
                     OWN_STATE_W:OWN_STATE_W + U]
        parts = [raw, out.reshape(MAX_N, MAX_N * Wp)]
        if inbox is not None:
            parts.append(inbox.reshape(MAX_N, MAX_N * self._raw_w))
        obs = np.concatenate(parts, axis=1).astype(np.float32)
        assert obs.shape == (MAX_N, self._obs_w), (
            f"built obs {obs.shape}, spec says (MAX_N={MAX_N}, "
            f"{self._obs_w}) -- om2g_obs_width and this builder disagree.")
        return obs

    # ── telemetry ────────────────────────────────────────────────────────────

    def _record_telemetry(self, obs_json, n_fresh, n_stale, n_blank):
        ctrl_tx = float(np.sum(obs_json.get("ctrl_bytes_tx", [0.0])))
        ctrl_rx = float(np.sum(obs_json.get("ctrl_bytes_rx", [0.0])))
        total_rx = float(np.sum(obs_json.get("throughput", [0.0])))
        self._telemetry.append(dict(
            step=int(obs_json.get("step", 0)),
            ctrl_bps_tx=ctrl_tx * 8.0,
            ctrl_bps_rx=ctrl_rx * 8.0,
            # throughput counts every frame a drone receives, beacons included,
            # so the data plane is the remainder. Reported, not asserted: the
            # two are metered at different layers (UDP payload vs MAC frame),
            # so treat this as an estimate and channel_busy as the truth.
            data_bps_est=max(0.0, (total_rx - ctrl_rx) * 8.0),
            channel_busy=float(np.mean(obs_json.get("channel_busy", [0.0]))),
            peers_fresh=n_fresh, peers_stale=n_stale, peers_blank=n_blank,
        ))

    def _flush_telemetry(self):
        if not self._telemetry or not self._telemetry_csv:
            self._telemetry = []
            return
        rows = self._telemetry
        agg = {k: float(np.mean([r[k] for r in rows]))
               for k in rows[0] if k != "step"}
        n_pairs = max(1.0, agg["peers_fresh"] + agg["peers_stale"] + agg["peers_blank"])
        out = dict(
            track="OM2G", arm=self._arm,
            worker=self._worker_index, episode=self._episode_index,
            steps=len(rows),
            ctrl_share_of_channel=agg["ctrl_bps_tx"] / T.OM2G_BITRATE_BPS,
            peer_available_frac=(agg["peers_fresh"] + agg["peers_stale"]) / n_pairs,
            peer_fresh_frac=agg["peers_fresh"] / n_pairs,
            **agg,
        )
        write_header = not os.path.exists(self._telemetry_csv)
        try:
            with open(self._telemetry_csv, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(out))
                if write_header:
                    w.writeheader()
                w.writerow(out)
        except OSError:
            pass          # telemetry must never take a training run down
        self._telemetry = []

    # ── episode boundaries ───────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        self._flush_telemetry()
        self._msg_history.clear()
        self._raw_history.clear()
        self._episode_index += 1
        return super().reset(seed=seed, options=options)

    def step(self, actions):
        out = super().step(actions)
        if not self.agents:               # episode ended this step
            self._flush_telemetry()
            self._msg_history.clear()
            self._raw_history.clear()
        return out
