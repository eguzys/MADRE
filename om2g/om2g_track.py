"""OM2G track — constants and the anti-confusion guard.

╔══════════════════════════════════════════════════════════════════════════╗
║  OM2G  =  the IEEE 802.15.4 narrowband message-compression track.        ║
║  It is NOT the OMNET track (802.11g).  Numbers are NEVER comparable.     ║
║  See code/om2g/OM2G_README.md.                                           ║
╚══════════════════════════════════════════════════════════════════════════╝

This module is the SINGLE SOURCE OF TRUTH for every OM2G number. The
simulator config (om2g.ini), the reward radio model (max_flow_reward), the
environment (om2g_env) and the docs all derive from the constants below --
nothing may re-type one of them. Every constant carries the derivation that
produced it, so that a physical scale never exists in two places.
"""
from __future__ import annotations

import math
import os

# ── The guard ────────────────────────────────────────────────────────────────
TRACK = "OM2G"
_TRACK_ENV = "OM2G_TRACK"


def require_om2g(what: str = "") -> None:
    """Refuse to run unless the caller has explicitly opted into the OM2G track.

    Paired with the mirror-image check in omnet_module/omnet_env.py, which
    refuses to run WITH $OM2G_TRACK set. Between them, an OMNET component and
    an OM2G component can never be loaded into the same process by accident,
    and no OM2G artefact can be produced by an OMNET launcher (or vice versa).
    """
    if os.environ.get(_TRACK_ENV) != "1":
        raise RuntimeError(
            f"{what or 'this component'} belongs to the OM2G track "
            f"(IEEE 802.15.4 narrowband) but {_TRACK_ENV}=1 is not exported.\n"
            "OM2G and OMNET are two different experiments whose numbers are "
            "NOT comparable -- see code/om2g/OM2G_README.md.\n"
            "If you meant OM2G:   export OM2G_TRACK=1\n"
            "If you meant OMNET:  use code/omnet_module/, not code/om2g/.")


# ── Radio: IEEE 802.15.4-2006 narrowband, 2.45 GHz, DSSS-OQPSK-16 ────────────
# Every value below is INET 4.6's own Ieee802154NarrowbandRadio.ned default,
# quoted here so the reward model and the simulator cannot disagree. The
# citations in that file are to 802.15.4-2006 and the ATmega256RFR2 datasheet.
OM2G_CENTER_FREQ_GHZ   = 2.45     # Ieee802154NarrowbandRadio.ned: 2450 MHz
OM2G_BANDWIDTH_HZ      = 2.8e6    # B_20dB, ATmega256RFR2 p.564
OM2G_BITRATE_BPS       = 250e3    # 802.15.4-2006 p.28 -- THE fixed PHY rate
OM2G_SENSITIVITY_DBM   = -100.0   # ATmega256RFR2 p.565
OM2G_SNIR_THRESHOLD_DB = -8.0     # DSSS spreading gain: decodes below noise
OM2G_NOISE_FLOOR_DBM   = -96.616  # Ieee802154NarrowbandScalarRadioMedium.ned
OM2G_MTU_BYTES         = 127 - 9  # aMaxPHYPacketSize - aMinMPDUOverhead = 118
OM2G_IP_UDP_HEADER_BYTES_FWD = 20 + 8   # IPv4 + UDP, used by the demand model below

# Propagation: UNCHANGED from the OMNET track. Holding the channel model
# fixed is what makes "only the PHY capacity moved" a true statement.
OM2G_PATHLOSS_ALPHA = 2.5
OM2G_PATHLOSS_SIGMA = 4.0

# Which gate binds: max(sensitivity, noise + snir_threshold)
#   max(-100, -96.616 + (-8) = -104.616) = -100  -> SENSITIVITY binds.
# Same structure as the OMNET track (-85 vs -90+4 = -86 -> sensitivity), so
# the reward's decode-gate geometry carries over unchanged.
OM2G_DECODE_FLOOR_DBM = max(OM2G_SENSITIVITY_DBM,
                            OM2G_NOISE_FLOOR_DBM + OM2G_SNIR_THRESHOLD_DB)


def fspl_1m_db(freq_ghz: float = OM2G_CENTER_FREQ_GHZ) -> float:
    """Free-space loss at the 1 m reference distance, in dB.

    Matches INET's LogNormalShadowing::computePathLoss exactly, which computes
    PL(d) = PL(1m) + 10*alpha*log10(d) + N(0, sigma) with
    PL(1m) = 20*log10(4*pi/lambda). Verified against
    inet4.6/src/inet/physicallayer/wireless/common/pathloss/LogNormalShadowing.cc.
    """
    lam = 3e8 / (freq_ghz * 1e9)
    return 20.0 * math.log10(4.0 * math.pi / lam)


def tx_power_mw_for_range(range_m: float,
                          floor_dbm: float = OM2G_DECODE_FLOOR_DBM,
                          freq_ghz: float = OM2G_CENTER_FREQ_GHZ,
                          alpha: float = OM2G_PATHLOSS_ALPHA) -> float:
    """TX power (mW) whose median decode range is `range_m`."""
    dbm = floor_dbm + fspl_1m_db(freq_ghz) + 10.0 * alpha * math.log10(range_m)
    return 10.0 ** (dbm / 10.0)


def range_m_for_tx_power_mw(tx_mw: float,
                            floor_dbm: float = OM2G_DECODE_FLOOR_DBM,
                            freq_ghz: float = OM2G_CENTER_FREQ_GHZ,
                            alpha: float = OM2G_PATHLOSS_ALPHA) -> float:
    """Median decode range (m) for a given TX power. Inverse of the above."""
    dbm = 10.0 * math.log10(tx_mw)
    return 10.0 ** ((dbm - floor_dbm - fspl_1m_db(freq_ghz)) / (10.0 * alpha))


# ── Geometry: IDENTICAL to the frozen OMNET scenario ─────────────────────────
#
# The experiment is defined by dimensionless ratios (comm_range/field,
# speed/comm_range-per-interval, control-plane share). OM2G deliberately holds
# the first two EXACTLY equal to the OMNET track's frozen scenario and
# changes only the third. That is why the arena, the drone count,
# the user count, the cluster cap, the mobility and the episode length are all
# byte-identical to omnetpp.ini -- and why the C1-C6 task-validation criteria
# can be re-run against the same geometry.
#
# Target ranges are the OMNET track's own, recomputed from its constants:
#   drone TX 200 mW = 23.0103 dBm, floor -85 dBm, FSPL_1m(2.4 GHz) = 40.0460 dB
#     25*log10(r_d) = 23.0103 - 40.0460 + 85  ->  r_d = 523.06 m
#   user  TX  10 mW = 10.0000 dBm
#     25*log10(r_u) = 10.0000 - 40.0460 + 85  ->  r_u = 157.82 m
# ⚠️ These are the "v2 trainable" values from launch_v2_learnable.sh -- the
# config that ACTUALLY TRAINS and has hover/random/greedy baselines -- NOT the
# older L=1700 defaults sitting in omnetpp.ini, which no training run used.
# $OM2G_ARENA / $OM2G_SPREAD override the geometry. Default is 1200 m / 550 m,
# not 2000/350: at 2000 m / 350 m a drone hears no user (r_u=158 m) over 73% of
# the arena. Shrinking the arena alone flattens the task (900/350 -> ratio
# 1.11x); the clusters must widen with it. 1200 m / 550 m: blind 26.3%, ratio
# 1.47x, D_0 sd 29.8. This is the canonical world the training runs set
# explicitly.
# NOTE userSpread is EFFECTIVE METRES at any arena: om2g_config passes
# EFF*2000/L and RLBridge.cc:423 multiplies by L/2000, so the two cancel.
OM2G_ARENA_M          = float(os.environ.get("OM2G_ARENA", "1200.0"))
# $OM2G_DRONES overrides the drone count, so the drone budget can be swept --
# drone count is the knob that decides whether relaying is AFFORDABLE: a
# bridge in empty space costs a whole drone, so at N=5 covering 8 clusters the
# optimum avoids relays for budget reasons alone.
#
# MUST stay in lockstep with the simulator's *.numDrones and with
# omnet_env.MAX_N; config_knobs emits `--*.numDrones={n}` from this value and
# OMNET_MAX_N is exported alongside, so setting this one variable moves all
# three. Check the run header before trusting any result.
OM2G_NUM_DRONES       = int(os.environ.get("OM2G_DRONES", "5"))
# 40 users, together with the control-plane airtime charge in
# max_flow_reward. Measured by offline replay over the 12 real
# layouts: at 17 users the swarm offers ~99 kbps against a ~191 kbps per-radio
# application budget, so the radio rows NEVER BIND and the charge is literally
# invisible -- compressed and uncompressed score identically to the decimal.
# The channel has to be scarce before airtime spent on chatter can cost user
# throughput. Replay of the kmeans-5 ceiling, charge on:
#
#   users   compressed   uncompressed   uncompressed penalty
#      17        92.2           92.2     0.0%   <- charge invisible
#      30       165.1          162.3     1.7%   <- too weak to detect
#      40       221.5          175.3    20.9%   <- CHOSEN
#      60       315.9           41.1    87.0%   <- degenerate: the uncompressed
#                                                  radio is saturated by its own
#                                                  chatter, ratio 1.00x, nothing
#                                                  left to learn
#
# 40 keeps BOTH arms learnable (kmeans5/hover 1.52x compressed, 1.21x
# uncompressed) while making the flood cost real user throughput.
# $OM2G_USERS overrides the population (cap follows it). Default 40 keeps the
# `cc` and `cc2` families byte-identical; the `cc3` family runs 80.
# ⚠️ Any EVAL of a checkpoint must export the SAME value the run trained with,
# or the obs width and the padded message size disagree with the checkpoint.
OM2G_NUM_USERS        = int(os.environ.get("OM2G_USERS", "40"))
# Overridable so a world sweep can vary the number of user groups and how
# much traffic crosses between them. Defaults: 2 clusters / 0.5 cross.
# ⚠️ Like OM2G_USERS, any EVAL must export the SAME values the run trained with.
OM2G_MAX_CLUSTERS     = int(os.environ.get("OM2G_CLUSTERS", "2"))
OM2G_USER_SPREAD_EFF  = float(os.environ.get("OM2G_SPREAD", "550.0"))  # effective metres
OM2G_P_CROSS          = float(os.environ.get("OM2G_PCROSS", "0.5"))
# $OM2G_SPAWN. "staging" bunches ALL drones on the largest cluster: from a
# bunched start the difference reward pays for covering users nobody else
# covers, so the gradient says "separate", and separating from a bunch means
# moving radially OUTWARD (measured: +2.07 m/step outward drift, random -1.21).
# "cluster" deals drones round-robin across clusters, so they START distributed
# and the separate-from-your-peers gradient is absent.
OM2G_SPAWN_MODE       = os.environ.get("OM2G_SPAWN", "staging")
OM2G_EP_STEPS         = 250
OM2G_JAMMER_ACTIVATE  = 999999   # > ep_steps => jammer never fires
# $OM2G_STEP_M (2026-09-11, search-cost experiment): metres per RL step at
# |action| = 1. Default 60.0 reproduces every existing number byte-identically.
# Lower it to make FINDING demand expensive (20 = a drone crosses the arena
# ~1.3x per episode instead of ~4x). maxSpeed follows it, as it must
# (maxSpeed is a ceiling and must not sit below step_m). Set identically on
# train AND eval -- ew.sh re-exports the cell's OM2G_* lines, so it carries.
OM2G_STEP_M           = float(os.environ.get("OM2G_STEP_M", "60.0"))
OM2G_MAX_SPEED_MPS    = OM2G_STEP_M  # maxSpeed is set equal to step_m
                                 # (maxSpeed is a CEILING and must not be
                                 # below step_m)
OM2G_SYNC_INTERVAL_S  = float(os.environ.get("OM2G_SYNC_INTERVAL_S", "1.0"))
# The RL DECISION period. RLBridge.ned's syncInterval defaults to 1 s and
# nothing in this track overrides it, so the step clock is 1 s even when
# $OM2G_SYNC_INTERVAL_S (the BEACON period, above) is raised. The env's
# age -> rx_step lookup must divide by THIS, never by the beacon period:
# at a 2 s beacon it would otherwise deliver a half-age snapshot.
OM2G_RL_STEP_S        = 1.0


# $OM2G_RANGE_DRONE / $OM2G_RANGE_USER override these (metres).
#
# WHY THE DRONE RANGE IS SWEEPABLE: at the default 523 m,
# with 5 drones in a 1200-1600 m arena and clusters 750-1250 m apart, the
# backbone CONNECTS BY ACCIDENT -- drones spread out to cover users and end up
# in radio range of each other for free. The connectivity constraint is
# therefore never binding in those worlds: pCross=1.0 makes traffic cross
# clusters, but no drone ever has to sacrifice coverage to carry it. Measured: greedy stays
# 8-14% above `aggregate` in every world, and hover->greedy spans only ~2x.
#
# Shrinking the drone-drone range forces the chain to be built DELIBERATELY: a
# relay in empty space stops being a nicety and becomes the only way traffic
# flows, and a marginal-coverage hill-climb scores that move at exactly zero.
# This is the one lever that attacks greedy's known structural blindness
# instead of just rescaling the scenario.
# Archipelago world: controlled-separation user spawn (RLBridge.cc). "padre"
# is the production uniform-random sampler. "archipelago" keeps per-episode
# geometry entropy but pins the MINIMUM pairwise cluster separation, which
# removes the separation lottery that made the baselines bimodal, and draws
# Dirichlet cluster sizes -- the DEMAND ENTROPY the learned message must carry.
OM2G_USER_SPAWN_MODE = os.environ.get("OM2G_USER_SPAWN", "padre")
OM2G_MIN_SEPARATION_M = float(os.environ.get("OM2G_MINSEP", "1465.0"))
OM2G_WALL_MARGIN_M    = float(os.environ.get("OM2G_MARGIN", "300.0"))
# Upper edge of the cluster-separation band. A MINIMUM alone leaves the tail
# unbounded, which produced bridges 3.7-6.2 r_d deep where the first relay rung
# pays 0.2-8.4 kbps instead of the ~33 the world was designed around.
OM2G_MAX_NEIGHBOUR_SEP_M = float(os.environ.get("OM2G_MAXNN", "1750.0"))

OM2G_TARGET_RANGE_DRONE_M = float(os.environ.get("OM2G_RANGE_DRONE", "523.06"))
OM2G_TARGET_RANGE_USER_M  = float(os.environ.get("OM2G_RANGE_USER", "157.82"))

# The two powers that reproduce those ranges on the 802.15.4 link budget.
# Both are ordinary settable 802.15.4 transmit levels (chips cover roughly
# -25..+5 dBm on their own; +8 dBm is a standard front-end/PA part).
OM2G_DRONE_TX_MW = tx_power_mw_for_range(OM2G_TARGET_RANGE_DRONE_M)  # ~6.585 mW

# Control-plane spreading factor. 1 = no spreading (no change anywhere).
OM2G_CTRL_SPREADING = float(os.environ.get("OM2G_CTRL_SPREADING", "1.0"))
# Processing gain, applied to the SIMULATOR's drone transmitter only. The LP's
# $OMNET_DRONE_TX_mW is deliberately NOT touched: data-plane capacity must not
# move, or the boost feeds coverage and every arm lifts together (measured:
# closer clusters gave a uniform 2.3x).
OM2G_CTRL_GAIN_DB = 10.0 * math.log10(max(1.0, OM2G_CTRL_SPREADING))
OM2G_CTRL_TX_MW = OM2G_DRONE_TX_MW * (10.0 ** (OM2G_CTRL_GAIN_DB / 10.0))
OM2G_USER_TX_MW  = tx_power_mw_for_range(OM2G_TARGET_RANGE_USER_M)   # ~0.3293 mW

# Dimensionless ratios -- the actual definition of the experiment (scale
# invariance). Asserted in om2g_selftest().
OM2G_RHO_DRONE = OM2G_TARGET_RANGE_DRONE_M / OM2G_ARENA_M   # ~0.3077
OM2G_RHO_USER  = OM2G_TARGET_RANGE_USER_M / OM2G_ARENA_M    # ~0.0928
OM2G_MU        = OM2G_STEP_M / OM2G_TARGET_RANGE_DRONE_M    # ~0.1147


# ── Offered load: the demand model ───────────────────────────────────────────
# Bonn/MELPE voice, from om2g.ini: 21 payload bytes every 67.5 ms per talker.
# On the wire each is a UDP/IPv4 datagram, so the network must carry
# 21 + 8 (UDP) + 20 (IPv4) = 49 B at that interval.
#
# This is what caps a commodity in the max-flow LP (max_flow_reward.TAU_CAP_BPS
# on the OM2G track). A conversation cannot absorb more than one voice stream,
# so extra parallel capacity is worthless once its demand is met -- which is
# what makes the reward depend on COVERAGE rather than on how much aggregate
# capacity the swarm can pile up in one place.
OM2G_MELPE_PAYLOAD_BYTES = 21
OM2G_MELPE_INTERVAL_S    = 0.0675
OM2G_VOICE_BPS = ((OM2G_MELPE_PAYLOAD_BYTES + OM2G_IP_UDP_HEADER_BYTES_FWD)
                  * 8.0 / OM2G_MELPE_INTERVAL_S)          # ~5807 bps


# ── The per-node radio budget, in APPLICATION bits ───────────────────────────
# The LP's flow variables are counted in the same units as OM2G_VOICE_BPS, i.e.
# payload + UDP + IPv4 = 49 B per voice packet. Putting those 49 B on the air
# also costs the 9-byte 802.15.4 MAC header and the 6-byte PHY header+SFD, so a
# voice packet occupies 64 B of airtime to deliver the 49 B the LP sees.
#
# Charging the control plane at ON-AIR bytes (which is what airtime is) while
# crediting user flows at payload bytes would mix units, so the radio budget is
# converted into the flows' own units instead:
#
#     RADIO_CAP(app bps) = 250 kbps * 49/64 = 191 406.25
#
# Both sides of the constraint are then application bits and the subtraction in
# max_flow_reward._build_b_ub is dimensionally consistent.
OM2G_VOICE_ON_AIR_BYTES = (OM2G_MELPE_PAYLOAD_BYTES
                           + OM2G_IP_UDP_HEADER_BYTES_FWD + 9 + 6)   # 64 B
OM2G_RADIO_CAP_APP_BPS = (OM2G_BITRATE_BPS
                          * (OM2G_MELPE_PAYLOAD_BYTES
                             + OM2G_IP_UDP_HEADER_BYTES_FWD)
                          / OM2G_VOICE_ON_AIR_BYTES)      # 191 406.25 bps


# ── The inter-agent state message ────────────────────────────────────────────
#
# Column provenance, read off omnet_env.build_drone_tokens' inner loop. A
# column is part of the MESSAGE iff its value is peer j's own property; it is
# observer-local (and therefore free) iff observer i measures or computes it.
#
#   row[0], row[1]  peer j absolute position          -> MESSAGE (2)
#   row[2], row[3]  offset (jx-ix, jy-iy)             -> local (derived)
#   row[4], row[5]  peer RSSI + validity, measured    -> local (i's radio)
#   row[6]          peer j noise floor                -> MESSAGE (1)
#   user block      peer j's sensed user set          -> MESSAGE (D or 25*13)
#   event latent    peer j's MAC/PHY event summary    -> MESSAGE (8)
#   self_flag       j == i                            -> local
#   relay_thru      peer j forwarded throughput       -> MESSAGE (1)
#   channel_busy    peer j channel-busy fraction      -> MESSAGE (1)
#   time_since_rx   peer j seconds since last rx      -> MESSAGE (1)
#   mask            j is live                         -> local
OM2G_MSG_FIXED_FLOATS = 2 + 1 + 8 + 1 + 1 + 1   # = 14, both arms
OM2G_MAX_USERS_CAP    = OM2G_NUM_USERS   # must equal omnet_env.MAX_USERS_CAP
# ⚠️ In lockstep with OM2G_NUM_USERS. "Pad to max users" is the
# no-compressor baseline's definition, so the cap IS the uncompressed message
# size: 14 + 40*13 = 534 floats = 2136 B = 23 frames, against the compressed
# arm's unchanged 22 floats = 88 B = 1 frame. This is the compression story
# stated in bytes -- the latent is O(1) in the user population, the padded
# alternative is O(max_users) -- and it is why raising the population widens
# the gap on BOTH axes at once rather than only loading the channel.
# Must equal omnet_env.OBS_FEATURES, which is GATED on the partner tag --
# so this is gated identically: with $OM2G_USER_PARTNER_TAG set a user token
# is 17 features (40x17=680 for the uncompressed arm), otherwise 13 (520).
# The uncompressed arm's airtime charge is computed from this width.
OM2G_OBS_FEATURES     = 17 if os.environ.get("OM2G_USER_PARTNER_TAG") else 13
OM2G_FLOAT_BYTES      = 4    # float32 on the wire

OM2G_IP_UDP_HEADER_BYTES = 20 + 8
OM2G_MAX_UDP_PAYLOAD_ONE_FRAME = OM2G_MTU_BYTES - OM2G_IP_UDP_HEADER_BYTES  # 90


def om2g_message_floats(arm: str, latent_dim: int = 8) -> int:
    """Number of float32s drone j broadcasts about itself, per arm."""
    if arm == "compressed":
        return OM2G_MSG_FIXED_FLOATS + latent_dim
    if arm == "uncompressed":
        return OM2G_MSG_FIXED_FLOATS + OM2G_MAX_USERS_CAP * OM2G_OBS_FEATURES
    raise ValueError(f"arm must be 'compressed' or 'uncompressed', got {arm!r}")


# ── Geometry-channel pricing ────────────────────────────────────────────────
# $OM2G_GEO_BYTES: total ON-WIRE bytes for the sender position (BOTH coords
# together), transmitted quantized to OM2G_GEO_BYTES*4 bits per coordinate.
# Default 8 = two float32s = the historical layout, and reproduces every
# existing message size BYTE-IDENTICALLY (8 + 4*(12+L) = 4L + 56). Sweeping
# it prices the previously-fixed geometry channel the same way L prices the
# user summary: bits-per-coordinate is a standard, a-priori-justifiable
# protocol knob (GPSR/OLSR position beacons quantize routinely). 0 = the
# position is not transmitted at all. The env quantizes the DELIVERED
# content to match (om2g_env._wire_message), so charge == content always.
OM2G_GEO_BYTES = int(os.environ.get("OM2G_GEO_BYTES", "8"))
if not (0 <= OM2G_GEO_BYTES <= 16):
    raise RuntimeError(
        f"OM2G_GEO_BYTES={OM2G_GEO_BYTES} outside 0..16 (bytes for both "
        f"coordinates together; 8 = full float32 pair).")


def om2g_message_bytes(arm: str, latent_dim: int = 8) -> int:
    """UDP payload size of one state message, in bytes.

    GEO_BYTES for the position pair + float32 for every other message float
    (the 12 fixed non-position floats and the L-float user summary)."""
    other_floats = om2g_message_floats(arm, latent_dim) - 2
    return OM2G_GEO_BYTES + other_floats * OM2G_FLOAT_BYTES


def om2g_frames_per_message(arm: str, latent_dim: int = 8) -> int:
    """How many 802.15.4 frames one state message costs after IPv4 fragmentation.

    IPv4 fragments on 8-byte boundaries: each fragment carries a 20-byte IP
    header, so the payload budget per fragment is
    ((MTU - 20) // 8) * 8 bytes. The first fragment also spends 8 bytes on the
    UDP header.
    """
    payload = om2g_message_bytes(arm, latent_dim)
    total = payload + 8                                  # UDP header
    per_frag = ((OM2G_MTU_BYTES - 20) // 8) * 8          # = 96
    return max(1, math.ceil(total / per_frag))


def om2g_control_plane_bps(arm: str, latent_dim: int = 8,
                           n_drones: int = OM2G_NUM_DRONES,
                           interval_s: float = OM2G_SYNC_INTERVAL_S) -> float:
    """Offered control-plane bitrate ON AIR, including framing overhead.

    Counts whole 802.15.4 frames (each fragment pays the 9-byte MAC header and
    the 6-byte PHY header+SFD), because airtime -- not payload -- is what the
    channel actually spends. Excludes MAC ACKs and CSMA backoff, so this is a
    LOWER bound on the true occupancy; the measured figure comes from
    om2g_regime_probe.py.
    """
    return (n_drones * om2g_message_on_air_bytes(arm, latent_dim)
            * 8.0 / interval_s)


def om2g_message_on_air_bytes(arm: str, latent_dim: int = 8) -> float:
    """Bytes ONE state message actually puts on the air, framing included.

    Each IPv4 fragment pays its own 20-byte IP header, 9-byte 802.15.4 MAC
    header and 6-byte PHY header+SFD; the datagram pays one 8-byte UDP header.
    Airtime -- not payload -- is what the channel spends, which is why this,
    and not om2g_message_bytes, is what gets charged against a radio budget.
    """
    payload = om2g_message_bytes(arm, latent_dim)
    n_frag = om2g_frames_per_message(arm, latent_dim)
    return payload + 8 + 20 * n_frag + (9 + 6) * n_frag


def om2g_beacon_on_air_bytes(n_floats: int = 2) -> float:
    """Bytes ONE position beacon puts on the air, framing included.

    The realistic `greedy` baseline (signal_baselines.greedy_realistic_
    velocities) coordinates via one-hop HELLO beaconing -- RFC 6130 / OLSR
    RFC 3626 neighbour discovery, the same GPSR-style position beacon any
    deployed MANET already runs. Its payload is the sender's (x, y); it does
    NOT carry any user state, which is exactly why it is cheap and exactly why
    the policy that DOES carry user state has to earn its larger bill.

    Same framing arithmetic as om2g_message_on_air_bytes: one UDP header for
    the datagram, and IP + MAC + PHY headers per fragment (a beacon is always
    one fragment).
    """
    payload = n_floats * OM2G_FLOAT_BYTES
    return payload + 8 + 20 + (9 + 6)


def om2g_beacon_share(n_floats: int = 2,
                      n_drones: int = OM2G_NUM_DRONES,
                      interval_s: float = OM2G_SYNC_INTERVAL_S) -> float:
    """Beacon control-plane share of the channel, as a fraction."""
    return (n_drones * om2g_beacon_on_air_bytes(n_floats) * 8.0
            / interval_s) / OM2G_BITRATE_BPS


def om2g_control_plane_bps_per_node(arm: str, latent_dim: int = 8,
                                    n_neighbors: float = 0.0,
                                    interval_s: float = OM2G_SYNC_INTERVAL_S
                                    ) -> float:
    """Airtime ONE drone's half-duplex radio spends on the control plane, bps.

    A node's single radio pays for every state message it TRANSMITS and every
    one it RECEIVES -- the broadcast it sends itself, plus one per drone
    neighbour it can hear:

        control_bps(d) = (1 + n_neighbors(d)) * on_air_bits / interval

    This is the quantity charged against RADIO_CAP_BPS in the reward LP (see
    max_flow_reward.CONTROL_ON_AIR_BYTES), which is what makes an uncompressed
    state message cost the swarm real USER throughput rather than only its own
    delivery ratio. It is per-node and position-dependent, unlike
    om2g_control_plane_bps, which is the aggregate offered load on the medium.
    """
    return ((1.0 + n_neighbors) * om2g_message_on_air_bytes(arm, latent_dim)
            * 8.0 * OM2G_CTRL_SPREADING / interval_s)


def om2g_control_plane_share(arm: str, latent_dim: int = 8, **kw) -> float:
    """Control-plane share of the 250 kbps channel, as a fraction."""
    return om2g_control_plane_bps(arm, latent_dim, **kw) / OM2G_BITRATE_BPS


# ── Self-test: run `python om2g_track.py` ────────────────────────────────────
def om2g_selftest(verbose: bool = True) -> None:
    r_d = range_m_for_tx_power_mw(OM2G_DRONE_TX_MW)
    r_u = range_m_for_tx_power_mw(OM2G_USER_TX_MW)
    assert abs(r_d - OM2G_TARGET_RANGE_DRONE_M) < 0.5, (r_d, OM2G_TARGET_RANGE_DRONE_M)
    assert abs(r_u - OM2G_TARGET_RANGE_USER_M) < 0.5, (r_u, OM2G_TARGET_RANGE_USER_M)
    assert OM2G_DECODE_FLOOR_DBM == OM2G_SENSITIVITY_DBM, "SNIR gate must not bind"
    assert om2g_frames_per_message("compressed", 8) == 1, "D=8 must fit one frame"
    assert om2g_frames_per_message("uncompressed") > 10, "uncompressed must fragment"
    if not verbose:
        return
    print(f"OM2G track self-test  ({TRACK})")
    print(f"  FSPL(1 m) @ {OM2G_CENTER_FREQ_GHZ} GHz   = {fspl_1m_db():.4f} dB")
    print(f"  decode floor                  = {OM2G_DECODE_FLOOR_DBM:.3f} dBm "
          f"(sensitivity binds)")
    print(f"  drone TX {OM2G_DRONE_TX_MW:.4f} mW "
          f"({10*math.log10(OM2G_DRONE_TX_MW):+.3f} dBm) -> r_d = {r_d:.2f} m")
    print(f"  user  TX {OM2G_USER_TX_MW:.4f} mW "
          f"({10*math.log10(OM2G_USER_TX_MW):+.3f} dBm) -> r_u = {r_u:.2f} m")
    print(f"  rho_drone = {OM2G_RHO_DRONE:.4f}   rho_user = {OM2G_RHO_USER:.4f}"
          f"   mu = {OM2G_MU:.4f}")
    print(f"  one-frame UDP payload budget  = {OM2G_MAX_UDP_PAYLOAD_ONE_FRAME} B")
    print()
    print(f"  {'arm':<14} {'floats':>7} {'bytes':>7} {'frames':>7} {'ctrl share':>11}")
    for arm, d in (("compressed", 8), ("uncompressed", 8)):
        print(f"  {arm:<14} {om2g_message_floats(arm, d):>7} "
              f"{om2g_message_bytes(arm, d):>7} "
              f"{om2g_frames_per_message(arm, d):>7} "
              f"{om2g_control_plane_share(arm, d)*100:>10.2f}%")
    print()
    print("  compression sweep (latent_dim -> frames, control share):")
    for d in (2, 4, 8, 12, 16, 32, 64):
        print(f"    D={d:<3} {om2g_message_bytes('compressed', d):>5} B  "
              f"{om2g_frames_per_message('compressed', d)} frame(s)  "
              f"{om2g_control_plane_share('compressed', d)*100:.2f}%")


if __name__ == "__main__":
    om2g_selftest()
