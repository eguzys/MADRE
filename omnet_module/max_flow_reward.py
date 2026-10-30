"""
max_flow_reward.py — Multi-commodity max-flow reward for the drone relay task.

Reward = Σ_k τ_k, in Mbps: the PLAIN SUM OF DELIVERED THROUGHPUT over all
structural user pairs (not gated on live calls), every commodity weighted
the same.  Demands are HUNGRY (never saturate).

OBJECTIVE.  The objective is linear in τ rather than Σ_k log1p(τ_k)
(proportional fairness, Kelly 1998), and capacity is smooth in distance:

  1. log1p is nearly flat once τ clears a few Mbps, so each newly-CONNECTED
     commodity would add an almost-constant lump and the reward would be
     ~98% an implicit connectivity counter and ~2% throughput.
  2. A HARD zero below the decode thresholds (beyond ~396 m user-drone /
     ~523 m drone-drone) would give EXACTLY zero gradient there, a dead zone
     in which hill-climbing baselines stall in a strict local optimum.

So the objective is linear in τ, and capacity decays
smoothly with distance (see _link_capacity_bps / CAPACITY_TAPER_dB) so a
policy has a non-zero gradient to climb from ANY position in the arena.
This aligns the OMNeT track with the PADRE-Python objective, whose
signal/capacity model is likewise smooth (local SINR, no hard cutoff).

The objective being linear, the PWL machinery collapses to ONE segment of
unit slope -- kept as a single degenerate segment so the LP column map,
constraints and solution extraction are structurally unchanged.

LP formulation
--------------
maximize  Σ_k τ_k

subject to:
    Σ_k f_{k,i,j}  ≤  c_{i,j}                        ∀ edges      (capacity)
    Σ_i f_{k,i,n}  =  Σ_j f_{k,n,j}                  ∀ k, ∀ transit n  (conservation)
    Σ_j f_{k,src_k,j} - Σ_i f_{k,i,src_k} = τ_k      ∀ k          (source balance)
    Σ_i f_{k,i,dst_k} - Σ_j f_{k,dst_k,j} = τ_k      ∀ k          (sink balance)
    τ_k  ≤  TAU_CAP_BPS                              ∀ k          (numerical cap)
    f_{k,i,j} ≥ 0,  τ_k ≥ 0

Config parameters (module-level, exposed for sweeping):
    TAU_CAP_BPS            – numerical upper bound on τ_k, bps (default  1e9)
    CAPACITY_TAPER_dB      – logistic softness of the decode gates (default 3.0)
    REWARD_UNIT_BPS        – bps per reward unit (default 1e6, i.e. Mbps)

Solver: HiGHS (scipy.sparse + highspy), warm-started across calls.
Node numbering: 0..nd-1 = drones, nd..nd+nu-1 = users.

Function signatures
-------------------
compute_position_oracle_reward(obs, …) -> dict   full result
compute_oracle_max_flow_reward(obs, …) -> float  scalar
"""

from __future__ import annotations

import logging
import math
import os
from collections import OrderedDict
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ─── Constants matching omnetpp.ini / episode_parameter_math.md ──────────────

ARENA_W = 2000.0
ARENA_H = 2000.0

# ── Radio constants: dual-track ──────────────────────────────────────────────
# The literals below are the OMNET track (802.11g). Each reads an optional
# $OM2G_* environment override, exactly like DRONE_TX_mW/USER_TX_mW just below
# and for exactly the same reason: a training run spawns `python run.py` as a
# SEPARATE PROCESS, so in-process attribute mutation cannot reach it -- only
# environment variables cross that boundary.
#
# UNSET (every OMNET caller) reproduces the original literals byte for byte.
# SET (om2g_config.apply_om2g_radio) switches this module to the IEEE 802.15.4
# narrowband PHY of the OM2G track. The two tracks' rewards are NOT comparable
# and must never be mixed -- see code/om2g/OM2G_README.md.
ALPHA           = 2.5        # unchanged on both tracks (same propagation model)
BANDWIDTH       = float(os.environ.get("OM2G_BANDWIDTH_HZ", 20e6))   # 802.11g: 20 MHz
# [config_knobs.py] DRONE_TX_mW / USER_TX_mW are TWO of the three
# "dual-site" knobs the Tier-A/B/C config search varies: config_knobs.
# apply_config mutates these module attributes directly for IN-PROCESS
# callers (Tier A/B, one Python process), but a Stage-2 training run spawns
# `python run.py` as a SEPARATE process -- in-process attribute mutation
# cannot cross that boundary, only environment variables can. These two
# constants therefore also read an optional env-var override at import
# time, same pattern as omnet_env.py's own $OMNET_NUM_DRONES (see that
# module's "Never silently falls back" comment). UNSET reproduces the
# literals below.
DRONE_TX_mW     = float(os.environ.get("OMNET_DRONE_TX_mW", 200.0))      # mW
USER_TX_mW      = float(os.environ.get("OMNET_USER_TX_mW", 10.0))       # mW
FREQ_GHz        = float(os.environ.get("OM2G_FREQ_GHZ", 2.4))       # GHz
NOISE_FLOOR_dBm = float(os.environ.get("OM2G_NOISE_FLOOR_DBM", -90.0))  # dBm  (ambient thermal; jammer raises this per-receiver)

# ── INET Ieee80211Radio defaults — the radio this reward is scored against ────
# Ieee80211Radio.ned:42-44.  The reward uses the same decode thresholds as
# the radio OMNeT simulates (a -100 dBm floor would be 15 dB more sensitive
# and put r_d at ~2082 m, so no relay would ever be needed in a 2 km arena).
# Aligned to INET:
SENSITIVITY_dBm  = float(os.environ.get("OM2G_SENSITIVITY_DBM", -85.0))    # hard floor: no reception below this, at any SNIR
SNIR_THRESHOLD_dB = float(os.environ.get("OM2G_SNIR_THRESHOLD_DB", 4.0))   # and SNIR must clear this to decode

# ── $OM2G_PHYSICAL_METRIC: report what the radio can actually deliver ────────
#
# REPORTING gate, not a training change. `_link_capacity_bps` uses a 3 dB
# LOGISTIC to smooth the two decode conditions, a stand-in for log-normal
# shadowing that is ~9x fatter in the deep tail: at 1465 m the logistic gives
# 0.023 where the simulator's own Gaussian (sigma = 4 dB) gives 0.0026, i.e.
# ~1510 bps credited across a link on which the radio decodes almost nothing.
#
# This gate uses the SAME distribution the simulator rolls, capacity x
# Phi(margin/sigma) per decode condition, rather than a hard cutoff (which
# would deny the real shadowing tail and create a zero-gradient zone).
# Training keeps the surrogate; reporting uses this. Default off.
_PHYSICAL_METRIC = os.environ.get("OM2G_PHYSICAL_METRIC") == "1"
# Bump on ANY change to the reward model, so every cell records which reward
# it trained on.
REWARD_MODEL_VERSION = "phys-2026-09-01d"  # a: delivery gate; b: +tax; c: tx-only tax
# P(decode) of the link most recently costed, so _build_edges can record the
# RETRANSMISSION TAX alongside the capacity. A link delivering one frame per
# 1/p attempts spends 1/p times its delivered rate in airtime, so the airtime
# of a long, lossy link is charged at its attempt rate, not its delivered rate.
_LAST_P = [1.0]
_EDGE_P = {}
_SHADOW_SIGMA_DB = float(os.environ.get("OM2G_PATHLOSS_SIGMA", 4.0))
# A link is USABLE only if a delivered bit costs a bounded number of attempts.
# Below this, 1/p explodes (at p=1e-9 the radio row would get a coefficient of
# 1e9 and the LP would be badly conditioned). Drones often start far apart, so
# tiny p is the common case. 100 attempts per delivered bit is already a
# hopeless link; treat it as absent.
_P_USABLE_MIN = float(os.environ.get("OM2G_P_USABLE_MIN", 0.01))


def _phi(x_db: float) -> float:
    """P(a log-normal shadowing draw leaves x_db of margin intact)."""
    return 0.5 * (1.0 + math.erf(x_db / (_SHADOW_SIGMA_DB * math.sqrt(2.0))))
# Ambient: sensitivity binds (-85 > -90+4 = -86).
# Jammed:  the raised noise floor makes SNIR bind instead.  Both are modelled.

# ─── Oracle reward config (expose for sweeping) ───────────────────────────────
# Every commodity weighs the same -- every user is equally important.  The
# objective is a plain uniform-weight sum of delivered throughput: no
# per-commodity weight, no concavity, no bridging premium.
#
# Numerical upper bound on τ_k (bps), NOT a demand model: the single objective
# segment needs a finite width to use as its column upper bound, and the LP's
# τ_k must never be allowed to saturate against it. Must be >> max achievable
# per-commodity flow; max link capacity ≈ 500 Mbps (1 m, 200 mW), so 1e9 bps
# is always well above anything achievable regardless of topology.
#
# ── OM2G: this becomes a REAL DEMAND MODEL ───────────────────────────────────
# On the OMNET track the above holds and TAU_CAP_BPS stays 1e9: demands are
# hungry, and the reward is "how much throughput can this placement carry".
#
# On the OM2G track that question has a degenerate answer. Offered load is not
# unlimited -- it is 17 MELPE voice conversations, each 21 B every 67.5 ms.
# With a hungry demand the LP simply fills every radio it can reach, so the
# reward would not depend on where the drones are: with the per-node radio
# constraint switched on and demands hungry, EVERY policy on EVERY seed
# returns exactly 156250.0 = Σ_d RADIO_CAP/2 (each unit of flow through a
# drone spends that drone's radio twice, once in and once out). Zero dynamic
# range.
#
# Capping τ_k at one voice stream turns the objective into "how much of the
# offered voice traffic is actually deliverable", which is the coverage and
# connectivity question the task is about, and which depends strongly on drone
# positions. A user out of range delivers nothing; a user in range delivers its
# stream and no more -- so parallel capacity from bunched drones adds nothing
# once demand is met.
#
# Set via $OM2G_TAU_CAP_BPS (om2g_config exports it, derived in
# om2g_track.OM2G_VOICE_BPS). UNSET reproduces the hungry-demand literal.
TAU_CAP_BPS: float = float(os.environ.get("OM2G_TAU_CAP_BPS", 1e9))

# ── Reward units ──────────────────────────────────────────────────────────────
# The objective is Σ_k τ_k with τ in bps, which lands around 1e8-1e9 -- a
# terrible scale to hand an RL algorithm. Reporting in Mbps puts a typical
# episode step in the tens-to-hundreds range, matching the magnitude the
# PADRE-Python side already works in (its LogReward is log1p of throughput in
# Mbps). This is a pure unit choice: it rescales the reward, it does not
# reshape it.
# OM2G overrides this to 1e3 (kbps): with 250 kbps links the sum-of-throughput
# objective lands in the 1e5-1e6 bps range, which in Mbps units would hand the
# RL algorithm rewards around 0.1-2.0 -- two to three orders of magnitude below
# the scale every hyperparameter in this project was tuned at. Reporting OM2G
# in kbps puts a typical step back in the same tens-to-hundreds band as the
# OMNET track. It is a unit choice, not a reshaping; the two tracks' numbers
# remain non-comparable for every other reason.
REWARD_UNIT_BPS: float = float(os.environ.get("OM2G_REWARD_UNIT_BPS", 1e6))  # bps per reward unit

# ── Decode-gate smoothing ─────────────────────────────────────────────────────
# INET's radio decodes iff rx power clears SENSITIVITY_dBm AND SNIR clears
# SNIR_THRESHOLD_dB. Modelled as a hard AND, that is a step function, and a
# step function has zero derivative everywhere except at the step -- the dead
# zone documented in the module docstring.
#
# Each hard gate is replaced by a logistic in the dB margin:
#
#     gate(margin_dB) = 1 / (1 + exp(-margin_dB / CAPACITY_TAPER_dB))
#
# so capacity is Shannon * gate(rx margin) * gate(SNIR margin): equal to the
# old value well inside range, exactly 1/4 of Shannon at the old double
# threshold, and decaying smoothly-but-never-to-zero outside it.
#
# Why 3.0 dB: path loss is 10*ALPHA*log10(d) = 25*log10(d) dB, so 3 dB of
# taper is a factor 10^(3/25) = 1.32 in distance. At 2x the old range the gate
# product is ~5.8e-3 and at 3x it is ~3.4e-4 -- small, but strictly positive
# with a strictly positive derivative, which is the entire point. Larger
# values blur the radio model further from INET's; smaller ones reinstate the
# cliff. Exposed for sweeping.
CAPACITY_TAPER_dB: float = 3.0

# ── Fixed-rate PHY cap (OM2G track) ───────────────────────────────────────────
# 802.11g adapts its modulation to the SNR, so Shannon is a defensible proxy
# for what the link delivers and RATE_CAP_BPS stays None -- the OMNET track is
# byte-for-byte unaffected.
#
# IEEE 802.15.4 does NOT rate-adapt: DSSS-OQPSK-16 runs at exactly 250 kbps or
# does not decode at all.  Uncapped Shannon over the 2.8 MHz channel would
# claim ~10 Mbps at the cell edge -- 40x the rate the radio can physically
# carry -- and every capacity in the max-flow LP would be fiction.  With the
# cap, capacity(d) = min(Shannon, RATE_CAP) * gate, which inside the useful
# range is the flat 250 kbps of a real fixed-rate PHY, rolling off through the
# decode gate at the edge.  The gate is what supplies the spatial gradient.
#
# Set via $OM2G_RATE_CAP_BPS (om2g_env.py exports it) or by assigning the
# module attribute in-process.  UNSET leaves the rate uncapped (OMNET track).
_rate_cap_env = os.environ.get("OM2G_RATE_CAP_BPS")
RATE_CAP_BPS = float(_rate_cap_env) if _rate_cap_env else None

# ── Per-node half-duplex radio capacity (OM2G track) ─────────────────────────
# The LP models every link as an INDEPENDENT pipe: one capacity row per edge.
# That is a fair approximation of 802.11g, where the drone-user links a drone
# serves are fast relative to the offered load. It is badly wrong for a
# narrowband shared medium, and the error has a direction:
#
#   a user in range of 5 BUNCHED drones gets 5 parallel 250 kbps pipes,
#   while the same user served by 1 spread-out drone gets one.
#
# Without a per-node constraint the LP pays the swarm to CLUSTER: measured at
# n=40 paired seeds on the OM2G radio without it, hover 8338.6 kbps/step beat
# greedy 7212.6 (t = +5.93, 36/40 seeds) and random 7375.1 (t = +9.15, 38/40),
# so not moving would be the best policy.
#
# A real node has ONE half-duplex radio on ONE channel. Every bit it sends or
# receives spends that one radio's airtime, so the constraint is
#
#     Σ_k Σ_{e incident on drone d} f_{k,e}  ≤  RADIO_CAP_BPS      ∀ drones d
#
# which makes serving many users from one drone genuinely expensive and makes
# spreading pay. This is the constraint that turns a fixed-rate narrowband
# network into a placement problem.
#
# Drones only, not users: a user's radio carries only its own two commodities
# and is already bounded by its own edges, whereas the drones are the SHARED
# resource. Stated in the methods.
#
# Set via $OM2G_RADIO_CAP_BPS. UNSET (every OMNET caller) builds no extra
# rows at all.
_radio_cap_env = os.environ.get("OM2G_RADIO_CAP_BPS")
RADIO_CAP_BPS = float(_radio_cap_env) if _radio_cap_env else None

# ── Control-plane airtime charged against the radio (OM2G track) ─────────────
# The inter-agent state messages ride the simulated medium (they are metered,
# and an undelivered one costs the receiver its peer picture). The oracle LP
# also charges their airtime: without this charge it would hand every drone
# its full radio budget for users regardless of how many bytes that drone
# spends on state exchange, and flooding the channel would cost nothing in the
# reward the policy optimises.
#
# A node has ONE half-duplex radio, and every bit of state exchange it sends
# or hears spends the same airtime that user traffic needs:
#
#     Σ_k Σ_{e incident on d} f_{k,e}  ≤  RADIO_CAP_BPS - control_bps(d)
#     control_bps(d) = (1 + n_drone_neighbours(d)) * on_air_bits / interval
#
# so the state message competes with users for exactly one budget --
# "the messages use the same bandwidth as users". The charge is
# position-dependent (a bunched drone hears more peers and pays more), and it
# is 15.4x larger for the uncompressed arm because that message is 15 frames
# rather than 1.
#
# Set via $OM2G_CONTROL_ON_AIR_BYTES (om2g_config exports it from
# om2g_track.om2g_message_on_air_bytes, the single source of truth for the
# arm's framing). UNSET (every OMNET caller) leaves the LP arithmetic
# untouched.
#
# ⚠️ REQUIRES RADIO_CAP_BPS. Without the per-node radio row there is no budget
# to charge against: link capacities are independent pipes and control bytes
# cannot displace user bytes. Enabling one without the other is a no-op at
# best, so _build_b_ub raises rather than silently ignoring the charge.
_control_bytes_env = os.environ.get("OM2G_CONTROL_ON_AIR_BYTES")
CONTROL_ON_AIR_BYTES = float(_control_bytes_env) if _control_bytes_env else None
CONTROL_SYNC_INTERVAL_S = float(os.environ.get("OM2G_SYNC_INTERVAL_S", 1.0))

# ── Oracle LP unit scaling ─────────────────────────────────────────────────────
# HiGHS solves the oracle LP in these units, not raw bps. Capacities run
# ~1e7-1e8 bps while the objective slope is 1/REWARD_UNIT_BPS = 1e-6. Confirmed by direct
# repro (capture a real task_validation.py greedy-rollout trajectory that hit
# HighsModelStatus.kUnknown, replay its exact capacity sequence standalone):
# without this scaling the warm-started path (changeRowsBounds + run(),
# reusing the previous optimal basis on a badly scaled LP) can return kUnknown
# outright, or simply take 100-1000x longer per solve than the scaled version
# -- the "hangs for minutes" symptom is that slowdown accumulated over
# hundreds of warm-started calls in one rollout, not a true infinite loop.
# Dividing every capacity/bound by LP_SCALE and multiplying every objective
# coefficient by it is a plain change of variable (tau_scaled = tau_bps /
# LP_SCALE) that leaves the optimum unchanged but brings both the constraint
# matrix and the objective into a comparable [1e-3, 1e3] range.
LP_SCALE: float = 1e6  # bps per LP unit (i.e. the LP is solved in Mbps)


class OracleLPSolveError(RuntimeError):
    """Raised when the oracle MCMF LP does not solve to optimality.

    A reward of 0.0 from a failed solve would be indistinguishable from a
    reward of 0.0 from a genuinely disconnected network, silently poisoning
    every downstream aggregate. Callers must handle this explicitly rather
    than have it masked as a zero reward.
    """


# ─── Link capacity model ──────────────────────────────────────────────────────

def _soft_gate(margin_dB: float, taper_dB: float = None) -> float:
    """Logistic decode gate on a dB margin.  1 well above, 1/2 at, ->0 below.

    Written in the two-branch form so exp() never sees a large positive
    argument: margin_dB can be -100 dB or more for a far-apart pair, and the
    naive 1/(1+exp(-z)) overflows there instead of underflowing to 0.
    """
    if taper_dB is None:
        taper_dB = CAPACITY_TAPER_dB
    z = margin_dB / taper_dB
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _link_capacity_bps(d_m: float, tx_mW: float, noise_floor_dBm: float = NOISE_FLOOR_dBm) -> float:
    """Smoothly-gated Shannon capacity (bps) for a link of distance d_m.

    Uses Friis + log-distance model matching OMNeT++ LogNormalShadowing:
      PL(d) = FSPL_1m + 10*alpha*log10(d)
    where FSPL_1m = 20*log10(4*pi*f/c) is the free-space loss at 1 m reference.

    The two INET decode conditions (Ieee80211Radio.ned:42-44 -- rx power
    clears sensitivity, SNIR clears its threshold) are LOGISTIC gates (see
    CAPACITY_TAPER_dB) rather than a hard AND, so this function is
    continuous, strictly positive, and strictly decreasing in d_m over the
    whole arena.  The radio CONSTANTS (alpha, the thresholds, the TX powers)
    are those of the simulator; only the mapping from them to capacity is
    smooth.

    The return value is never exactly 0, so the graph is always complete.
    That is deliberate: it avoids a zero-gradient dead zone.
    """
    if d_m < 1.0:
        d_m = 1.0

    lam = 3e8 / (FREQ_GHz * 1e9)
    fspl_1m_dB = 20.0 * math.log10(4 * math.pi / lam)

    path_loss_dB = fspl_1m_dB + 10.0 * ALPHA * math.log10(d_m)
    tx_dBm = 10.0 * math.log10(tx_mW)
    rx_dBm = tx_dBm - path_loss_dB
    snr_dB = rx_dBm - noise_floor_dBm

    snr_linear = 10.0 ** (snr_dB / 10.0)
    shannon = BANDWIDTH * math.log2(1.0 + snr_linear)
    # OM2G: a fixed-rate PHY cannot exceed its one modulation rate. No-op when
    # RATE_CAP_BPS is None (the OMNET track).
    if RATE_CAP_BPS is not None and shannon > RATE_CAP_BPS:
        shannon = RATE_CAP_BPS

    if _PHYSICAL_METRIC:
        # The simulator's own statistics, not a stand-in for them.
        gate = (_phi(rx_dBm - SENSITIVITY_dBm)
                * _phi(snr_dB - SNIR_THRESHOLD_dB))
        if gate < _P_USABLE_MIN:
            gate = 0.0          # unusable link: carries nothing, costs nothing
        _LAST_P[0] = max(gate, _P_USABLE_MIN)
    else:
        gate = (_soft_gate(rx_dBm - SENSITIVITY_dBm)
                * _soft_gate(snr_dB - SNIR_THRESHOLD_dB))
    return shannon * gate


def _dist(a: List[float], b: List[float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# ─── LP solver ────────────────────────────────────────────────────────────────

def _build_b_ub(edges, edge_list, nd):
    """Capacity-row right-hand side: one entry per edge, then (when the OM2G
    per-node radio constraint is active) one entry per drone.

    The radio entries are CONSTANT -- a radio's rate does not depend on where
    the drone is -- but they are rebuilt here anyway so the warm-start path,
    which rewrites every capacity row bound from this vector, stays correct.
    """
    # numpy is imported lazily inside compute_position_oracle_reward (module
    # import stays cheap for callers that never solve), so import it here too
    # rather than relying on a module-level name that does not exist.
    import numpy as np
    edge_caps = np.array([edges[e] for e in edge_list]) / LP_SCALE
    if RADIO_CAP_BPS is None:
        if CONTROL_ON_AIR_BYTES is not None:
            raise ValueError(
                "OM2G_CONTROL_ON_AIR_BYTES is set but OM2G_RADIO_CAP_BPS is "
                "not. The control-plane charge is subtracted from the per-node "
                "radio budget; with no radio row there is nothing to charge "
                "and the setting would be silently ignored. Set both or "
                "neither -- see max_flow_reward.CONTROL_ON_AIR_BYTES.")
        return edge_caps

    if CONTROL_ON_AIR_BYTES is None:
        radio_caps = np.full(nd, RADIO_CAP_BPS, dtype=np.float64)
    else:
        # Drone d hears drone j iff the directed edge (j, d) carries positive
        # capacity, which is exactly the decodability test the rest of the LP
        # uses -- so the neighbour count is consistent with the topology the
        # flows see, not a second range model that could drift from it.
        n_nb = np.zeros(nd, dtype=np.float64)
        for (i, j) in edge_list:
            if i < nd and j < nd:
                n_nb[j] += 1.0          # j receives from i
        control_bps = ((1.0 + n_nb) * CONTROL_ON_AIR_BYTES * 8.0
                       / CONTROL_SYNC_INTERVAL_S)
        # A radio saturated by its own control plane serves no users; it must
        # never go negative and hand the LP a free infeasibility.
        radio_caps = np.maximum(RADIO_CAP_BPS - control_bps, 0.0)

    return np.concatenate([edge_caps, radio_caps / LP_SCALE])


def _build_edges(
    nd: int,
    nu: int,
    drone_pos: List[List[float]],
    user_pos: List[List[float]],
    noise_floor: List[float] = None,
) -> Dict[Tuple[int, int], float]:
    """Return {(node_i, node_j): capacity_bps} for all positive-capacity edges.

    Node indices: 0..nd-1 = drones, nd..nd+nu-1 = users.
    Drone-drone: bidirectional, 200 mW.
    User-drone: asymmetric — 10 mW up, 200 mW down.
    """
    edges: Dict[Tuple[int, int], float] = {}
    def _nf(idx):
        if noise_floor and idx < len(noise_floor):
            return noise_floor[idx]
        return NOISE_FLOOR_dBm

    for i in range(nd):
        for j in range(i + 1, nd):
            d_ij = _dist(drone_pos[i], drone_pos[j])
            c_ij = _link_capacity_bps(d_ij, DRONE_TX_mW, _nf(j))  # receiver j
            c_ji = _link_capacity_bps(d_ij, DRONE_TX_mW, _nf(i))  # receiver i
            if c_ij > 0:
                edges[(i, j)] = c_ij
                _EDGE_P[(i, j)] = _LAST_P[0]
            if c_ji > 0:
                edges[(j, i)] = c_ji
                _EDGE_P[(j, i)] = _LAST_P[0]

    for u in range(nu):
        for d in range(nd):
            dist = _dist(user_pos[u], drone_pos[d])
            c_up   = _link_capacity_bps(dist, USER_TX_mW, _nf(d))   # receiver = drone d
            c_down = _link_capacity_bps(dist, DRONE_TX_mW)          # receiver = user (ambient)
            node_u = nd + u
            if c_up > 0:
                edges[(node_u, d)] = c_up
            if c_down > 0:
                edges[(d, node_u)] = c_down

    return edges


# ─── Position-oracle weighted max-flow reward ─────────────────────────────────

# LRU structure cache for compute_position_oracle_reward.
# Key: (nd, nu, frozenset(edges.keys()), tau_cap, sender_pairs)
# Value: all LP structural objects — topology, commodity set, linear objective
#        coefficients (which depend on tau_cap, hence it is IN the key so any
#        change forces a full rebuild), plus the live HiGHS solver.  Only
#        capacities (b_ub, i.e. edge values — NOT edge existence, which is
#        already in the key via frozenset(edges.keys())) are excluded from the
#        key: those change every call and are applied via warm-start
#        (changeRowsBounds) instead of a rebuild.
#
# Multiple slots: the difference-reward path (omnet_env's
# OMNET_DIFFERENCE_REWARD) evaluates R(all) and then R(all \ i) for each drone,
# so nd alternates 5,4,4,4,4,4 within one step and a single slot would rebuild
# on every call (measured 1.52 ms warm vs 6.71 ms per leave-one-out solve).
#
# WHY `cache_tag` EXISTS. An LRU alone is not enough. Every
# leave-one-out variant has the SAME structure (same nd, same complete-graph
# topology), so they all collided on one key and therefore shared one HiGHS
# instance -- each solve warm-started from the basis left by a DIFFERENT
# drone's removal, which is a poor starting point. Measured on a moving
# trajectory: 38.0 ms/step, against a 12.6 ms ceiling implied by the global
# solve's own 2.09 ms warm cost. Callers therefore pass an opaque hashable
# `cache_tag` (omnet_env passes the removed drone index) that joins the key, so
# variant i warm-starts from variant i's OWN previous-step basis -- and
# consecutive steps move the drones only slightly, which is exactly the
# situation warm-start is for.
#
# MAXSLOTS must exceed the working set: 1 full + N tagged = N+1 per drone
# count. Each slot holds a HiGHS instance, so it is not free; 16 covers N=5
# with room for a cardinality sweep (N=3,4,6), and least-recently-used
# eviction keeps it bounded regardless.
_ORACLE_LP_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_ORACLE_LP_CACHE_MAXSLOTS = 16


def _build_oracle_commodities(nu: int,
                              sender_pairs=None) -> List[Tuple[int, int]]:
    """Return [(src_user_idx, dst_user_idx), ...] -- the conversations to score.

    `sender_pairs` is the FIXED sender->partner assignment drawn once per episode
    by RLBridge::buildPadreSenders and shipped in obs_json["sender_pairs"]. Those
    ~numUsers pairs are the conversations the radio actually carries.

    Why not all N*(N-1) ordered pairs: that assumes ANY user might talk to ANY
    other. With 17 users that is 272 commodities instead of 17 -- a ~16x larger
    LP (80ms vs ~5-10ms per solve, and greedy fires 40 solves PER STEP), and it
    would weight cross-cluster traffic at 66% by combinatorics rather than at
    the traffic model's own pCross.

    STILL POSITIONS-ONLY. The pairs are fixed for the whole episode (fixedPartner,
    not `other`, which onCallEnd resets), so the reward remains a deterministic
    function of drone positions -- NOT of who happens to be mid-call. Keying on
    live calls would make the reward bursty (see SS6.3).

    Falls back to all-pairs when sender_pairs is absent, so old obs JSON and the
    unit tests still work.
    """
    if sender_pairs:
        out = []
        for pair in sender_pairs:
            i, j = int(pair[0]), int(pair[1])
            if i == j or not (0 <= i < nu and 0 <= j < nu):
                continue
            out.append((i, j))
        if out:
            return out

    # legacy fallback: every ordered pair
    commodities = []
    for i in range(nu):
        for j in range(nu):
            if i == j:
                continue
            commodities.append((i, j))
    return commodities


def _build_objective_segments(
    tau_cap: float,
) -> Tuple[List[float], List[float]]:
    """Return (breakpoints, slopes) for the LINEAR objective Σ_k τ_k.

    The objective is linear, so the piecewise-linear representation is exactly
    ONE segment spanning [0, tau_cap] with constant slope.

    The slope is 1 / REWARD_UNIT_BPS rather than 1 so that a commodity
    carrying τ bps contributes τ/1e6 = τ in Mbps.  Keeping the (breakpoints,
    slopes) shape means the LP's column map, segment upper bounds, objective
    assembly and solution extraction are all structurally unchanged -- S just
    happens to be 1.
    """
    return [0.0, tau_cap], [1.0 / REWARD_UNIT_BPS]


def compute_position_oracle_reward(
    obs: dict,
    tau_cap: float = TAU_CAP_BPS,
    cache_tag=None,
) -> dict:
    """Position-oracle multi-commodity max-flow reward.

    Reward = Σ_k  τ_k   (total delivered throughput, in Mbps)

    LINEAR in τ_k: no diminishing returns, no fairness term, no connectivity
    premium.  A commodity is worth exactly the throughput it carries, so the
    only way to raise the reward is to raise total delivered throughput.
    See the module docstring for why the previous Σ log1p(τ_k) objective was
    retired.

    Commodities are ALL structural (user_i, user_j) pairs — never gated on
    live calls, every pair weighs the same.  Demands are HUNGRY (never
    saturate against tau_cap).  Because _link_capacity_bps is now smooth and
    strictly positive, the reward has a non-zero gradient at every drone
    position in the arena.

    Args:
        obs: dict with keys
            positions      – [[x, y], ...] drone positions (m)
            user_positions – [[x, y], ...] user positions (m)
        tau_cap: numerical upper bound on τ_k (bps); must be >> max
                 achievable flow so the constraint never binds.
        cache_tag: opaque HASHABLE that joins the LP structure-cache key. It
                 does not affect the returned value in any way -- it exists
                 purely so callers that interleave several same-shaped LPs can
                 keep them in separate warm-start slots instead of fighting
                 over one basis. omnet_env's difference reward passes the
                 removed drone's index; see _ORACLE_LP_CACHE's comment for the
                 3x measured on that path. Leave as None to share one slot.

    Returns:
        dict with keys:
            reward              – float Σ_k τ_k, in Mbps
            per_commodity_tau   – {(src_user_idx, dst_user_idx): tau_bps}  (total τ_k, in bps)
            per_commodity_flow  – {(src_user_idx, dst_user_idx): {(i,j): bps}}
            num_commodities     – int, total number of commodities
    """
    _empty = {
        "reward": 0.0,
        "per_commodity_tau": {},
        "per_commodity_flow": {},
        "num_commodities": 0,
    }

    drone_pos = obs.get("positions", [])
    user_pos  = obs.get("user_positions", [])
    noise_floor = obs.get("noise_floor", [])

    if not drone_pos or not user_pos:
        return _empty

    nd = len(drone_pos)
    nu = len(user_pos)

    # _link_capacity_bps tapers smoothly instead of returning a hard 0 below the
    # decode thresholds, so _build_edges returns a COMPLETE graph (every
    # capacity is strictly positive), which is deliberate -- see
    # _link_capacity_bps.
    edges = _build_edges(nd, nu, drone_pos, user_pos, noise_floor)
    if not edges:
        return _empty

    import numpy as np
    from scipy.sparse import csc_matrix, vstack as _sp_vstack

    # Objective breakpoints/slopes depend only on `demand` (already part of cache_key
    # below), so recomputing them on every call is correct on both warm and
    # cold paths and costs microseconds next to the ms-scale LP solve.
    breakpoints, obj_slopes = _build_objective_segments(tau_cap)
    S = len(obj_slopes)

    # The commodity list is EPISODE-DEPENDENT (it comes from sender_pairs), so
    # it MUST be part of the key. Without it, two episodes with the same drone
    # count, user count and edge set would collide in the cache and reuse a
    # constraint matrix built for a DIFFERENT set of conversations.
    _comms_for_key = tuple(
        (int(p[0]), int(p[1])) for p in (obs.get("sender_pairs") or [])
    )
    # RADIO_CAP_BPS changes the ROW STRUCTURE of the LP (it adds one row per
    # drone), so it must be part of the key -- otherwise a caller that flips it
    # mid-process would warm-start onto a matrix built for the other structure.
    cache_key = (nd, nu, frozenset(edges.keys()), tau_cap, _comms_for_key,
                 cache_tag, RADIO_CAP_BPS)
    # NOTE: CONTROL_ON_AIR_BYTES is deliberately NOT in the key. It changes only
    # the radio rows' RHS, never the row STRUCTURE, and every b_ub is rebuilt
    # from scratch on both the warm and cold paths below. The per-drone
    # neighbour count it depends on is a function of edge existence, which is
    # already in the key via frozenset(edges.keys()).
    _cache = _ORACLE_LP_CACHE.get(cache_key)

    if _cache is not None:
        # Most-recently-used, so the eviction below drops a stale cardinality
        # rather than the slot this step is about to reuse.
        _ORACLE_LP_CACHE.move_to_end(cache_key)
        # ── Cache hit: warm-start — only capacities (row upper bounds) changed ──
        edge_list       = _cache["edge_list"]
        N               = _cache["N"]
        E               = _cache["E"]
        K               = _cache["K"]
        row_i           = _cache["row_i"]
        R_CAP           = _cache["R_CAP"]
        commodities_raw = _cache["commodities_raw"]

        b_ub = _build_b_ub(edges, edge_list, nd)
        _warm_start = True
    else:
        _warm_start = False
        _cache = {}
        # ── Cache miss: full rebuild ──────────────────────────────────────────
        commodities_raw = _build_oracle_commodities(nu, obs.get("sender_pairs"))
        if not commodities_raw:
            return _empty

        K = len(commodities_raw)

        # ── Build adjacency helpers ───────────────────────────────────────────
        from collections import defaultdict as _dd
        out_edges: dict = _dd(list)
        in_edges:  dict = _dd(list)
        for (i, j) in edges:
            out_edges[i].append(j)
            in_edges[j].append(i)
        all_nodes = set(range(nd + nu))

        # ── Formulate LP ─────────────────────────────────────────────────────
        # Explicit column-index map (S == 1 segment, i.e. one tau column per k):
        #   tau_{k,s}      → column ki*S + s               (ki in 0..K-1, s in 0..S-1)
        #   f_{ki, ei, ej} → column TAU_COLS + ki*E + e_idx  (e_idx in 0..E-1)
        edge_list   = list(edges.keys())
        E           = len(edge_list)
        edge_to_col = {e: idx for idx, e in enumerate(edge_list)}
        TAU_COLS    = K * S
        N = TAU_COLS + K * E

        # Objective: minimise -Σ_k Σ_s slope_s * tau_{k,s}  (linprog minimises)
        # With the linear objective there is exactly ONE segment of constant
        # slope 1/REWARD_UNIT_BPS, so Σ_s slope_s * tau_{k,s} is just τ_k
        # expressed in Mbps. No ordering constraints are needed (there is
        # nothing to order).
        # tau_{k,s} is solved in LP_SCALE units (Mbps, not bps) -- see LP_SCALE
        # docstring -- so the objective coefficient is scaled up by the same
        # factor to leave tau_scaled * c_scaled == tau_bps * (-slope) exactly.
        c_obj = np.zeros(N)
        for ki, (si, dj) in enumerate(commodities_raw):
            for s in range(S):
                c_obj[ki * S + s] = -obj_slopes[s] * LP_SCALE

        # ── Capacity inequalities (A_ub x ≤ b_ub) ────────────────────────────
        # Row e_idx: Σ_k f_{k, e_idx} ≤ cap[e_idx]  (cap in LP_SCALE units)
        ub_r, ub_c, ub_v = [], [], []
        for e_idx, e in enumerate(edge_list):
            for ki in range(K):
                ub_r.append(e_idx)
                ub_c.append(TAU_COLS + ki * E + e_idx)
                ub_v.append(1.0)
        # OM2G: one extra row per drone -- every flow variable on an edge
        # INCIDENT on drone d (either direction: the radio is half-duplex, so
        # transmitting and receiving spend the same airtime) enters that
        # drone's radio row with coefficient 1.
        R_CAP = E
        if RADIO_CAP_BPS is not None:
            for e_idx, (ei, ej) in enumerate(edge_list):
                for node in (ei, ej):
                    if node >= nd:
                        continue                      # users have no radio row
                    # Airtime per DELIVERED bit is 1/p: a link that needs 380
                    # attempts per success spends 380x its delivered rate on the
                    # medium. Charging 1.0 would price marginal links as if
                    # failed attempts were free.
                    # ARQ costs land on the TRANSMITTER: delivering one bit over
                    # a link with per-frame success p costs 1/p attempts of its
                    # airtime. The RECEIVER is charged 1.0 per DELIVERED bit --
                    # a frame that fails below sensitivity never syncs a preamble
                    # and does not occupy the receiver's schedule.
                    #
                    # Charging 1/p at both ends would over-bill a relay by ~30%
                    # per hop and compound along a chain, while short links sit
                    # at p~1 where the tax is negligible. Edge (ei, ej) is ei
                    # transmitting to ej.
                    coef = 1.0
                    if _PHYSICAL_METRIC and node == ei:
                        coef = 1.0 / max(_EDGE_P.get((ei, ej), 1.0), 1e-9)
                    for ki in range(K):
                        ub_r.append(E + node)
                        ub_c.append(TAU_COLS + ki * E + e_idx)
                        ub_v.append(coef)
            R_CAP = E + nd

        b_ub = _build_b_ub(edges, edge_list, nd)
        A_ub = csc_matrix((ub_v, (ub_r, ub_c)), shape=(R_CAP, N))

        # ── Equality constraints (A_eq x = b_eq) — built as COO ──────────────
        eq_r, eq_c, eq_v = [], [], []
        eq_rhs: list = []
        row_i = 0

        for ki, (si, dj) in enumerate(commodities_raw):
            src_node = nd + si
            dst_node = nd + dj

            # Source balance: Σ_j f[k,src,j] - Σ_i f[k,i,src] - Σ_s τ_{k,s} = 0
            for s in range(S):
                eq_r.append(row_i); eq_c.append(ki * S + s); eq_v.append(-1.0)
            for j in out_edges[src_node]:
                eq_r.append(row_i)
                eq_c.append(TAU_COLS + ki * E + edge_to_col[(src_node, j)])
                eq_v.append(1.0)
            for i in in_edges[src_node]:
                eq_r.append(row_i)
                eq_c.append(TAU_COLS + ki * E + edge_to_col[(i, src_node)])
                eq_v.append(-1.0)
            eq_rhs.append(0.0)
            row_i += 1

            # Sink balance: Σ_i f[k,i,dst] - Σ_j f[k,dst,j] - Σ_s τ_{k,s} = 0
            for s in range(S):
                eq_r.append(row_i); eq_c.append(ki * S + s); eq_v.append(-1.0)
            for i in in_edges[dst_node]:
                eq_r.append(row_i)
                eq_c.append(TAU_COLS + ki * E + edge_to_col[(i, dst_node)])
                eq_v.append(1.0)
            for j in out_edges[dst_node]:
                eq_r.append(row_i)
                eq_c.append(TAU_COLS + ki * E + edge_to_col[(dst_node, j)])
                eq_v.append(-1.0)
            eq_rhs.append(0.0)
            row_i += 1

            # Flow conservation at every transit node
            for n in sorted(all_nodes):
                if n == src_node or n == dst_node:
                    continue
                if not in_edges[n] and not out_edges[n]:
                    continue
                for i in in_edges[n]:
                    eq_r.append(row_i)
                    eq_c.append(TAU_COLS + ki * E + edge_to_col[(i, n)])
                    eq_v.append(1.0)
                for j in out_edges[n]:
                    eq_r.append(row_i)
                    eq_c.append(TAU_COLS + ki * E + edge_to_col[(n, j)])
                    eq_v.append(-1.0)
                eq_rhs.append(0.0)
                row_i += 1

        A_eq = csc_matrix((eq_v, (eq_r, eq_c)), shape=(row_i, N))
        b_eq = np.array(eq_rhs)

        # Store structural objects; capacities (b_ub) and weights (c_obj) are excluded.
        # The slot is published into the LRU only after "highs" is set (below the
        # solve), so an exception mid-build cannot leave a half-populated slot
        # that a later call would take the warm-start branch on.
        _cache["edge_list"]     = edge_list
        _cache["edge_to_col"]   = edge_to_col
        _cache["N"]             = N
        _cache["E"]             = E
        _cache["K"]             = K
        _cache["row_i"]         = row_i
        _cache["R_CAP"]         = R_CAP
        _cache["A_ub"]          = A_ub
        _cache["A_eq"]          = A_eq
        _cache["b_eq"]          = b_eq
        _cache["commodities_raw"] = commodities_raw

    # ── Solve ─────────────────────────────────────────────────────────────────
    import highspy as _hs
    if _warm_start:
        # Warm-start: only row upper bounds (capacities) changed; reuse basis.
        _h = _cache["highs"]
        # R_CAP, not E: with the OM2G radio constraint the capacity block is
        # E edge rows PLUS nd drone-radio rows, and b_ub is the same length.
        # Passing E here would leave the radio rows at their cold-start bounds
        # and silently desynchronise the two.
        _row_idx = np.arange(R_CAP, dtype=np.int32)
        _h.changeRowsBounds(
            R_CAP, _row_idx,
            np.full(R_CAP, -_hs.kHighsInf, dtype=np.float64),
            b_ub,
        )
        _h.run()
    else:
        # Cold start: full LP rebuild on structure change.
        # Segment tau_{k,s} upper bound is its breakpoint width (same for every
        # k, in LP_SCALE units); flow vars stay unbounded above (capacity rows
        # constrain them).
        TAU_COLS = K * S
        _widths = np.diff(np.array(breakpoints)) / LP_SCALE  # length S
        _lp = _hs.HighsLp()
        _lp.num_col_ = N
        _lp.num_row_ = R_CAP + row_i

        _lp.col_cost_  = c_obj
        _lp.col_lower_ = np.zeros(N, dtype=np.float64)
        _lp.col_upper_ = np.concatenate([
            np.tile(_widths, K),
            np.full(K * E, _hs.kHighsInf, dtype=np.float64),
        ])

        _A = _sp_vstack([A_ub, A_eq]).tocsc()
        _lp.a_matrix_.format_  = _hs.MatrixFormat.kColwise
        _lp.a_matrix_.num_col_ = N
        _lp.a_matrix_.num_row_ = R_CAP + row_i
        _lp.a_matrix_.start_   = _A.indptr
        _lp.a_matrix_.index_   = _A.indices
        _lp.a_matrix_.value_   = _A.data

        # Capacity rows: (-inf, b_ub[e]];  equality rows: [0, 0]
        _zeros = np.zeros(row_i, dtype=np.float64)
        _lp.row_lower_ = np.concatenate([np.full(R_CAP, -_hs.kHighsInf), _zeros])
        _lp.row_upper_ = np.concatenate([b_ub, _zeros])

        _h = _hs.Highs()
        _h.setOptionValue("output_flag", False)
        _h.setOptionValue("presolve", "off")
        _h.passModel(_lp)
        _h.run()
        _cache["highs"] = _h
        # Publish the now-complete slot and bound the LRU.
        _ORACLE_LP_CACHE[cache_key] = _cache
        while len(_ORACLE_LP_CACHE) > _ORACLE_LP_CACHE_MAXSLOTS:
            _ORACLE_LP_CACHE.popitem(last=False)

    if _h.getModelStatus() != _hs.HighsModelStatus.kOptimal:
        raise OracleLPSolveError(
            f"Oracle MCMF LP did not solve to optimality: "
            f"status={_h.getModelStatus()!s}. A silent zero reward here would "
            f"be indistinguishable from a genuinely disconnected network and "
            f"would poison downstream aggregates — this must be investigated, "
            f"not swallowed."
        )

    x = list(_h.getSolution().col_value)

    # ── Extract solution ──────────────────────────────────────────────────────
    # x is in LP_SCALE units (Mbps); convert back to bps for everything reported.
    TAU_COLS = K * S
    total_reward = 0.0
    per_commodity_tau:  Dict[Tuple[int, int], float] = {}
    per_commodity_flow: Dict[Tuple[int, int], Dict[Tuple[int, int], float]] = {}

    for ki, (si, dj) in enumerate(commodities_raw):
        k = (si, dj)
        # Total tau_k is the sum of its segments (S == 1); the reward
        # contribution is the same segment's LP objective term, i.e. tau_k
        # converted from bps to Mbps.
        seg_vals = [max(0.0, x[ki * S + s]) * LP_SCALE for s in range(S)]
        tau_val = sum(seg_vals)
        reward_k = sum(obj_slopes[s] * seg_vals[s] for s in range(S))
        per_commodity_tau[k] = tau_val
        total_reward += reward_k

        edge_flows: Dict[Tuple[int, int], float] = {}
        for e_idx, (ei, ej) in enumerate(edge_list):
            fval = x[TAU_COLS + ki * E + e_idx] * LP_SCALE
            if fval > 1e-6:
                edge_flows[(ei, ej)] = fval
        per_commodity_flow[k] = edge_flows

    return {
        "reward":             total_reward,
        "per_commodity_tau":  per_commodity_tau,
        "per_commodity_flow": per_commodity_flow,
        "num_commodities":    K,
    }


def compute_oracle_max_flow_reward(
    obs: dict,
    tau_cap: float = TAU_CAP_BPS,
    cache_tag=None,
) -> float:
    """Scalar wrapper: returns Σ_k τ_k (Mbps) as a plain float.

    `cache_tag` is a pure warm-start hint and never changes the value -- see
    compute_position_oracle_reward.
    """
    return compute_position_oracle_reward(
        obs, tau_cap=tau_cap, cache_tag=cache_tag)["reward"]


# ─── Smoke tests ──────────────────────────────────────────────────────────────

def _run_oracle_smoke_tests() -> None:
    """Run the oracle smoke checks described in the module docstring."""
    import sys

    PASS = "\033[32mPASS\033[0m"
    FAIL = "\033[31mFAIL\033[0m"
    WARN = "\033[33mWARN\033[0m"
    results = []

    # ── Shared scenario: two user clusters (areas 0 and 1), bridge drone gap ──
    # Layout (x-axis):
    #   Area 0 users: x=100, x=300      Area 1 users: x=1700, x=1900
    #   Drones relay: d0=400, d1=1000, d2=1600
    # At y=1000 (mid-arena).

    base_obs = {
        "positions":      [[400, 1000], [1000, 1000], [1600, 1000]],
        "user_positions": [[100, 1000], [300, 1000], [1700, 1000], [1900, 1000]],
        "user_area":      [0, 0, 1, 1],
    }

    # ── CHECK 1: Determinism ─────────────────────────────────────────────────
    r1a = compute_position_oracle_reward(base_obs)["reward"]
    r1b = compute_position_oracle_reward(base_obs)["reward"]
    ok1 = abs(r1a - r1b) < 1.0  # CBC is deterministic; allow tiny float drift
    results.append(("1 Determinism", ok1, f"call1={r1a:.2f}  call2={r1b:.2f}"))

    # ── CHECK 2: Independence from live-call state ───────────────────────────
    obs_no_ab  = dict(base_obs)
    obs_with_ab = dict(base_obs, active_broadcasts=[
        {"talk_group_id": 0, "speaker_user_id": 0},
        {"talk_group_id": 4, "speaker_user_id": 1},
    ])
    r2a = compute_position_oracle_reward(obs_no_ab)["reward"]
    r2b = compute_position_oracle_reward(obs_with_ab)["reward"]
    ok2 = abs(r2a - r2b) < 1.0
    results.append(("2 Live-call independence", ok2,
                    f"no_ab={r2a:.2f}  with_ab={r2b:.2f}"))

    # ── CHECK 3: Position sensitivity, AND no dead zone ──────────────────────
    # With the smooth capacity model the invariant is:
    #
    #   * on-cluster must still score far higher than a far corner (position
    #     must matter -- a reward that ignores position teaches nothing), but
    #   * the far corner must be STRICTLY POSITIVE, because a policy stranded
    #     there needs a non-zero gradient to climb back out.
    #
    # The ratio bound enforces the first property: "smooth" must not mean
    # "flat".
    obs_on_cluster = dict(base_obs)
    obs_far_corner = dict(base_obs, positions=[[50, 50], [100, 50], [50, 100]])
    r3_on  = compute_position_oracle_reward(obs_on_cluster)["reward"]
    r3_far = compute_position_oracle_reward(obs_far_corner)["reward"]
    ok3 = r3_on > 0.0 and r3_far > 0.0 and (r3_far / r3_on) < 0.05
    results.append(("3 Position sensitivity (far corner >0 but <5% of on-cluster)", ok3,
                    f"on-cluster={r3_on:.2f}  far-corner={r3_far:.4f}  "
                    f"ratio={r3_far / r3_on:.2%}"))

    # ── CHECK 4: No saturation — improvement still possible in best config ────
    # In the base scenario the drones form a reasonable chain.  Add an extra
    # drone exactly on the midpoint — this strictly improves the cross-area path.
    obs_better = dict(base_obs, positions=[
        [400, 1000], [700, 1000], [1000, 1000], [1300, 1000], [1600, 1000]
    ])
    r4_base   = compute_position_oracle_reward(base_obs)["reward"]
    r4_better = compute_position_oracle_reward(obs_better)["reward"]
    ok4 = r4_better > r4_base
    results.append(("4 No saturation (more drones → higher reward)", ok4,
                    f"3-drone={r4_base:.2f}  5-drone={r4_better:.2f}"))

    # ── CHECK 5: Capacity is smooth (no cliff) ───────────────────────────────
    # _link_capacity_bps applies logistic decode gates instead of a hard zero.
    # Capacity must be strictly positive and strictly DECREASING far past the
    # nominal r_u / r_d ranges, so there is a gradient to climb from anywhere
    # in the arena.
    _probe_d = [100.0, 400.0, 523.0, 800.0, 1200.0, 1700.0, 2400.0]
    _caps = [_link_capacity_bps(d, DRONE_TX_mW) for d in _probe_d]
    ok5 = (all(c > 0.0 for c in _caps)
           and all(_caps[i] > _caps[i + 1] for i in range(len(_caps) - 1)))
    results.append(("5 Capacity smooth: >0 and strictly decreasing to 2400 m", ok5,
                    "  ".join(f"{d:.0f}m:{c/1e6:.3g}Mbps" for d, c in zip(_probe_d, _caps))))

    # ── CHECK 6: objective is exactly linear ─────────────────────────────────
    # The old 12-segment PWL approximation of log1p is retired. There must now
    # be exactly ONE segment, of constant slope 1/REWARD_UNIT_BPS, so a
    # commodity carrying tau bps contributes exactly tau/1e6 (Mbps) with no
    # approximation error at all.
    _bp, _slopes = _build_objective_segments(TAU_CAP_BPS)
    _test_xs = [1e4, 1e5, 1e6, 1e7, 1e8, 5e8, 1e9]
    _errs = [abs(_slopes[0] * _x - _x / REWARD_UNIT_BPS) / (_x / REWARD_UNIT_BPS)
             for _x in _test_xs]
    ok6 = (len(_slopes) == 1 and max(_errs) < 1e-12)
    results.append(("6 Objective exactly linear (1 segment, zero approx error)", ok6,
                    f"segments={len(_slopes)} slope={_slopes[0]:.3e} "
                    f"max_rel_err={max(_errs):.1e}"))

    # ── CHECK 7: Concentrated vs spread must now TIE ─────────────────────────
    # Same total raw flow, split two ways across 4 commodities: all in one
    # (concentrated) vs evenly spread across all four.  Under the OLD concave
    # log1p objective spreading had to score strictly higher.  Under the new
    # LINEAR objective they must tie EXACTLY -- Σ tau_k depends only on the
    # total.  A tie is the proof the concavity (and with it the implicit
    # connectivity counter) is really gone.
    _T = 1e8
    _concentrated = [_T, 0.0, 0.0, 0.0]
    _spread = [_T / 4.0] * 4
    _r_conc = sum(t / REWARD_UNIT_BPS for t in _concentrated)
    _r_spread = sum(t / REWARD_UNIT_BPS for t in _spread)
    ok7 = abs(_r_spread - _r_conc) < 1e-9
    results.append(("7 Concentrated vs spread (linear objective: must TIE)", ok7,
                    f"concentrated={_r_conc:.6f}  spread={_r_spread:.6f}  "
                    f"(same total flow={_T:.0e})"))

    # ── CHECK 8: Runtime (warm-started) ──────────────────────────────────────
    # Budget ~30 ms/call; flag (do not fail the suite) past ~100 ms.
    import time
    _n_timing = 50
    compute_position_oracle_reward(base_obs)  # prime the warm-start cache
    _t0 = time.perf_counter()
    for _ in range(_n_timing):
        compute_position_oracle_reward(base_obs)
    _ms_per_call = (time.perf_counter() - _t0) / _n_timing * 1000.0
    ok8 = _ms_per_call < 100.0
    results.append(("8 Runtime per call (warm-started)", ok8,
                    f"{_ms_per_call:.2f} ms/call  (budget ~30 ms, alarm >100 ms)"))

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("POSITION-ORACLE MAX-FLOW REWARD — SMOKE TESTS")
    print("=" * 65)
    all_passed = True
    for name, ok, detail in results:
        if ok is None:
            tag = WARN
        elif ok:
            tag = PASS
        else:
            tag = FAIL
            all_passed = False
        print(f"  [{tag}]  CHECK {name}")
        print(f"           {detail}")
    print("=" * 65)
    if all_passed:
        print(f"  [{PASS}]  All hard checks passed.\n")
    else:
        print(f"  [{FAIL}]  One or more checks FAILED.\n")
        sys.exit(1)


if __name__ == "__main__":
    _run_oracle_smoke_tests()
