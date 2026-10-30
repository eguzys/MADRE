"""OM2G run configuration — the ONE place a run's knobs are assembled.

╔══════════════════════════════════════════════════════════════════════════╗
║  OM2G TRACK (IEEE 802.15.4 narrowband).  NOT the OMNET track.           ║
║  See code/om2g/OM2G_README.md.                                          ║
╚══════════════════════════════════════════════════════════════════════════╝

This module exists so that a physical constant is written once and cannot
drift between the places that use it (config_knobs.py calls this "dual-site
drift").  OM2G has FOUR sites --

  1. the simulator      (om2g.ini, via $OMNET_EXTRA_ARGS)
  2. the reward oracle  (max_flow_reward module constants, via $OM2G_* env vars)
  3. the environment    (om2g_env / omnet_env module constants, via env vars)
  4. the model          (encoder arm + latent width, via env vars)

-- and every one of them is derived HERE from om2g_track.py.  Nothing else may
re-type a number.  `python om2g_config.py --arm compressed` prints the exact
environment a run will see, and `--selftest` checks it against om2g.ini.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "omnet_module"))

import om2g_track as T

ARMS = ("compressed", "uncompressed", "nocontrol")

# Demand multiplier -- see OM2G_TAU_CAP_BPS below. 1.0 reproduces the `cc`
# family exactly; the `cc2` family runs 2.0.
# Default is 2.0, not 1.0: at 1x the swarm already meets ~93% of demand, the
# difference reward collapses (D_0 sd 9.7 vs 414), and neither arm learns --
# 1x is a degenerate task, not a neutral/lower setting (see the block below).
# The campaign runs at 2x, so a run with no OM2G_TAU_MULT set reproduces that.
TAU_MULT = float(os.environ.get("OM2G_TAU_MULT", "2.0"))

# Radio-capacity multiplier -- the SUPPLY-side mirror of TAU_MULT (demand).
# Scales ONLY the reward LP's per-node application-bit budget
# (OM2G_RADIO_CAP_BPS below), never the physical PHY bitrate
# (T.OM2G_BITRATE_BPS / OM2G_RATE_CAP_BPS) that message fragmentation and
# on-air-byte accounting derive from. That keeps "how much channel capacity
# exists" independent of "how big is our own message" -- the D-sweep already
# owns the latter axis. 1.0 reproduces every existing family byte-identical.
CAP_MULT = float(os.environ.get("OM2G_CAP_MULT", "1.0"))

# Opt-in: send only sensed users on the wire instead of padding to
# OM2G_MAX_USERS_CAP. The MODEL side is untouched -- it always reconstructs a
# padded tensor; only the transmitted/on-air byte count varies with the
# sender's live sensed-user count. Only meaningful for arm="uncompressed"
# (the compressed arm's latent is already O(1) in population). Default off
# reproduces every existing run byte-identical -- this only affects a run
# that opts in.
DYNAMIC_SEND = os.environ.get("OM2G_DYNAMIC_SEND", "0") == "1"

# Opt-in: skip a beacon entirely (no packet, no airtime) when the drone
# hasn't moved more than OM2G_SEND_DELTA_M and its sensed-user count is
# unchanged since its last real send. The "cheap" (rule-based) half of the
# send-gating research question; the learned-gate half is a separate model
# change. Applies to compressed/nocontrol arms (uncompressed already pays
# full price by design and isn't part of this question). Default off
# reproduces every existing run byte-identical.
SEND_ON_DELTA = os.environ.get("OM2G_SEND_ON_DELTA", "0") == "1"
SEND_DELTA_M = float(os.environ.get("OM2G_SEND_DELTA_M", "10.0"))

# Opt-in: the "smart"/learned send-gate. The policy gets a 3rd action
# dimension (see om2g_env.py's OM2G_LEARNED_GATE, which must be set the
# SAME way for the env to actually widen its action space -- this flag only
# controls the SIMULATOR side reading that 3rd value; om2g_env.py's own
# env var is the one that makes the model emit it in the first place. Only
# meaningful for arm="compressed". Default off, byte-identical otherwise.
LEARNED_GATE = os.environ.get("OM2G_LEARNED_GATE", "0") == "1"

# Opt-in: PRESENCE BEACON on every ground user (UserBeaconApp).
# On this track a user only puts a frame on the medium while it is the current
# SPEAKER of a conversation, and only users drawn as senders get a conversation
# at all -- a sender's partner never speaks either. A non-calling user is
# therefore radio-silent and a drone, whose only sensing channel is the
# received frame, cannot know it exists. $OM2G_USER_BEACON=1 adds a third app
# to every user that emits a small fixed-size multicast frame every
# $OM2G_USER_BEACON_INTERVAL seconds, so presence becomes observable. The drone
# side is untouched -- RLBridge's radio hook already sets user_rssi_valid for
# any reception whose source MAC maps to a user.
#
# Default off emits NOTHING into $OMNET_EXTRA_ARGS, so a run without the
# variable is byte-identical to before this existed. The user's beacon uses
# port 5003 / group 239.1.2.2, deliberately distinct from the drone beacon's
# 5002 / 239.1.2.1, so user presence traffic can never be counted as drone
# control traffic.
USER_BEACON = os.environ.get("OM2G_USER_BEACON", "0") == "1"
USER_BEACON_INTERVAL_S = float(os.environ.get("OM2G_USER_BEACON_INTERVAL", "1.0"))
USER_BEACON_BYTES = int(os.environ.get("OM2G_USER_BEACON_BYTES", "1"))

# The latent width per arm.  For the compressed arm this is the encoder
# bottleneck.  For the uncompressed arm it is STRUCTURAL: the passthrough
# "encoder" emits max_users * user_token_dim numbers, so every downstream width
# (reconstructed_input_dim, self_flag_col, the checkpoint's parameter shapes)
# must be told the same value or the model will not even build.
# $OM2G_LATENT_DIM overrides the COMPRESSED arm's bottleneck width, for the
# compression sweep (the thesis question on the TASK axis: how small can the
# message get before performance degrades?). Default 8 = the one-frame limit,
# so every existing family is unchanged. The uncompressed arm is structural and
# never overridden.
ARM_LATENT_DIM = {
    "compressed":   int(os.environ.get("OM2G_LATENT_DIM", "8")),
    "uncompressed": T.OM2G_MAX_USERS_CAP * T.OM2G_OBS_FEATURES,   # 325
    # "nocontrol" never sends a real beacon (see OM2G_NOCONTROL in om2g.ini:
    # beaconInterval=3600s, far beyond any episode), so peer tokens are always
    # empty regardless of latent width -- reuse compressed's model shape.
    "nocontrol":    int(os.environ.get("OM2G_LATENT_DIM", "8")),
}

def _wire_latent(arm):
    """Latent-equivalent size of what actually goes ON THE WIRE.

    The MODEL always encodes ARM_LATENT_DIM[arm] floats about itself, and that
    is unchanged by forwarding -- so multi-hop reach is kept separate from
    model capacity.

    But a drone that RELAYS must put the forwarded records on the air. With
    OM2G_FORWARD_MODE=flood the transmitted beacon therefore carries the drone's
    own record plus one record per peer it relays, and the extra bytes are real
    802.15.4 airtime: real fragmentation, real collisions, real loss. Extra
    reach is therefore PAID FOR in frames (~1,800 reward per frame, measured).

    OM2G_FORWARD_PAYLOAD:
      digest (default) -- 2 floats (x, y) per relayed peer
      full             -- the peer's entire latent record
    """
    base = ARM_LATENT_DIM[arm]
    mode = os.environ.get("OM2G_FORWARD_MODE", "none").strip().lower()
    if arm == "nocontrol" or mode != "flood":
        return base
    n_drones = int(float(os.environ.get("OM2G_DRONES", "8") or 8))
    n_relayed = max(0, n_drones - 1)
    per_peer = base if os.environ.get("OM2G_FORWARD_PAYLOAD", "digest") == "full" else 2
    return base + n_relayed * per_peer


ARM_ENCODER = {
    "compressed":   "learned",       # LearnableUserSetEncoder
    "uncompressed": "passthrough",   # PassthroughUserSetEncoder (0 parameters)
    "nocontrol":    "learned",
}
ARM_INI_CONFIG = {
    "compressed":   "OM2G_COMPRESSED",
    "uncompressed": "OM2G_UNCOMPRESSED",
    # The never-communicate floor: beaconBytes=4 (sender ID only, never
    # actually sent within an episode), beaconInterval=3600s. Tests whether
    # inter-drone communication is worth anything at all -- the lower anchor
    # for both the compression axis and the new send-gating axis below.
    "nocontrol":    "OM2G_NOCONTROL",
}


def om2g_extra_args(arm: str, n_drones: int | None = None,
                    ep_steps: int | None = None) -> str:
    """The $OMNET_EXTRA_ARGS string for an OM2G run.

    Appended LAST by omnet_env._launch_omnet, so these win over om2g.ini.
    Same escaping rules as config_knobs.build_omnet_extra_args: the string
    parameter's quotes must survive shlex.split(), hence the backslashes.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    n = n_drones if n_drones is not None else T.OM2G_NUM_DRONES
    steps = ep_steps if ep_steps is not None else T.OM2G_EP_STEPS
    # "nocontrol" never actually fires a beacon within an episode (see
    # ARM_INI_CONFIG); 4 B matches the ini's own OM2G_NOCONTROL block and is
    # the sender-ID-only floor -- om2g_message_bytes() only knows
    # compressed/uncompressed, so this is special-cased rather than routed
    # through it.
    beacon = 4 if arm == "nocontrol" else T.om2g_message_bytes(arm, _wire_latent(arm))
    args = [
        f"--*.rlBridge.arenaWidth={T.OM2G_ARENA_M}",
        f"--*.rlBridge.arenaHeight={T.OM2G_ARENA_M}",
        f"--**.mobility.constraintAreaMaxX={T.OM2G_ARENA_M}m",
        f"--**.mobility.constraintAreaMaxY={T.OM2G_ARENA_M}m",
        f"--*.rlBridge.maxClusters={T.OM2G_MAX_CLUSTERS}",
        # Archipelago spawn. The C++ reuses dumbbellD/dumbbellJitter as the
        # minimum pairwise separation and the wall margin (RLBridge.ned).
        f'--*.rlBridge.userSpawnMode=\\"{T.OM2G_USER_SPAWN_MODE}\\"',
        f"--*.rlBridge.dumbbellD={T.OM2G_MIN_SEPARATION_M}",
        f"--*.rlBridge.dumbbellJitter={T.OM2G_WALL_MARGIN_M}",
        # Upper edge of the separation band. BOTH builders must emit every
        # world parameter: baselines are launched through config_knobs and
        # TRAINING through this file, so a param added to only one of them
        # would give the two paths DIFFERENT WORLDS. maxNeighbourSep bounds
        # the separation band together with dumbbellD.
        f"--*.rlBridge.maxNeighbourSep={T.OM2G_MAX_NEIGHBOUR_SEP_M}",
        f"--*.rlBridge.pCross={T.OM2G_P_CROSS}",
        # base == effective only because L == 2000; keep the inversion explicit
        # so a future L change cannot silently mis-scale the user cloud.
        f"--*.rlBridge.userSpread={T.OM2G_USER_SPREAD_EFF * 2000.0 / T.OM2G_ARENA_M}",
        # The SIMULATOR's drone transmitter carries CONTROL FRAMES ONLY -- user
        # traffic is scored by the position-LP in max_flow_reward, which reads
        # $OMNET_DRONE_TX_mW and is deliberately NOT raised here: raising the
        # LP's drone range 523->1000 m at pCross 0.9 lifts HOVER from 18.10 to
        # 98.02 (5.4x) because cross-cluster calls become relayable. Control
        # reach moves without data reach.
        f"--*.host[*].wlan[*].radio.transmitter.power={T.OM2G_CTRL_TX_MW:.6g}mW",
        f"--*.user[*].wlan[*].radio.transmitter.power={T.OM2G_USER_TX_MW:.6g}mW",
        f"--**.host[*].mobility.maxSpeed={T.OM2G_MAX_SPEED_MPS}mps",
        f"--*.numUsers={T.OM2G_NUM_USERS}",
        f"--*.numDrones={n}",
        f"--*.jammer.mobility.activate={T.OM2G_JAMMER_ACTIVATE}",
        f"--*.jammer.mobility.startMoving={T.OM2G_JAMMER_ACTIVATE}",
        f'--*.rlBridge.droneSpawnMode=\\"{T.OM2G_SPAWN_MODE}\\"',
        # THE ARM.  This single number is the entire manipulation.
        f"--*.host[*].app[0].beaconBytes={beacon}",
        # $OM2G_DYNAMIC_SEND (uncompressed arm only): stop padding the
        # transmitted message to OM2G_MAX_USERS_CAP -- size it to each
        # drone's live sensed-user count instead. beaconBytes above still
        # sets the padded ceiling (used when this is off); the two constants
        # below mirror OM2G_MSG_FIXED_FLOATS/OM2G_OBS_FEATURES exactly so the
        # C++ side can never drift from this Python source of truth.
        f"--*.host[*].app[0].dynamicBeaconSize="
        f"{'true' if (DYNAMIC_SEND and arm == 'uncompressed') else 'false'}",
        f"--*.host[*].app[0].beaconFixedBytes="
        f"{T.OM2G_MSG_FIXED_FLOATS * T.OM2G_FLOAT_BYTES}",
        f"--*.host[*].app[0].beaconBytesPerUser="
        f"{T.OM2G_OBS_FEATURES * T.OM2G_FLOAT_BYTES}",
        # $OM2G_SEND_ON_DELTA: the cheap send-gating heuristic (see
        # DroneBeaconApp.cc). Meaningless for "uncompressed" (that arm is
        # defined as paying full price every step) and for "nocontrol"
        # (already never sends), so only wire it for "compressed".
        f"--*.host[*].app[0].sendOnDeltaEnabled="
        f"{'true' if (SEND_ON_DELTA and arm == 'compressed') else 'false'}",
        f"--*.host[*].app[0].sendOnDeltaPosThresholdM={SEND_DELTA_M}",
        f"--*.host[*].app[0].learnedGateEnabled="
        f"{'true' if (LEARNED_GATE and arm == 'compressed') else 'false'}",
        # Send rate == charge rate, from ONE constant. See om2g_track's
        # OM2G_SYNC_INTERVAL_S: every control-plane charge function divides by
        # this, so a run that beacons 4x faster is billed 4x. Never split these.
        # nocontrol arm: beaconInterval is passed explicitly as 3600 s ("never
        # fires inside an episode", as in [Config OM2G_NOCONTROL]) because an
        # OMNeT command-line option overrides the ini section; every other arm
        # gets the sync interval.
        f"--*.host[*].app[0].beaconInterval="
        f"{'3600' if arm == 'nocontrol' else T.OM2G_SYNC_INTERVAL_S}s",
        f"--*.rlBridge.maxSteps={steps}",
    ]
    # $OM2G_USER_BEACON: the user PRESENCE beacon (see USER_BEACON above).
    # Appended only when opted in, so an un-set run produces the exact same
    # string as a run without this option.
    if USER_BEACON:
        args += [
            "--*.user[*].numApps=3",
            '--*.user[*].app[2].typename=\\"UserBeaconApp\\"',
            # userId is left at its NED default of -1: UserBeaconApp then asks
            # the containing UserHost for its own id, which avoids having to
            # push a `parent.userId` NED expression through shlex.split().
            f"--*.user[*].app[2].beaconInterval={USER_BEACON_INTERVAL_S}s",
            f"--*.user[*].app[2].beaconBytes={USER_BEACON_BYTES}",
            "--*.user[*].app[2].beaconPort=5003",
            '--*.user[*].app[2].beaconGroup=\\"239.1.2.2\\"',
        ]
    return " ".join(args)


def om2g_env_vars(arm: str, n_drones: int | None = None,
                  ep_steps: int | None = None) -> dict:
    """Every environment variable an OM2G run needs, as one dict.

    Covers all four sites.  Anything a training subprocess must know has to be
    in here: `python run.py` is spawned as a SEPARATE PROCESS, so in-process
    attribute mutation cannot reach the reward oracle or the model.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    n = n_drones if n_drones is not None else T.OM2G_NUM_DRONES
    steps = ep_steps if ep_steps is not None else T.OM2G_EP_STEPS
    return {
        # ── site 0: the track guard ──────────────────────────────────────
        "OM2G_TRACK": "1",
        # omnet_env.py reads OMNET_NUM_DRONES (omnet_env.py:137) while the
        # simulator is driven by `--*.numDrones` built from T.OM2G_NUM_DRONES.
        # Those are two different names for one quantity: without this line a
        # $OM2G_DRONES override moves the SIMULATOR to N drones while the
        # Python env stays at 5, and the obs/action shapes disagree with the
        # world. Emit both from the single source of truth.
        "OMNET_NUM_DRONES": str(T.OM2G_NUM_DRONES),
        "OM2G_ARM": arm,
        "OMNET_INI": "om2g.ini",
        "OMNET_CONFIG": ARM_INI_CONFIG[arm],

        # ── site 1: the simulator ────────────────────────────────────────
        "OMNET_EXTRA_ARGS": om2g_extra_args(arm, n, steps),
        "OMNET_MAX_STEPS": str(steps),
        "OMNET_NUM_DRONES": str(n),

        # ── site 2: the reward oracle (max_flow_reward) ──────────────────
        # The 802.15.4 PHY.  Without these the reward would score an 802.11g
        # world while the simulator carries an 802.15.4 one.
        "OM2G_BANDWIDTH_HZ":      repr(T.OM2G_BANDWIDTH_HZ),
        "OM2G_FREQ_GHZ":          repr(T.OM2G_CENTER_FREQ_GHZ),
        "OM2G_SENSITIVITY_DBM":   repr(T.OM2G_SENSITIVITY_DBM),
        "OM2G_SNIR_THRESHOLD_DB": repr(T.OM2G_SNIR_THRESHOLD_DB),
        "OM2G_NOISE_FLOOR_DBM":   repr(T.OM2G_NOISE_FLOOR_DBM),
        # A fixed-rate PHY: DSSS-OQPSK-16 runs at 250 kbps or not at all.
        "OM2G_RATE_CAP_BPS":      repr(T.OM2G_BITRATE_BPS),
        # ── THE CONTROL-PLANE CHARGE ─────────────────────────────────────
        # These three knobs are ONE mechanism and must be set together. They
        # are what makes an uncompressed state message cost the swarm real
        # USER throughput instead of only its own delivery ratio.
        #
        #   TAU_CAP    one MELPE voice stream per conversation, so extra
        #              parallel capacity piled up in one place is worthless
        #              and the reward depends on COVERAGE.
        #   RADIO_CAP  one half-duplex radio per drone instead of one
        #              independent pipe per link, expressed in application
        #              bits (see T.OM2G_RADIO_CAP_APP_BPS).
        #   CONTROL_   the airtime this arm's state message spends on that
        #   ON_AIR_    radio, subtracted from the budget:
        #   BYTES          (1 + n_drone_neighbours) * on_air_bits / interval
        #
        # Earlier measurement said RADIO_CAP alone was degenerate (it let the
        # LP saturate every radio regardless of position, a constant 156250.0)
        # and that with TAU_CAP on it was irrelevant. Both were true AT 17
        # USERS, where offered load (~99 kbps) sits far under the ~191 kbps
        # budget so the radio rows never bind. At the 40-user population (see
        # T.OM2G_NUM_USERS) they bind, and the charge separates the arms:
        #
        #   kmeans-5 ceiling, 12 real layouts   compressed   uncompressed
        #     charge off                            221.9         221.9
        #     charge on                             221.5         175.3
        #     task ratio kmeans5/hover              1.52x         1.21x
        #
        # i.e. compression is free (-0.2%) and the flood costs 21%.
        # $OM2G_TAU_MULT scales the per-conversation demand. Measured: at 1x
        # the swarm already meets ~93% of the 40-user demand,
        # so removing one drone costs almost nothing and the DIFFERENCE reward
        # collapses (D_0 sd 9.7 vs 414 pre-cc) -- the policy gets no placement
        # signal and neither arm learns. 2x restores it (D_0 sd 31.2) while
        # keeping the placement gradient (kmeans5/hover 1.40x vs 1.49x) AND
        # doubling the control-plane penalty (uncompressed ceiling -50.4% vs
        # -24.4%). At 4x+ the gradient collapses (1.07x -> 1.00x): hover
        # becomes as good as kmeans5 and the task is degenerate.
        "OM2G_TAU_CAP_BPS":          repr(T.OM2G_VOICE_BPS * TAU_MULT),
        "OM2G_RADIO_CAP_BPS":        repr(T.OM2G_RADIO_CAP_APP_BPS * CAP_MULT),
        # "nocontrol" never fires a beacon in an episode -> zero real charge,
        # not the compressed-arm formula (see om2g_extra_args' beacon var).
        "OM2G_CONTROL_ON_AIR_BYTES": repr(
            0.0 if arm == "nocontrol"
            else T.om2g_message_on_air_bytes(arm, _wire_latent(arm))),
        "OM2G_SYNC_INTERVAL_S":      repr(T.OM2G_SYNC_INTERVAL_S),
        # kbps, not Mbps -- see max_flow_reward.REWARD_UNIT_BPS's comment.
        "OM2G_REWARD_UNIT_BPS":   "1000.0",
        "OMNET_DRONE_TX_mW":      repr(T.OM2G_DRONE_TX_MW),
        "OMNET_USER_TX_mW":       repr(T.OM2G_USER_TX_MW),

        # ── site 3: the environment ──────────────────────────────────────
        "OMNET_STEP_M":        repr(T.OM2G_STEP_M),
        # Reaches omnet_env.MAX_USERS_CAP; train_omnet_arton.sh passes the same
        # value to model.max_users, so the padded block and the model agree.
        "OMNET_MAX_USERS_CAP": str(T.OM2G_MAX_USERS_CAP),
        "OMNET_ENCODER_MODE":  "learned",   # raw user block reaches the model
        # Peer visibility is decided by MEASURED beacon delivery in om2g_env,
        # not by the OMNET track's geometric component mask. Two mechanisms
        # gating the same thing would double-count.
        "OMNET_PEER_LINK_MASK": "0",
        # $OM2G_DIFF_REWARD=0 switches to the plain GLOBAL reward. The default
        # is the per-drone difference reward D_i, which pays a drone for
        # covering users nobody else covers (this rewards separation from
        # peers regardless of where they start).
        "OMNET_DIFFERENCE_REWARD": os.environ.get("OM2G_DIFF_REWARD", "1"),
        #                                   per-drone credit assignment,
                                            # as on the OMNET track

        # ── site 4: the model ────────────────────────────────────────────
        "OM2G_ENCODER":             ARM_ENCODER[arm],
        "OMNET_USER_ENC_OUT_DIM":   str(ARM_LATENT_DIM[arm]),
    }


def om2g_run_name(arm: str, arch: str = "st", tag: str = "") -> str:
    """Run name / save suffix. Always carries the track token."""
    base = f"OM2G_{arm}_{arch}_N{T.OM2G_NUM_DRONES}_L{int(T.OM2G_ARENA_M)}"
    return base + (f"_{tag}" if tag else "")


def om2g_selftest(verbose: bool = True) -> None:
    """Check the derived numbers against om2g.ini's literals.

    The ini carries defaults so a human reading it sees a complete config; the
    authority is om2g_track.py.  If they disagree, the ini is stale and this
    fails loudly rather than letting a run use one number while the docs quote
    another.
    """
    ini_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "omnet_smoke", "om2g.ini")
    ini = open(ini_path).read()

    def ini_val(pattern):
        m = re.search(pattern, ini)
        assert m, f"om2g.ini has no match for {pattern!r}"
        return m.group(1)

    checks = []
    checks.append(("drone TX mW", float(ini_val(
        r"\*\.host\[\*\]\.wlan\[\*\]\.radio\.transmitter\.power\s*=\s*([\d.]+)mW")),
        T.OM2G_DRONE_TX_MW, 1e-3))
    checks.append(("user TX mW", float(ini_val(
        r"\*\.user\[\*\]\.wlan\[\*\]\.radio\.transmitter\.power\s*=\s*([\d.]+)mW")),
        T.OM2G_USER_TX_MW, 1e-4))
    checks.append(("arena m", float(ini_val(
        r"\*\.rlBridge\.arenaWidth\s*=\s*([\d.]+)")), T.OM2G_ARENA_M, 1e-6))
    checks.append(("compressed beaconBytes", float(re.search(
        r"\[Config OM2G_COMPRESSED\][\s\S]*?beaconBytes\s*=\s*(\d+)", ini).group(1)),
        T.om2g_message_bytes("compressed", ARM_LATENT_DIM["compressed"]), 0.5))
    checks.append(("uncompressed beaconBytes", float(re.search(
        r"\[Config OM2G_UNCOMPRESSED\][\s\S]*?beaconBytes\s*=\s*(\d+)", ini).group(1)),
        T.om2g_message_bytes("uncompressed", ARM_LATENT_DIM["uncompressed"]), 0.5))

    # The ini's numUsers/beaconBytes are DEFAULTS for the 40-user families; a
    # run with $OM2G_USERS set overrides both through OMNET_EXTRA_ARGS (which
    # omnet_env._launch_omnet appends LAST, so it wins). Comparing the literal
    # to a track value computed for a different population is then a false
    # alarm, so those two checks are dropped -- the ones that matter (radio,
    # arena, powers) still run.
    # $OM2G_USER_PARTNER_TAG widens the per-user token 13 -> 17, so the
    # uncompressed message grows 2136 -> 2776 B. Same situation as the two
    # above: the ini literal is a 13-wide DEFAULT and OMNET_EXTRA_ARGS
    # carries the real value, so comparing them is a false alarm.
    if (os.environ.get("OM2G_USERS") or os.environ.get("OM2G_LATENT_DIM")
            or os.environ.get("OM2G_USER_PARTNER_TAG")):
        checks = [c for c in checks if "beaconBytes" not in c[0]]
    if os.environ.get("OM2G_ARENA"):
        checks = [c for c in checks if "arena" not in c[0]]

    # $OM2G_RANGE_DRONE / $OM2G_RANGE_USER deliberately move the TX powers away
    # from the values written into om2g.ini, so comparing the two is guaranteed
    # to fail. That is not a drift -- config_knobs.build_omnet_extra_args()
    # emits `--*.host[*]...transmitter.power=<computed>mW` and
    # `--*.user[*]...transmitter.power=<computed>mW`, and an OMNeT++ command
    # line override always beats the ini, so the SIMULATOR gets the overridden
    # power either way. Drop only the check that the override invalidates, the
    # same way OM2G_ARENA above drops the arena check; every other constant
    # stays guarded. Verify in the run header that transmitter.power matches
    # the intended range before trusting any result.
    if os.environ.get("OM2G_RANGE_DRONE"):
        checks = [c for c in checks if c[0] != "drone TX mW"]
    if os.environ.get("OM2G_RANGE_USER"):
        checks = [c for c in checks if c[0] != "user TX mW"]

    bad = []
    for name, got, want, tol in checks:
        if abs(got - want) > tol:
            bad.append(f"    {name}: om2g.ini={got}  om2g_track={want}")
        elif verbose:
            print(f"  ok  {name:<26} {got}")
    if bad:
        raise AssertionError("om2g.ini is out of sync with om2g_track.py:\n"
                             + "\n".join(bad))

    # The passthrough arm's latent width is structural, not a free choice.
    assert ARM_LATENT_DIM["uncompressed"] == T.OM2G_MAX_USERS_CAP * T.OM2G_OBS_FEATURES
    if verbose:
        print("  ok  arm latent dims             "
              f"{ARM_LATENT_DIM['compressed']} / {ARM_LATENT_DIM['uncompressed']}")
        print("OM2G config self-test PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="compressed", choices=ARMS)
    ap.add_argument("--arch", default="st")
    ap.add_argument("--n-drones", type=int, default=None)
    ap.add_argument("--ep-steps", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--shell", action="store_true",
                    help="print as `export K=V` lines")
    a = ap.parse_args()
    if a.selftest:
        T.om2g_selftest(verbose=False)
        om2g_selftest()
        return
    env = om2g_env_vars(a.arm, a.n_drones, a.ep_steps)
    env["OMNET_SAVE_SUFFIX"] = om2g_run_name(a.arm, a.arch)
    for k, v in env.items():
        print(f"export {k}='{v}'" if a.shell else f"{k}={v}")


if __name__ == "__main__":
    main()
