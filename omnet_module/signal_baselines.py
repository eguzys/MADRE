"""
signal_baselines.py — cheap, myopic, signal-only baselines for the OMNeT++ track.

TWO baselines, selected by --mode:

  greedy : each drone probes 8 compass directions, picks the one that maximizes
           ITS OWN local signal (sum of link-capacity to users it can serve).
           Robust, discrete, slightly more compute (8 signal evals/drone).

  spsa   : each drone feels the local signal GRADIENT (two random-direction
           pokes) and steps along it. No fixed directions, cheapest, but
           noisier (shadowing can mislead the gradient). The "efficient but
           weaker" baseline.

Both are LOCAL and UNCOORDINATED: a drone optimizes only its OWN signal to
nearby users. No drone sees the global reward, the LP, or the other drones'
choices. Myopia is the point — this is what the learned joint policy beats.

WHY THIS IS FAST: it reuses `_link_capacity_bps` (distance -> signal -> capacity),
a pure formula. NO max-flow LP is solved. Greedy-on-reward solved 40 LPs/step;
this solves zero. Episodes run in seconds.

Run order (same as smoke_test.py): start this FIRST, then launch OMNeT++.

Local signal = "if I (one drone) sit here, how much capacity can I offer to the
users near me?" = sum over users of link-capacity(me<->user). Each drone
optimizes that independently. It does NOT account for being a relay/bridge —
that omission is exactly why it can't form relay chains, which is the thesis point.
"""

import os
import sys
import math
import json
import argparse

# Reuse the project's signal math. Pure distance->capacity, no LP.
from max_flow_reward import (_link_capacity_bps, NOISE_FLOOR_dBm,
                             ALPHA, FREQ_GHz, SENSITIVITY_dBm,
                             SNIR_THRESHOLD_dB)

# --- knobs -------------------------------------------------------------------
NUM_PROBE_DIRS = 8     # greedy mode: compass directions
# STEP_M reads $OMNET_STEP_M exactly like greedy_oracle.STEP_M and
# omnet_env.STEP_M (default 20.0), so the baseline probes the same distance
# the drones actually move per step.
STEP_M = float(os.environ.get("OMNET_STEP_M", 20.0))  # metres/decision-step
# $OM2G_BASELINE_TRUEPOS=1: baselines read every heard user's true CURRENT
# position. In a static world the two modes are identical by construction.
_BASELINE_TRUE_POS = os.environ.get("OM2G_BASELINE_TRUEPOS") == "1"
# See relay_ladder_velocities: restricts the relay arms to heard users only, so
# they do not read where the clusters are.
_RELAY_HONEST = os.environ.get("OM2G_RELAY_HONEST") == "1"
# $OM2G_LEVER_NO_DEDUP=1: fuse peers' summaries but do NOT discount users a
# peer already covers. Isolates shared sensing from de-confliction; see the
# module docstring note on hop_limit.
_NO_DEDUP = os.environ.get("OM2G_LEVER_NO_DEDUP") == "1"
# Drone TX power the scripted baselines PLAN with. It reads $OMNET_DRONE_TX_mW,
# the same variable max_flow_reward.py uses to SCORE, so planning and scoring
# use one radio (e.g. 6.585 mW in the OM2G world: r_d = 523 m, so a one-hop
# beacon does not cross a 1465 m cluster separation). evaluate_omnet.py
# asserts the two copies agree.
DRONE_TX_mW = float(os.environ.get("OMNET_DRONE_TX_mW", 200.0))  # mW
SPSA_C = 20.0          # spsa: probe perturbation distance (m)
# -----------------------------------------------------------------------------

DIRS = [(math.cos(2 * math.pi * k / NUM_PROBE_DIRS),
         math.sin(2 * math.pi * k / NUM_PROBE_DIRS))
        for k in range(NUM_PROBE_DIRS)]


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _local_signal(drone_xy, user_pos, noise_floor_for_drone):
    """How much capacity can THIS drone offer from position drone_xy?

    Sum of link capacity to every user. Pure signal math, no LP. This is the
    drone's own myopic objective — it has no idea about relay chains or other
    drones; it just wants strong links to nearby users.
    """
    total = 0.0
    for u in user_pos:
        d = _dist(drone_xy, u)
        total += _link_capacity_bps(d, DRONE_TX_mW, noise_floor_for_drone)
    return total


def _nf_for(noise_floor, i):
    if noise_floor and i < len(noise_floor):
        return noise_floor[i]
    return NOISE_FLOOR_dBm


def _marginal_signal(drone_xy, user_pos, nf, already):
    """Coverage THIS drone adds beyond what the rest of the swarm already gives.

    `already[u]` is the best capacity user u already receives from any OTHER
    drone. A drone parked on top of an already-well-covered user therefore
    scores ~0 for it and has to go find someone else.

    Information note: the raw obs does carry per-user state, but every
    learned architecture compresses the 25-user
    block 40:1 into LEARNED_USER_ENCODER_OUT_DIM=8 numbers via a masked SUM
    before its first layer (architectures/user_encoder.py). Individual user
    positions are NOT recoverable from that, so greedy_velocities operates on
    strictly MORE user information than any learned policy here.
    Report greedy as an ORACLE / upper bound, never as a baseline. The
    information-matched comparator is aggregate_velocities() below.
    """
    total = 0.0
    for u, cov in zip(user_pos, already):
        gain = _link_capacity_bps(_dist(drone_xy, u), DRONE_TX_mW, nf) - cov
        if gain > 0.0:
            total += gain
    return total


def greedy_velocities(obs):
    """8-direction MARGINAL-coverage hill-climb, independent per drone.

    With _local_signal as the score (each drone maximises its OWN total
    signal, unaware of the others) every drone climbs toward the same densest
    user region and the swarm collapses onto one point; on 20 seeds that
    objective lost to HOVER on 7/20 seeds at every user_spread_eff swept
    (150/200/250/350).

    Scoring the MARGINAL coverage instead (what this drone adds on top of the
    rest of the swarm) removes the incentive to stack, and is still purely
    local/sensory. On the same 20 seeds it beats random 20/20 at every
    spread and beats hover 19/20 at spread 350.

    _local_signal is kept -- spsa_velocities still uses it, and it is the
    faithful "no idea the others exist" objective when that is what is wanted.
    """
    drone_pos = obs.get("positions", [])
    user_pos = obs.get("user_positions", [])
    noise_floor = obs.get("noise_floor", [])
    vels = []

    for i, dp in enumerate(drone_pos):
        nf = _nf_for(noise_floor, i)
        # Best capacity each user already gets from some OTHER drone.
        already = [
            max((_link_capacity_bps(_dist(dj, u), DRONE_TX_mW, _nf_for(noise_floor, j))
                 for j, dj in enumerate(drone_pos) if j != i), default=0.0)
            for u in user_pos
        ]
        best = _marginal_signal(dp, user_pos, nf, already)   # stay-put baseline
        best_vel = (0.0, 0.0)
        for (dx, dy) in DIRS:
            cand = (dp[0] + dx * STEP_M, dp[1] + dy * STEP_M)
            s = _marginal_signal(cand, user_pos, nf, already)
            if s > best:
                best = s
                # Unit vector. The bridge does not scale actions in [-1,1]: the
                # simulator reads the action as a velocity in m/s and only clips
                # at maxSpeed, so a unit vector is a 1 m step. Scaling is applied by
                # OMNeTEnv.step(), and by main() below for this standalone
                # driver, which bypasses OMNeTEnv.
                best_vel = (dx, dy)
        vels.append([best_vel[0], best_vel[1]])
    return vels


def _user_moments(user_pos):
    """First two moments of the user cloud: (cx, cy, sxx, syy, sxy, n).

    Five numbers plus a count -- deliberately comparable to the 8-wide pooled
    summary the learned policies get, and like that summary it is a symmetric
    function of the set that cannot identify any individual user.
    """
    n = len(user_pos)
    if n == 0:
        return None
    cx = sum(u[0] for u in user_pos) / n
    cy = sum(u[1] for u in user_pos) / n
    d = max(n - 1, 1)
    sxx = sum((u[0] - cx) ** 2 for u in user_pos) / d
    syy = sum((u[1] - cy) ** 2 for u in user_pos) / d
    sxy = sum((u[0] - cx) * (u[1] - cy) for u in user_pos) / d
    return (cx, cy, sxx, syy, sxy, n)


def _moment_quadrature(mom):
    """Reconstruct a stand-in user cloud from the moments alone.

    Centroid + 8 points on the 1-sigma ellipse (principal axes from the 2x2
    covariance), each standing for n/9 real users. This is the best any policy
    holding only aggregate statistics could do: it knows WHERE the mass is and
    HOW SPREAD it is, and nothing about who is individually underserved.
    """
    cx, cy, sxx, syy, sxy, n = mom
    tr, det = sxx + syy, sxx * syy - sxy * sxy
    disc = math.sqrt(max(tr * tr / 4.0 - det, 0.0))
    l1, l2 = tr / 2.0 + disc, max(tr / 2.0 - disc, 0.0)
    if abs(sxy) > 1e-12:
        v1 = (l1 - syy, sxy)
    else:
        v1 = (1.0, 0.0) if sxx >= syy else (0.0, 1.0)
    nrm = math.hypot(*v1) or 1.0
    e1 = (v1[0] / nrm, v1[1] / nrm)
    e2 = (-e1[1], e1[0])
    a, b = math.sqrt(max(l1, 0.0)), math.sqrt(max(l2, 0.0))
    pts = [(cx, cy)]
    for k in range(8):
        th = 2.0 * math.pi * k / 8.0
        ox, oy = a * math.cos(th), b * math.sin(th)
        pts.append((cx + ox * e1[0] + oy * e2[0], cy + ox * e1[1] + oy * e2[1]))
    return pts, n / float(len(pts))


def aggregate_velocities(obs):
    """INFORMATION-MATCHED baseline. The fair comparator for a learned policy.

    Identical 8-direction marginal hill-climb to greedy_velocities, with one
    change: the true user set is replaced by its first two moments (centroid +
    covariance), reconstructed as 9 weighted stand-in users. Drone positions
    stay exact -- those are NOT compressed in the learned observation (each peer
    is its own token); only the 25-user block is pooled to 8 numbers.

    So this baseline sees exactly the class of user information a learned policy
    can actually recover, and greedy_velocities sees strictly more. Reporting
    all three together (aggregate < learned < greedy-oracle) is what makes the
    architecture comparison legible: the learned policies should BEAT this, and
    trail the oracle by a stated margin.
    """
    drone_pos = obs.get("positions", [])
    user_pos = obs.get("user_positions", [])
    noise_floor = obs.get("noise_floor", [])

    mom = _user_moments(user_pos)
    if mom is None:
        return [[0.0, 0.0] for _ in drone_pos]
    pseudo, weight = _moment_quadrature(mom)

    vels = []
    for i, dp in enumerate(drone_pos):
        nf = _nf_for(noise_floor, i)
        already = [
            max((_link_capacity_bps(_dist(dj, u), DRONE_TX_mW, _nf_for(noise_floor, j))
                 for j, dj in enumerate(drone_pos) if j != i), default=0.0)
            for u in pseudo
        ]
        best = _marginal_signal(dp, pseudo, nf, already) * weight
        best_vel = (0.0, 0.0)
        for (dx, dy) in DIRS:
            cand = (dp[0] + dx * STEP_M, dp[1] + dy * STEP_M)
            s = _marginal_signal(cand, pseudo, nf, already) * weight
            if s > best:
                best, best_vel = s, (dx, dy)
        vels.append([best_vel[0], best_vel[1]])
    return vels


def spsa_velocities(obs, rng):
    """Gradient-by-feel: two random-direction pokes estimate the slope,
    step uphill along it. Independent per drone."""
    drone_pos = obs.get("positions", [])
    user_pos = obs.get("user_positions", [])
    noise_floor = obs.get("noise_floor", [])
    vels = []

    for i, dp in enumerate(drone_pos):
        nf = _nf_for(noise_floor, i)
        # Rademacher random direction (each component +1 or -1), normalized
        gx = 1.0 if rng.random() < 0.5 else -1.0
        gy = 1.0 if rng.random() < 0.5 else -1.0
        norm = math.hypot(gx, gy)
        gx, gy = gx / norm, gy / norm

        plus = (dp[0] + SPSA_C * gx, dp[1] + SPSA_C * gy)
        minus = (dp[0] - SPSA_C * gx, dp[1] - SPSA_C * gy)
        s_plus = _local_signal(plus, user_pos, nf)
        s_minus = _local_signal(minus, user_pos, nf)

        # gradient estimate sign along (gx,gy); step uphill
        ghat = (s_plus - s_minus)
        if ghat == 0.0:
            vels.append([0.0, 0.0])
            continue
        direction = 1.0 if ghat > 0 else -1.0
        vels.append([direction * gx, direction * gy])   # unit; bridge scales
    return vels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-drones", type=int, required=True)
    ap.add_argument("--mode", choices=["greedy", "spsa", "static"], default="greedy")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import zmq
    import random
    import csv
    import time
    rng = random.Random(args.seed)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://127.0.0.1:{args.port}")
    print(f"[{args.mode}] bound tcp://127.0.0.1:{args.port}, waiting for "
          f"{args.num_drones} drones", flush=True)

    csv_path = f"signal_{args.mode}_run_{int(time.time())}.csv"
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.writer(csv_f)
    header = ["step", "total_signal"]
    for d in range(args.num_drones):
        header += [f"d{d}_x", f"d{d}_y"]
    csv_w.writerow(header)
    print(f"[{args.mode}] logging to {csv_path}", flush=True)

    step = 0
    while True:
        msg = sock.recv()
        obs = json.loads(msg.decode("utf-8"))

        if obs.get("done", False):
            reply = {"type": "action",
                     "actions": [[0.0, 0.0]] * args.num_drones,
                     "reward": 0.0}
            sock.send_string(json.dumps(reply))
            csv_f.close()
            print(f"[{args.mode}] episode done at step {step}; wrote {csv_path}",
                  flush=True)
            break

        if args.mode == "static":
            vels = [[0.0, 0.0]] * args.num_drones
        elif args.mode == "greedy":
            vels = greedy_velocities(obs)
        else:
            vels = spsa_velocities(obs, rng)

        assert len(vels) == args.num_drones, \
            f"produced {len(vels)} vels, need {args.num_drones}"

        # diagnostic: total local signal across drones (NOT the training reward,
        # just a progress readout for the baseline)
        dpos = obs.get("positions", [])
        upos = obs.get("user_positions", [])
        nf = obs.get("noise_floor", [])
        total_sig = sum(_local_signal(dpos[i], upos, _nf_for(nf, i))
                        for i in range(len(dpos)))

        row = [step, total_sig]
        for p in dpos:
            row += [float(p[0]), float(p[1])]
        csv_w.writerow(row)
        csv_f.flush()

        # Same scaling OMNeTEnv.step() applies -- see the note in
        # _signal_velocities. This driver talks to the simulator directly, so
        # without this the drones move 1 m/step regardless of STEP_M.
        # STEP_M reads $OMNET_STEP_M (see the knobs block at the top).
        scaled = [[vx * STEP_M, vy * STEP_M] for vx, vy in vels]
        reply = {"type": "action", "actions": scaled, "reward": total_sig / 1e6}
        sock.send_string(json.dumps(reply))

        if step % 25 == 0:
            print(f"[{args.mode}] step {step:4d}  total_signal {total_sig:.3e}",
                  flush=True)
        step += 1


if __name__ == "__main__":
    main()


# ── Realistic distributed greedy (the DEPLOYABLE baseline) ───────────────────
#
# `greedy_velocities` above is an ORACLE and is reported as one: it reads every user's exact position (including users no drone can
# hear -- a drone's uplink decode range is r_u = 157.8 m against a 550 m user
# cloud in a 1200 m arena, so a real drone detects a minority of them) and
# every peer's exact position, at zero airtime cost. Nothing in a deployment
# supplies that.
#
# `greedy_realistic_velocities` is the same marginal hill-climb restricted to
# information a drone can actually obtain, with the obtaining charged:
#
#   USERS  -- only those this drone has actually heard (user_rssi_valid[i][u]).
#             Positions of heard users are taken as exact. A real receiver has
#             RSSI (range) and angle-of-arrival, both already columns 1 and 0
#             of the learned policy's own token, so the recoverable quantity is
#             a noisy version of this. Using the noise-free version isolates
#             the variable to WHICH users are known, with no localisation error
#             added, and gives the baseline the more favourable input.
#
#   PEERS  -- only drones within decode range, i.e. the same positive-capacity
#             test the reward LP itself uses for an edge (max_flow_reward.
#             _build_edges), so the neighbour set cannot drift from the
#             topology the flows see. hop_limit=1 is one-hop HELLO beaconing
#             (RFC 6130 / OLSR RFC 3626 neighbour discovery, GPSR-style
#             position beacons); hop_limit>1 floods the beacon that many hops,
#             at correspondingly higher airtime.
#
#   COST   -- the beacon is charged against the same radio budget as user
#             traffic via $OM2G_CONTROL_ON_AIR_BYTES, exactly as the learned
#             arms are. See om2g_track.om2g_beacon_on_air_bytes.
#
# Prior art for the algorithm itself, so this reads as a standard method
# rather than a bespoke construction: distributed Lloyd / Voronoi coverage
# control (Cortes, Martinez, Karatas & Bullo, IEEE T-RA 2004) and virtual-force
# relay deployment (Howard, Mataric & Sukhatme 2002; Zou & Chakrabarty,
# INFOCOM 2003) are both "move along the local gradient of a coverage
# objective using one-hop neighbour state", which is what this is.
#
# Expected consequence: this baseline is WEAKER than the oracle by
# construction, because the information is not there. `_local_signal` above
# is the opposite extreme -- a drone with NO peer knowledge collapses the swarm
# and loses to hover on 7/20 seeds. The pair brackets the realistic range.


def _heard_users(obs, i):
    """Indices of users drone i has actually received a frame from.

    Falls back to "all users" only when the simulator did not ship the
    validity mask at all (old obs JSON / unit tests), never when the mask is
    present but empty -- an empty mask means this drone heard nobody, which is
    a real state that must not be upgraded to omniscience.
    """
    rssi_valid = obs.get("user_rssi_valid")
    if rssi_valid is None:
        return list(range(len(obs.get("user_positions", []))))
    if i >= len(rssi_valid):
        return []
    return [u for u, v in enumerate(rssi_valid[i]) if v]


# ── What a drone BELIEVES about where the users are ───────────────────────────
#
# `user_rssi_valid` is CUMULATIVE: the simulator sets it on the first frame ever
# received from a user and never clears it inside an episode. Reading
# `user_positions[u]` off the back of that bit therefore hands the baseline the
# user's TRUE CURRENT position forever, from a single contact -- perfect
# tracking, for free. With static users that is exact and harmless, because the
# position never changes. Under $OM2G_USER_ROAM_M it is a free oracle, and it is
# the reason a mobility sweep would report a null no matter what the truth was:
# the scripted arms simply cannot feel staleness.
#
# So each drone remembers, per user, only where that user WAS when it last
# received a frame from them. Freshness is detected from `user_rssi_age`
# DECREASING, which means a new frame arrived; that test needs no knowledge of
# the step duration and cannot drift from it.
_LAST_HEARD = {}   # (drone, user) -> [x, y] at last contact
_LAST_AGE = {}     # (drone, user) -> previous age reading


def reset_baseline_memory():
    """Forget every remembered user position. Call at each episode reset.

    Memory that survives an episode boundary is memory of a DIFFERENT world --
    the users are resampled -- so this is not an optimisation, it is required
    for correctness.
    """
    _LAST_HEARD.clear()
    _LAST_AGE.clear()
    _COMMITTED.clear()


def _heard_user_positions(obs, i):
    """Positions of the users drone i has heard, AS DRONE i LAST SAW THEM."""
    user_pos = obs.get("user_positions", [])
    idxs = [u for u in _heard_users(obs, i) if u < len(user_pos)]
    ages = obs.get("user_rssi_age")
    if _BASELINE_TRUE_POS or ages is None or i >= len(ages):
        # No age channel (a simulator predating user_rssi_age) or the explicit
        # override. Correct and exact whenever users are static.
        return [user_pos[u] for u in idxs]
    row = ages[i]
    out = []
    for u in idxs:
        key = (i, u)
        age = row[u] if u < len(row) else 0.0
        prev = _LAST_AGE.get(key)
        if prev is None or age < prev:      # a NEW frame arrived -> refresh
            _LAST_HEARD[key] = list(user_pos[u])
        _LAST_AGE[key] = age
        out.append(_LAST_HEARD.get(key, user_pos[u]))
    return out


def _decodable(d_m, tx_mW, noise_floor_dBm):
    """Can a receiver at noise_floor_dBm actually decode a frame from d_m away?

    The HARD INET condition (Ieee80211Radio.ned:42-44): received power clears
    sensitivity AND SNIR clears its threshold. Deliberately NOT
    `_link_capacity_bps(...) > 0` -- that function is a smooth logistic (no
    zero-gradient zone in the REWARD), and its own docstring notes it "is never
    exactly 0, so callers that treated 0 as 'no edge' now get a full graph".
    Used as a reachability test it would make every drone a one-hop neighbour
    of every other and turn hop_limit into a no-op. Reception is a hard event; only the reward is smoothed.
    """
    d_m = max(d_m, 1.0)
    lam = 3e8 / (FREQ_GHz * 1e9)
    fspl_1m_dB = 20.0 * math.log10(4 * math.pi / lam)
    rx_dBm = 10.0 * math.log10(tx_mW) - (fspl_1m_dB + 10.0 * ALPHA * math.log10(d_m))
    return (rx_dBm >= SENSITIVITY_dBm
            and (rx_dBm - noise_floor_dBm) >= SNIR_THRESHOLD_dB)


def _peer_neighbours(drone_pos, noise_floor, i, hop_limit=1):
    """Indices of drones whose beacon reaches drone i within hop_limit hops."""
    # hop_limit <= 0 means the drone is DEAF: it fuses nothing from peers and
    # sees only what its own radio heard. This is the zero point of the
    # communication lever; without it max(1, ...) below would promote it to a
    # full one-hop round.
    if int(hop_limit) <= 0:
        return []
    nd = len(drone_pos)
    adj = [[] for _ in range(nd)]
    for a in range(nd):
        for b in range(nd):
            if a == b:
                continue
            if _decodable(_dist(drone_pos[a], drone_pos[b]),
                          DRONE_TX_mW, _nf_for(noise_floor, b)):
                adj[a].append(b)          # b hears a

    seen, frontier = {i}, [i]
    for _ in range(max(1, int(hop_limit))):
        nxt = []
        for node in frontier:
            for src in range(nd):
                if node in adj[src] and src not in seen:
                    seen.add(src)
                    nxt.append(src)
        frontier = nxt
        if not frontier:
            break
    seen.discard(i)
    return sorted(seen)


def greedy_realistic_velocities(obs, hop_limit=1):
    """DEPLOYABLE greedy: marginal hill-climb on locally-obtainable state only.

    Same 8-direction marginal-coverage step as greedy_velocities, but each
    drone reasons about only the users it has heard and only the peers whose
    beacon reaches it. Report this as `greedy`; report greedy_velocities as
    `oracle`.
    """
    drone_pos = obs.get("positions", [])
    user_pos = obs.get("user_positions", [])
    noise_floor = obs.get("noise_floor", [])
    vels = []

    for i, dp in enumerate(drone_pos):
        nf = _nf_for(noise_floor, i)
        mine = _heard_user_positions(obs, i)
        if not mine:
            # Heard nobody: no local gradient exists. Hold station rather than
            # inventing a direction -- moving on no evidence is what collapsed
            # _local_signal's swarm.
            vels.append([0.0, 0.0])
            continue

        peers = _peer_neighbours(drone_pos, noise_floor, i, hop_limit)
        already = [0.0] * len(mine) if _NO_DEDUP else [
            max((_link_capacity_bps(_dist(drone_pos[j], u), DRONE_TX_mW,
                                    _nf_for(noise_floor, j))
                 for j in peers), default=0.0)
            for u in mine
        ]

        best = _marginal_signal(dp, mine, nf, already)   # stay-put baseline
        best_vel = (0.0, 0.0)
        for (dx, dy) in DIRS:
            cand = (dp[0] + dx * STEP_M, dp[1] + dy * STEP_M)
            s = _marginal_signal(cand, mine, nf, already)
            if s > best:
                best, best_vel = s, (dx, dy)
        vels.append([best_vel[0], best_vel[1]])
    return vels


# ── The discriminating ablation: greedy with the POLICY's information ────────
#
# Where `greedy_realistic_velocities` outscores a learned policy, two very
# different causes can predict that, and they call for opposite responses:
#
#   (A) INFORMATION. Greedy does exact per-user marginal-capacity arithmetic
#       over the users it hears. The learned policy cannot: every architecture
#       masked-SUMS its raw user block into LEARNED_USER_ENCODER_OUT_DIM=8
#       numbers before its first layer (architectures/user_encoder.py:38) --
#       and that pooling is applied to the drone's OWN observation, not only to
#       the message it transmits. If this is the cause, the policy is at the
#       ceiling OF ITS INPUT and no amount of training closes the gap.
#
#   (B) OPTIMISATION. The policy has the information it needs and simply has
#       not learned to use it (under-training, entropy collapse, bad seeds).
#       Then more frames / a better schedule closes the gap.
#
# This function holds the USER SET fixed (same heard users as realistic greedy)
# and removes ONLY the per-user access, replacing the heard set with its first
# two moments reconstructed as 9 pseudo-users -- the same moment trick
# aggregate_velocities uses, applied locally instead of globally. It is
# therefore "greedy with exactly the class of user information the learned
# policy can hold".
#
# The prediction that decides it:
#   score(greedy_pooled) ~ score(policy)  => cause (A), pooling. Change the
#                                            encoder, not the training.
#   score(greedy_pooled) ~ score(greedy)  => cause (B), optimisation. The
#                                            information was never the limit.
#
# Everything else -- peers, beacon, charge, step, marginal objective -- is
# identical to greedy_realistic_velocities, so the contrast is clean.


def greedy_pooled_velocities(obs, hop_limit=1):
    """Realistic greedy, but it may only hold POOLED statistics of its users.

    Same heard-user set and same one-hop peers as greedy_realistic_velocities;
    the only change is that the heard users are replaced by the 9-point
    moment reconstruction of themselves. Isolates per-user access as a
    variable.
    """
    drone_pos = obs.get("positions", [])
    user_pos = obs.get("user_positions", [])
    noise_floor = obs.get("noise_floor", [])
    vels = []

    for i, dp in enumerate(drone_pos):
        nf = _nf_for(noise_floor, i)
        heard = _heard_user_positions(obs, i)
        if not heard:
            vels.append([0.0, 0.0])
            continue

        mom = _user_moments(heard)
        pseudo, weight = _moment_quadrature(mom)

        peers = _peer_neighbours(drone_pos, noise_floor, i, hop_limit)
        already = [0.0] * len(pseudo) if _NO_DEDUP else [
            max((_link_capacity_bps(_dist(drone_pos[j], u), DRONE_TX_mW,
                                    _nf_for(noise_floor, j))
                 for j in peers), default=0.0)
            for u in pseudo
        ]

        best = _marginal_signal(dp, pseudo, nf, already) * weight
        best_vel = (0.0, 0.0)
        for (dx, dy) in DIRS:
            cand = (dp[0] + dx * STEP_M, dp[1] + dy * STEP_M)
            s = _marginal_signal(cand, pseudo, nf, already) * weight
            if s > best:
                best, best_vel = s, (dx, dy)
        vels.append([best_vel[0], best_vel[1]])
    return vels


# ── The DEPLOYABLE engineered protocol: `engineered` ─────────────────────────
#
# `aggregate_velocities` is ALL-SEEING and pays NO airtime: it reads the full
# 40-user set and compresses it centrally. That makes it an upper BOUND on what
# a moment-based code can achieve, not a deployable protocol.
#
# `engineered_velocities` is the deployable version of the same idea, and it is
# the information-matched opponent for "learned message vs hand-designed
# message":
#
#   1. each drone computes the two moments of the users IT ACTUALLY HEARS
#      (_heard_users -> _user_moments): 5 floats + a count = ~24 B payload,
#      the same order as the learned arm's 88 B and far above greedy's 8 B;
#   2. it broadcasts them to peers within hop_limit (the same neighbour
#      relation greedy's position beacon uses);
#   3. each drone FUSES the received moment sets into a stand-in user cloud
#      and hill-climbs on marginal coverage of that cloud.
#
# Fusion is by count-weighted pooling of the per-drone quadratures rather than
# by merging moments into one Gaussian: merging would re-create the single-blob
# failure mode this baseline exists to avoid, and would understate the
# engineered protocol. Keeping each peer's quadrature separate is strictly
# stronger, which is the direction a baseline should err in.
#
# It is charged exactly like the learned arm (OM2G_FORCE_CHARGE_BYTES).
def engineered_velocities(obs, hop_limit=1):
    drone_pos = obs.get("positions", [])
    user_pos = obs.get("user_positions", [])
    noise_floor = obs.get("noise_floor", [])

    # Step 1: every drone summarises only what it heard.
    local = []
    for i in range(len(drone_pos)):
        mine = _heard_user_positions(obs, i)
        local.append(_user_moments(mine) if mine else None)

    vels = []
    for i, dp in enumerate(drone_pos):
        nf = _nf_for(noise_floor, i)
        # Step 2: fuse my own summary with those of peers I can hear.
        sources = [i] + list(_peer_neighbours(drone_pos, noise_floor, i, hop_limit))
        peers = _peer_neighbours(drone_pos, noise_floor, i, hop_limit)

        # Per-source quadratures, each kept with ITS OWN weight. The weight is
        # applied INSIDE the sum: a single global scalar would multiply all 8
        # candidate directions equally and cancel in the argmax, so a peer
        # that heard 2 users would carry the same mass as one that heard 20.
        parts = []
        for s in sources:
            if s < len(local) and local[s] is not None:
                p, w = _moment_quadrature(local[s])
                if p:
                    already_s = [0.0] * len(p) if _NO_DEDUP else [
                        max((_link_capacity_bps(_dist(drone_pos[j], u), DRONE_TX_mW,
                                                _nf_for(noise_floor, j))
                             for j in peers), default=0.0)
                        for u in p
                    ]
                    parts.append((p, w, already_s))
        if not parts:
            vels.append([0.0, 0.0])
            continue

        def _score(xy):
            return sum(w * _marginal_signal(xy, p, nf, a) for p, w, a in parts)

        best = _score(dp)
        best_vel = (0.0, 0.0)
        for (dx, dy) in DIRS:
            cand = (dp[0] + dx * STEP_M, dp[1] + dy * STEP_M)
            v = _score(cand)
            if v > best:
                best, best_vel = v, (dx, dy)
        vels.append([best_vel[0], best_vel[1]])
    return vels


# ── `scripted_relay`: an OMNISCITENT hand-placed upper bound ─────────────────
#
# [2026-08-30] Every baseline here is a hill-climb, and in the relay-forced
# world (short drone-drone range) every hill-climb severs its own backbone --
# greedy scores BELOW aggregate there. So none of them measures how much
# throughput the world can actually deliver, and the analytic ceiling
# (nd*RADIO_CAP/2 per hop-count) assumes a uniform hop count that is optimistic.
#
# This policy is not a baseline and must never be reported as one. It is a
# FALSIFIER: a hand-designed, omniscient placement that deliberately does the
# thing no coverage heuristic will -- park drones in empty space to bridge --
# so that its score is an empirical lower bound on the achievable optimum.
#
#   if scripted_relay cannot reach the target, no trained policy will.
#
# Construction: k-means the true user positions; rank clusters by population;
# cover as many of the top clusters as the drone budget allows; spend the
# remaining drones on midpoints of the longest covered-pair gaps, which is
# exactly the relay role. Then move each drone toward its assigned target.

def engineered_exact_velocities(obs, hop_limit=1):
    """`engineered`, but peers exchange the EXACT positions of the users
    they heard instead of a six-number moment summary.

    ONE change from engineered_velocities: no Summarise, no Reconstruct.
    Same 8-direction marginal hill-climb, same one-hop peer set, same
    airtime charge. It still only ever knows users some drone actually
    HEARD -- it is not an oracle.

    Mass is preserved: _moment_quadrature returns 9 points weighted n/9
    (total n); this returns n points weighted 1.0 (total n), so _score is
    on the same scale and the two are directly comparable.
    """
    drone_pos = obs.get("positions", [])
    user_pos = obs.get("user_positions", [])
    noise_floor = obs.get("noise_floor", [])

    # Step 1: every drone summarises only what it heard.
    local = []
    for i in range(len(drone_pos)):
        mine = _heard_user_positions(obs, i)
        local.append(list(mine) if mine else None)   # EXACT positions, no moments

    vels = []
    for i, dp in enumerate(drone_pos):
        nf = _nf_for(noise_floor, i)
        # Step 2: fuse my own summary with those of peers I can hear.
        sources = [i] + list(_peer_neighbours(drone_pos, noise_floor, i, hop_limit))
        peers = _peer_neighbours(drone_pos, noise_floor, i, hop_limit)

        # Per-source quadratures, each kept with ITS OWN weight. The weight is
        # applied INSIDE the sum: a single global scalar would multiply all 8
        # candidate directions equally and cancel in the argmax, so a peer
        # that heard 2 users would carry the same mass as one that heard 20.
        parts = []
        for s in sources:
            if s < len(local) and local[s] is not None:
                p, w = list(local[s]), 1.0   # each heard user stands for itself
                if p:
                    already_s = [0.0] * len(p) if _NO_DEDUP else [
                        max((_link_capacity_bps(_dist(drone_pos[j], u), DRONE_TX_mW,
                                                _nf_for(noise_floor, j))
                             for j in peers), default=0.0)
                        for u in p
                    ]
                    parts.append((p, w, already_s))
        if not parts:
            vels.append([0.0, 0.0])
            continue

        def _score(xy):
            return sum(w * _marginal_signal(xy, p, nf, a) for p, w, a in parts)

        best = _score(dp)
        best_vel = (0.0, 0.0)
        for (dx, dy) in DIRS:
            cand = (dp[0] + dx * STEP_M, dp[1] + dy * STEP_M)
            v = _score(cand)
            if v > best:
                best, best_vel = v, (dx, dy)
        vels.append([best_vel[0], best_vel[1]])
    return vels


# ── `scripted_relay`: an OMNISCITENT hand-placed upper bound ─────────────────
#
# [2026-08-30] Every baseline here is a hill-climb, and in the relay-forced
# world (short drone-drone range) every hill-climb severs its own backbone --
# greedy scores BELOW aggregate there. So none of them measures how much
# throughput the world can actually deliver, and the analytic ceiling
# (nd*RADIO_CAP/2 per hop-count) assumes a uniform hop count that is optimistic.
#
# This policy is not a baseline and must never be reported as one. It is a
# FALSIFIER: a hand-designed, omniscient placement that deliberately does the
# thing no coverage heuristic will -- park drones in empty space to bridge --
# so that its score is an empirical lower bound on the achievable optimum.
#
#   if scripted_relay cannot reach the target, no trained policy will.
#
# Construction: k-means the true user positions; rank clusters by population;
# cover as many of the top clusters as the drone budget allows; spend the
# remaining drones on midpoints of the longest covered-pair gaps, which is
# exactly the relay role. Then move each drone toward its assigned target.

def _kmeans(pts, k, iters=25, seed=0):
    import random as _r
    rng = _r.Random(seed)
    if len(pts) <= k:
        return [list(p) for p in pts]
    cent = [list(p) for p in rng.sample(list(pts), k)]
    for _ in range(iters):
        groups = [[] for _ in range(k)]
        for p in pts:
            j = min(range(k), key=lambda c: _dist(cent[c], p))
            groups[j].append(p)
        moved = False
        for j, g in enumerate(groups):
            if not g:
                continue
            nx = [sum(q[0] for q in g) / len(g), sum(q[1] for q in g) / len(g)]
            if _dist(nx, cent[j]) > 1e-6:
                moved = True
            cent[j] = nx
        if not moved:
            break
    return cent, [len(g) for g in groups]


_SR_CACHE = {"key": None, "targets": None}


def _coordinate_ascent_targets(user_pos, sender_pairs, noise_floor, nd,
                               iters=140, restarts=3, seed=11):
    """Near-optimal drone placement by coordinate ascent on the TRUE reward.

    The users are static for the whole episode (RLBridge.cc:432 assigns
    sampledUserPositions_ once and nothing updates it), so the optimal
    placement is a FIXED POINT: it can be solved once per episode and then
    flown to. That is what makes a true near-optimum affordable in the
    simulator instead of a hand-written heuristic.

    A hand-written heuristic (cover the top clusters, bridge the widest gap)
    scores 29.7, below hover, so it cannot serve as an upper bound; this
    construction is used instead.

    This is the only construction in this file allowed to read the true reward.
    It bounds the achievable; it is NOT a baseline and must never be reported
    as one.
    """
    import numpy as _np
    import max_flow_reward as _M
    W = float(os.environ.get("OM2G_ARENA", "1200"))
    U = _np.asarray(user_pos, float)
    base = dict(user_positions=user_pos, sender_pairs=sender_pairs,
                noise_floor=noise_floor)

    def R(P):
        d = dict(base); d["positions"] = [list(map(float, q)) for q in P]
        return float(_M.compute_position_oracle_reward(d)["reward"])

    rng = _np.random.default_rng(seed)
    lo, hi = U.min(0), U.max(0)
    best, bp = -1e18, None
    for _ in range(restarts):
        P = rng.uniform(lo, hi, (nd, 2))
        cur, step = R(P), W / 6.0
        for _ in range(iters):
            moved = False
            for i in range(nd):
                for (dx, dy) in DIRS:
                    Q = P.copy()
                    Q[i] = _np.clip(Q[i] + _np.array([dx, dy]) * step, 0.0, W)
                    v = R(Q)
                    if v > cur:
                        cur, P, moved = v, Q, True
            if not moved:
                step *= 0.55
                if step < 12.0:
                    break
        if cur > best:
            best, bp = cur, P.copy()
    return [list(map(float, q)) for q in bp]


def scripted_relay_velocities(obs, n_clusters=None, link_r_m=None):
    drone_pos = obs.get("positions", [])
    user_pos = obs.get("user_positions", [])
    nd = len(drone_pos)
    if nd == 0 or not user_pos:
        return [[0.0, 0.0] for _ in drone_pos]

    # Solve the placement ONCE per episode; the users never move.
    key = (len(user_pos), round(user_pos[0][0], 3), round(user_pos[-1][1], 3))
    if _SR_CACHE["key"] != key:
        _SR_CACHE["key"] = key
        _SR_CACHE["targets"] = _coordinate_ascent_targets(
            user_pos, obs.get("sender_pairs", []), obs.get("noise_floor", []), nd)
    targets = _SR_CACHE["targets"]

    free = list(range(len(targets)))
    vels = [[0.0, 0.0] for _ in range(nd)]
    for i in sorted(range(nd),
                    key=lambda i: min((_dist(drone_pos[i], targets[t]) for t in free),
                                      default=0.0)):
        if not free:
            break
        t = min(free, key=lambda t: _dist(drone_pos[i], targets[t]))
        free.remove(t)
        dx = targets[t][0] - drone_pos[i][0]
        dy = targets[t][1] - drone_pos[i][1]
        n = math.hypot(dx, dy)
        vels[i] = [0.0, 0.0] if n < 1e-6 else [dx / n, dy / n]
    return vels


# ── The RELAY LADDER: two hand-designed relay-aware comparators ───────────────
#
# These are the rungs that make a win ATTRIBUTABLE rather than merely large.
# Both build the same structure -- a head on every cluster, then a relay chain
# across ONE bridge -- and differ only in HOW THE BRIDGE IS CHOSEN:
#
#   OM2G_RELAY_MODE=blind     pick the CLOSEST feasible pair.  Needs only peer
#                             POSITIONS, which a drone can see directly.
#                             This is the hand-designed relay heuristic, i.e.
#                             a baseline with an explicit connectivity term.
#   OM2G_RELAY_MODE=informed  pick the HIGHEST-CROSS-DEMAND feasible pair.
#                             Needs per-cluster user counts, which exist only
#                             as heard-user state at the drones sitting on
#                             clusters and reach anyone else ONLY through the
#                             transmitted message. This is an ORACLE.
#
# The gap between them is the cash value of the payload's content. Measured
# offline at n=20: blind 112.09, informed 185.72 -- and they pick DIFFERENT
# bridges on 17/20 seeds. A learned policy scoring above `blind` has an
# advantage that positions alone cannot explain.
#
# Neither is a coverage controller, so neither is bounded by the confinement
# cap; report `blind` as a BASELINE rung and `informed` as an upper bound.
def _kmeans_centres(pts, k, iters=40, seed=0, restarts=6):
    """k-means with k-means++ init and restarts, kept best by inertia.

    Why k-means++: with four TIGHT, well-separated crowds, uniformly random
    seeding (`rng.sample(pts, k)`) frequently draws two seeds from the SAME
    crowd, and k-means cannot recover (e.g. centres at (3239,1119) and
    (3277,1138), 42 m apart, so one real cluster gets no centre). k-means++
    picks each new seed with probability proportional to squared distance
    from the nearest existing seed, which handles this world.
    """
    import random as _r
    pts = [list(p) for p in pts]
    if len(pts) <= k:
        return pts, [1] * len(pts)

    def _run(rs):
        rng = _r.Random(rs)
        cent = [list(pts[rng.randrange(len(pts))])]
        while len(cent) < k:
            d2 = [min(_dist(p, c) ** 2 for c in cent) for p in pts]
            tot = sum(d2)
            if tot <= 0:
                cent.append(list(pts[rng.randrange(len(pts))])); continue
            t, acc = rng.random() * tot, 0.0
            for p, w in zip(pts, d2):
                acc += w
                if acc >= t:
                    cent.append(list(p)); break
        groups = [[] for _ in range(k)]
        for _ in range(iters):
            groups = [[] for _ in range(k)]
            for p in pts:
                groups[min(range(k), key=lambda c: _dist(cent[c], p))].append(p)
            moved = False
            for j, g in enumerate(groups):
                if not g:
                    continue
                nx = [sum(q[0] for q in g) / len(g), sum(q[1] for q in g) / len(g)]
                if _dist(nx, cent[j]) > 1e-6:
                    moved = True
                cent[j] = nx
            if not moved:
                break
        inertia = sum(min(_dist(p, c) ** 2 for c in cent) for p in pts)
        return inertia, cent, [len(g) for g in groups]

    best = min((_run(seed + r) for r in range(restarts)), key=lambda t: t[0])
    return best[1], best[2]


# Per-episode plan cache. RESTORED: my k-means++ replacement spliced from the
# start of _kmeans_centres to the start of this function and deleted this
# declaration, which sat between them -- every shard died with
# "NameError: name '_RL_CACHE' is not defined". When replacing a span of a
# file, check what else lives inside the span.
_RL_CACHE = {"key": None, "targets": None, "mode": None}


# ── `dispatch`: realistic sensing, multi-step actuator ───────────────────────
# Per-conversation demand ceiling. Pinned as a literal rather than read from
# max_flow_reward.TAU_CAP_BPS, which becomes 1e9 when the OM2G env is absent.
_TAU_CAP_BPS = float(os.environ.get("OM2G_TAU_CAP_BPS", 11614.814814814814))
_CLUSTER_LINK_M = float(os.environ.get("OM2G_CLUSTER_LINK_M", 500.0))


def _single_linkage(points, link_m):
    """Cluster points by single linkage. Not k-means: its result depends on the
    random init (one case gave a 55% bridge rate where the truth was 0%)."""
    n = len(points)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(n):
        for b in range(a + 1, n):
            if _dist(points[a], points[b]) < link_m:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    groups = {}
    for a in range(n):
        groups.setdefault(find(a), []).append(points[a])
    return list(groups.values())


# One drone's application-layer carrying capacity. A cluster worth more than
# this cannot be absorbed by a single drone, and that saturation is exactly the
# signal the allocation needs in order to send the surplus somewhere useful.
_RADIO_CAP_APP_BPS = float(os.environ.get("OM2G_RADIO_CAP_APP_BPS", 191406.0))
# Retarget only when a rival cluster is clearly better, so that ordinary churn
# in the cohort does not swap targets under a drone mid-flight.
_DISPATCH_SWITCH_RATIO = float(os.environ.get("OM2G_DISPATCH_SWITCH", 1.5))
_DISPATCH_BEACON_ONLY = os.environ.get("OM2G_DISPATCH_BEACON_ONLY") == "1"

_COMMITTED = {}   # drone -> ((x, y), value_at_commit)


def dispatch_velocities(obs, hop_limit=1):
    """Target-selection policy: pick a cluster, then actually go there."""
    drone_pos = obs.get("positions", [])
    noise_floor = obs.get("noise_floor", [])
    vels = []

    for i, dp in enumerate(drone_pos):
        nf = _nf_for(noise_floor, i)
        peers = _peer_neighbours(drone_pos, noise_floor, i, hop_limit)

        known = list(_heard_user_positions(obs, i))
        if not _DISPATCH_BEACON_ONLY:
            for j in peers:
                known.extend(_heard_user_positions(obs, j))
        # A user heard by two cohort members is ONE user. Without this the
        # clusters are weighted by how many drones saw them.
        seen, uniq = set(), []
        for p in known:
            k = (round(float(p[0]), 3), round(float(p[1]), 3))
            if k not in seen:
                seen.add(k)
                uniq.append(p)
        known = uniq
        if not known:
            vels.append([0.0, 0.0])
            continue

        cands = []
        for grp in _single_linkage(known, _CLUSTER_LINK_M):
            cx = sum(p[0] for p in grp) / len(grp)
            cy = sum(p[1] for p in grp) / len(grp)
            raw = 0.0
            for u in grp:
                mine_here = min(_link_capacity_bps(_dist((cx, cy), u), DRONE_TX_mW, nf),
                                _TAU_CAP_BPS)
                theirs = max((min(_link_capacity_bps(_dist(drone_pos[j], u),
                                                     DRONE_TX_mW, _nf_for(noise_floor, j)),
                                  _TAU_CAP_BPS) for j in peers), default=0.0)
                raw += max(0.0, mine_here - theirs)
            cands.append([(cx, cy), raw])

        # Sequential marginal-greedy over a saturating objective: iterate the
        # cohort in INDEX order -- a fixed priority every drone computes the
        # same way, so no negotiation and no fixed point -- and let each claim
        # consume one drone's worth of the cluster it takes. Cohort members hold
        # different `known` sets, so the allocations only approximately agree;
        # that residual mis-coordination is what a real protocol has, and it
        # belongs in the measurement rather than being engineered away.
        residual = [c[1] for c in cands]
        cohort = sorted(set(peers) | {i})
        mine_idx = 0
        for d in cohort:
            pick = max(range(len(cands)), key=lambda c: min(residual[c], _RADIO_CAP_APP_BPS))
            if d == i:
                mine_idx = pick
                break
            residual[pick] = max(0.0, residual[pick] - _RADIO_CAP_APP_BPS)

        tgt, tgt_val = cands[mine_idx][0], min(cands[mine_idx][1], _RADIO_CAP_APP_BPS)
        prev = _COMMITTED.get(i)
        if prev is not None and _dist(dp, prev[0]) > STEP_M:
            # Still in transit. Keep going unless the new pick is clearly better
            # -- otherwise a cohort change mid-flight turns the drone around and
            # it oscillates instead of arriving.
            if tgt_val <= _DISPATCH_SWITCH_RATIO * prev[1]:
                tgt, tgt_val = prev
        _COMMITTED[i] = (tgt, tgt_val)

        dx, dy = tgt[0] - dp[0], tgt[1] - dp[1]
        nrm = math.hypot(dx, dy)
        vels.append([0.0, 0.0] if nrm < 1e-6 else [dx / nrm, dy / nrm])
    return vels


def relay_explore_velocities(obs, mode=None):
    """Realistic-sensing relay that SEARCHES for clusters before bridging them.

    See the module note. Falls through to relay_ladder_velocities as soon as
    the pooled heard-set spans two or more clusters.
    """
    drone_pos = obs.get("positions", [])
    nd = len(drone_pos)
    if nd == 0:
        return []

    pooled, seen = [], set()
    for i in range(nd):
        for p in _heard_user_positions(obs, i):
            k = (round(float(p[0]), 3), round(float(p[1]), 3))
            if k not in seen:
                seen.add(k)
                pooled.append(list(p))

    groups = _single_linkage(pooled, _CLUSTER_LINK_M) if len(pooled) >= 1 else []
    if len(groups) >= 2:
        # Enough of the map is known: bridge it, using ONLY the pooled set.
        sub = dict(obs)
        sub["user_positions"] = pooled
        sub["user_rssi_valid"] = [[1.0] * len(pooled) for _ in range(nd)]
        return relay_ladder_velocities(sub, mode=mode)

    # SEARCH. Fewer than two clusters known, so there is nothing to bridge yet.
    # Each drone moves away from the centroid of what the swarm already knows,
    # fanning out by drone index so they do not all search the same direction.
    if pooled:
        cx = sum(p[0] for p in pooled) / len(pooled)
        cy = sum(p[1] for p in pooled) / len(pooled)
    else:
        cx = sum(p[0] for p in drone_pos) / nd
        cy = sum(p[1] for p in drone_pos) / nd
    vels = []
    for i, dp in enumerate(drone_pos):
        ang = 2.0 * math.pi * i / max(1, nd)
        dx = (dp[0] - cx) + 200.0 * math.cos(ang)
        dy = (dp[1] - cy) + 200.0 * math.sin(ang)
        nrm = math.hypot(dx, dy)
        vels.append([0.0, 0.0] if nrm < 1e-6 else [dx / nrm, dy / nrm])
    return vels


def relay_ladder_velocities(obs, mode=None):
    """NOTE: the plan is solved ONCE PER EPISODE and cached.

    k-means labels are only defined up to a permutation, so re-solving every
    step could flip cluster indices between consecutive steps and swap the
    head/relay TARGETS under the drones. Users are static, so one solve per
    episode is exact, stable and ~250x cheaper.
    """
    import itertools
    drone_pos = obs.get("positions", [])
    user_pos = obs.get("user_positions", [])
    nd = len(drone_pos)
    if nd == 0 or not user_pos:
        return [[0.0, 0.0] for _ in drone_pos]

    # $OM2G_RELAY_HONEST=1 -- the information-matched variant.
    #
    # By default this arm reads the GLOBAL user array to build its cluster map,
    # i.e. it knows where every cluster is (an oracle). The learned policy hears
    # roughly 4 of 40 (user->drone decode range 157.8 m against a 3800 m arena).
    #
    # This mode pools only what the swarm has ACTUALLY heard: the union of
    # every drone's last-heard set, which is the information a COMMUNICATING
    # swarm holds (each drone hears its own users, the message carries the
    # rest). That is the right comparator for the talking arm.
    if _RELAY_HONEST:
        seen, pooled = set(), []
        for _i in range(nd):
            for _p in _heard_user_positions(obs, _i):
                _k = (round(float(_p[0]), 3), round(float(_p[1]), 3))
                if _k not in seen:
                    seen.add(_k)
                    pooled.append(list(_p))
        if len(pooled) < 2:
            return [[0.0, 0.0] for _ in drone_pos]
        user_pos = pooled

    # One-shot diagnostic: the policy scores exactly hover in-simulator while
    # working offline, which means it is bailing out early on something in the
    # observation. Print what it actually received, once per process.
    if os.environ.get("OM2G_RELAY_DEBUG") == "1" and not _RL_CACHE.get("dbg"):
        _RL_CACHE["dbg"] = True
        print("[relay-dbg] nd=%d n_users=%d keys=%s sender_pairs=%d"
              % (nd, len(user_pos), sorted(obs.keys())[:12],
                 len(obs.get("sender_pairs", []))), flush=True)

    mode = (mode or os.environ.get("OM2G_RELAY_MODE", "blind")).strip().lower()
    if mode not in ("blind", "informed"):
        raise RuntimeError(f"OM2G_RELAY_MODE={mode!r} must be blind|informed")
    k = int(os.environ.get("OM2G_CLUSTERS", "4"))
    r_d = float(os.environ.get("OM2G_RANGE_DRONE", "523.06"))

    key = (mode, len(user_pos), round(user_pos[0][0], 3), round(user_pos[-1][1], 3))
    if _RL_CACHE["key"] == key:
        return _fly_to(drone_pos, _RL_CACHE["targets"])

    cent, _sizes = _kmeans_centres(user_pos, max(1, min(k, len(user_pos))))
    ncl = len(cent)

    # Cross-cluster demand per pair -- ONLY available to the informed arm.
    lab = [min(range(ncl), key=lambda c: _dist(cent[c], u)) for u in user_pos]
    dem = {}
    for a, b in obs.get("sender_pairs", []):
        if a < len(lab) and b < len(lab) and lab[a] != lab[b]:
            key = tuple(sorted((lab[a], lab[b])))
            dem[key] = dem.get(key, 0) + 1

    spare = max(0, nd - ncl)
    if os.environ.get("OM2G_RELAY_DEBUG") == "1" and _RL_CACHE.get("dbg") == True:
        _RL_CACHE["dbg"] = 2
        import itertools as _it
        _seps = sorted(_dist(cent[a], cent[b]) for a, b in _it.combinations(range(ncl), 2))
        print("[relay-dbg] ncl=%d spare=%d centre-seps=%s r_d=%.0f dem=%s"
              % (ncl, spare, [int(x) for x in _seps], r_d, dict(list(dem.items())[:6])), flush=True)
    feas = []
    for p in itertools.combinations(range(ncl), 2):
        gap = _dist(cent[p[0]], cent[p[1]])
        need = max(1, int(math.ceil(gap / (0.95 * r_d))) - 1)
        if need <= spare:
            feas.append((p, need))
    if not feas:
        targets = [cent[c % ncl] for c in range(nd)]
    else:
        if mode == "informed":
            pair, need = max(feas, key=lambda t: dem.get(t[0], 0))
        else:
            pair, need = min(feas, key=lambda t: _dist(cent[t[0][0]], cent[t[0][1]]))
        a, b = pair
        # `need` above is the MINIMUM hop count under hard-range physics
        # (r_d = 523 m, links right at the edge). Under the smooth Phi model
        # that is the wrong optimum: more, SHORTER hops deliver far more.
        # At a 1465 m gap, 2 relays gives 497 m links at p~0.56, while 3 relays
        # gives 366 m links at p~0.83 -- and both fit in 8 drones. Spending
        # every spare drone on the chosen chain strictly shortens each hop, so
        # it strictly raises per-hop delivery. Leaving spares parked was an
        # artifact of the old cliff model, and it made this arm a LOWER BOUND
        # that was then quoted as a ceiling.
        need = max(need, min(spare, need + 2))
        targets = [list(cent[c]) for c in range(ncl)]
        for j in range(1, need + 1):
            targets.append([cent[a][0] + (cent[b][0] - cent[a][0]) * j / (need + 1),
                            cent[a][1] + (cent[b][1] - cent[a][1]) * j / (need + 1)])
        while len(targets) < nd:
            targets.append(list(cent[a]))
        targets = targets[:nd]

    _RL_CACHE.update(key=key, targets=targets, mode=mode)
    return _fly_to(drone_pos, targets)


def _fly_to(drone_pos, targets):
    """Stable nearest-target assignment, then a unit step toward it."""
    nd = len(drone_pos)
    free = list(range(len(targets)))
    vels = [[0.0, 0.0] for _ in range(nd)]
    for i in sorted(range(nd), key=lambda i: min(
            (_dist(drone_pos[i], targets[t]) for t in free), default=0.0)):
        if not free:
            break
        t = min(free, key=lambda t: _dist(drone_pos[i], targets[t]))
        free.remove(t)
        dx, dy = targets[t][0] - drone_pos[i][0], targets[t][1] - drone_pos[i][1]
        nrm = math.hypot(dx, dy)
        vels[i] = [0.0, 0.0] if nrm < 1e-6 else [dx / nrm, dy / nrm]
    return vels
