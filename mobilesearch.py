"""Measure the VALUE OF INFORMATION as a function of user mobility and sensing.

WHY
---
`sampledUserPositions_` is assigned once per episode in RLBridge.cc:432 and
never updated: the ground users are STATIC. That makes the task a static
optimisation problem, which favours a myopic hill-climb:

  * the optimal drone placement is a FIXED POINT, so a one-step hill-climb has
    250 steps to walk to it and no moving target to chase;
  * every drone can re-derive what it needs from its own local observations,
    so a message from a peer carries information the receiver could also
    collect by exploring.

This sweep measures how much user mobility and limited sensing change that.

WHAT THIS MEASURES
------------------
The quantity that has to be non-zero before ANY communication method can help:

    VOI  =  oracle_greedy  -  greedy_1hop

Both are the SAME hill-climb algorithm. They differ only in what they can see:
`greedy_1hop` reasons over the users it has heard (K nearest), `oracle_greedy`
over all NU users. So their difference is purely the value of the user state a
drone does NOT have -- exactly the payload learned communication transmits.
If VOI is ~0, no message can help.

EXPECTATIONS
---------------------------------------------
 P1  VOI grows as sensing K shrinks (fewer users heard => more to tell).
 P2  VOI grows with user speed: a static world lets an unheard user be
     discovered by wandering, a moving one must be re-reported continuously.
 P3  greedy/hover shrinks with user speed -- the hill-climb chases a stale
     optimum. NOTE: STEP_M is 20 m/step against a 1 s step, so a pedestrian at
     1.4 m/s is 14x slower than the drones and should cost greedy nothing.
     If P3 only appears at speeds above ~10 m/s, the relevant setting is
     vehicular users (responder convoys), or the lever is sensing, not motion.
     Reported either way.

MOBILITY MODEL
--------------
Reference Point Group Mobility: each cluster centroid follows a Gauss-Markov
walk (correlation ALPHA, so headings persist rather than jittering white),
and every user keeps its offset from its own centroid plus a small independent
diffusion. Cluster structure therefore survives, which matters because the
task's geometry is built on it. USPEED=0 must exactly reproduce the static
world -- that is the control, and it is checked first.
"""
import os, sys
import numpy as np
sys.path.insert(0, "omnet_module"); sys.path.insert(0, "om2g")
import max_flow_reward as M, signal_baselines as SB

W       = float(os.environ.get("WW", "1200"))
SPEFF   = float(os.environ.get("SPEFF", "550"))
SPB     = SPEFF * 2000.0 / W
NCL     = int(os.environ.get("NCL", "2"))
PCROSS  = float(os.environ.get("PCROSS", "0.5"))
SPAWN   = os.environ.get("SPAWN", "cluster")
ND      = int(os.environ.get("ND", "5"))
NU      = int(os.environ.get("NU", "40"))
EPS     = int(os.environ.get("EPS", "6"))
STEPS   = int(os.environ.get("STEPS", "250"))
K       = int(os.environ.get("KSENSE", "13"))       # users a drone can hear
USPEED  = float(os.environ.get("USPEED", "0"))      # metres per decision step
ALPHA   = float(os.environ.get("UALPHA", "0.85"))   # Gauss-Markov correlation
UJITTER = float(os.environ.get("UJITTER", "0.15"))  # individual diffusion, x speed


def scene(ep):
    r = np.random.default_rng(4200 + ep)
    sp = SPB * (W / 2000.0)
    centres = np.array([[r.uniform(0, W), r.uniform(0, W)] for _ in range(NCL)])
    g = r.gamma(10.0, 1.0, NCL); sizes = np.maximum(1, np.round(g / g.sum() * NU).astype(int))
    while sizes.sum() != NU:
        i = int(np.argmax(sizes)) if sizes.sum() > NU else int(np.argmin(sizes))
        sizes[i] += -1 if sizes.sum() > NU else 1
    U = []; lab = []
    for c, (ctr, n) in enumerate(zip(centres, sizes)):
        U.append(ctr + r.normal(0, sp, (n, 2))); lab += [c] * n
    U = np.clip(np.vstack(U), 0, W); lab = np.array(lab)
    pairs = []
    for u in range(NU):
        cross = (r.random() < PCROSS and NCL > 1)
        pool = np.where(lab != lab[u])[0] if cross else np.where(lab == lab[u])[0]
        pool = [c for c in pool if c != u]
        pairs.append([u, int(r.choice(pool)) if len(pool) else u])
    if SPAWN == "corner":
        P = np.tile(np.array([60., 60.]), (ND, 1)) + r.normal(0, 20, (ND, 2))
    elif SPAWN == "staging":
        big = centres[int(np.argmax(sizes))]; P = np.tile(big, (ND, 1)) + r.normal(0, 40, (ND, 2))
    else:
        P = centres[np.arange(ND) % NCL] + r.normal(0, 40, (ND, 2))
    o = dict(positions=np.clip(P, 0, W).tolist(), user_positions=U.tolist(),
             sender_pairs=pairs, noise_floor=[M.NOISE_FLOOR_dBm] * ND)
    return o, U, lab, r


class Crowd:
    """Reference-Point Group Mobility over the cluster centroids."""

    def __init__(self, U, lab, rng):
        self.U = U.copy()
        self.lab = lab
        self.rng = rng
        self.vel = np.zeros((int(lab.max()) + 1, 2))     # per-cluster heading

    def step(self):
        if USPEED <= 0:
            return self.U
        # Gauss-Markov: keep a fraction ALPHA of the previous heading so groups
        # travel rather than vibrate in place (white noise averages to zero and
        # would leave the centroid where it started).
        drive = self.rng.normal(0, 1.0, self.vel.shape)
        self.vel = ALPHA * self.vel + np.sqrt(1 - ALPHA ** 2) * drive
        n = np.linalg.norm(self.vel, axis=1, keepdims=True)
        heading = self.vel / np.maximum(n, 1e-9)
        self.U = self.U + heading[self.lab] * USPEED
        self.U = self.U + self.rng.normal(0, UJITTER * USPEED, self.U.shape)
        # Reflect at the arena wall so the crowd stays in the scenario.
        for ax in (0, 1):
            lo = self.U[:, ax] < 0; hi = self.U[:, ax] > W
            self.U[lo, ax] = -self.U[lo, ax]
            self.U[hi, ax] = 2 * W - self.U[hi, ax]
            self.vel[:, ax] *= -1 if (lo.any() or hi.any()) else 1
        self.U = np.clip(self.U, 0, W)
        return self.U


MASK_MODE = os.environ.get("MASK_MODE", "knn")     # "knn" | "ttl"
R_OBS = float(os.environ.get("R_OBS", "157.8"))    # hard sensitivity radius, m
TTL = int(os.environ.get("TTL", "250"))            # steps a sighting stays valid


def hmask(U, P):
    """user_rssi_valid: each drone hears its K nearest users."""
    out = []
    for d in P:
        order = np.argsort(np.linalg.norm(U - np.asarray(d), axis=1))[:K]
        s = set(int(x) for x in order)
        out.append([u in s for u in range(len(U))])
    return out


class TTLMask:
    """Physical observability: in range NOW, or heard within the last TTL steps.

    WHY THIS REPLACES THE K-NEAREST MASK
    ------------------------------------
    `userRssiValid_[i][u]` in RLBridge.cc is set at exactly one site
    (RLBridge.cc:2167-2169) and **cleared nowhere**. It is monotone-increasing
    over the episode: once drone i has heard user u it retains exact access to
    u's CURRENT ground-truth position for the rest of the episode, even if u is
    a kilometre away and silent. `userRssiTime_` is already maintained and used
    for nothing.

    Two consequences, both of which this class exists to test:

    1. The greedy baseline already has perfect free memory. That is why the
       memoryful-greedy variant returns a bit-identical score -- there is
       nothing to add. A K-nearest probe models sensing as a per-step limit,
       which the simulator does not implement.

    2. A TTL on validity models RSSI-derived user state that ages out, as is
       physical and already true of the traffic model, where users only
       transmit while in a call.

    A TTL shrinks OBSERVABILITY with provably zero effect on SERVICE capacity --
    unlike user TX power, where r_observe/r_service is pinned at 1.74 for every
    power level, so lowering it shrinks both together and can never create an
    information gap.
    """

    def __init__(self, n_drones, n_users):
        self.last = np.full((n_drones, n_users), -10 ** 9, dtype=np.int64)
        self.t = 0

    def __call__(self, U, P):
        d = np.linalg.norm(np.asarray(P)[:, None, :] - U[None, :, :], axis=2)
        self.last[d <= R_OBS] = self.t
        vis = (self.t - self.last) <= TTL
        self.t += 1
        return vis.tolist()


def rollout(o0, U0, lab, seed, fn, moving=True, all_seeing=False, memory=False):
    """Run one policy for STEPS steps; return mean per-step reward.

    `memory=True` is the FALSIFIER for the whole sensing-range result. A
    memoryless hill-climb reasons only about users it hears RIGHT NOW, so
    shrinking K necessarily hurts it -- but in a static world a real drone
    could simply remember every user it has ever heard and, after 250 steps of
    wandering, reconstruct most of the map for free. If a remembering greedy
    closes the VOI gap, then blinding creates only a TRANSIENT information
    asymmetry and communication adds nothing asymptotically. (Foerster 2016 /
    IC3Net's "vision=0" works because their hidden state is not re-derivable by
    exploration; here it might be.)

    Memory here is an upper bound for the baseline: perfect recall of the last
    known position of every user ever heard, with no decay and no cost.
    """
    rng = np.random.default_rng(seed)
    crowd = Crowd(U0, lab, rng)
    P = np.asarray(o0["positions"], float).copy()
    d = dict(o0)
    tot = 0.0
    seen = [dict() for _ in range(len(P))] if memory else None   # per drone: u -> xy
    ttl = TTLMask(len(P), len(U0)) if MASK_MODE == "ttl" else None
    for _ in range(STEPS):
        U = crowd.step() if moving else crowd.U
        d["user_positions"] = U.tolist()
        d["positions"] = P.tolist()
        # An all-seeing drone hears every user; otherwise the configured mask.
        if all_seeing:
            mask = [[True] * len(U) for _ in P]
        elif ttl is not None:
            mask = ttl(U, P)
        else:
            mask = hmask(U, P)
        d["user_rssi_valid"] = mask
        # Reward is always scored on the TRUE current positions.
        tot += float(M.compute_position_oracle_reward(d)["reward"])
        if fn is not None:
            if memory:
                # Fold this step's sightings into each drone's recalled map,
                # then let it act on recalled positions for everything it
                # remembers but cannot currently hear.
                Umem = [list(U[u]) for u in range(len(U))]
                for i in range(len(P)):
                    for u in range(len(U)):
                        if mask[i][u]:
                            seen[i][u] = list(U[u])
                dm = dict(d)
                # One hill-climb per drone, each on its own recalled map.
                vel = []
                for i in range(len(P)):
                    recalled = sorted(seen[i].keys())
                    if not recalled:
                        vel.append([0.0, 0.0]); continue
                    sub = dict(dm)
                    sub["user_positions"] = [seen[i][u] for u in recalled]
                    sub["user_rssi_valid"] = [[True] * len(recalled)
                                              for _ in range(len(P))]
                    vel.append(list(np.asarray(fn(sub), float)[i]))
                V = np.asarray(vel, float)
            else:
                V = np.asarray(fn(d), float)
            P = np.clip(P + V * SB.STEP_M, 0, W)
    return tot / STEPS


HOV = []; GRD = []; ORC = []; MEM = []; AGG = []
for ep in range(EPS):
    o, U, lab, r = scene(ep)
    sd = 90000 + ep
    HOV.append(rollout(o, U, lab, sd, None))
    GRD.append(rollout(o, U, lab, sd, SB.greedy_realistic_velocities))
    ORC.append(rollout(o, U, lab, sd, SB.greedy_realistic_velocities, all_seeing=True))
    MEM.append(rollout(o, U, lab, sd, SB.greedy_realistic_velocities, memory=True))
    # `aggregate` is the information-matched comparator. It compresses the WHOLE user set to centroid + covariance,
    # re-expanded as 9 weighted stand-in users (signal_baselines.py
    # aggregate_velocities). That is a single Gaussian: near-exact for one
    # cluster, and increasingly inaccurate as the crowd becomes multi-modal, because
    # a Gaussian fitted to k separated clusters puts its mass in the EMPTY
    # space between them. It is all-seeing, so this is a pure test of the
    # CODE, not of observability.
    AGG.append(rollout(o, U, lab, sd, SB.aggregate_velocities, all_seeing=True))

h, g, c, m, a = (float(np.mean(HOV)), float(np.mean(GRD)),
                 float(np.mean(ORC)), float(np.mean(MEM)), float(np.mean(AGG)))
# ROOM = headroom between the moment code and the same hill-climb given the
# true user set. This is the space a LEARNED 8-number code has to win in at
# equal bandwidth. If ROOM is ~0 no code can improve on the aggregate in this
# world, whatever the architecture does.
print("AGGR NCL=%d SPEFF=%d ND=%d NU=%d | hover=%7.1f aggregate=%7.1f greedy=%7.1f "
      "oracle=%7.1f | ROOM=%7.1f room/agg=%5.2f agg/hov=%5.2f"
      % (NCL, SPEFF, ND, NU, h, a, g, c, c - a,
         (c - a) / max(a, 1e-9), a / max(h, 1e-9)), flush=True)
# voi_live  = what a REMEMBERING greedy still cannot get on its own. This, not
#             VOI, is the headroom for communication that remains after
#             exploring and accumulating.
print("MOBI mask=%-3s K=%2d ttl=%3d W=%d NCL=%d ND=%d pX=%.2f spawn=%-7s uspeed=%5.1f | "
      "hover=%7.1f greedy=%7.1f mem=%7.1f oracle=%7.1f | VOI=%7.1f voi/grd=%5.2f "
      "VOI_LIVE=%7.1f voilive/mem=%5.2f grd/hov=%5.2f"
      % (MASK_MODE, K, TTL, W, NCL, ND, PCROSS, SPAWN, USPEED, h, g, m, c,
         c - g, (c - g) / max(g, 1e-9),
         c - m, (c - m) / max(m, 1e-9), g / max(h, 1e-9)), flush=True)
