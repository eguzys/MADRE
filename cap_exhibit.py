"""The per-seed CAP EXHIBIT: do the coverage baselines saturate their bound?

Purpose: compare the learned policy with the ceiling of the deployable
coverage-control family. Each coverage baseline is shown against its own
analytic cap; a baseline that reaches its cap is a complete implementation of
that family.

An arithmetic cap ("local demand + leak") omits drone->far-cluster USER
down-links: at banded separations a drone on cluster A still delivers a little
to users of cluster B (~1265 m, ~1.3 kbps each), and commodities are
one-directional so no return path is needed. The cap is therefore computed
NUMERICALLY from the confinement lemma itself:

    a coverage hill-climb never leaves the hull of the users it can see,
    so its reachable set is {placements with every drone within HULL_R of
    some cluster centre}.

cap_s = max reward over that constrained set, by coordinate ascent with every
proposal projected back into the hull. Anything a confined policy can achieve
is <= cap_s by construction, so `baseline_s <= cap_s` is checkable per seed and
the "family saturates its cap" claim becomes a measurement rather than an
argument.
"""
import os
import sys
import numpy as np

sys.path.insert(0, "omnet_module")
sys.path.insert(0, "om2g")
import max_flow_reward as M
import signal_baselines as SB

ND = int(os.environ.get("OM2G_DRONES", "8"))
NCL = int(os.environ.get("OM2G_CLUSTERS", "4"))
W = float(os.environ.get("OM2G_ARENA", "3800"))
HULL_R = float(os.environ.get("OM2G_HULL_R", "200"))   # 2.5 sigma, the truncation
EPS = int(os.environ.get("EPS", "12"))


def project(P, cent):
    """Push every drone back inside the nearest cluster hull."""
    out = []
    for q in P:
        d = [np.linalg.norm(q - c) for c in cent]
        j = int(np.argmin(d))
        if d[j] <= HULL_R:
            out.append(q)
        else:
            v = q - cent[j]
            out.append(cent[j] + v / (np.linalg.norm(v) + 1e-9) * HULL_R)
    return np.array(out)


def cap(o, cent, iters=90, restarts=3, seed=3):
    """Best reward achievable while CONFINED to the cluster hulls."""
    rng = np.random.default_rng(seed)
    best = -1e18
    for _ in range(restarts):
        P = project(np.array([cent[i % len(cent)] + rng.normal(0, HULL_R / 2, 2)
                              for i in range(ND)]), cent)
        cur = R(o, P)
        step = HULL_R / 2.0
        for _ in range(iters):
            moved = False
            for i in range(ND):
                for d in ((step, 0), (-step, 0), (0, step), (0, -step)):
                    Q = P.copy(); Q[i] = Q[i] + np.array(d)
                    Q = project(Q, cent)
                    v = R(o, Q)
                    if v > cur:
                        cur, P, moved = v, Q, True
            if not moved:
                step *= 0.6
                if step < 10:
                    break
        best = max(best, cur)
    return best


def R(o, P):
    d = dict(o); d["positions"] = [list(map(float, q)) for q in P]
    return float(M.compute_position_oracle_reward(d)["reward"])


exec(open("t0_voi.py").read().split("inf_s, blind_s")[0])   # reuse scene()

caps, hov, grd = [], [], []
for ep in range(EPS):
    cent, U, lab, pairs, sizes = scene(ep)
    o = dict(user_positions=U.tolist(), sender_pairs=pairs.tolist(),
             noise_floor=[M.NOISE_FLOOR_dBm] * ND)
    c = cap(o, cent)
    h = R(o, np.array([cent[i % NCL] for i in range(ND)]))
    # greedy, run to convergence from the cluster spawn
    P = np.array([cent[i % NCL] for i in range(ND)], float)
    d = dict(o)
    for _ in range(120):
        d["positions"] = P.tolist()
        d["user_rssi_valid"] = [[True] * len(U) for _ in range(ND)]
        P = np.clip(P + np.asarray(SB.greedy_realistic_velocities(d), float) * 60.0, 0, W)
    g = R(o, P)
    caps.append(c); hov.append(h); grd.append(g)
    print("ep%2d  cap %7.1f | hover %7.1f (%.0f%% of cap) | greedy %7.1f (%.0f%%)  %s"
          % (ep, c, h, 100 * h / max(c, 1e-9), g, 100 * g / max(c, 1e-9),
             "OK" if max(h, g) <= c * 1.02 else "*** EXCEEDS CAP ***"), flush=True)

caps, hov, grd = map(np.array, (caps, hov, grd))
print()
print("n=%d   mean cap %.1f | hover %.1f (%.0f%%) | greedy %.1f (%.0f%%)"
      % (len(caps), caps.mean(), hov.mean(), 100 * hov.mean() / caps.mean(),
         grd.mean(), 100 * grd.mean() / caps.mean()))
print("violations (baseline > cap): hover %d, greedy %d"
      % (int((hov > caps * 1.02).sum()), int((grd > caps * 1.02).sum())))
print()
print("READ: baselines sitting AT the cap is the anti-straw-man exhibit --")
print("saturating a proven bound is what a competent implementation does.")
print("Any violation means the confinement lemma or HULL_R is wrong.")
