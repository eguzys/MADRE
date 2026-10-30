"""T0 — the ZERO-GPU world-validity test for the archipelago world.

THE QUESTION THIS ANSWERS, before a single GPU hour is spent:
does the world's OPTIMUM actually need the information the message carries?
If the best achievable placement is just as good when chosen from peer
POSITIONS alone, then no policy can profit from the payload and the world is
open-loop, however large its baseline gap.

`payload_zero` leaves peer positions intact, and head
drones sit on clusters -- so once the heads settle, their positions REVEAL the
geometry and a relay can chain-follow without ever reading the payload.
Geometry entropy alone therefore does not pass the open-loop test, because
positions already carry the geometry. The quantity that positions cannot carry, even in principle, is
DEMAND: how many users each cluster holds and which pairs are talking. That is
observable only as heard-user state at the drones sitting on clusters.

So we score the same constructor twice per seed:
  informed    : bridge the pair with the largest cross-demand (needs the payload)
  demand-blind: bridge the CLOSEST pair (needs only peer positions)
and report VOI = (R_informed - R_blind) / R_informed.

Decision rule:
  VOI >= 0.15  -> the payload is load-bearing; proceed to T1/T2
  VOI <  0.15  -> open-loop world; do not train on it
Expected range: 0.25-0.35.
"""
import os
import sys
import itertools
import numpy as np

sys.path.insert(0, "omnet_module")
sys.path.insert(0, "om2g")

# The arm's 13 radio env vars MUST be exported before max_flow_reward is
# imported, or the whole test runs on the OMNET track's 802.11g physics
# (200 mW, -85 dBm, 20 MHz) instead of OM2G's 802.15.4 (6.59 mW, -100 dBm).
# VOI is a ratio so a pure unit error cancels -- but the PHYSICS does not:
# different TX power and sensitivity change which placements are good, and
# therefore change the ratio. Same export as calibrate_shaping.py.
import om2g_config as _C

for _k, _v in _C.om2g_env_vars(os.environ.get("OM2G_ARM", "compressed")).items():
    if _v is not None:
        os.environ.setdefault(_k, str(_v))

import max_flow_reward as M

W       = float(os.environ.get("OM2G_ARENA", "3800"))
SPREAD  = float(os.environ.get("OM2G_SPREAD", "25"))
NCL     = int(os.environ.get("OM2G_CLUSTERS", "4"))
ND      = int(os.environ.get("OM2G_DRONES", "8"))
NU      = int(os.environ.get("OM2G_NUM_USERS", "40"))
PCROSS  = float(os.environ.get("OM2G_PCROSS", "0.9"))
MINSEP  = float(os.environ.get("OM2G_MINSEP", "1465"))
MARGIN  = float(os.environ.get("OM2G_MARGIN", "300"))
R_D     = float(os.environ.get("OM2G_RANGE_DRONE", "523.06"))
EPS     = int(os.environ.get("EPS", "12"))

# OM2G_SPREAD is EFFECTIVE metres. config_knobs passes userSpread = eff*2000/L
# and RLBridge.cc then multiplies by L/2000, so the two cancel and the
# SIMULATOR's sigma is eff itself. Applying the scaling here as well made the
# offline clusters 1.9x wider than the simulator's at L=3800 -- a mirror that
# does not mirror. This is the same eff-vs-base inversion the project has been
# bitten by before.
SP = SPREAD


def scene(ep):
    """Mirror of the C++ archipelago branch: rejection-sampled min separation,
    Dirichlet sizes (the demand entropy), 2.5-sigma truncated users."""
    r = np.random.default_rng(7000 + ep)
    cent = []
    for _ in range(NCL):
        for _try in range(4000):
            c = np.array([r.uniform(MARGIN, W - MARGIN), r.uniform(MARGIN, W - MARGIN)])
            if all(np.linalg.norm(c - p) >= MINSEP for p in cent):
                cent.append(c); break
        else:
            cent.append(c)
    cent = np.array(cent)

    g = r.gamma(1.0, 1.0, NCL) + 0.15
    sizes = np.maximum(1, np.round(g / g.sum() * NU).astype(int))
    while sizes.sum() != NU:
        i = int(np.argmax(sizes)) if sizes.sum() > NU else int(np.argmin(sizes))
        sizes[i] += -1 if sizes.sum() > NU else 1

    U, lab = [], []
    for c in range(NCL):
        for _ in range(sizes[c]):
            while True:
                d = r.normal(0, SP, 2)
                if np.hypot(*d) <= 2.5 * SP:
                    break
            U.append(np.clip(cent[c] + d, 0, W)); lab.append(c)
    U = np.array(U); lab = np.array(lab)

    pairs = []
    for u in range(NU):
        cross = (r.random() < PCROSS and NCL > 1)
        pool = np.where(lab != lab[u])[0] if cross else np.where(lab == lab[u])[0]
        pool = [c for c in pool if c != u]
        pairs.append([u, int(r.choice(pool)) if len(pool) else u])
    return cent, U, lab, np.array(pairs), sizes


def cross_demand(pairs, lab):
    """Conversations per unordered cluster pair -- the DEMAND the payload carries."""
    d = {}
    for a, b in pairs:
        ca, cb = lab[a], lab[b]
        if ca != cb:
            d[tuple(sorted((ca, cb)))] = d.get(tuple(sorted((ca, cb))), 0) + 1
    return d


def bridgeable(cent, pair):
    """Can this pair actually be chained with the spare drone budget?

    The informed pick is argmax(demand x chain FEASIBILITY). With clusters up
    to ~3200 m apart and only ND-NCL spare drones, the highest-demand pair can
    need a chain too long to connect at all, which would carry zero flow, so
    feasibility is part of the choice.
    """
    gap = np.linalg.norm(cent[pair[0]] - cent[pair[1]])
    need = max(1, int(np.ceil(gap / (0.95 * R_D))) - 1)
    return need <= (ND - NCL), need


def place(cent, pair, spare_to=None):
    """Heads on every cluster, then relays evenly along the chosen bridge."""
    P = [cent[c].copy() for c in range(NCL)]
    a, b = pair
    gap = np.linalg.norm(cent[a] - cent[b])
    nrel = max(1, int(np.ceil(gap / (0.95 * R_D))) - 1)
    nrel = min(nrel, ND - NCL)
    for k in range(1, nrel + 1):
        P.append(cent[a] + (cent[b] - cent[a]) * k / (nrel + 1))
    while len(P) < ND:
        P.append(cent[spare_to if spare_to is not None else a].copy())
    return np.array(P[:ND])


def R(o, P):
    d = dict(o); d["positions"] = [list(map(float, q)) for q in P]
    # NOT /250. compute_position_oracle_reward returns a PER-STEP rate in
    # kbps already; only the baseline CSVs' `episode_reward` is a 250-step
    # total (a fully-served single cluster reads 465.0, the 464.6 kbps offered
    # load).
    return float(M.compute_position_oracle_reward(d)["reward"])


def hover_placement(cent, nd):
    """Spawn placement: drones round-robin onto cluster centres (spawn=cluster).
    Scoring this offline gives a direct calibration point against the
    simulator's measured hover -- if the mirror is faithful the two agree."""
    return np.array([cent[i % len(cent)] for i in range(nd)])


inf_s, blind_s, hov_s = [], [], []
for ep in range(EPS):
    cent, U, lab, pairs, sizes = scene(ep)
    o = dict(user_positions=U.tolist(), sender_pairs=pairs.tolist(),
             noise_floor=[M.NOISE_FLOOR_dBm] * ND)
    dem = cross_demand(pairs, lab)
    allp = list(itertools.combinations(range(NCL), 2))
    if not dem:
        continue
    # INFORMED: needs per-cluster demand -> only reachable through the payload.
    feas = [p for p in allp if bridgeable(cent, p)[0]]
    if not feas:
        continue                      # no pair is reachable with this budget
    best_inf = max(feas, key=lambda p: dem.get(p, 0))
    # DEMAND-BLIND: closest pair -> reachable from peer POSITIONS alone.
    best_bl = min(feas, key=lambda p: np.linalg.norm(cent[p[0]] - cent[p[1]]))
    ri, rb = R(o, place(cent, best_inf)), R(o, place(cent, best_bl))
    rh = R(o, hover_placement(cent, ND))
    inf_s.append(ri); blind_s.append(rb); hov_s.append(rh)
    print("ep%2d sizes=%s demand=%s informed=%s(%6.1f) blind=%s(%6.1f)"
          % (ep, list(sizes), {f"{k[0]}{k[1]}": v for k, v in sorted(dem.items())},
             best_inf, ri, best_bl, rb), flush=True)

i, b = float(np.mean(inf_s)), float(np.mean(blind_s))
voi = (i - b) / max(i, 1e-9)
same = sum(1 for x, y in zip(inf_s, blind_s) if abs(x - y) < 1e-9)
h = float(np.mean(hov_s))
print("\nT0 n=%d  informed=%.2f  demand-blind=%.2f  hover=%.2f  VOI=%.3f  (identical picks: %d/%d)"
      % (len(inf_s), i, b, h, voi, same, len(inf_s)))
# The reference is PER WORLD. The archipelago's own n=100 simulator ladder is
#   engineered 76.5 | greedy1hop 77.2 | hover 77.4 | random 81.5
# and this script's offline hover in that world is 76.45, i.e. 1.2%. Only the
# original archipelago has a measured reference; the 700/1000 m variants have
# none yet, so they print the value and say so.
_HOVER_REF = {1465: 77.4}
_ref = _HOVER_REF.get(int(MINSEP))
if _ref:
    print("CALIBRATION: simulator hover at MINSEP=%d is %.2f (n=100 ladder)."
          " Offline hover here = %.2f -> %.1f%% apart. Above a few %% the mirror"
          " is not faithful and no offline number transfers."
          % (int(MINSEP), _ref, h, 100.0 * abs(h - _ref) / _ref))
else:
    print("CALIBRATION: no simulator hover measured for this world"
          " (MINSEP=%s). Offline hover here = %.2f -- record it, do not compare"
          " it to another world's number."
          % (os.environ.get("OM2G_MINSEP", "?"), h))
print("DECISION GATE: VOI >= 0.15 -> payload load-bearing, proceed."
      "   VOI < 0.15 -> OPEN-LOOP WORLD, KILL IT.")
print("VERDICT:", "PASS - proceed to T1/T2" if voi >= 0.15 else "FAIL - do not train here")
