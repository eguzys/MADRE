"""Stage 0(b): CALIBRATE the shaping weights against the reward's real scale.

The weights are measured, not assumed: each term's magnitude is calibrated
against the measured reward scale. Target:

    sigma(F_fiedler)  = 0.10 * sigma(D_i)
    sigma(F_voronoi)  = 0.05 * sigma(D_i)
    sigma(F_edge)     = 0.02 * sigma(D_i)
    sigma(F_spread)   = 0          (off; redundant with voronoi)

where D_i is the per-agent difference reward the policy actually receives and
F is the potential-based shaping increment gamma*Phi(s') - Phi(s).

Shaping should WHISPER. It only has to tilt the valley the drones must cross;
it provably cannot change the optimum (Ng et al. 1999), but a term that shouts
drowns the real gradient and the run learns the potential instead of the task.

Method: roll out hover and random policies in the archipelago (both visit the
relevant state distribution -- random in particular produces the excursions
that shaping exists to reward), measure sigma of each raw potential increment
at unit weight, and solve for the weight that hits the target ratio.
"""
import os
import sys
import numpy as np

sys.path.insert(0, "omnet_module")
sys.path.insert(0, "om2g")

# ── The 13 radio env vars MUST be exported BEFORE max_flow_reward is imported ─
# It reads them at module import time, and without them it falls back to the
# OMNET track's 802.11g physics (20 MHz, 200 mW, -85 dBm sensitivity) and
# REWARD_UNIT_BPS=1e6 instead of OM2G's 1e3 (sigma(D_i) = 107.54 unset vs 3.78
# with the arm exported). If the caller has already exported them
# (om2g_eval_arton.sh does, via `eval "$(python om2g_config.py --arm ... --shell)"`)
# this is a no-op -- setdefault never overrides. The full set is exported so the
# offline replay matches the simulated runs exactly.
import om2g_config as _C

for _k, _v in _C.om2g_env_vars(os.environ.get("OM2G_ARM", "compressed")).items():
    if _v is not None:
        os.environ.setdefault(_k, str(_v))

import max_flow_reward as M

W = float(os.environ.get("OM2G_ARENA", "3800"))
NCL = int(os.environ.get("OM2G_CLUSTERS", "4"))
ND = int(os.environ.get("OM2G_DRONES", "8"))
NU = int(os.environ.get("OM2G_NUM_USERS", "40"))
PCROSS = float(os.environ.get("OM2G_PCROSS", "0.9"))
MINSEP = float(os.environ.get("OM2G_MINSEP", "1465"))
MAXNN = float(os.environ.get("OM2G_MAXNN", "1750"))
MARGIN = float(os.environ.get("OM2G_MARGIN", "300"))
SPREAD = float(os.environ.get("OM2G_SPREAD", "25"))
STEP_M = float(os.environ.get("OMNET_STEP_M", "60"))
EPS = int(os.environ.get("EPS", "20"))
STEPS = int(os.environ.get("STEPS", "60"))

os.environ.setdefault("OM2G_SHAPE_VORONOI", "1")
os.environ.setdefault("OM2G_SHAPE_FIEDLER", "1")
os.environ.setdefault("OM2G_SHAPE_EDGE", "1")
os.environ.setdefault("OM2G_SHAPE_SPREAD", "1")
import shaping as SH   # noqa: E402  (weights must be set before import)


def scene(ep):
    """Archipelago layout WITH the separation band, mirroring RLBridge."""
    r = np.random.default_rng(9000 + ep)
    for _attempt in range(600):
        cent = []
        for _ in range(NCL):
            for _t in range(4000):
                c = np.array([r.uniform(MARGIN, W - MARGIN),
                              r.uniform(MARGIN, W - MARGIN)])
                if all(np.linalg.norm(c - p) >= MINSEP for p in cent):
                    cent.append(c)
                    break
        cent = np.array(cent)
        nn = [min(np.linalg.norm(cent[i] - cent[j]) for j in range(NCL) if j != i)
              for i in range(NCL)]
        if max(nn) <= MAXNN:
            break
    g = r.gamma(1.0, 1.0, NCL) + 0.15
    sizes = np.maximum(1, np.round(g / g.sum() * NU).astype(int))
    while sizes.sum() != NU:
        i = int(np.argmax(sizes)) if sizes.sum() > NU else int(np.argmin(sizes))
        sizes[i] += -1 if sizes.sum() > NU else 1
    U, lab = [], []
    for c in range(NCL):
        for _ in range(sizes[c]):
            while True:
                d = r.normal(0, SPREAD, 2)
                if np.hypot(*d) <= 2.5 * SPREAD:
                    break
            U.append(np.clip(cent[c] + d, 0, W)); lab.append(c)
    U = np.array(U); lab = np.array(lab)
    pairs = []
    for u in range(NU):
        cross = (r.random() < PCROSS and NCL > 1)
        pool = np.where(lab != lab[u])[0] if cross else np.where(lab == lab[u])[0]
        pool = [c for c in pool if c != u]
        pairs.append([u, int(r.choice(pool)) if len(pool) else u])
    return cent, U, np.array(pairs), r


def d_i(o, P):
    """Per-agent difference reward: R(all) - R(all \\ i), the real signal."""
    d = dict(o); d["positions"] = [list(map(float, q)) for q in P]
    full = float(M.compute_position_oracle_reward(d)["reward"])
    out = []
    for i in range(len(P)):
        Q = [list(map(float, q)) for j, q in enumerate(P) if j != i]
        d2 = dict(o); d2["positions"] = Q
        out.append(full - float(M.compute_position_oracle_reward(d2)["reward"]))
    return out


def term_sigma(name):
    """sigma of one shaping term's per-step increment at UNIT weight."""
    for k in ("VORONOI", "FIEDLER", "EDGE", "SPREAD"):
        os.environ["OM2G_SHAPE_" + k] = "1" if k == name else "0"
    import importlib
    importlib.reload(SH)
    inc = []
    for ep in range(EPS):
        cent, U, pairs, r = scene(ep)
        o = dict(user_positions=U.tolist(), sender_pairs=pairs.tolist(),
                 noise_floor=[M.NOISE_FLOOR_dBm] * ND)
        P = np.array([cent[i % NCL] for i in range(ND)], float)
        sh = SH.Shaper(ND)
        o["positions"] = P.tolist(); sh.reset(o)
        for _ in range(STEPS):
            P = np.clip(P + r.normal(0, 1, (ND, 2)) * STEP_M, 0, W)  # random walk
            o["positions"] = P.tolist()
            base = {f"drone_{i}": 0.0 for i in range(ND)}
            f = sh.shape(base, o)
            inc.extend(f[f"drone_{i}"] for i in range(ND))
    return float(np.std(inc))


# --- sigma(D_i) on the same state distribution -------------------------------
ds = []
for ep in range(min(EPS, 8)):
    cent, U, pairs, r = scene(ep)
    o = dict(user_positions=U.tolist(), sender_pairs=pairs.tolist(),
             noise_floor=[M.NOISE_FLOOR_dBm] * ND)
    P = np.array([cent[i % NCL] for i in range(ND)], float)
    for t in range(0, STEPS, 12):
        ds.extend(d_i(o, P))
        P = np.clip(P + r.normal(0, 1, (ND, 2)) * STEP_M, 0, W)
sd = float(np.std(ds))
print("sigma(D_i) = %.2f   (mean %.2f, max %.2f)" % (sd, np.mean(ds), np.max(ds)))
print()

TARGET = {"FIEDLER": 0.10, "VORONOI": 0.05, "EDGE": 0.02, "SPREAD": 0.0}
print("%-9s %12s %12s %12s" % ("term", "sigma@w=1", "target sigma", "WEIGHT"))
for name, frac in TARGET.items():
    if frac == 0.0:
        print("%-9s %12s %12s %12s" % (name, "-", "0 (off)", "0"))
        continue
    s1 = term_sigma(name)
    want = frac * sd
    w = want / s1 if s1 > 0 else 0.0
    print("%-9s %12.4f %12.2f %12.4g" % (name, s1, want, w))
print()
print("Set these as OM2G_SHAPE_<TERM>. Re-run this whenever the world changes:")
print("the potentials scale with arena, radio range and drone count.")
