#!/usr/bin/env python3
"""WHICH geometric feature should the shaping actually pay for?

qual is a verified predictor of reward (qual_vs_reward.py: within-arm r=0.626,
t=7.44, n=88), and the p-kernel was built to raise it. But qual is not the only
candidate: hop_geometry.py showed relay_explore also differs on reach% (72.6 vs
33.9) and compactness. Those are DIFFERENT interventions and shaping can only
whisper one thing at a time, so the choice has to be measured, not argued.

This ranks candidate per-episode features by how well each predicts that
episode's own reward, WITHIN arm (arm-demeaned), and then reports the partial
correlation of each against reward controlling for qual -- i.e. what a feature
adds ONCE qual is already accounted for. A feature with a strong raw r but no
partial r is just a proxy for qual and shaping it buys nothing new.

Features, all per episode (mean over steps):
  qual     widest-path bottleneck p between the two best-connected clusters
  reach    fraction of drones in earshot of >=1 user
  nn       median nearest-neighbour drone spacing (compactness)
  span     max pairwise drone separation
  hopfrac  fraction of steps whose bottleneck hop is under 500 m (p>=0.54)
"""
import argparse, glob, json, math, os, sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
AMBIENT = -100.0


def configure(arm="compressed"):
    sys.path.insert(0, str(_HERE / "om2g"))
    sys.path.insert(0, str(_HERE / "omnet_module"))
    import om2g_config as C
    env = C.om2g_env_vars(arm)
    for k, v in env.items():
        os.environ[k] = str(v)
    os.environ["OM2G_PHYSICAL_METRIC"] = "1"
    return env


def components(adj):
    n = len(adj); comp = -np.ones(n, dtype=int); c = 0
    for s in range(n):
        if comp[s] >= 0: continue
        stack, comp[s] = [s], c
        while stack:
            u = stack.pop()
            for v in np.nonzero(adj[u] & (comp < 0))[0]:
                comp[v] = c; stack.append(v)
        c += 1
    return comp


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    return r, r * math.sqrt((n - 2) / max(1 - r * r, 1e-12))


def demean(v, groups):
    v = np.array(v, float)
    for g in set(groups):
        m = np.array([x == g for x in groups])
        if m.sum() >= 2:
            v[m] -= v[m].mean()
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-glob", action="append", required=True)
    ap.add_argument("--link", type=float, default=500.0)
    a = ap.parse_args()
    env = configure()
    import max_flow_reward as M
    tx_d = float(env.get("OMNET_DRONE_TX_mW", 200.0))
    tx_u = float(env.get("OMNET_USER_TX_mW", tx_d))

    def hears(d):
        return M._link_capacity_bps(float(d), tx_u, AMBIENT) > 0.0

    F = {k: [] for k in ("qual", "reach", "nn", "span", "hopfrac")}
    rew, arms = [], []
    for spec in a.arm_glob:
        name, pat = spec.split("=", 1)
        for f in sorted(glob.glob(pat)):
            try:
                recs = [r for r in json.load(open(f)) if r.get("kind") != "obs"]
            except Exception:
                continue
            eps = {}
            for r in recs:
                e = eps.setdefault(r["seed"], {"u": None, "s": []})
                if r.get("users") is not None: e["u"] = r["users"]
                e["s"].append(r)
            for seed, e in sorted(eps.items()):
                if e["u"] is None or not e["s"]: continue
                U = np.asarray(e["u"], float)
                du = np.linalg.norm(U[:, None, :] - U[None, :, :], axis=-1)
                lab = components(du <= a.link); k = int(lab.max()) + 1
                q, rc, nnv, spv, hp = [], [], [], [], []
                for r in e["s"]:
                    pos = np.asarray(r["pos"], float)
                    n = len(pos)
                    dd = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
                    P = np.zeros((n, n))
                    for i in range(n):
                        for j in range(i + 1, n):
                            cap = M._link_capacity_bps(float(dd[i, j]), tx_d, AMBIENT)
                            P[i, j] = P[j, i] = 0.0 if cap <= 0 else M._LAST_P[0]
                    dU = np.linalg.norm(pos[:, None, :] - U[None, :, :], axis=-1)
                    heard = np.vectorize(hears)(dU)
                    acc = [np.nonzero(heard[:, lab == c].any(1))[0] for c in range(k)]
                    W = P.copy(); np.fill_diagonal(W, 1.0)
                    for kk in range(n):
                        W = np.maximum(W, np.minimum(W[:, kk][:, None], W[kk, :][None, :]))
                    best = 0.0
                    for x in range(k):
                        for y in range(x + 1, k):
                            if len(acc[x]) and len(acc[y]):
                                best = max(best, float(W[np.ix_(acc[x], acc[y])].max()))
                    q.append(best)
                    cand = np.abs(P - best) < 1e-9
                    hp.append(1.0 if (best > 0 and cand.any()
                                      and float(dd[cand].max()) < 500.0) else 0.0)
                    off = dd + np.eye(n) * 1e9
                    nnv.append(float(np.median(off.min(1))))
                    spv.append(float(dd.max()))
                    rc.append(float(heard.any(1).mean()))
                rr = [x["reward"] for x in e["s"] if x.get("reward") is not None]
                if not rr: continue
                F["qual"].append(np.mean(q)); F["reach"].append(np.mean(rc))
                F["nn"].append(np.mean(nnv)); F["span"].append(np.mean(spv))
                F["hopfrac"].append(np.mean(hp))
                rew.append(float(np.mean(rr))); arms.append(name)

    if len(rew) < 5:
        print("not enough episodes"); return
    y = demean(rew, arms)
    dq = demean(F["qual"], arms)
    print(f"n={len(rew)} episodes across {len(set(arms))} arms   (all arm-demeaned)")
    print(f"{'feature':<10}{'r':>8}{'t':>8}{'partial r|qual':>17}{'t':>8}")
    for k in ("qual", "hopfrac", "reach", "nn", "span"):
        x = demean(F[k], arms)
        r, t = pearson(x, y)
        if k == "qual":
            print(f"{k:<10}{r:>8.3f}{t:>8.2f}{'—':>17}{'—':>8}")
            continue
        # residualise both x and y on qual, then correlate
        b1 = float(np.dot(dq, x) / max(np.dot(dq, dq), 1e-12))
        b2 = float(np.dot(dq, y) / max(np.dot(dq, dq), 1e-12))
        pr, pt = pearson(x - b1 * dq, y - b2 * dq)
        print(f"{k:<10}{r:>8.3f}{t:>8.2f}{pr:>17.3f}{pt:>8.2f}")


if __name__ == "__main__":
    sys.exit(main())
