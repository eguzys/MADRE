#!/usr/bin/env python3
"""Does bridge QUALITY actually predict SCORE?

The p-kernel shaping assumes that raising the widest-path bottleneck decode
probability raises the reward. That is plausible (the LP charges ARQ 1/p at the
transmitter); this script measures it.

So: per EPISODE, compute the mean bottleneck quality `qual` exactly as
bridge_quality.py does, pair it with that episode's own mean reward from the
trace, and report the correlation -- within each arm (so it is not merely
recovering "good arms score well") and pooled.

If qual does not predict reward WITHIN an arm, the p kernel is optimising a
number the reward does not follow.
"""
import argparse, json, math, os, sys
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


def widest(P):
    W = P.copy(); np.fill_diagonal(W, 1.0)
    for k in range(len(W)):
        W = np.maximum(W, np.minimum(W[:, k][:, None], W[k, :][None, :]))
    return W


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    t = r * math.sqrt((n - 2) / max(1 - r * r, 1e-12))
    return r, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-glob", action="append", required=True,
                    help="NAME=glob, repeatable")
    ap.add_argument("--link", type=float, default=500.0)
    a = ap.parse_args()

    env = configure()
    import max_flow_reward as M
    tx_d = float(env.get("OMNET_DRONE_TX_mW", 200.0))
    tx_u = float(env.get("OMNET_USER_TX_mW", tx_d))

    def link_p(d):
        cap = M._link_capacity_bps(float(d), tx_d, AMBIENT)
        return 0.0 if cap <= 0.0 else M._LAST_P[0]

    def hears(d):
        return M._link_capacity_bps(float(d), tx_u, AMBIENT) > 0.0

    import glob as G
    allq, allr, alla = [], [], []
    print(f"{'arm':<18}{'eps':>5}{'qual':>9}{'reward':>11}{'r':>8}{'t':>8}")
    for spec in a.arm_glob:
        name, pat = spec.split("=", 1)
        qs, rs = [], []
        for f in sorted(G.glob(pat)):
            try:
                recs = [r for r in json.load(open(f)) if r.get("kind") != "obs"]
            except Exception:
                continue
            eps = {}
            for r in recs:
                e = eps.setdefault(r["seed"], {"u": None, "s": []})
                if r.get("users") is not None:
                    e["u"] = r["users"]
                e["s"].append(r)
            for seed, e in sorted(eps.items()):
                if e["u"] is None or not e["s"]:
                    continue
                U = np.asarray(e["u"], float)
                du = np.linalg.norm(U[:, None, :] - U[None, :, :], axis=-1)
                lab = components(du <= a.link); k = int(lab.max()) + 1
                q = []
                for r in e["s"]:
                    pos = np.asarray(r["pos"], float)
                    dd = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
                    nD = len(pos); P = np.zeros((nD, nD))
                    for i in range(nD):
                        for j in range(i + 1, nD):
                            P[i, j] = P[j, i] = link_p(dd[i, j])
                    dU = np.linalg.norm(pos[:, None, :] - U[None, :, :], axis=-1)
                    heard = np.vectorize(hears)(dU)
                    acc = [np.nonzero(heard[:, lab == c].any(1))[0] for c in range(k)]
                    W = widest(P); best = 0.0
                    for x in range(k):
                        for y in range(x + 1, k):
                            if len(acc[x]) and len(acc[y]):
                                best = max(best, float(W[np.ix_(acc[x], acc[y])].max()))
                    q.append(best)
                rew = [r["reward"] for r in e["s"] if r.get("reward") is not None]
                if not q or not rew:
                    continue
                qs.append(float(np.mean(q))); rs.append(float(np.mean(rew)))
        if not qs:
            print(f"{name:<18}{0:>5}   no episodes"); continue
        r, t = pearson(qs, rs)
        print(f"{name:<18}{len(qs):>5}{np.mean(qs):>9.4f}{np.mean(rs):>11.1f}{r:>8.3f}{t:>8.2f}")
        allq += qs; allr += rs; alla += [name] * len(qs)
    if len(allq) >= 3:
        r, t = pearson(allq, allr)
        print(f"\n  POOLED n={len(allq)}  r={r:.3f}  t={t:.2f}")
        # within-arm (arm-demeaned) -- removes any between-arm confound
        aq, ar = np.array(allq), np.array(allr)
        for nm in set(alla):
            m = np.array([x == nm for x in alla])
            if m.sum() >= 2:
                aq[m] -= aq[m].mean(); ar[m] -= ar[m].mean()
        r2, t2 = pearson(aq, ar)
        print(f"  WITHIN-ARM (arm-demeaned) r={r2:.3f}  t={t2:.2f}"
              f"   <- the one that matters")


if __name__ == "__main__":
    sys.exit(main())
