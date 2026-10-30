#!/usr/bin/env python3
"""WHAT does relay_explore do that engineered does not? -- in metres.

`qual` is now a verified predictor of score (qual_vs_reward.py: within-arm
r=0.626, t=7.44, n=88), and relay_explore holds 0.27 against engineered's 0.11.
That is a target, but "raise qual" is not something a drone can execute. This
turns the gap into geometry a policy could actually be asked to produce.

Per arm, over every step of every episode:
  bothop_m   length of the weakest hop on the best inter-cluster path
             (the hop whose p sets the ARQ tax -- the thing that must shrink)
  nn_m       nearest-neighbour drone spacing, median over drones
  span_m     max pairwise drone separation (is the swarm spread or balled up?)
  reach%     fraction of drones within earshot of at least one user

The decode curve at the drone's 6.59 mW is the ruler: p=0.93 at 300 m, 0.49 at
500 m, 0.14 at 700 m. So a prescription in metres is a prescription in p.
"""
import argparse, glob, json, os, sys
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-glob", action="append", required=True)
    ap.add_argument("--link", type=float, default=500.0)
    a = ap.parse_args()
    env = configure()
    import max_flow_reward as M
    tx_d = float(env.get("OMNET_DRONE_TX_mW", 200.0))
    tx_u = float(env.get("OMNET_USER_TX_mW", tx_d))

    def pmat(dd):
        n = len(dd); P = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                cap = M._link_capacity_bps(float(dd[i, j]), tx_d, AMBIENT)
                P[i, j] = P[j, i] = 0.0 if cap <= 0 else M._LAST_P[0]
        return P

    def hears(d):
        return M._link_capacity_bps(float(d), tx_u, AMBIENT) > 0.0

    print("# decode ruler at 6.59 mW: "
          + "  ".join(f"{d}m->{float(M._link_capacity_bps(d, tx_d, AMBIENT) > 0 and M._LAST_P[0]):.2f}"
                      for d in (200, 300, 400, 500, 600, 700, 900)))
    print(f"{'arm':<18}{'eps':>5}{'hop10':>8}{'hop25':>8}{'hop50':>8}"
          f"{'nn_m':>9}{'span_m':>9}{'reach%':>8}{'hop<500%':>9}")
    for spec in a.arm_glob:
        name, pat = spec.split("=", 1)
        bh, nn, sp, rc = [], [], [], []
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
                if e["u"] is None: continue
                U = np.asarray(e["u"], float)
                du = np.linalg.norm(U[:, None, :] - U[None, :, :], axis=-1)
                lab = components(du <= a.link); k = int(lab.max()) + 1
                for r in e["s"]:
                    pos = np.asarray(r["pos"], float)
                    dd = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
                    P = pmat(dd)
                    dU = np.linalg.norm(pos[:, None, :] - U[None, :, :], axis=-1)
                    heard = np.vectorize(hears)(dU)
                    acc = [np.nonzero(heard[:, lab == c].any(1))[0] for c in range(k)]
                    # widest path WITH predecessor, to recover the actual hop
                    n = len(pos); W = P.copy(); np.fill_diagonal(W, 1.0)
                    for kk in range(n):
                        W = np.maximum(W, np.minimum(W[:, kk][:, None], W[kk, :][None, :]))
                    best, bi, bj = 0.0, None, None
                    for x in range(k):
                        for y in range(x + 1, k):
                            if not len(acc[x]) or not len(acc[y]): continue
                            sub = W[np.ix_(acc[x], acc[y])]
                            m = float(sub.max())
                            if m > best:
                                best = m
                                ii, jj = np.unravel_index(sub.argmax(), sub.shape)
                                bi, bj = acc[x][ii], acc[y][jj]
                    if best > 0 and bi is not None:
                        # the hop whose p equals the bottleneck
                        cand = np.abs(P - best) < 1e-9
                        if cand.any():
                            bh.append(float(dd[cand].max()))
                    off = dd + np.eye(n) * 1e9
                    nn.append(float(np.median(off.min(1))))
                    sp.append(float(dd.max()))
                    rc.append(float(heard.any(1).mean()))
        if not nn:
            print(f"{name:<18}{0:>5}  no episodes"); continue
        f = lambda v: (np.median(v) if v else float("nan"))
        # PERCENTILES of the bottleneck hop, not just the median. p(d) is
        # steeply nonlinear (0.93 at 300 m, 0.19 at 700 m), so a mean decode
        # probability is dominated by the SHORT tail while the median distance
        # can be identical across arms. Measured 2026-09-02: median bothop was
        # 763 / 719 / 741 m across three arms whose mean botP differed 2.1x --
        # the median was hiding the whole effect.
        q = (np.percentile(bh, [10, 25, 50]) if bh else [float("nan")] * 3)
        print(f"{name:<18}{len(nn)//250:>5}{q[0]:>8.0f}{q[1]:>8.0f}{q[2]:>8.0f}"
              f"{f(nn):>9.0f}{f(sp):>9.0f}{100*np.mean(rc):>8.1f}"
              f"{100.0*np.mean(np.asarray(bh) < 500) if bh else float('nan'):>9.1f}")


if __name__ == "__main__":
    sys.exit(main())
