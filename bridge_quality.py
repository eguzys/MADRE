#!/usr/bin/env python3
"""Bridge QUALITY, not bridge existence.

bridge_metric.py answers "are two clusters joined?" with a HARD geometric
radius. The reward does not work that way: `max_flow_reward._link_capacity_bps`
under $OM2G_PHYSICAL_METRIC gives every link a decode probability p and DROPS
it below _P_USABLE_MIN, then charges ARQ 1/p at the transmitter. So a chain can
be structurally complete at 520 m and carry essentially nothing.

That is exactly the gap this script measures. At n=40 the learned arms bridge
(struct 2.28%) and carry no more than the silent arm (2.68 vs 2.75), while
relay_explore bridges 31.40% and carries 14.39%. Two explanations survive:

  (a) TRANSIENCE -- the bridges are fine but too brief for a call to complete.
  (b) FICTION    -- the bridges exist geometrically but their bottleneck link
                    sits at the edge of decodability, so capacity is ~0.

(a) and (b) predict opposite things about the bottleneck p at bridged steps, so
one number separates them.

Reported per arm:
  struct%_geo    steps with >=2 clusters joined by links within R_GEO
  struct%_use    ... joined by links the RADIO can actually use (p >= p_min)
  botP           mean over bridged steps of the widest-path bottleneck p
  botCap         ... and the bottleneck capacity in kbps
  qual           mean over ALL steps of the best inter-cluster bottleneck p
                 (threshold-free: 0 when nothing is connected, so it is a
                 single continuous number comparable across arms)

Env setup is offline_score.py's, verbatim -- om2g_env_vars() before the import
and noise_floor rebuilt as the -100.0 dBm jammer-never-fired constant. The full
set is exported so the offline replay matches the simulated runs exactly.
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
_OM2G_DIR = _HERE / "om2g"
_OMNET_MODULE_DIR = _HERE / "omnet_module"
AMBIENT_NOISE_FLOOR_DBM = -100.0


def configure(arm="compressed"):
    sys.path.insert(0, str(_OM2G_DIR))
    sys.path.insert(0, str(_OMNET_MODULE_DIR))
    import om2g_config as C
    env = C.om2g_env_vars(arm)
    for k, v in env.items():
        os.environ[k] = str(v)
    # _LAST_P is only populated on the physical-metric path.
    os.environ["OM2G_PHYSICAL_METRIC"] = "1"
    return env


def components(adj):
    n = len(adj)
    comp = -np.ones(n, dtype=int)
    c = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        stack, comp[s] = [s], c
        while stack:
            u = stack.pop()
            for v in np.nonzero(adj[u] & (comp < 0))[0]:
                comp[v] = c
                stack.append(v)
        c += 1
    return comp


def widest(P):
    """Bottleneck (max-min) closure of a link-quality matrix -- Floyd-Warshall
    over the (max, min) semiring. P[i,j] in [0,1]."""
    W = P.copy()
    np.fill_diagonal(W, 1.0)
    for k in range(len(W)):
        W = np.maximum(W, np.minimum(W[:, k][:, None], W[k, :][None, :]))
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--arm", default="compressed")
    ap.add_argument("--r-geo", type=float, default=523.06,
                    help="geometric radius for the struct%%_geo column only")
    ap.add_argument("--link", type=float, default=500.0,
                    help="single-linkage threshold for user clustering")
    a = ap.parse_args()

    env = configure(a.arm)
    import max_flow_reward as M
    p_min = M._P_USABLE_MIN
    tx_d = float(env.get("OMNET_DRONE_TX_mW", 200.0))
    tx_u = float(env.get("OMNET_USER_TX_mW", tx_d))
    print(f"# arm={a.arm}  drone_tx={tx_d} mW  user_tx={tx_u} mW  "
          f"p_min={p_min}  noise_floor={AMBIENT_NOISE_FLOOR_DBM} dBm")

    def link_p(d):
        cap = M._link_capacity_bps(float(d), tx_d, AMBIENT_NOISE_FLOOR_DBM)
        return (0.0 if cap <= 0.0 else M._LAST_P[0]), cap

    def hears(d):
        return M._link_capacity_bps(float(d), tx_u, AMBIENT_NOISE_FLOOR_DBM) > 0.0

    print(f"{'trace':<40}{'seeds':>6}{'steps':>7}{'geo%':>8}{'use%':>8}"
          f"{'botP':>9}{'botCap_kbps':>13}{'qual':>9}")
    for t in a.traces:
        recs = [r for r in json.load(open(t)) if r.get("kind") != "obs"]
        eps = {}
        for r in recs:
            e = eps.setdefault(r["seed"], {"u": None, "s": []})
            if r.get("users") is not None:
                e["u"] = r["users"]
            e["s"].append(r)
        n_geo = n_use = n_all = 0
        botP, botC, qual = [], [], []
        for seed, e in sorted(eps.items()):
            U = np.asarray(e["u"], float)
            du = np.linalg.norm(U[:, None, :] - U[None, :, :], axis=-1)
            lab = components(du <= a.link)
            k = int(lab.max()) + 1
            for r in e["s"]:
                pos = np.asarray(r["pos"], float)
                dd = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
                nD = len(pos)
                P = np.zeros((nD, nD))
                C = np.zeros((nD, nD))
                for i in range(nD):
                    for j in range(i + 1, nD):
                        p, c = link_p(dd[i, j])
                        P[i, j] = P[j, i] = p
                        C[i, j] = C[j, i] = c
                dU = np.linalg.norm(pos[:, None, :] - U[None, :, :], axis=-1)
                heard = np.vectorize(hears)(dU) if dU.size else dU
                # access drones per cluster
                acc = [np.nonzero(heard[:, lab == c].any(1))[0] for c in range(k)]

                geo = components(dd <= a.r_geo)
                use = components(P >= p_min)
                W = widest(P)
                best = 0.0
                bg = bu = False
                for x in range(k):
                    for y in range(x + 1, k):
                        if len(acc[x]) == 0 or len(acc[y]) == 0:
                            continue
                        if set(geo[acc[x]]) & set(geo[acc[y]]):
                            bg = True
                        if set(use[acc[x]]) & set(use[acc[y]]):
                            bu = True
                        best = max(best, float(W[np.ix_(acc[x], acc[y])].max()))
                n_all += 1
                n_geo += bg
                n_use += bu
                qual.append(best)
                if bu:
                    botP.append(best)
                    # capacity of the weakest hop on a chain of that quality
                    botC.append(float(C[(P >= p_min) & (P <= best + 1e-12)].min())
                                if np.any((P >= p_min) & (P <= best + 1e-12)) else 0.0)
        f = lambda v: (np.mean(v) if v else float("nan"))
        print(f"{Path(t).name:<40}{len(eps):>6}{n_all:>7}"
              f"{100.0*n_geo/max(n_all,1):>8.2f}{100.0*n_use/max(n_all,1):>8.2f}"
              f"{f(botP):>9.3f}{f(botC)/1e3:>13.1f}{f(qual):>9.4f}")


if __name__ == "__main__":
    sys.exit(main())
