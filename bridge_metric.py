#!/usr/bin/env python3
"""% of steps with >=1 cluster pair BRIDGED, computed from an OM2G trace dump.

Reward and position statistics alone do not show bridging: drones can sit in
the inter-cluster gap without connecting two clusters. This measures
CONNECTIVITY, which is what max-flow pays for -- flow is zero until a chain
completes.

Two numbers are reported, and they answer different questions:

  struct_bridge%  -- any two user-clusters joined through the drone graph.
                     The structural measure ("zero bridges").
  demand_bridge%  -- the fraction of the 40 ACTUAL conversations whose two
                     endpoints have connected access drones. This is the one
                     that maps to reward: a bridge nobody is calling across
                     earns nothing.

Ranges come from the emitted run config, NOT from these defaults -- pass
--r-u/--r-d when scoring a run with an overridden radio (e.g. the
$OM2G_RANGE_DRONE curriculum sweep), so the metric uses the radio ranges the
policy actually ran with.

Usage:
  python bridge_metric.py trace.json [trace2.json ...] [--r-u 157.82] [--r-d 523.06]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def cluster_users(X, link=500.0):
    """Single-linkage at `link` metres.

    Single linkage rather than k-means: k-means with random init can split one
    physical cluster in two while merging another, which would count a single
    drone pair as a bridge. The archipelago's geometry makes the robust method
    simple: users sit in ~25 m-wide clumps separated by >=1465 m (MINSEP), so
    any threshold comfortably between those two scales recovers the true
    clusters exactly and deterministically.
    """
    n = len(X)
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    return components(d <= link)


def components(adj):
    """Connected components of a boolean adjacency matrix -> label per node."""
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


def score_trace(path, r_u, r_d, k=4):  # k kept for CLI compat; unused
    recs = [r for r in json.load(open(path)) if r.get("kind") is None]
    by_seed = {}
    for r in recs:
        by_seed.setdefault(r["seed"], []).append(r)

    out = {}
    for seed, rows in sorted(by_seed.items()):
        rows.sort(key=lambda r: r["step"])
        users = np.asarray(rows[0]["users"], dtype=float)
        # Cluster labels are a property of the SCENARIO, not the step: users
        # are static in OM2G, so this is computed once per episode.
        lab = cluster_users(users)
        k_found = int(lab.max()) + 1
        pairs = rows[0].get("pairs") or []

        n_struct = n_dem = 0
        dem_frac = []
        maxcomp = []
        for r in rows:
            pos = np.asarray(r["pos"], dtype=float)
            dd = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
            adj = dd <= r_d
            np.fill_diagonal(adj, True)
            comp = components(adj)
            _, cnt = np.unique(comp, return_counts=True)
            maxcomp.append(cnt.max())

            du = np.linalg.norm(pos[:, None, :] - users[None, :, :], axis=-1)
            heard = du <= r_u                      # (drone, user)

            # component id(s) serving each cluster
            cl_comps = [set(comp[np.nonzero(heard[:, lab == c].any(1))[0]])
                        for c in range(k_found)]
            bridged = any(cl_comps[a] & cl_comps[b]
                          for a in range(k_found) for b in range(a + 1, k_found))
            n_struct += bool(bridged)

            # demand side: a call is carried only if BOTH endpoints attach to
            # drones in the same component.
            ok = 0
            for u, v in pairs:
                cu = set(comp[np.nonzero(heard[:, u])[0]])
                cv = set(comp[np.nonzero(heard[:, v])[0]])
                ok += bool(cu & cv)
            if pairs:
                dem_frac.append(ok / len(pairs))
                n_dem += bool(ok)

        nst = len(rows)
        out[seed] = dict(
            steps=nst,
            k_found=k_found,
            struct_pct=100.0 * n_struct / nst,
            demand_pct=100.0 * n_dem / nst,
            demand_frac=100.0 * float(np.mean(dem_frac)) if dem_frac else float("nan"),
            maxcomp=float(np.mean(maxcomp)),
            reward=float(np.mean([r["reward"] for r in rows])),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--r-u", type=float, default=157.82)
    ap.add_argument("--r-d", type=float, default=523.06)
    ap.add_argument("--clusters", type=int, default=4)
    a = ap.parse_args()

    print(f"  r_u={a.r_u}  r_d={a.r_d}  k={a.clusters}")
    print(f"{'trace':<28}{'seed':>5}{'steps':>7}{'reward':>9}"
          f"{'k':>3}{'struct%':>9}{'demand%':>9}{'calls%':>8}{'maxcomp':>9}")
    allrows = []
    for t in a.traces:
        res = score_trace(t, a.r_u, a.r_d, a.clusters)
        for seed, v in res.items():
            allrows.append(v)
            print(f"{Path(t).name:<28}{seed:>5}{v['steps']:>7}{v['reward']:>9.1f}"
                  f"{v['k_found']:>3}{v['struct_pct']:>9.1f}{v['demand_pct']:>9.1f}"
                  f"{v['demand_frac']:>8.1f}{v['maxcomp']:>9.2f}")
    if allrows:
        n = len(allrows)
        print(f"\n  OVERALL n={n} seeds, {sum(r['steps'] for r in allrows)} steps")
        print(f"    struct bridge%  {np.mean([r['struct_pct'] for r in allrows]):.2f}")
        print(f"    demand bridge%  {np.mean([r['demand_pct'] for r in allrows]):.2f}")
        print(f"    calls carried%  {np.mean([r['demand_frac'] for r in allrows]):.2f}")
        print(f"    max component   {np.mean([r['maxcomp'] for r in allrows]):.2f}")


if __name__ == "__main__":
    sys.exit(main())
