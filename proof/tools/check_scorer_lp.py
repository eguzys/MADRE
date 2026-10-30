#!/usr/bin/env python3
"""Mechanical check: the LP the simulator's scorer solves IS the explicit-edge weighted LP
`P1EW` of `MadreProof/Statement/Weighted.lean`.

For random drone/user layouts (fixed seeds) and every tested message size L, this script
  1. calls the scorer `omnet_module/max_flow_reward.py` (imported READ-ONLY from the code
     repository) with the paper's environment (OM2G_PHYSICAL_METRIC=1, 8 drones, compressed
     arm, R, b(L), sync interval and demand cap from `om2g/om2g_track.py`), and reads back the
     exact LP that HiGHS solved (`Highs.getLp()` of the scorer's cache);
  2. rebuilds the LP independently from the P1EW definition: edge set E = the scorer's
     positive-capacity edges, capacities c = the scorer's capacities, weights a_i(e) = 1/p(e)
     on drone-drone edges that drone i transmits on, 1 on every other edge at i (drone-to-user
     transmit, and every receive), κ(i) = 8 b(L) (1 + |𝒟(i)|) with 𝒟(i) = drones with an edge
     into i, demand D = the scorer's τ cap;
  3. asserts that objective, column bounds, inequality rows and equality rows coincide (up to
     row order, the sign of equality rows, trivial 0 = 0 rows, and the scorer's internal unit
     scale), and that both LPs have the same optimum (the reference solved with scipy/HiGHS).

Usage (from the proof directory; --code is the MADRE code checkout, default ..):
    python tools/check_scorer_lp.py [--code ..] [--layouts 5]
"""
import argparse
import json
import os
import subprocess
import sys

TESTED_L = [8, 16, 32, 64, 128, 256, 512]


def child(code_dir: str, L: int, n_layouts: int) -> dict:
    sys.path.insert(0, os.path.join(code_dir, "om2g"))
    sys.path.insert(0, os.path.join(code_dir, "omnet_module"))
    import numpy as np
    from scipy.optimize import linprog
    import om2g_track as T
    import max_flow_reward as M

    assert M._PHYSICAL_METRIC, "OM2G_PHYSICAL_METRIC=1 was not applied"
    R = float(os.environ["OM2G_RADIO_CAP_BPS"])
    b = float(os.environ["OM2G_CONTROL_ON_AIR_BYTES"])
    interval = float(os.environ["OM2G_SYNC_INTERVAL_S"])
    tau_cap = M.TAU_CAP_BPS
    S = M.LP_SCALE

    def scorer_lp():
        cache = next(reversed(M._ORACLE_LP_CACHE.values()))
        lp = cache["highs"].getLp()
        edge_list, K, E = cache["edge_list"], cache["K"], cache["E"]
        comms = cache["commodities_raw"]
        names = [("tau", ki) for ki in range(K)] + \
                [("f", ki, edge_list[e]) for ki in range(K) for e in range(E)]
        A = lp.a_matrix_
        start, index, value = list(A.start_), list(A.index_), list(A.value_)
        rows = [dict() for _ in range(lp.num_row_)]
        for col in range(lp.num_col_):
            for p in range(start[col], start[col + 1]):
                rows[index[p]][names[col]] = value[p]
        return dict(names=names, rows=rows, rlo=list(lp.row_lower_), rhi=list(lp.row_upper_),
                    clo=list(lp.col_lower_), chi=list(lp.col_upper_), cost=list(lp.col_cost_),
                    edges=edge_list, K=K, comms=comms)

    def reference(nd, nu, edges, comms, weight):
        """P1EW rows in bps. weight(i, e) is a_i(e) for drone i and incident edge e."""
        E = list(edges.keys())
        K = len(comms)
        ineq = []
        for e in E:                                          # capacity rows
            ineq.append(({("f", k, e): 1.0 for k in range(K)}, edges[e]))
        for i in range(nd):                                  # weighted radio rows
            nb = sum(1 for (u, v) in E if u < nd and v == i)
            kappa = 8.0 * b * (1 + nb) / interval
            row = {}
            for e in E:
                if i in e:
                    for k in range(K):
                        row[("f", k, e)] = weight(i, e)
            ineq.append((row, R - kappa))
        eq = []
        for k, (su, du) in enumerate(comms):                 # flow conservation, all nodes
            src, dst = nd + su, nd + du
            for v in range(nd + nu):
                row = {}
                for e in E:
                    if e[0] == v:
                        row[("f", k, e)] = row.get(("f", k, e), 0.0) + 1.0
                    if e[1] == v:
                        row[("f", k, e)] = row.get(("f", k, e), 0.0) - 1.0
                sigma = 1.0 if v == src else (-1.0 if v == dst else 0.0)
                if sigma != 0.0:
                    row[("tau", k)] = -sigma
                row = {n: c for n, c in row.items() if c != 0.0}
                if row:
                    eq.append(row)
        return ineq, eq

    def close(a, b, rel=1e-9, ab=1e-9):
        return abs(a - b) <= ab + rel * max(abs(a), abs(b))

    def compare(sc, nd, nu, edges, comms, weight):
        K = len(comms)
        ineq_ref, eq_ref = reference(nd, nu, edges, comms, weight)
        # objective: scorer minimises cost; P1EW maximises Σ τ  ->  cost = -α on τ, 0 on f
        alpha = -sc["cost"][0]
        assert alpha > 0
        for n, c in zip(sc["names"], sc["cost"]):
            want = -alpha if n[0] == "tau" else 0.0
            assert close(c, want), ("objective", n, c, want)
        # column bounds (scorer columns are in units of LP_SCALE bps)
        for n, lo, hi in zip(sc["names"], sc["clo"], sc["chi"]):
            assert lo == 0.0, ("lower bound", n, lo)
            if n[0] == "tau":
                assert close(hi * S, tau_cap), ("tau cap", n, hi * S, tau_cap)
            else:
                assert hi > 1e29, ("flow upper bound", n, hi)
        # rows
        sc_ineq, sc_eq = [], []
        for r, lo, hi in zip(sc["rows"], sc["rlo"], sc["rhi"]):
            r = {n: c for n, c in r.items() if c != 0.0}
            if lo < -1e29:
                sc_ineq.append((r, hi * S))
            else:
                assert lo == hi == 0.0, ("equality rhs", lo, hi)
                if r:
                    sc_eq.append(r)

        def norm_eq(r):
            first = sorted(r)[0]
            s = 1.0 if r[first] > 0 else -1.0
            return {n: s * c for n, c in r.items()}

        def match(ref_rows, sc_rows, with_rhs):
            pool = {}
            for sr in sc_rows:
                srow = sr[0] if with_rhs else sr
                pool.setdefault(frozenset(srow), []).append(sr)
            bad = 0
            for rr in ref_rows:
                row, rhs = rr if with_rhs else (rr, None)
                cands = pool.get(frozenset(row), [])
                found = None
                for idx, sr in enumerate(cands):
                    srow, srhs = sr if with_rhs else (sr, None)
                    if all(close(srow[n], row[n]) for n in row) and \
                            (not with_rhs or close(srhs, rhs, rel=1e-9, ab=1e-6)):
                        found = idx
                        break
                if found is None:
                    bad += 1
                else:
                    cands.pop(found)
            return bad, sum(len(v) for v in pool.values())

        bad_i, extra_i = match(ineq_ref, sc_ineq, True)
        bad_e, extra_e = match([norm_eq(r) for r in eq_ref], [norm_eq(r) for r in sc_eq], False)
        return dict(ineq_unmatched=bad_i, ineq_extra=extra_i,
                    eq_unmatched=bad_e, eq_extra=extra_e,
                    ineq_rows=len(ineq_ref), eq_rows=len(eq_ref)), ineq_ref, eq_ref

    def solve_ref(ineq, eq, comms, edges):
        K = len(comms)
        names = [("tau", k) for k in range(K)] + [("f", k, e) for k in range(K) for e in edges]
        col = {n: j for j, n in enumerate(names)}
        from scipy.sparse import coo_matrix
        ri, ci, vi = [], [], []
        b_ub = np.zeros(len(ineq))
        for r, (row, rhs) in enumerate(ineq):
            for n, c in row.items():
                ri.append(r); ci.append(col[n]); vi.append(c)
            b_ub[r] = rhs
        A_ub = coo_matrix((vi, (ri, ci)), shape=(len(ineq), len(names))).tocsr()
        ri, ci, vi = [], [], []
        for r, row in enumerate(eq):
            for n, c in row.items():
                ri.append(r); ci.append(col[n]); vi.append(c)
        A_eq = coo_matrix((vi, (ri, ci)), shape=(len(eq), len(names))).tocsr()
        cost = np.array([-1.0 if n[0] == "tau" else 0.0 for n in names])
        bounds = [(0, tau_cap) if n[0] == "tau" else (0, None) for n in names]
        res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(len(eq)),
                      bounds=bounds, method="highs")
        assert res.status == 0, res.message
        return -res.fun

    out = dict(L=L, b=b, layouts=[], loo=[])
    nd, nu = 8, 40
    for seed in range(n_layouts):
        rng = np.random.default_rng(1000 + seed)
        drones_a = rng.uniform(1500.0, 2500.0, size=(nd, 2))
        drones = drones_a.tolist()
        users = (drones_a[rng.integers(0, nd, size=nu)]
                 + rng.normal(0, 100.0, size=(nu, 2))).tolist()
        pairs = [[u, int(rng.integers(0, nu - 1)) % nu] for u in range(nu)]
        pairs = [[u, v if v != u else (u + 1) % nu] for u, v in pairs]
        obs = dict(positions=drones, user_positions=users, sender_pairs=pairs,
                   noise_floor=[M.NOISE_FLOOR_dBm] * nd)
        res = M.compute_position_oracle_reward(obs)
        sc = scorer_lp()
        edges = M._build_edges(nd, nu, drones, users, obs["noise_floor"])
        assert list(edges.keys()) == list(sc["edges"]), "edge list differs from the scorer's"
        comms = sc["comms"]
        P = dict(M._EDGE_P)

        def w_p1ew(i, e):
            if e[0] == i and e[1] < nd:              # drone i transmits to a drone
                return 1.0 / max(P[e], 1e-9)
            return 1.0                               # transmits to a user, or receives
        cmp, ineq, eq = compare(sc, nd, nu, edges, comms, w_p1ew)
        opt_ref = solve_ref(ineq, eq, comms, list(edges.keys()))
        opt_sc = res["reward"] * M.REWARD_UNIT_BPS
        cmp.update(seed=seed, opt_scorer_bps=opt_sc, opt_reference_bps=opt_ref,
                   opt_match=abs(opt_sc - opt_ref) <= 1e-6 * max(1.0, abs(opt_ref)),
                   max_kappa=max(8.0 * b * (1 + sum(1 for (u, v) in edges if u < nd and v == i))
                                 / interval for i in range(nd)))
        cmp["pass"] = (cmp["ineq_unmatched"] == 0 and cmp["ineq_extra"] == 0 and
                       cmp["eq_unmatched"] == 0 and cmp["eq_extra"] == 0 and cmp["opt_match"])
        out["layouts"].append(cmp)

        # leave-one-out LPs of the difference reward, same process (stale _EDGE_P)
        for drop in range(nd):
            o2 = dict(obs)
            o2["positions"] = [p for k, p in enumerate(drones) if k != drop]
            o2["noise_floor"] = [p for k, p in enumerate(obs["noise_floor"]) if k != drop]
            M.compute_position_oracle_reward(o2, cache_tag=drop)
            sc2 = scorer_lp()
            nd2 = nd - 1
            edges2 = M._build_edges(nd2, nu, o2["positions"], users, o2["noise_floor"])
            P2 = dict(M._EDGE_P)

            def w2(i, e, P2=P2, nd2=nd2):
                if e[0] == i and e[1] < nd2:
                    return 1.0 / max(P2[e], 1e-9)
                return 1.0
            cmp2, _, _ = compare(sc2, nd2, nu, edges2, sc2["comms"], w2)

            def w_stale(i, e, P2=P2):
                # what the scorer actually uses: 1/p for EVERY transmit edge that has a
                # (possibly stale) _EDGE_P entry, 1 otherwise
                if e[0] == i and e in P2:
                    return 1.0 / max(P2[e], 1e-9)
                return 1.0
            cmp3, _, _ = compare(sc2, nd2, nu, edges2, sc2["comms"], w_stale)
            stale = [e for e in edges2 if e[0] < nd2 and e[1] >= nd2 and e in P2]
            out["loo"].append(dict(seed=seed, drop=drop,
                                   radio_rows_differing=cmp2["ineq_unmatched"],
                                   differing_with_stale_weights=cmp3["ineq_unmatched"],
                                   stale_user_edges=len(stale)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="..")
    ap.add_argument("--layouts", type=int, default=5)
    ap.add_argument("--child", type=int, default=None)
    args = ap.parse_args()
    if args.child is not None:
        print(json.dumps(child(args.code, args.child, args.layouts)))
        return
    sys.path.insert(0, os.path.join(args.code, "om2g"))
    import om2g_track as T
    all_ok = True
    for L in TESTED_L:
        env = dict(os.environ)
        env.update({
            "OM2G_PHYSICAL_METRIC": "1",
            "OM2G_DRONES": "8",
            "OM2G_RADIO_CAP_BPS": repr(T.OM2G_RADIO_CAP_APP_BPS),
            "OM2G_CONTROL_ON_AIR_BYTES": repr(float(T.om2g_message_on_air_bytes("compressed", L))),
            "OM2G_SYNC_INTERVAL_S": repr(T.OM2G_SYNC_INTERVAL_S),
            "OM2G_TAU_CAP_BPS": repr(T.OM2G_VOICE_BPS * 2.0),
            "OM2G_REWARD_UNIT_BPS": "1000.0",
            "CUDA_VISIBLE_DEVICES": "",
        })
        r = subprocess.run([sys.executable, __file__, "--code", args.code,
                            "--layouts", str(args.layouts), "--child", str(L)],
                           env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"L={L}: ERROR\n{r.stderr[-2000:]}")
            all_ok = False
            continue
        res = json.loads(r.stdout.strip().splitlines()[-1])
        ok = all(x["pass"] for x in res["layouts"])
        all_ok &= ok
        loo_diff = sum(x["radio_rows_differing"] for x in res["loo"])
        loo_stale = sum(x["stale_user_edges"] for x in res["loo"])
        loo_expl = sum(x["differing_with_stale_weights"] for x in res["loo"])
        print(f"L={L:4d} b={res['b']:.0f}: full LP {'PASS' if ok else 'FAIL'} "
              f"({len(res['layouts'])} layouts; max κ = {max(x['max_kappa'] for x in res['layouts']):.0f} bit/s; "
              f"optima {[round(x['opt_scorer_bps'], 3) for x in res['layouts']]} bps) | "
              f"leave-one-out: {loo_diff} radio rows differ from P1EW, "
              f"{loo_stale} drone->user edges carry a stale _EDGE_P entry, "
              f"{loo_expl} differ once the stale 1/p is used")
        if not ok:
            for x in res["layouts"]:
                if not x["pass"]:
                    print("   ", x)
    print("ALL FULL-LP CHECKS PASS" if all_ok else "SOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
