#!/usr/bin/env python3
"""Additional scripted baselines, run through the
unmodified evaluate_omnet.py harness by wrapping _select_actions. Selected by
SUITE_POLICY:

  gtm     PADRE's Greedy Traffic Maximizer (Leuenberger, Di Maio, Braun, WD
          2025, Alg. 1): 8 compass probes at kappa=OMNET_STEP_M, keep the one
          that raises the objective most, others frozen. SUITE_GTM_OBJ=
          hosted (g(m_i), PADRE-faithful) | global (Sigma tau). Free, exact
          global state.
  kmeans  Coverage control, no relay reasoning: k-means (k = #drones,
          k-means++ with restarts) on the UNION of heard users, solved once
          the pooled set stops changing, then fly to the centres. Local
          sensing; needs the heard positions shared (charge 67 B).
          Lineage: Lloyd / Voronoi coverage (Cortes, Martinez, Karatas,
          Bullo, IEEE T-RA 2004).
  force   Virtual-force / spring placement (Howard, Mataric, Sukhatme 2002;
          Zou, Chakrabarty INFOCOM 2003; Saffre, Hildmann, Anttonen 2023,
          PADRE's ref [12]): attraction to OWN heard users, spring to every
          peer within 2*d0 with rest length d0 (repel closer, attract
          farther). One parameter set, untuned. Needs peer positions
          (position beacon, charge 51 B).
  rsr     De Liso, Di Maio, Braun (MedComNet 2024) RSR-style SEQUENTIAL grid
          relocation: for each drone in turn, evaluate every point of a
          fixed grid over the user area by the global max-flow throughput
          (the reward LP) and move it to the best; two sweeps; solved once
          per episode (users are static), then fly there. Omniscient and
          free, like scripted_relay: an upper-bound reference, not a
          protocol. Differences from RSR: no jammer trigger, all drones
          relocate, no relocation-cost or centrality terms, no filters.
  lattice Static regular deployment over the arena, no information, never
          moves after arrival. Free.
  maxcov  Max-coverage sequential greedy (Nemhauser, Wolsey, Fisher 1978):
          place drones one at a time at the grid point adding the most
          capped user capacity beyond what earlier drones give, over the
          union of heard users. Coverage only, no flow or relay reasoning.
          Local sensing, shared heard positions (charge 67 B).
"""
import math
import os
import sys

sys.path.insert(0, os.getcwd())
import evaluate_omnet as E  # noqa: E402
import max_flow_reward as M  # noqa: E402
import signal_baselines as SB  # noqa: E402

POL = os.environ.get("SUITE_POLICY", "gtm").strip().lower()
STEP = float(os.environ.get("OMNET_STEP_M", 60.0))
OBJ = os.environ.get("SUITE_GTM_OBJ", "global").strip().lower()
DIRS = [(math.cos(2 * math.pi * k / 8), math.sin(2 * math.pi * k / 8)) for k in range(8)]
_orig = E._select_actions
_stats = {"steps": 0, "moved": 0, "solves": 0}
_CACHE = {"key": None, "targets": None, "nkey": None}


def _solve(obs, positions):
    o = dict(obs)
    o["positions"] = positions
    _stats["solves"] += 1
    return M.compute_position_oracle_reward(o)


def _R(obs, positions):
    return float(_solve(obs, positions)["reward"])


# ── GTM ──────────────────────────────────────────────────────────────────────
def _objective(res, i):
    if OBJ == "global":
        return float(res["reward"])
    tot = 0.0
    for edge_flows in res.get("per_commodity_flow", {}).values():
        for (a, b), f in edge_flows.items():
            if b == i:
                tot += float(f)
    return tot


def gtm(obs):
    pos = [[float(p[0]), float(p[1])] for p in obs.get("positions", [])]
    base = _solve(obs, pos)
    vels = []
    for i in range(len(pos)):
        best, bv = _objective(base, i), (0.0, 0.0)
        for dx, dy in DIRS:
            cand = [list(p) for p in pos]
            cand[i] = [pos[i][0] + dx * STEP, pos[i][1] + dy * STEP]
            r = _objective(_solve(obs, cand), i)
            if r > best:
                best, bv = r, (dx, dy)
        if bv != (0.0, 0.0):
            _stats["moved"] += 1
        vels.append([bv[0], bv[1]])
    return vels


# ── shared helpers ───────────────────────────────────────────────────────────
def _pooled_heard(obs, nd):
    seen, pooled = set(), []
    for i in range(nd):
        for p in SB._heard_user_positions(obs, i):
            k = (round(float(p[0]), 3), round(float(p[1]), 3))
            if k not in seen:
                seen.add(k)
                pooled.append([float(p[0]), float(p[1])])
    return pooled


def _episode_key(obs):
    up = obs.get("user_positions", [])
    return (len(up), round(up[0][0], 3), round(up[-1][1], 3)) if up else None


# ── k-means coverage control ─────────────────────────────────────────────────
def kmeans(obs):
    pos = obs.get("positions", [])
    nd = len(pos)
    pooled = _pooled_heard(obs, nd)
    if not pooled:
        return [[0.0, 0.0] for _ in pos]
    key = (_episode_key(obs), len(pooled))
    if _CACHE["key"] != key:
        cent, _ = SB._kmeans_centres(pooled, min(nd, len(pooled)))
        targets = [list(c) for c in cent]
        while len(targets) < nd:
            targets.append(list(targets[len(targets) % len(cent)]))
        _CACHE.update(key=key, targets=targets)
    return SB._fly_to(pos, _CACHE["targets"])


# ── virtual-force / spring placement ─────────────────────────────────────────
F_D0 = float(os.environ.get("SUITE_FORCE_D0", 400.0))      # spring rest length, m
F_K = float(os.environ.get("SUITE_FORCE_K", 0.01))         # spring constant per m
F_WU = float(os.environ.get("SUITE_FORCE_WU", 1.0))        # per-user attraction


def force(obs):
    pos = [[float(p[0]), float(p[1])] for p in obs.get("positions", [])]
    nd = len(pos)
    vels = []
    for i, p in enumerate(pos):
        fx = fy = 0.0
        for u in SB._heard_user_positions(obs, i):
            dx, dy = u[0] - p[0], u[1] - p[1]
            d = math.hypot(dx, dy)
            if d > 1e-6:
                s = F_WU * min(1.0, d / 100.0) / d
                fx += s * dx
                fy += s * dy
        for j, q in enumerate(pos):
            if j == i:
                continue
            dx, dy = q[0] - p[0], q[1] - p[1]
            d = math.hypot(dx, dy)
            if 1e-6 < d < 2.0 * F_D0:
                s = F_K * (d - F_D0) / d
                fx += s * dx
                fy += s * dy
        n = math.hypot(fx, fy)
        if n < 0.05:            # dead band: settle instead of jittering
            vels.append([0.0, 0.0])
        else:
            vels.append([fx / n, fy / n])
    return vels


# ── De Liso-style sequential grid relocation on the LP ───────────────────────
RSR_SPACING = float(os.environ.get("SUITE_RSR_SPACING", 200.0))
RSR_SWEEPS = int(os.environ.get("SUITE_RSR_SWEEPS", 2))
RSR_PAD = float(os.environ.get("SUITE_RSR_PAD", 200.0))


def rsr(obs):
    pos = [[float(p[0]), float(p[1])] for p in obs.get("positions", [])]
    nd = len(pos)
    key = _episode_key(obs)
    if key is None:
        return [[0.0, 0.0] for _ in pos]
    if _CACHE["key"] != key:
        up = obs.get("user_positions", [])
        xs = [u[0] for u in up]
        ys = [u[1] for u in up]
        x0, x1 = min(xs) - RSR_PAD, max(xs) + RSR_PAD
        y0, y1 = min(ys) - RSR_PAD, max(ys) + RSR_PAD
        nx = max(2, int((x1 - x0) / RSR_SPACING) + 1)
        ny = max(2, int((y1 - y0) / RSR_SPACING) + 1)
        grid = [[x0 + a * (x1 - x0) / (nx - 1), y0 + b * (y1 - y0) / (ny - 1)]
                for a in range(nx) for b in range(ny)]
        P = [list(p) for p in pos]
        cur = _R(obs, P)
        for _ in range(RSR_SWEEPS):
            for i in range(nd):
                best, bp = cur, P[i]
                for g in grid:
                    Q = [list(q) for q in P]
                    Q[i] = list(g)
                    r = _R(obs, Q)
                    if r > best:
                        best, bp = r, list(g)
                P[i] = bp
                cur = best
        print(f"[rsr] grid {nx}x{ny}, {_stats['solves']} solves, planned reward/step {cur:.1f}", flush=True)
        _CACHE.update(key=key, targets=P)
    return SB._fly_to(pos, _CACHE["targets"])



# ── static lattice deployment (no information at all) ───────────────────────
def lattice(obs):
    pos = obs.get("positions", [])
    nd = len(pos)
    if _CACHE["key"] != "lattice":
        W = float(os.environ.get("OM2G_ARENA", "3800"))
        nx = int(math.ceil(math.sqrt(nd)))
        ny = int(math.ceil(nd / nx))
        targets = [[W * (a + 0.5) / nx, W * (b + 0.5) / ny] for b in range(ny) for a in range(nx)][:nd]
        _CACHE.update(key="lattice", targets=targets)
    return SB._fly_to(pos, _CACHE["targets"])


# ── max-coverage sequential greedy on a grid (coverage only, no flow) ────────
MC_SPACING = float(os.environ.get("SUITE_MC_SPACING", 100.0))


def maxcov(obs):
    pos = obs.get("positions", [])
    nd = len(pos)
    noise = obs.get("noise_floor", [])
    pooled = _pooled_heard(obs, nd)
    if not pooled:
        return [[0.0, 0.0] for _ in pos]
    key = ("maxcov", _episode_key(obs), len(pooled))
    if _CACHE["key"] != key:
        xs = [u[0] for u in pooled]
        ys = [u[1] for u in pooled]
        x0, x1 = min(xs) - RSR_PAD, max(xs) + RSR_PAD
        y0, y1 = min(ys) - RSR_PAD, max(ys) + RSR_PAD
        nx = max(2, int((x1 - x0) / MC_SPACING) + 1)
        ny = max(2, int((y1 - y0) / MC_SPACING) + 1)
        grid = [[x0 + a * (x1 - x0) / (nx - 1), y0 + b * (y1 - y0) / (ny - 1)]
                for a in range(nx) for b in range(ny)]
        cap = SB._TAU_CAP_BPS
        nf = SB._nf_for(noise, 0)
        # capacity each grid point offers each user, capped at one conversation
        # The binding link is the user UPLINK (0.33 mW, ~158 m), not the drone
        # downlink: scoring with DRONE_TX_mW let two grid points "cover" every
        # user and sent the other drones to a corner.
        cov = [[min(cap, SB._link_capacity_bps(SB._dist(g, u), M.USER_TX_mW, nf)) for u in pooled] for g in grid]
        best_so_far = [0.0] * len(pooled)
        targets = []
        for _ in range(nd):
            gi = max(range(len(grid)), key=lambda g: sum(max(0.0, c - b) for c, b in zip(cov[g], best_so_far)))
            targets.append(list(grid[gi]))
            best_so_far = [max(b, c) for b, c in zip(best_so_far, cov[gi])]
        _CACHE.update(key=key, targets=targets)
    return SB._fly_to(pos, _CACHE["targets"])


FN = {"gtm": gtm, "kmeans": kmeans, "force": force, "rsr": rsr,
      "lattice": lattice, "maxcov": maxcov}[POL]


def _sel(policy, action_fn, obs_dict, num_drones, env):
    vels = FN(env._last_obs)
    if len(vels) != num_drones:
        raise RuntimeError(f"{POL} returned {len(vels)} velocities, expected {num_drones}")
    _stats["steps"] += 1
    if _stats["steps"] % 250 == 0:
        print(f"[{POL}] steps={_stats['steps']} moved={_stats['moved']} solves={_stats['solves']}", flush=True)
    return {f"drone_{i}": vels[i] for i in range(num_drones)}


E._select_actions = _sel
print(f"[suite] policy={POL} gtm_obj={OBJ} step={STEP} m  d0={F_D0} k={F_K} wu={F_WU}  rsr spacing={RSR_SPACING} sweeps={RSR_SWEEPS}", flush=True)
E.main()
