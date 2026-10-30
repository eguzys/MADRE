#!/usr/bin/env python3
"""Priced and information-matched relay planners.

Runs through the unmodified evaluate_omnet.py harness by wrapping
_select_actions, like evaluate_suite.py. Selected by SUITE_POLICY.

PURPOSE
-------
`relay_explore` (RXP) and the blind relay ladder (RBH/RBG) pool every drone's
heard-user set and act on the union, and are charged a flat 67 B or a position
beacon (51 B). A position beacon carries no user state
(om2g_track.om2g_beacon_on_air_bytes); swarm-wide pooling of heard user positions
is user state, and shipping it costs the uncompressed user block (3799 B at 40
users). The planners below run the same relay algorithm with a charge or an
information channel that matches what they exchange:

  relay_priced    Same algorithm as relay_explore. Only the charge changes: the
                  pooled heard-set is billed as the uncompressed user block.

  relay_moments   Information-matched relay planner, the analogue of
                  `aggregate`. Each drone compresses its own heard set to the
                  first two moments (the same 8-wide symmetric summary a
                  learned message carries) and transmits that. The swarm map
                  is the union of the per-drone reconstructions, so multi-cluster
                  structure survives whenever different drones sit on different
                  clusters. The relay ladder then runs unchanged on that map.
                  Charged as one compressed message.

  relay_1hop      Same planner, but drones exchange heard-sets only within a
                  connected component of the one-hop peer graph, and each
                  component plans on what it collectively knows. Charged the
                  position beacon, which is what it sends. Identical to
                  relay_explore whenever the swarm is connected; differs when the
                  swarm is split.
"""
import math
import os
import sys

sys.path.insert(0, os.getcwd())
import evaluate_omnet as E  # noqa: E402
import signal_baselines as SB  # noqa: E402

POL = os.environ.get("SUITE_POLICY", "relay_priced").strip().lower()
MODE = os.environ.get("OM2G_RELAY_MODE", "blind").strip().lower()
_stats = {"steps": 0, "solves": 0}
_VIEW = {"key": None, "pooled": None, "comps": None}


def _pooled_heard(obs, idxs, nd):
    """Union of the heard-user positions of the drones in idxs."""
    seen, pooled = set(), []
    for i in idxs:
        for p in SB._heard_user_positions(obs, i):
            k = (round(float(p[0]), 3), round(float(p[1]), 3))
            if k not in seen:
                seen.add(k)
                pooled.append([float(p[0]), float(p[1])])
    return pooled


def _ladder_on(obs, pooled, nd):
    """Run the unmodified relay ladder against a given user map."""
    if not pooled:
        return None
    sub = dict(obs)
    sub["user_positions"] = pooled
    sub["user_rssi_valid"] = [[1.0] * len(pooled) for _ in range(nd)]
    _stats["solves"] += 1
    return SB.relay_ladder_velocities(sub, mode=MODE)


def _search(obs, drone_pos, pooled, nd):
    """relay_explore's fan-out when fewer than two clusters are known."""
    if pooled:
        cx = sum(p[0] for p in pooled) / len(pooled)
        cy = sum(p[1] for p in pooled) / len(pooled)
    else:
        cx = sum(p[0] for p in drone_pos) / nd
        cy = sum(p[1] for p in drone_pos) / nd
    vels = []
    for i, dp in enumerate(drone_pos):
        ang = 2.0 * math.pi * i / max(1, nd)
        dx = (dp[0] - cx) + 200.0 * math.cos(ang)
        dy = (dp[1] - cy) + 200.0 * math.sin(ang)
        nrm = math.hypot(dx, dy)
        vels.append([0.0, 0.0] if nrm < 1e-6 else [dx / nrm, dy / nrm])
    return vels


# ── relay_priced: identical algorithm, priced pooling ────────────────────────
def relay_fair(obs):
    return SB.relay_explore_velocities(obs, mode=MODE)


def relay_priced(obs):
    return SB.relay_explore_velocities(obs, mode=MODE)


# ── relay_moments: information-matched ───────────────────────────────────────
def relay_moments(obs):
    drone_pos = obs.get("positions", [])
    nd = len(drone_pos)
    if nd == 0:
        return []
    # Each drone compresses its OWN heard set to two moments and sends that.
    pooled, seen = [], set()
    for i in range(nd):
        mine = SB._heard_user_positions(obs, i)
        mom = SB._user_moments(mine)
        if mom is None:
            continue
        pts, _w = SB._moment_quadrature(mom)
        for p in pts:
            k = (round(float(p[0]), 3), round(float(p[1]), 3))
            if k not in seen:
                seen.add(k)
                pooled.append([float(p[0]), float(p[1])])
    groups = SB._single_linkage(pooled, SB._CLUSTER_LINK_M) if pooled else []
    if len(groups) >= 2:
        v = _ladder_on(obs, pooled, nd)
        if v is not None:
            return v
    return _search(obs, drone_pos, pooled, nd)


# ── relay_1hop: pooling does not cross the peer graph for free ───────────────
def _components(drone_pos, noise_floor, nd):
    adj = {i: set(SB._peer_neighbours(drone_pos, noise_floor, i, 1))
           for i in range(nd)}
    for i in range(nd):                      # symmetrise: a shared link is bidi
        for j in adj[i]:
            adj[j].add(i)
    seen, comps = set(), []
    for i in range(nd):
        if i in seen:
            continue
        stack, comp = [i], []
        seen.add(i)
        while stack:
            n = stack.pop()
            comp.append(n)
            for j in adj[n]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        comps.append(sorted(comp))
    return comps


def relay_1hop(obs):
    drone_pos = obs.get("positions", [])
    nd = len(drone_pos)
    if nd == 0:
        return []
    comps = _components(drone_pos, obs.get("noise_floor", []), nd)
    vels = [[0.0, 0.0] for _ in range(nd)]
    for comp in comps:
        pooled = _pooled_heard(obs, comp, nd)
        groups = SB._single_linkage(pooled, SB._CLUSTER_LINK_M) if pooled else []
        if len(groups) >= 2:
            v = _ladder_on(obs, pooled, nd)
        else:
            v = None
        if v is None:
            v = _search(obs, drone_pos, pooled, nd)
        for i in comp:
            vels[i] = v[i]
    return vels


FN = {"relay_priced": relay_priced,
      "relay_fair": relay_fair,
      "relay_moments": relay_moments,
      "relay_1hop": relay_1hop}[POL]


def _sel(policy, action_fn, obs_dict, num_drones, env):
    vels = FN(env._last_obs)
    if len(vels) != num_drones:
        raise RuntimeError(f"{POL} returned {len(vels)} velocities, expected {num_drones}")
    _stats["steps"] += 1
    if _stats["steps"] % 250 == 0:
        print(f"[{POL}] steps={_stats['steps']} solves={_stats['solves']}", flush=True)
    return {f"drone_{i}": vels[i] for i in range(num_drones)}


E._select_actions = _sel
print(f"[relay2] policy={POL} mode={MODE} link_m={SB._CLUSTER_LINK_M}", flush=True)
E.main()
