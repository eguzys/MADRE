#!/usr/bin/env python3
"""C4 — the offline planner-information ladder.

╔══════════════════════════════════════════════════════════════════════════╗
║  OM2G TRACK.  Runs BEFORE any GPU-hour: upper-bounds the compression     ║
║  ladder's achievable dynamic range and adjudicates e2e-vs-frozen         ║
║  disagreements without a relaunch.                                       ║
╚══════════════════════════════════════════════════════════════════════════╝

Question: how much team performance does each TIER of inter-drone
information unlock, for a competent non-RL placer? The tiers mirror what a
beacon's CONTENT can carry (the kinematic core is free in all arms):

  t0 spawn      no planning at all -- score the cluster-spawn positions.
  t1 geom       cluster GEOMETRY only (what accumulating position beacons +
                own flight reveal): centroids known, demand masses and
                pairings unknown (assumed uniform).
  t2 hist       t1 + the per-cluster demand HISTOGRAM: true per-cluster
                user masses and the true cluster-level cross-traffic
                matrix -- i.e. what O(#clusters) floats of message can
                carry.  This is the tier a small-L codec could encode.
  t3 full       the true per-user world (positions + pairings) -- what the
                uncompressed arm's 680 floats carry.

Method, per layout (exact offline replay -- users are static, the oracle
reward is a pure function of drone positions; env vars must be bound by the
caller exactly as a training cell binds them, see the sbatch wrapper):
  1. dump the real layout from the live simulator (OM2GEnv reset).
  2. for each tier, joint hill-climb the 8 drone positions against the
     oracle reward computed on that tier's REDACTED world model.
  3. score every tier's final placement on the TRUE world.

The (t1 -> t3) gap upper-bounds what ANY user-content message adds over
pure geometry for this planner class; t2 ~= t3 means a histogram-sized
message (~#clusters floats) suffices and the compression ladder's left arm
can only live below L ~= #clusters.

Output: --out CSV (world, seed, tier, score_true, score_redacted) plus a
paired summary. An empty layout dump or a failed LP must CRASH, not pass.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "omnet_module"))

import om2g_track as T  # noqa: E402

T.require_om2g("om2g_info_ladder.py")

if not os.environ.get("OMNET_EXTRA_ARGS"):
    raise SystemExit(
        "FATAL: $OMNET_EXTRA_ARGS unset -- the world is not bound. Run "
        "eval \"$(python om2g/om2g_config.py --arm nocontrol --shell)\" "
        "under the SAME world exports as the training cells (see "
        "arch_out/cells/*.sh); an unbound world scores a different task.")


def dump_layouts(n_seeds: int, seed_start: int):
    from om2g_env import OM2GEnv
    env = OM2GEnv(worker_index=0, encoder_mode="learned", arm="nocontrol")
    layouts = []
    for s in range(seed_start, seed_start + n_seeds):
        env.reset(seed=s)
        o = env._last_obs
        lay = dict(
            seed=s,
            positions=[list(map(float, p)) for p in o["positions"]],
            user_positions=[list(map(float, p)) for p in o["user_positions"]],
            noise_floor=[float(v) for v in o["noise_floor"]],
            sender_pairs=[[int(p[0]), int(p[1])]
                          for p in (o.get("sender_pairs") or [])],
            arena_width=float(o.get("arena_width", T.OM2G_ARENA_M)),
            arena_height=float(o.get("arena_height", T.OM2G_ARENA_M)),
        )
        assert lay["user_positions"] and lay["positions"], (
            f"seed {s}: empty layout -- an empty dump is a broken dump")
        assert lay["sender_pairs"], f"seed {s}: no sender_pairs in obs"
        layouts.append(lay)
        print(f"  layout seed {s}: {len(lay['user_positions'])} users, "
              f"{len(lay['positions'])} drones, "
              f"{len(lay['sender_pairs'])} pairs", flush=True)
    env.close()
    return layouts


# ── world models per tier ───────────────────────────────────────────────────

def _clusters(user_pos):
    from omnet_env import _user_cluster_labels
    labels = _user_cluster_labels([tuple(p) for p in user_pos])
    labels = np.asarray(labels)
    cents = {}
    for c in sorted(set(labels.tolist())):
        pts = np.asarray(user_pos)[labels == c]
        cents[c] = pts.mean(axis=0)
    return labels, cents


def _pseudo_world(lay, masses, pair_matrix):
    """Users collapsed onto their cluster centroids; per-cluster user counts
    from `masses`; pairs realised deterministically from `pair_matrix`
    (counts of conversations cluster_a -> cluster_b)."""
    labels, cents = _clusters(lay["user_positions"])
    cl = sorted(cents)
    n_users = len(lay["user_positions"])
    # user slots per cluster, largest-remainder rounding to keep n fixed
    total = sum(masses[c] for c in cl)
    exact = {c: masses[c] * n_users / max(total, 1e-9) for c in cl}
    counts = {c: int(exact[c]) for c in cl}
    rem = n_users - sum(counts.values())
    for c in sorted(cl, key=lambda c: exact[c] - counts[c], reverse=True):
        if rem <= 0:
            break
        counts[c] += 1
        rem -= 1
    upos, owner = [], []
    for c in cl:
        for k in range(counts[c]):
            # tiny deterministic jitter so users are not literally identical
            ang = 2 * np.pi * k / max(counts[c], 1)
            upos.append([float(cents[c][0] + 10.0 * np.cos(ang)),
                         float(cents[c][1] + 10.0 * np.sin(ang))])
            owner.append(c)
    owner = np.asarray(owner)
    # realise pair_matrix as sender pairs over the pseudo users
    free = {c: [i for i in range(len(upos)) if owner[i] == c] for c in cl}
    pairs = []
    for a in cl:
        for b in cl:
            want = int(round(pair_matrix.get((a, b), 0.0)))
            for _ in range(want):
                if not free[a] or not free[b]:
                    break
                sa = free[a].pop()
                if a == b and free[b] and free[b][-1] == sa:
                    # avoid self-pairing on the last element
                    if len(free[b]) < 2:
                        free[a].append(sa)
                        break
                sb = free[b].pop() if free[b] else None
                if sb is None or sb == sa:
                    free[a].append(sa)
                    break
                pairs.append([sa, sb])
    w = dict(lay)
    w["user_positions"] = upos
    w["sender_pairs"] = pairs
    return w


def tier_world(lay, tier):
    if tier in ("t0", "t3"):
        return lay
    labels, cents = _clusters(lay["user_positions"])
    cl = sorted(cents)
    if tier == "t2p":
        # TRUE per-user positions, but demand known only at CLUSTER level:
        # pairings re-drawn (seeded, deterministic per layout) to match the
        # true cluster-level pair matrix while scrambling WHICH user calls
        # WHICH. t2p-t2 isolates the value of fine-grained WHERE;
        # t3-t2p isolates the value of exact WHO-CALLS-WHOM (traffic
        # structure a cluster histogram destroys).
        rng = np.random.default_rng(1000003 + int(lay["seed"]))
        users_of = {c: [i for i in range(len(lay["user_positions"]))
                        if labels[i] == c] for c in cl}
        free_senders = {c: list(users_of[c]) for c in cl}
        for c in cl:
            rng.shuffle(free_senders[c])
        pairs = []
        for sa, sb in lay["sender_pairs"]:
            a, b = int(labels[sa]), int(labels[sb])
            if not free_senders[a]:
                continue
            na = free_senders[a].pop()
            cand = [u for u in users_of[b] if u != na]
            if not cand:
                continue
            pairs.append([na, int(rng.choice(cand))])
        w = dict(lay)
        w["sender_pairs"] = pairs
        return w
    if tier == "t1":
        masses = {c: 1.0 for c in cl}
        n_pairs = max(1, len(lay["sender_pairs"]))
        pm = {}
        # uniform cross-heavy traffic (the world's pCross is not message
        # content -- it is a public design parameter, so t1 may assume it)
        pc = T.OM2G_P_CROSS
        for a in cl:
            share = n_pairs / len(cl)
            pm[(a, a)] = share * (1.0 - pc)
            for b in cl:
                if b != a:
                    pm[(a, b)] = share * pc / (len(cl) - 1)
        return _pseudo_world(lay, masses, pm)
    if tier == "t2":
        masses = {c: float((labels == c).sum()) for c in cl}
        pm = {}
        for sa, sb in lay["sender_pairs"]:
            a, b = int(labels[sa]), int(labels[sb])
            pm[(a, b)] = pm.get((a, b), 0) + 1
        return _pseudo_world(lay, masses, pm)
    raise ValueError(tier)


# ── the planner ─────────────────────────────────────────────────────────────

def score(positions, world, tag, strict=True):
    """Oracle score of a placement.

    strict=False (hill-climb CANDIDATES): a rare HiGHS kSolveError on one of
    thousands of probe placements returns -inf -- an unscorable candidate
    cannot be chosen, and one numerically hard probe must not stop a
    100-layout job (an OracleLPSolveError in a worker would kill the pool).
    strict=True (FINAL results, t0): one retry with a 1 mm nudge, then raise,
    so a final number is always a real solve.
    """
    from max_flow_reward import (compute_oracle_max_flow_reward,
                                 OracleLPSolveError)
    obs = dict(world)
    obs["positions"] = [list(map(float, p)) for p in positions]
    try:
        return float(compute_oracle_max_flow_reward(obs, cache_tag=tag))
    except OracleLPSolveError:
        if not strict:
            return float("-inf")
        obs["positions"] = [[p[0] + 0.001, p[1] + 0.001]
                            for p in obs["positions"]]
        return float(compute_oracle_max_flow_reward(obs, cache_tag=tag))


def hill_climb(lay, world, tag, rounds=25):
    """Joint coordinate hill-climb of all drones against `world`'s oracle."""
    arena_w, arena_h = lay["arena_width"], lay["arena_height"]
    labels, cents = _clusters(world["user_positions"])
    cl = sorted(cents)
    pos = []
    n = len(lay["positions"])
    for i in range(n):  # round-robin init on the tier's visible centroids
        c = cents[cl[i % len(cl)]]
        pos.append([float(c[0] + 40.0 * ((i // len(cl)) % 3 - 1)),
                    float(c[1] + 40.0 * ((i // (3 * len(cl))) % 3 - 1))])
    best = score(pos, world, tag, strict=False)
    if best == float("-inf"):
        pos = [[p[0] + 1.0, p[1] + 1.0] for p in pos]
        best = score(pos, world, tag, strict=False)
    dirs = [(np.cos(a), np.sin(a)) for a in np.linspace(0, 2 * np.pi, 8,
                                                        endpoint=False)]
    for r in range(rounds):
        improved = False
        steps = (120.0, 360.0) if r % 3 else (120.0, 360.0, 800.0)
        for i in range(n):
            cur = pos[i]
            cands = [[cur[0] + d[0] * st, cur[1] + d[1] * st]
                     for st in steps for d in dirs]
            for cx, cy in cands:
                if not (0 <= cx <= arena_w and 0 <= cy <= arena_h):
                    continue
                trial = pos[:i] + [[cx, cy]] + pos[i + 1:]
                v = score(trial, world, tag, strict=False)
                if v > best + 1e-9:
                    best, pos, improved = v, trial, True
        if not improved:
            break
    return pos, best


def run_layout(args):
    lay, tiers = args
    out = []
    for tier in tiers:
        tag = f"{lay['seed']}_{tier}"
        if tier == "t0":
            pos = lay["positions"]
            red = float("nan")
        else:
            w = tier_world(lay, tier)
            pos, red = hill_climb(lay, w, tag)
        true_v = score(pos, lay, tag + "_true")
        out.append(dict(seed=lay["seed"], tier=tier,
                        score_true=true_v, score_redacted=red))
        print(f"  seed {lay['seed']} {tier}: true={true_v:.1f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--world-tag", required=True,
                    help="label for the CSV (e.g. 1000/1200); the WORLD "
                    "itself comes from the environment exports")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=int(os.environ.get(
        "SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--tiers", default="t0,t1,t2,t3",
                    help="comma list from t0,t1,t2,t2p,t3 (t2p = true "
                    "positions + cluster-level demand only)")
    a = ap.parse_args()

    print(f"── info ladder: world={a.world_tag} seeds={a.seeds} "
          f"minsep={os.environ.get('OM2G_MINSEP')} "
          f"arena={os.environ.get('OM2G_ARENA')}", flush=True)
    layouts = dump_layouts(a.seeds, a.seed_start)

    tiers = tuple(a.tiers.split(","))
    import multiprocessing as mp
    with mp.Pool(a.procs) as pool:
        rows = [r for chunk in pool.imap_unordered(
            run_layout, [(lay, tiers) for lay in layouts])
            for r in chunk]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["world", "seed", "tier",
                                           "score_true", "score_redacted"])
        w.writeheader()
        for r in rows:
            w.writerow(dict(world=a.world_tag, **r))

    # paired summary
    by = {}
    for r in rows:
        by.setdefault(r["tier"], {})[r["seed"]] = r["score_true"]
    seeds = sorted(set.intersection(*(set(v) for v in by.values())))
    assert len(seeds) == a.seeds, (
        f"only {len(seeds)}/{a.seeds} seeds completed all tiers -- a "
        f"partial ladder is not a result")
    print(f"\n══ INFO LADDER {a.world_tag} (n={len(seeds)}, TRUE-world "
          f"scores, paired) ══")
    for t in tiers:
        v = np.array([by[t][s] for s in seeds])
        print(f"  {t}: {v.mean():9.1f} ± {v.std(ddof=1)/np.sqrt(len(v)):.1f}")
    _all_gaps = (("t3", "t1"), ("t3", "t2"), ("t2", "t1"), ("t1", "t0"),
                 ("t2p", "t2"), ("t3", "t2p"))
    for hi, lo in [(h, l) for h, l in _all_gaps if h in by and l in by]:
        d = np.array([by[hi][s] - by[lo][s] for s in seeds])
        t_stat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
        print(f"  {hi}-{lo}: {d.mean():+9.1f} ± "
              f"{d.std(ddof=1)/np.sqrt(len(d)):.1f}  (paired t={t_stat:.2f})")
    print("══ interpretation: t3-t1 bounds ALL message user-content value; "
          "t2~=t3 means ~#clusters floats suffice ══", flush=True)


if __name__ == "__main__":
    main()
