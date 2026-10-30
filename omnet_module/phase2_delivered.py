"""PHASE 2 -- does the oracle headroom survive the REAL radio?

Phase 1 asked a MATHS MODEL whether the scripted relay beats greedy. It said
2.12x. But that model computes "how much traffic COULD flow given these
positions." It does not send a packet.

Phase 2 asks the actual radio simulator: fly the drones to those exact
positions, run real 802.11 packets for 1000 s -- DO THE PACKETS ARRIVE?

WHY IT MATTERS: the oracle reward IS the training signal. If the reward says
"P2 is great" but the radio delivers 20% of packets at P2, the reward does not
match the radio and a policy would optimise the wrong quantity.

PASS CONDITION (set before any data):
  P2 delivers substantially more real packets than P1, AND P2's absolute
  delivery is high (>70%). If oracle says P2 >> P1 but delivered% disagrees,
  the reward does not match the radio at this geometry.

The JAMMER CRITERION rides along here -- it has NEVER been tested against a
criterion (S6.4). Phase 1 ran no jammer at all. The post-jammer window gives it
to us free.

No learned model is loaded anywhere in this script.
"""
import argparse, json, os, random, re, sys, tempfile, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# OMNET_LOG_STDOUT is read at IMPORT time in omnet_env (module constant), so it
# must be set BEFORE the import or the env uses its own tempfile and our
# TRAFFIC:/DELIVERED: parse finds nothing.
LOG_PATH = os.environ.setdefault(
    "OMNET_LOG_STDOUT", os.path.join(tempfile.gettempdir(), f"p2_{os.getpid()}.log"))

from omnet_env import OMNeTEnv
from greedy_oracle import greedy_velocities
from task_validation import (get_omnet_users, assert_users_match_omnet,
                             spawn_staging, probe_greedy, probe_scripted_relay,
                             score, R_U, R_D)

MAX_SPEED = 14.0          # m/s -- must match *.host[*].mobility.maxSpeed
ARRIVE_M  = 5.0           # within this of target => hold
FLY_STEPS = 200           # excluded from every window: this is transit time
PRE_JAM   = (200, 500)    # steady state, jammer OFF
POST_JAM  = (700, 1000)   # steady state, jammer ON (fires at step 500)


def fly_to(pos, target):
    """Velocity toward target, clipped to MAX_SPEED. Zero once arrived.

    NOT a teleport. Teleporting to the target would dodge the flight transient
    and measure a world the policy can never actually reach.
    """
    d = np.asarray(target, float) - np.asarray(pos, float)
    mag = float(np.linalg.norm(d))
    if mag < ARRIVE_M:
        return np.zeros(2, dtype=np.float32)
    return (d / mag * MAX_SPEED).astype(np.float32)


def parse_log(path):
    """TRAFFIC:t,uid,dst,bytes  /  DELIVERED:t,uid,bytes,ttl=N  (printf, S13.4)."""
    sent, deliv, ttls = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("TRAFFIC:"):
                sent.append(float(line.split(":", 1)[1].split(",")[0]))
            elif line.startswith("DELIVERED:"):
                body = line.split(":", 1)[1]
                deliv.append(float(body.split(",")[0]))
                m = re.search(r"ttl=(-?\d+)", body)
                if m:
                    ttls.append(int(m.group(1)))
    return np.asarray(sent), np.asarray(deliv), np.asarray(ttls)


def window(ts, lo, hi):
    return int(((ts >= lo) & (ts < hi)).sum())


def run_probe(seed, targets, steps, tag, expected_users):
    """Fly to `targets`, hold, run the real radio. Return delivery stats.

    `expected_users` is the real layout get_omnet_users() already fetched for
    this seed (used to compute `targets`). This launch is a SECOND, INDEPENDENT
    OMNeT process for the same seed -- its own first observation must place
    the SAME users, or `targets` was optimised against a world this episode
    never simulates (the case the AGREEMENT ASSERT exists to catch).
    """
    open(LOG_PATH, "w").close()          # truncate: one probe, one log
    env = OMNeTEnv(worker_index=0)
    env.reset(seed=seed)
    sim_users = [list(p) for p in env._last_obs.get("user_positions", [])]
    assert_users_match_omnet(expected_users, sim_users, seed=seed)
    t0 = time.time()
    for step in range(steps):
        pos = env._last_obs.get("positions", [])
        act = {f"drone_{i}": fly_to(pos[i], targets[i]) for i in range(len(pos))}
        env.step(act)
        if step % 200 == 0:
            print(f"    [{tag} seed {seed}] step {step:4d}  {time.time()-t0:5.1f}s",
                  flush=True)
    env.close()

    sent, deliv, ttls = parse_log(LOG_PATH)
    if len(sent) == 0:
        print(f"    !! [{tag} seed {seed}] NO TRAFFIC: lines in {LOG_PATH} -- "
              f"the sim ran but sent nothing, or the log is not being captured.",
              flush=True)
    out = {"tag": tag, "seed": seed, "sent_total": len(sent), "deliv_total": len(deliv)}
    for name, (lo, hi) in (("pre_jam", PRE_JAM), ("post_jam", POST_JAM)):
        s, d = window(sent, lo, hi), window(deliv, lo, hi)
        out[f"{name}_sent"] = s
        out[f"{name}_deliv"] = d
        out[f"{name}_pct"] = (100.0 * d / s) if s else 0.0
    # ttl SPREAD is the multi-hop evidence. A CONSTANT ttl was the 100m-arena
    # artefact (S3.3). At L=1700 expect a spread.
    out["ttl_mean"] = float(ttls.mean()) if len(ttls) else None
    out["ttl_std"]  = float(ttls.std())  if len(ttls) else None
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--users", type=int, default=17)
    p.add_argument("--drones", type=int, default=5)
    p.add_argument("--arena", type=float, default=1700.0)
    p.add_argument("--max-clusters", type=int, default=2)
    p.add_argument("--out", type=str, required=True)
    a = p.parse_args()

    # REAL users OMNeT places for this seed -- NOT a resampled python layout.
    # A python-side random.Random(...) layout would not match the users OMNeT
    # simulates, so p1/p2 are optimised against the real layout. See
    # task_validation.get_omnet_users().
    users, areas, pairs, actual_seed = get_omnet_users(
        a.seed, a.users, a.drones, a.arena, a.max_clusters)
    assert_users_match_omnet(users, users, seed=actual_seed)

    # Probe-placement randomness only (drone staging jitter) -- NEVER user
    # layout. A separate, clearly-named stream so nobody reuses it for users.
    rng = random.Random(hash(("probe", int(a.arena), a.seed)) & 0xFFFFFFFF)

    start = spawn_staging(rng, a.drones, users, areas)
    obs = {"positions": [q[:] for q in start], "user_positions": users,
           "user_area": areas, "sender_pairs": pairs}
    p1 = probe_greedy(obs, a.drones, a.arena)          # converged greedy positions
    plan = probe_scripted_relay(users, a.drones, a.arena)
    p2 = plan.positions

    oracle_p1 = score(p1, users, areas, pairs)
    oracle_p2 = score(p2, users, areas, pairs)
    print(f"seed {a.seed}  ORACLE  P1 {oracle_p1:.1f}  P2 {oracle_p2:.1f}  "
          f"ratio {oracle_p2/oracle_p1 if oracle_p1 else float('inf'):.2f}x  "
          f"relays={plan.n_relay}  infeasible={plan.infeasible}", flush=True)

    r1 = run_probe(a.seed, p1, a.steps, "P1", expected_users=users)
    r2 = run_probe(a.seed, p2, a.steps, "P2", expected_users=users)

    res = {"seed": a.seed, "oracle_P1": oracle_p1, "oracle_P2": oracle_p2,
           "oracle_ratio": (oracle_p2 / oracle_p1) if oracle_p1 else None,
           "n_relay": plan.n_relay, "infeasible": bool(plan.infeasible),
           "P1": r1, "P2": r2}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)

    print(f"\nseed {a.seed}  DELIVERED%  (pre-jammer / post-jammer)")
    print(f"  P1  {r1['pre_jam_pct']:5.1f}% / {r1['post_jam_pct']:5.1f}%   "
          f"ttl {r1['ttl_mean']} +- {r1['ttl_std']}")
    print(f"  P2  {r2['pre_jam_pct']:5.1f}% / {r2['post_jam_pct']:5.1f}%   "
          f"ttl {r2['ttl_mean']} +- {r2['ttl_std']}")
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
