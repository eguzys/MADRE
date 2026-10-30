"""
greedy_oracle.py — greedy-on-reward baseline for the OMNeT++ track.

Each drone INDEPENDENTLY probes 8 compass directions, evaluates the GLOBAL
position-oracle reward for that tentative move, and commits the best.
No information is exchanged between drones: each drone optimizes the reward
as if only its own move changes (the other drones are frozen at their last
committed positions during one drone's probe). This is the myopic,
uncoordinated baseline — the same decision structure the learned policy gets,
minus the learned joint coordination.

Cost note: the reward is a PURE FUNCTION OF POSITIONS, so all 8 probes per
drone are evaluated in Python with zero sim steps. The sim advances exactly
ONCE per decision tick (to apply the committed velocities), same as the AI.

Run order (same as smoke_test.py): start this FIRST (it binds the ZMQ REP /
connects REQ exactly as smoke_test does), then launch the OMNeT++ exe.

==============================================================================
TWO KEY NAMES BELOW ARE ASSUMED — search for "FIXME_KEY" and adjust them to
match what your bridge actually sends/expects.
==============================================================================
"""

import sys
import os
import math
import argparse

# Reuse the project's reward. Same import smoke_test.py uses.
from max_flow_reward import compute_position_oracle_reward as compute_reward

# --- knobs (parameters, not magic constants) ---------------------------------
NUM_PROBE_DIRS = 8          # compass directions. 8 is conventional; change freely.
# [config_knobs.py] STEP_M is the third "dual-site" knob (drone
# speed): config_knobs.apply_config mutates this module attribute directly
# for IN-PROCESS callers, and ALSO sets $OMNET_STEP_M for a Stage-2 training
# subprocess (a separate `python run.py` process) to pick up here at its own
# import time -- see max_flow_reward.py's DRONE_TX_mW/USER_TX_mW comment for
# the identical reasoning. Unset reproduces the original literal (20.0).
STEP_M = float(os.environ.get("OMNET_STEP_M", 20.0))  # metres/decision-step == m/s at a 1s tick
# -----------------------------------------------------------------------------

# FIXME_KEY: the key under which OMNeT++ sends per-drone positions in `obs`.
#   Expected shape: list of [x, y] (or [x, y, z]) per drone. Run smoke_test once
#   with `print(obs.keys())` and swap this if it's not "drone_positions".
OBS_POS_KEY = "positions"

# FIXME_KEY: the key the bridge expects for the action/velocity dict you send
#   back. RLBridge::parseActionJson reads N actions; confirm the wrapper name.
#   Common candidates: "velocities", "actions", "vel".
ACTION_KEY = "actions"


def _unit_dirs(n):
    """n directions evenly spread around the circle, as (dx, dy) unit vectors."""
    return [(math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n))
            for k in range(n)]


DIRS = _unit_dirs(NUM_PROBE_DIRS)


def _positions(obs):
    """Pull the per-drone position list out of obs. Returns list of [x, y]."""
    pos = obs[OBS_POS_KEY]
    # normalize to mutable [x, y] lists (ignore z for planar movement)
    return [[float(p[0]), float(p[1])] for p in pos]


def _reward_with_positions(obs, positions):
    """Recompute the oracle reward for a hypothetical set of drone positions.

    We mutate a shallow copy of obs's position field only; every other field
    (user_area, noise_floor, ...) is left exactly as the sim reported it.
    """
    probe_obs = dict(obs)
    probe_obs[OBS_POS_KEY] = positions
    return compute_reward(probe_obs)["reward"]


def greedy_velocities(obs):
    """Independent per-drone 8-dir greedy on the global oracle reward.

    For drone i: hold all OTHER drones at their reported positions, try moving
    drone i by STEP_M in each of the 8 directions (plus stay-put), pick the
    direction that maximizes the global reward, emit that as drone i's velocity.

    Independence: drone i's probe does NOT see drones j<i's tentative choices.
    Every drone probes against the same frozen snapshot. No info exchange.
    """
    base_pos = _positions(obs)
    n = len(base_pos)
    vels = []

    for i in range(n):
        best_score = _reward_with_positions(obs, base_pos)  # stay-put baseline
        best_vel = (0.0, 0.0)

        for (dx, dy) in DIRS:
            trial = [p[:] for p in base_pos]          # copy snapshot
            trial[i][0] = base_pos[i][0] + dx * STEP_M
            trial[i][1] = base_pos[i][1] + dy * STEP_M
            score = _reward_with_positions(obs, trial)
            if score > best_score:
                best_score = score
                best_vel = (dx, dy)

        vels.append([best_vel[0], best_vel[1]])

    return vels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-drones", type=int, required=True,
                    help="MUST equal *.numDrones in omnetpp.ini")
    ap.add_argument("--port", type=int, default=5555,
                    help="ZMQ bind/connect string (match smoke_test.py)")
    args = ap.parse_args()

    # --- ZMQ setup: COPY whatever smoke_test.py does here -------------------
    # The exact socket setup (REQ vs REP, bind vs connect, linger, the JSON
    # send/recv helpers) already lives in smoke_test.py. Rather than guess it,
    # mirror those few lines. The loop below is the only new logic.
    import zmq
    import json

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)          # FIXME if smoke uses REQ — match it.
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://127.0.0.1:{args.port}")
    print(f"[greedy] bound tcp://127.0.0.1:{args.port}, waiting for {args.num_drones} drones", flush=True)

    import csv, time
    csv_path = f"greedy_run_{int(time.time())}.csv"
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.writer(csv_f)
    header = ["step", "reward"]
    for d in range(args.num_drones):
        header += [f"d{d}_x", f"d{d}_y"]
    csv_w.writerow(header)
    print(f"[greedy] logging to {csv_path}", flush=True)

    step = 0
    while True:
        msg = sock.recv()
        obs = json.loads(msg.decode("utf-8"))

        if obs.get("done", False):
            reply = {"type": "action",
                     "actions": [[0.0, 0.0]] * args.num_drones,
                     "reward": 0.0}
            sock.send_string(json.dumps(reply))
            print(f"[greedy] episode done at step {step}", flush=True)
            csv_f.close()
            print(f"[greedy] wrote {csv_path}", flush=True)
            break

        vels = greedy_velocities(obs)
        assert len(vels) == args.num_drones, \
            f"produced {len(vels)} vels, need {args.num_drones}"

        reward = compute_reward(obs)["reward"] / 1e6
        row = [step, reward]
        for p in obs.get("positions", []):
            row += [float(p[0]), float(p[1])]
        csv_w.writerow(row)
        csv_f.flush()

        # greedy_velocities returns UNIT direction vectors. The simulator reads
        # the action as a velocity in m/s, so a unit vector would be a 1 m step.
        # OMNeTEnv.step() scales actions by STEP_M; this standalone ZMQ driver
        # bypasses OMNeTEnv, so it applies the same scale itself.
        scaled = [[vx * STEP_M, vy * STEP_M] for vx, vy in vels]
        reply = {"type": "action", "actions": scaled, "reward": reward}
        sock.send_string(json.dumps(reply))

        if step % 50 == 0:
            print(f"[greedy] step {step:4d}  reward {reward:10.2f}", flush=True)
        step += 1


if __name__ == "__main__":
    main()
