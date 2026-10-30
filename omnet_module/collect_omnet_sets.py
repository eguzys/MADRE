"""Collect user-set + event-set tokens at the FROZEN arena, under greedy.

Frozen config (OMNET.md S8): L=1700, max_clusters=2, cluster spawn, N=5,
senderProb=1.0, pCross=0.5.

POLICY = greedy (greedy_oracle.greedy_velocities -- the SAME one task_validation
uses; NOTE signal_baselines.py has a DIFFERENT function of the same name at the
same line number. Do not import that one.)

Caveat, carried from PADRE: the encoder is fit on greedy-visited states,
and a trained ST/DS spreads drones out into the gaps. Distribution shift is
MEASURED post-hoc (PADRE's check found only `angle` degrades, worst for the
winning archs -- i.e. the comparison stays CONSERVATIVE). Not assumed away.

Episode-aligned split is enforced at TRAIN time via the episode_id column.
A random-row split leaks: step 437 and 438 are near-identical.
"""
import argparse
import os
import time

import numpy as np

from omnet_env import (OMNeTEnv, MAX_N, MAX_USERS_CAP, MAX_EVENTS_CAP,
                       build_raw_user_tokens, build_raw_event_tokens)
from greedy_oracle import greedy_velocities


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-start", type=int, required=True)
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()

    env = OMNeTEnv(worker_index=0)
    U, E, UM, EM, EPID, STEP = [], [], [], [], [], []
    t0 = time.time()

    for ep in range(a.n_episodes):
        seed = a.seed_start + ep
        env.reset(seed=seed)
        for step in range(a.steps):
            oj = env._last_obs
            vels = greedy_velocities(oj)
            act = {f"drone_{i}": np.asarray(vels[i], dtype=np.float32)
                   for i in range(len(vels))}
            env.step(act)
            oj = env._last_obs

            u = build_raw_user_tokens(oj, seed)          # (MAX_N, MAX_USERS_CAP, 13)
            e = build_raw_event_tokens(oj)               # tuple: (tokens, mask)
            e_tok, e_mask = (e if isinstance(e, tuple) else (e, e[..., -1]))
            U.append(np.asarray(u, dtype=np.float32))
            E.append(np.asarray(e_tok, dtype=np.float32))
            # `present` is the last col of the user token; that IS the mask.
            UM.append(np.asarray(u, dtype=np.float32)[..., -1])
            EM.append(np.asarray(e_mask, dtype=np.float32))
            EPID.append(seed)
            STEP.append(step)

            if step % 200 == 0:
                print(f"  ep {ep+1}/{a.n_episodes} (seed {seed})  step {step:4d}  "
                      f"{time.time()-t0:6.1f}s", flush=True)
        print(f"episode {ep+1}/{a.n_episodes} done (seed {seed})  "
              f"{time.time()-t0:6.1f}s  rows={len(U)}", flush=True)

    env.close()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(
        a.out,
        user_tokens=np.stack(U), event_tokens=np.stack(E),
        user_mask=np.stack(UM), event_mask=np.stack(EM),
        episode_id=np.asarray(EPID, dtype=np.int64),
        step=np.asarray(STEP, dtype=np.int64),
    )
    print(f"wrote {a.out}  rows={len(U)}  {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
