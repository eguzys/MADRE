#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np

from omnet_env import OMNeTEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    env = OMNeTEnv(worker_index=0)
    obs, _infos = env.reset()

    counts: list[int] = []

    for step in range(args.steps):
        actions = {
            a: rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
            for a in env.agents
        }

        obs, _rewards, terms, truncs, _infos = env.step(actions)

        raw = env._last_obs
        if raw is None:
            continue

        urv = raw.get("user_rssi_valid", [])
        for drone_row in urv:
            counts.append(int(sum(1 for v in drone_row if v)))

        if any(terms.values()) or any(truncs.values()):
            break

    env.close()

    if not counts:
        print("NO DATA -- env produced no user_rssi_valid rows. Check the run.")
        return

    c = np.asarray(counts, dtype=np.int32)

    print()
    print("=" * 58)
    print(f"  SENSED SET  (n = {len(c)} drone-steps)")
    print("=" * 58)
    print(f"  mean        {c.mean():.2f}")
    print(f"  p50         {np.percentile(c, 50):.0f}")
    print(f"  p90         {np.percentile(c, 90):.0f}")
    print(f"  max         {c.max():d}")
    print(f"  frac_empty  {(c == 0).mean():.3f}")
    print("=" * 58)
    print()
    print("  2026-07-12 (dead sim): mean 0.16 | frac_empty 0.841")
    print()

    if c.mean() < 0.5:
        print("  >> STILL COLLAPSED. The user encoder has nothing to encode.")
        print("     Next: senderProb 0.2 -> 1.0 (PADRE-Python already runs 1.0)")
        print("     and re-run. See OMNET.md 5.4.")
    elif c.mean() < 2.0:
        print("  >> MARGINAL. Populated but thin. Report and discuss.")
    else:
        print("  >> RECOVERED. Blocker #1 is dead; the encoder question is real.")
    print()


if __name__ == "__main__":
    main()
