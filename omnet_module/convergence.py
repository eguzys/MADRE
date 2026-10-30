#!/usr/bin/env python3
"""convergence.py — split a policy's score into WHERE it converges and HOW FAST.

Usage:
  python3 omnet_module/convergence.py <trace.csv> [<trace2.csv> ...] [--last-frac 0.25]
                                      [--thresh 0.9] [--window 11] [--per-seed]

WHY THIS EXISTS. The episode mean averages two unrelated skills:

  * PLATEAU  — the quality of the configuration the swarm settles into.
  * T_CONV   — how many steps it burns getting there.

Greedy peaks in 1-14 steps of a 250-step episode, so its mean is ~95% steady
state. A learned policy that reaches the SAME configuration but takes 40 steps
to do it loses only ~12% of the episode, while one that arrives instantly at a
worse configuration loses far more -- and the episode mean reports both as
simply "below greedy". They have opposite fixes: transit is an actuation /
step-size problem, plateau is a credit-assignment problem.

DEFINITIONS (deliberately simple, and stated because the numbers mean nothing
without them):
  plateau   = mean reward over the last --last-frac of the episode.
  t_conv    = first step whose SMOOTHED reward reaches --thresh * plateau.
              Smoothing is a centred rolling mean of --window steps; per-step
              reward is noisy (link-level fading), and an unsmoothed crossing
              mostly measures that noise.
              FIRST crossing, not "crosses and never returns": with live
              per-step reward there is almost always a late dip below the
              threshold, and requiring the stricter condition returned NaN on
              every policy including hover. Read t_conv as "when it first got
              there", and transit as the cost actually paid.
  transit   = 1 - episode_mean / plateau. The fraction of the achievable score
              given up while travelling. 0 = arrived instantly.

t_conv is meaningless when a policy never moves: hover's plateau IS its start,
so it "converges" at step 1 by construction. That is expected -- read
hover's row as the no-transit reference, not as a fast policy.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict[int, np.ndarray]:
    per_seed: dict[int, list[tuple[int, float]]] = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            per_seed[int(r["seed"])].append((int(r["step"]), float(r["reward"])))
    out = {}
    for s, rows in per_seed.items():
        rows.sort()
        out[s] = np.asarray([v for _, v in rows], dtype=float)
    return out


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Centred rolling mean, normalised by the count actually in the window.

    A plain np.convolve(x, ones(w)/w, "same") divides the first and last w//2
    entries by w while summing fewer than w terms, so it drags both ends
    toward zero. That is not cosmetic here: it made a CONSTANT hover trace
    appear to "converge" at step 5, i.e. it invented a transit for a policy
    that never moves. Dividing by the same-shaped convolution of ones fixes
    the edges exactly.
    """
    if w <= 1 or x.size < w:
        return x
    num = np.convolve(x, np.ones(w), mode="same")
    den = np.convolve(np.ones_like(x), np.ones(w), mode="same")
    return num / den


def analyse(x: np.ndarray, last_frac: float, thresh: float, window: int) -> tuple:
    n = x.size
    tail = max(1, int(round(n * last_frac)))
    plateau = float(x[-tail:].mean())
    mean = float(x.mean())
    transit = 1.0 - mean / plateau if plateau > 0 else float("nan")

    sm = _smooth(x, window)
    hit = np.flatnonzero(sm >= thresh * plateau)
    t_conv = float(hit[0] + 1) if hit.size else float("nan")  # 1-indexed steps
    return plateau, mean, transit, t_conv


def main() -> None:
    args = [a for a in sys.argv[1:]]
    def _opt(name, default, cast):
        if name in args:
            i = args.index(name)
            v = cast(args[i + 1]); del args[i:i + 2]
            return v
        return default
    last_frac = _opt("--last-frac", 0.25, float)
    thresh = _opt("--thresh", 0.9, float)
    window = _opt("--window", 11, int)
    per_seed = "--per-seed" in args
    if per_seed:
        args.remove("--per-seed")
    if not args:
        raise SystemExit(__doc__)

    print(f"[plateau = mean of last {last_frac:.0%} | t_conv = first step holding "
          f"{thresh:.0%} of plateau, smoothed w={window}]\n")
    print(f"{'policy':<14} {'n':>3} {'plateau':>9} {'ep_mean':>9} "
          f"{'transit':>8} {'t_conv':>7} {'t_conv_med':>10}")
    for p in args:
        path = Path(p)
        data = _load(path)
        rows = [analyse(x, last_frac, thresh, window) for x in data.values()]
        plat = np.mean([r[0] for r in rows])
        mean = np.mean([r[1] for r in rows])
        tr = np.nanmean([r[2] for r in rows])
        tc = np.asarray([r[3] for r in rows], dtype=float)
        name = path.stem.replace("trace_", "")
        print(f"{name:<14} {len(rows):>3} {plat:>9.1f} {mean:>9.1f} "
              f"{tr:>7.1%} {np.nanmean(tc):>7.1f} {np.nanmedian(tc):>10.1f}")
        if per_seed:
            for s, r in sorted(zip(data.keys(), rows)):
                print(f"    seed {s:>3}  plateau={r[0]:>8.1f}  mean={r[1]:>8.1f}  "
                      f"transit={r[2]:>6.1%}  t_conv={r[3]:>6.1f}")


if __name__ == "__main__":
    main()
