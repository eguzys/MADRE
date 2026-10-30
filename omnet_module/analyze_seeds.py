#!/usr/bin/env python3
"""Analyse multi-training-seed OMNeT results correctly.

THE PROBLEM THIS EXISTS FOR
---------------------------
A single training run scored on 100 evaluation seeds, with a paired t-test
over those 100, measures WORLD seeds: they measure how one trained network copes with
different user layouts. They say nothing about whether a re-run with a
different initialisation would give the same network.

An architecture claim needs the TRAINING RUN as the unit of analysis. With 3
runs per arm the sample size is n=3, not n=100, and quoting t=2.91 on
99 degrees of freedom overstates the evidence by a wide margin.

WHAT THIS REPORTS
-----------------
Three views, deliberately all of them, because each answers a different
question and the gap between them IS the finding:

1. per-run means  -- shows the training-seed spread explicitly. If two arms
                     differ by less than their own seed spread, no test will
                     rescue the claim.
2. arm mean +- SE over RUNS (n=3)
                  -- the conservative claim. Low power by construction; a
                     non-significant result here is "underpowered", NOT
                     "no effect".
3. paired by WORLD seed on training-seed-averaged scores
                  -- the powered view. Averaging the 3 runs per arm first
                     removes most training-seed noise, then pairing on the
                     shared world seeds exploits the fact that every arm saw
                     the identical 100 layouts. Report alongside (2), never
                     instead of it.

Plus a VARIANCE DECOMPOSITION: how much of the spread is training-seed vs
world-seed. That single number tells you whether n=1-training-seed results
were ever trustworthy.

USAGE
  python analyze_seeds.py --glob 'out/frozen_eval/*_n100.csv' \
      --pattern 'frozen_(?P<arch>st|deepsets|ffn)_ts(?P<seed>\\d+)'
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
from collections import defaultdict

STEPS = 250.0


def load(p):
    d = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                d[int(r["seed"])] = float(r["episode_reward"]) / STEPS
            except (TypeError, ValueError):
                pass
    return d


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def sd(xs):
    xs = list(xs)
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def sem(xs):
    xs = list(xs)
    return sd(xs) / math.sqrt(len(xs))


def paired(a, b):
    ks = sorted(set(a) & set(b))
    if len(ks) < 2:
        return None
    df = [a[k] - b[k] for k in ks]
    m = mean(df)
    s = sem(df)
    return m, s, (m / s if s else float("nan")), sum(1 for x in df if x > 0), len(df)


def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log1p(-x) * b - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front / a * _betacf(a, b, x)
    return 1.0 - math.exp(math.log1p(-x) * b + math.log(x) * a - lbeta) / b \
        * _betacf(b, a, 1.0 - x)


def t_sf2(t, df):
    """Two-sided tail probability of Student-t. No scipy on the cluster env."""
    if df <= 0 or not math.isfinite(t):
        return float("nan")
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def welch(xs, ys):
    """Two-sample Welch t on RUN means. n is tiny (3 v 3) -- that is the point.

    Returns the Welch-SATTERTHWAITE df, not the pooled n1+n2-2. With unequal
    run-level variances (deepsets sd 23.9 vs ffn sd 9.0 at n=3) the two differ
    a lot: pooled df=4 gives p=0.022 where the correct df=2.55 gives p=0.046 --
    a factor of two, straddling 0.05. The previous version hardcoded a 2.78
    cutoff labelled "t(0.05,2)"; 2.78 is in fact t(0.05,4), so the comment and
    the constant disagreed and the test was anticonservative.
    """
    if len(xs) < 2 or len(ys) < 2:
        return None
    nx, ny = len(xs), len(ys)
    vx, vy = sd(xs) ** 2 / nx, sd(ys) ** 2 / ny
    d = mean(xs) - mean(ys)
    se = math.sqrt(vx + vy)
    if se == 0:
        return d, se, float("nan"), float("nan"), float("nan")
    df = (vx + vy) ** 2 / (vx ** 2 / (nx - 1) + vy ** 2 / (ny - 1))
    t = d / se
    return d, se, t, df, t_sf2(t, df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="glob over per-run eval CSVs")
    ap.add_argument("--pattern", required=True,
                    help="regex with named groups 'arch' and 'seed', matched "
                         "against each CSV's basename")
    ap.add_argument("--baseline", default=None,
                    help="optional CSV to normalise against (e.g. hover)")
    ap.add_argument("--ceiling", default=None,
                    help="optional CSV for the gap denominator (e.g. greedy)")
    a = ap.parse_args()

    rx = re.compile(a.pattern)
    runs = defaultdict(dict)          # arch -> seed -> {world_seed: reward}
    for p in sorted(glob.glob(a.glob)):
        m = rx.search(os.path.basename(p))
        if not m:
            continue
        d = load(p)
        if d:
            runs[m.group("arch")][m.group("seed")] = d
    if not runs:
        raise SystemExit(f"no CSV matched {a.glob!r} with {a.pattern!r}")

    H = load(a.baseline) if a.baseline else None
    G = load(a.ceiling) if a.ceiling else None

    def gap(v):
        if not (H and G):
            return float("nan")
        return (mean(v.values()) - mean(H.values())) / (
            mean(G.values()) - mean(H.values()))

    print("=" * 78)
    print("1. PER-RUN MEANS  (reward/step) -- the training-seed spread")
    print("=" * 78)
    run_means = {}
    for arch in sorted(runs):
        rm = []
        for s in sorted(runs[arch]):
            v = runs[arch][s]
            mu = mean(v.values())
            rm.append(mu)
            print(f"  {arch:10} seed {s:>3}  n={len(v):3}  {mu:8.1f}"
                  f"   gap={gap(v):+.3f}" if H and G else
                  f"  {arch:10} seed {s:>3}  n={len(v):3}  {mu:8.1f}")
        run_means[arch] = rm
        if len(rm) > 1:
            print(f"  {arch:10} {'':>8}  spread across seeds: "
                  f"min={min(rm):.1f} max={max(rm):.1f} "
                  f"range={max(rm)-min(rm):.1f} sd={sd(rm):.1f}")
        print()

    print("=" * 78)
    print("2. ARM MEAN +- SE over RUNS  (n = number of training seeds)")
    print("   The CONSERVATIVE claim. Non-significant here means UNDERPOWERED,")
    print("   not 'no effect' -- with n=3 the test can only see large effects.")
    print("=" * 78)
    for arch in sorted(run_means):
        rm = run_means[arch]
        print(f"  {arch:10} n={len(rm)}  {mean(rm):8.1f} +- {sem(rm):5.1f}")
    archs = sorted(run_means)
    print()
    for i in range(len(archs)):
        for j in range(i + 1, len(archs)):
            r = welch(run_means[archs[i]], run_means[archs[j]])
            if r:
                d, se, t, df, p = r
                sig = "SIG " if p < 0.05 else "n.s."
                print(f"  {archs[i]:>10} - {archs[j]:<10} {d:+8.1f} +- {se:5.1f}"
                      f"  t={t:6.2f}  {sig}  (Welch df={df:.2f}, p={p:.4f})")

    print()
    print("=" * 78)
    print("3. PAIRED BY WORLD SEED on training-seed-AVERAGED scores")
    print("   The POWERED view: averaging runs first removes most training-seed")
    print("   noise, then pairing exploits the shared 100 layouts.")
    print("   Report ALONGSIDE (2), never instead of it.")
    print("=" * 78)
    avg = {}
    for arch in archs:
        ws = set.intersection(*[set(d) for d in runs[arch].values()])
        avg[arch] = {k: mean(runs[arch][s][k] for s in runs[arch]) for k in ws}
    for i in range(len(archs)):
        for j in range(i + 1, len(archs)):
            r = paired(avg[archs[i]], avg[archs[j]])
            if r:
                m, s, t, w, n = r
                sig = "SIG " if abs(t) >= 1.98 else "n.s."
                print(f"  {archs[i]:>10} - {archs[j]:<10} {m:+8.1f} +- {s:5.1f}"
                      f"  t={t:6.2f}  {sig}  {w}/{n}")

    print()
    print("=" * 78)
    print("4. VARIANCE DECOMPOSITION -- was n=1 training seed ever safe?")
    print("=" * 78)
    for arch in archs:
        seeds = sorted(runs[arch])
        if len(seeds) < 2:
            print(f"  {arch:10} only {len(seeds)} run(s); cannot decompose")
            continue
        ws = sorted(set.intersection(*[set(runs[arch][s]) for s in seeds]))
        # between-run: sd of run means. within-run: mean over world seeds of the
        # sd across runs at that same world seed.
        between = sd([mean(runs[arch][s][k] for k in ws) for s in seeds])
        within = mean(sd([runs[arch][s][k] for s in seeds]) for k in ws)
        world = sd([mean(runs[arch][s][k] for s in seeds) for k in ws])
        print(f"  {arch:10} training-seed sd of run means = {between:6.1f}")
        print(f"  {arch:10} world-seed sd (run-averaged)  = {world:6.1f}")
        print(f"  {arch:10} per-world-seed sd ACROSS runs = {within:6.1f}")
        print(f"  {arch:10} -> a single run's mean carries +-{between:.1f} of "
              f"training-seed uncertainty that n=100 world seeds CANNOT see.")
        print()


if __name__ == "__main__":
    main()
