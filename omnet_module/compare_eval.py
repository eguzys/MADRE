#!/usr/bin/env python3
"""compare_eval.py — score a trained-policy CSV against the control baselines.

Usage: python3 omnet_module/compare_eval.py <trained.csv> <baseline_dir> [--json out.json]

Reports per-step reward (episode_reward / steps) on the SEEDS THE TWO SHARE, so
a partially-finished baseline cannot silently compare different worlds. Paired
per seed, because between-seed variance dominates here -- greedy ranges 169-1294
/step across seeds while the gap it is being tested for is ~2x, so an unpaired
mean comparison is mostly measuring which seeds landed in each set.

Deliberately NOT a hypothesis test with a p-value: the decision rule is applied
separately. This prints the paired win rate and the effect size and stops.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

POLICIES = ("greedy", "hover", "random")


def _load(path: Path) -> dict:
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            steps = float(r["steps"])
            if steps <= 0:
                raise SystemExit(f"FATAL: {path} seed {r['seed']} has steps={steps}")
            out[int(r["seed"])] = float(r["episode_reward"]) / steps
    return out


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    trained_path, base_dir = Path(sys.argv[1]), Path(sys.argv[2])
    trained = _load(trained_path)
    bases = {}
    for p in POLICIES:
        f = base_dir / f"{p}_n5.csv"
        if f.exists():
            bases[p] = _load(f)

    # --seed-limit N restricts every series to seeds < N. Needed for a
    # like-for-like number against a historical run scored on a smaller seed
    # set: aa732995's "-0.25 of the hover->greedy gap" is on seeds 0-9, and the
    # baselines shift with the seed set (hover 361.7 on 0-9 vs 416.2 on 0-19),
    # so comparing a 20-seed result against it directly would be comparing two
    # different worlds' worth of difficulty.
    if "--seed-limit" in sys.argv:
        lim = int(sys.argv[sys.argv.index("--seed-limit") + 1])
        trained = {s: v for s, v in trained.items() if s < lim}
        bases = {p: {s: v for s, v in d.items() if s < lim} for p, d in bases.items()}
        print(f"[restricted to seeds < {lim}]")

    if not trained:
        raise SystemExit(f"FATAL: {trained_path} has no episodes")

    print(f"\ntrained: {trained_path}  ({len(trained)} episodes)")
    print(f"baselines: {base_dir}")
    for p, b in bases.items():
        print(f"   {p:<7} {len(b)} episodes")

    print(f"\n{'policy':<10} {'n':>4} {'mean':>9} {'median':>9} {'min':>8} {'max':>9}")
    rows = [("trained", trained)] + [(p, bases[p]) for p in POLICIES if p in bases]
    for name, d in rows:
        v = np.array(list(d.values()))
        print(f"{name:<10} {len(v):>4} {v.mean():>9.1f} {np.median(v):>9.1f} "
              f"{v.min():>8.1f} {v.max():>9.1f}")

    print(f"\npaired against trained, on shared seeds only:")
    print(f"{'vs':<10} {'n':>4} {'trained':>9} {'baseline':>9} {'ratio':>7} "
          f"{'wins':>9} {'median Δ':>10}")
    summary = {"trained_csv": str(trained_path), "n_trained": len(trained),
               "per_step": {}, "paired": {}}
    summary["per_step"]["trained"] = float(np.mean(list(trained.values())))
    for p in POLICIES:
        if p not in bases:
            continue
        shared = sorted(set(trained) & set(bases[p]))
        if not shared:
            print(f"{p:<10} {'--':>4}  no shared seeds")
            continue
        t = np.array([trained[s] for s in shared])
        b = np.array([bases[p][s] for s in shared])
        wins = int((t > b).sum())
        ratio = t.mean() / b.mean() if b.mean() else float("nan")
        print(f"{p:<10} {len(shared):>4} {t.mean():>9.1f} {b.mean():>9.1f} "
              f"{ratio:>7.2f} {wins:>4}/{len(shared):<4} {np.median(t - b):>10.1f}")
        summary["per_step"][p] = float(b.mean())
        summary["paired"][p] = {"n": len(shared), "wins": wins,
                                "ratio": float(ratio),
                                "median_delta": float(np.median(t - b))}

    if "hover" in bases and "greedy" in bases:
        shared = sorted(set(trained) & set(bases["hover"]) & set(bases["greedy"]))
        if shared:
            t = np.mean([trained[s] for s in shared])
            h = np.mean([bases["hover"][s] for s in shared])
            g = np.mean([bases["greedy"][s] for s in shared])
            frac = (t - h) / (g - h) if g != h else float("nan")
            print(f"\nfraction of the hover->greedy gap closed: {frac:+.2f}"
                  f"   (0.0 = hover, 1.0 = greedy)")
            print(f"   hover {h:.1f}  ->  trained {t:.1f}  ->  greedy {g:.1f} /step")
            summary["gap_closed_vs_greedy"] = float(frac)

    print("\nREMINDER: this is the TEAM reward (infos['global_reward']). The "
          "training curve\n'mean return' is the per-drone difference reward D_i "
          "and is ~4-5x smaller —\nthe two are not comparable.")

    if "--json" in sys.argv:
        dest = sys.argv[sys.argv.index("--json") + 1]
        Path(dest).write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
