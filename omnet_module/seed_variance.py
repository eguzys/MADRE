#!/usr/bin/env python3
"""Seed-robustness checks for the OMNeT architecture comparison.

Companion to analyze_seeds.py, which owns the Welch / world-paired tests.
This script answers the separate question "does the training seed change the
ANSWER", via three checks that analyze_seeds.py does not print:

  1. worst-case overlap  -- each arm's worst seed vs the comparator's best,
                            i.e. is there any seed draw that reverses the result
  2. paired-by-seed      -- does the winner change from seed to seed
  3. variance decomposition -- architecture vs common-mode seed vs their
                            interaction. The interaction term is the dangerous
                            one: it is what "wins only on lucky seeds" looks
                            like.

  usage: seed_variance.py <csv_dir> <group_prefix> <seed> [<seed> ...]
  e.g.   seed_variance.py out/n100_learned learned 69 70 71 72
         seed_variance.py out/n100_frozen  frozen  69 70 71

Scores are mean episode_reward/250 (reward per step) over the world seeds in
each CSV.  At n=3-4 the decomposition is DESCRIPTIVE, not inferential -- do
not attach a p-value to the shares.
"""
import csv
import os
import statistics as st
import sys

ARCHS = ["st", "deepsets", "ffn"]
# Every pair, ordered so the first-named arm is the one expected to lead. The
# third is the equivalence pair and is expected to overlap.
PAIRS = [("st", "ffn"), ("deepsets", "ffn"), ("st", "deepsets")]
STEPS = 250  # OMNET_MAX_STEPS; episode_reward is a 250-step sum


def load(csv_dir, group, seeds):
    """-> {(arch, seed): mean reward/step}. Refuses partial grids."""
    out = {}
    for arch in ARCHS:
        for seed in seeds:
            path = os.path.join(csv_dir, f"{group}_{arch}_ts{seed}_n100.csv")
            if not os.path.exists(path):
                sys.exit(f"!! missing {path}\n"
                         f"   the grid must be complete -- a decomposition over "
                         f"an unbalanced design is not interpretable")
            with open(path) as fh:
                rows = [r for r in csv.reader(fh)][1:]
            vals = [float(r[1]) / STEPS for r in rows if r]
            if not vals:
                sys.exit(f"!! {path} has 0 data rows (a header-only CSV)")
            out[(arch, seed)] = sum(vals) / len(vals)
            out.setdefault("_n", {})[(arch, seed)] = len(vals)
    return out


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    csv_dir, group = sys.argv[1], sys.argv[2]
    seeds = [int(s) for s in sys.argv[3:]]
    d = load(csv_dir, group, seeds)
    n_worlds = d.pop("_n")

    print(f"\n{'=' * 78}\nSEED ROBUSTNESS -- group '{group}', "
          f"{len(seeds)} training seeds, {len(ARCHS)} architectures")
    print(f"world seeds per cell: {sorted(set(n_worlds.values()))}\n{'=' * 78}")

    # ---- per-arm summary ----------------------------------------------------
    print("\n1. PER-ARM (mean reward/step over world seeds)\n")
    print("            " + "".join(f"   s{s} " for s in seeds)
          + "    mean      sd   range")
    for arch in ARCHS:
        v = [d[(arch, s)] for s in seeds]
        sd = st.stdev(v) if len(v) > 1 else float("nan")
        print(f"{arch:>10}  " + "".join(f"{x:7.1f}" for x in v)
              + f" {st.mean(v):8.1f} {sd:7.1f} {max(v) - min(v):7.1f}")

    # ---- common-mode seed quality ------------------------------------------
    print("\n2. COMMON-MODE SEED QUALITY (mean over architectures)")
    print("   a consistent ranking here means the seed noise is SHARED and so"
          "\n   cancels when arms are paired by seed\n")
    for s in seeds:
        print(f"   s{s}: {st.mean([d[(a, s)] for a in ARCHS]):7.1f}")

    # ---- worst-case overlap -------------------------------------------------
    print("\n3. WORST-CASE OVERLAP (no distributional assumption)")
    print("   'separated' => no seed draw reverses the result\n")
    for a, b in PAIRS:
        # Test in the direction the data actually points, not the direction the
        # pair happens to be written in: whichever arm leads on the mean has its
        # WORST seed compared against the other's BEST. Comparing a trailing
        # arm's worst against the leader's best always "overlaps" and says
        # nothing (frozen deepsets-vs-ffn was mislabelled an equivalence by an
        # earlier version of this check).
        hi_arm, lo_arm = ((a, b) if st.mean([d[(a, s)] for s in seeds])
                          >= st.mean([d[(b, s)] for s in seeds]) else (b, a))
        lo = min(d[(hi_arm, s)] for s in seeds)
        hi = max(d[(lo_arm, s)] for s in seeds)
        verdict = (f"SEPARATED by {lo - hi:+.1f}" if lo > hi
                   else f"OVERLAP by {hi - lo:.1f}  <-- report as equivalence")
        print(f"   worst {hi_arm:>8} = {lo:6.1f}  vs  best {lo_arm:>8} = {hi:6.1f}"
              f"   -> {verdict}")

    # ---- paired by seed -----------------------------------------------------
    print("\n4. PAIRED BY SEED (does the winner change?)\n")
    for a, b in PAIRS:
        diffs = [d[(a, s)] - d[(b, s)] for s in seeds]
        sd = st.stdev(diffs) if len(diffs) > 1 else float("nan")
        wins = sum(x > 0 for x in diffs)
        print(f"   {a:>8} - {b:<8}" + "".join(f"{x:+7.1f}" for x in diffs)
              + f"   mean {st.mean(diffs):+6.1f}  sd {sd:5.1f}"
              + f"  wins {wins}/{len(seeds)}")

    # ---- variance decomposition --------------------------------------------
    # Balanced two-way layout with one observation per cell, so the residual IS
    # the architecture x seed interaction -- there is no separate error term.
    # That is only legitimate because evaluation is deterministic
    # (omnetSeeds.md 7.5): duplicate scoring jobs produced byte-identical CSVs.
    print("\n5. VARIANCE DECOMPOSITION")
    print("   DESCRIPTIVE ONLY at this n -- do not attach a p-value.")
    print("   the interaction term is the dangerous one: it is what")
    print("   'this architecture only wins on lucky seeds' would look like.\n")
    cells = [d[(a, s)] for a in ARCHS for s in seeds]
    gm = st.mean(cells)
    ss_arch = sum(len(seeds) * (st.mean([d[(a, s)] for s in seeds]) - gm) ** 2
                  for a in ARCHS)
    ss_seed = sum(len(ARCHS) * (st.mean([d[(a, s)] for a in ARCHS]) - gm) ** 2
                  for s in seeds)
    ss_tot = sum((x - gm) ** 2 for x in cells)
    ss_int = ss_tot - ss_arch - ss_seed
    for label, ss in (("architecture", ss_arch),
                      ("training seed (common-mode)", ss_seed),
                      ("architecture x seed (interaction)", ss_int)):
        print(f"   {label:<36} {100 * ss / ss_tot:5.1f} %")
    print()


if __name__ == "__main__":
    main()
