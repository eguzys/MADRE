#!/usr/bin/env python3
"""Outage share of the non-learned baselines in the fig:outages world.
World: 40 users, 1200 m, cluster spawn, 4 clusters, beacon ON, 250-step episodes, ep seeds 0..99 per family.
Arms: GTMG (GTM, global objective), AGG (aggregate / heuristic message), R1H (group relay, blind) = families
NDUB<chi>_{GTMG,AGG,R1H} (submit_outage_baselines.sh); for comparison ENG (greedy) = NDUB<chi>_ENG and
L64 = NDUB<chi>_L64s<k> pooled over an explicit seed set (config below).
Outage = episode with episode_reward == 0 (same definition as the fig:outages scorer). Throughput = reward / 250 (kb/s).
CI: Wilson 95% on the pooled share.
CSVs come from several dirs (per family the FIRST dir that has it wins, so nothing is counted twice);
outputs go to <scoring root>/four_seed_scored/outage_baselines_<mode>/. Baseline arms (ENG, GTMG, AGG, R1H) are unaffected.
Usage: score_outage_baselines_4seed.py --mode current|4seed [--seeds 1,2,4,5] [--with-extras] [--csv DIR ...]
"""
# =========================================================================================================
# SEED SETS -- the ONLY place to change which L64 training seeds are pooled (or --seeds 1,2,4,5 on the CLI)
SEEDS_L64_4SEED = [1, 2, 3, 4]
# L64 training runs pooled (edit the line above or pass --seeds).
EXTRA_SEEDS_L64 = [5]            # NDUB*_L64s5 exist (extras), NOT pooled unless --with-extras
SEEDS_L64_CURRENT = [1, 2, 3]
# =========================================================================================================
import argparse, glob, os, re, sys
from math import sqrt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["current", "4seed"], required=True)
ap.add_argument("--seeds", action="append", default=[])
ap.add_argument("--with-extras", action="store_true")
ap.add_argument("--csv", action="append", default=None)
ap.add_argument("--outdir", default=None)
ap.add_argument("--spreading", type=int, choices=[8, 1], default=8, help="1 = score the spreading-1 re-evaluations (S1_*), staged under the original family names in four_seed_staging/pull_raw_s1 (--mode 4seed only)")
A = ap.parse_args()
S1 = A.spreading == 1
if S1 and A.mode != "4seed":
    raise SystemExit("--spreading 1 needs --mode 4seed")

def _l64_override(specs):
    """--seeds accepts '1,2,4,5' or 'L64=1,2,4,5' (other rungs are ignored here: only L64 is pooled by this scorer)"""
    out = None
    for sp in specs or []:
        if "=" in sp:
            r, v = sp.split("=")
            if r != "L64":
                continue
            sp = v
        out = [int(x) for x in sp.split(",")]
    return out

SEEDS = SEEDS_L64_CURRENT if A.mode == "current" else (_l64_override(A.seeds) or list(SEEDS_L64_4SEED))
if A.with_extras:
    SEEDS = SEEDS + [k for k in EXTRA_SEEDS_L64 if k not in SEEDS]
SUF = A.mode + ("_extras" if A.with_extras else "") + ("_s1" if S1 else "")
CSVS = A.csv or ([os.path.join(RES, "four_seed_staging", "pull_raw_s1", "csv")] if S1 else
                 [os.path.join(HERE, "csv"), os.path.join(RES, "four_seed_staging", "pull_raw", "csv")])
OUT = A.outdir or os.path.join(RES, "four_seed_scored", f"outage_baselines_{SUF}")
os.makedirs(OUT, exist_ok=True)
FILES = {}
for CSV in CSVS:
    seen_here = {}
    for f in sorted(glob.glob(os.path.join(CSV, "OM2G_*.csv"))):
        m = re.search(r"_NDUB(\d+)_(GTMG|AGG|R1H|ENG|L64s\d)_(?:n10_)?seed(\d+)\.csv$", os.path.basename(f))
        if not m:
            continue
        if m.group(2).startswith("L64") and int(m.group(2)[-1]) not in SEEDS:
            continue
        seen_here.setdefault(f"NDUB{m.group(1)}_{m.group(2)}", []).append((f, m))
    for fam, lst in seen_here.items():
        FILES.setdefault(fam, lst)
rows = []
for fam, lst in sorted(FILES.items()):
    for f, m in lst:
        d = pd.read_csv(f)
        assert list(d.columns[:3]) == ["seed", "episode_reward", "steps"], f
        assert (d.steps == 250).all(), f
        arm = "L64" if m.group(2).startswith("L64") else m.group(2)
        for _, r in d.iterrows():
            rows.append(dict(chi=int(m.group(1)) / 100, arm=arm, fam=f"NDUB{m.group(1)}_{m.group(2)}",
                             ep=int(r.seed), reward=float(r.episode_reward)))
D = pd.DataFrame(rows)
D["out"] = (D.reward == 0).astype(int)
# integrity: every family has ep seeds 0..99 exactly once; the only allowed gap is NDUB100_GTMG ep 64, where the
# GTM global LP returns no solution (OracleLPSolveError, HiGHS kNotset) -- reported in the output.
MISSING = {}
DROPPED_L64 = []
for fam, g in D.groupby("fam"):
    assert len(set(g.ep)) == len(g), f"{fam}: duplicate ep seeds"
    miss = sorted(set(range(100)) - set(g.ep))
    if fam == "NDUB100_GTMG":
        assert set(miss) <= {64}, f"{fam}: missing {miss}"
    elif "_L64s" in fam and miss:
        DROPPED_L64.append(f"{fam}: {100 - len(miss)} of 100 ep seeds -> NOT pooled (incomplete)")
        continue
    else:
        assert not miss, f"{fam}: missing ep seeds {miss}"
    if miss:
        MISSING[fam] = miss
D = D[~D.fam.isin([x.split(":")[0] for x in DROPPED_L64])]
D.to_csv(os.path.join(OUT, "episodes_outage_baselines_long.csv"), index=False)


def wilson(k, n, z=1.959964):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0, c - h), min(1, c + h)


arms = ["ENG", "GTMG", "AGG", "R1H", "L64"]
label = {"ENG": "Greedy (ENG)", "GTMG": "GTM (GTMG)", "AGG": "Heuristic msg (AGG)", "R1H": "1-hop relay (R1H)",
         "L64": f"MADRE L64 (k={len(SEEDS)})"}
summ = []
for chi in [0.1, 0.2, 0.4, 0.6, 1.0]:
    for arm in arms:
        s = D[(D.chi == chi) & (D.arm == arm)]
        if len(s) == 0:
            summ.append(dict(chi=chi, arm=arm, n=0)); continue
        n, k = len(s), int(s.out.sum()); lo, hi = wilson(k, n)
        summ.append(dict(chi=chi, arm=arm, n=n, k=k, share=k / n, wlo=lo, whi=hi, mean_kbps=s.reward.mean() / 250))
S = pd.DataFrame(summ)
S.to_csv(os.path.join(OUT, "outage_baselines_summary.csv"), index=False, float_format="%.4f")
lines = ["Outage share (%) in the fig:outages world: 40 users, 1200 m, cluster spawn, 4 clusters, beacon ON,",
         f"250-step episodes, ep seeds 0..99 (n=100 per family; L64 pooled over training seeds {SEEDS}, n={100 * len(SEEDS)}).",
         "Outage = episode_reward == 0 (score_outages.py definition). [Wilson 95% CI]; mean throughput in kb/s.",
         f"ENG and L64 rows: NDUB*_ENG / NDUB*_L64s{{{','.join(map(str, SEEDS))}}} CSVs. MODE {SUF}.", ""]
hdr = f"{'chi':>4} | " + " | ".join(f"{label[a]:>24}" for a in arms)
lines += [hdr, "-" * len(hdr)]
for chi in [0.1, 0.2, 0.4, 0.6, 1.0]:
    cells = []
    for a in arms:
        r = S[(S.chi == chi) & (S.arm == a)].iloc[0]
        cells.append(f"{'missing':>24}" if r.n == 0 else f"{100*r.share:5.1f} [{100*r.wlo:4.1f},{100*r.whi:5.1f}] n={int(r.n):3d}")
    lines.append(f"{chi:>4} | " + " | ".join(cells))
lines += ["", "Mean throughput (kb/s):", f"{'chi':>4} | " + " | ".join(f"{a:>8}" for a in arms)]
for chi in [0.1, 0.2, 0.4, 0.6, 1.0]:
    vals = []
    for a in arms:
        r = S[(S.chi == chi) & (S.arm == a)].iloc[0]
        vals.append(f"{'-':>8}" if r.n == 0 else f"{r.mean_kbps:8.2f}")
    lines.append(f"{chi:>4} | " + " | ".join(vals))
lines += ["", "Missing episodes: " + ("; ".join(f"{k}: ep seed(s) {v} (GTM oracle LP OracleLPSolveError, HiGHS kNotset, "
          "failed identically on 3 attempts on 2 nodes; seeds 60-63 and 65-69 rerun as separate shards)" for k, v in MISSING.items())
          if MISSING else "none")]
lines += ["Incomplete L64 families dropped: " + ("; ".join(DROPPED_L64) if DROPPED_L64 else "none")]
if S1:
    lines.insert(0, "SPREADING 1 (drone TX 6.59004 mW, same checkpoints): every family below is its spreading-1 re-evaluation (S1_<family>, or the verified SPR1 twin), staged under the original family name by pull_from_cluster_s1.sh (four_seed_staging/pull_raw_s1/MANIFEST.tsv). Paper values quoted below are the spreading-8 numbers of ieee v14.")
txt = "\n".join(lines) + "\n"
open(os.path.join(OUT, "outage_baselines_table.txt"), "w").write(txt)
print(txt)
