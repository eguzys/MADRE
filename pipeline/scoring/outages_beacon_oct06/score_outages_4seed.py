#!/usr/bin/env python3
"""fig:outages scorer, four-seed version. The L64 training seeds are an explicit set (config below); CSV files are
read in sorted order; the figure rows are written by the script and cross-checked against the paper source; outputs
go to <scoring root>/four_seed_scored/outages_<mode>/. Estimator: outage = episode_reward == 0, plotted value = pooled
share over all episodes of the arm (ENG n=100, L64 n=100*k).
Beacon OFF (ND*) exists only for L64 seeds 1-3, so the OFF columns keep those seeds in every mode (not plotted).

Usage: score_outages_4seed.py --mode current|4seed [--seeds 1,2,4,5] [--with-extras] [--csv DIR] [--episodes EPISODES.tsv]
"""
# =========================================================================================================
# SEED SETS -- the ONLY place to change which L64 training seeds are pooled (or --seeds 1,2,4,5 on the CLI)
SEEDS_L64_4SEED = [1, 2, 3, 4]
# L64 training runs pooled (edit the line above or pass --seeds).
EXTRA_SEEDS_L64 = [5]            # NDUB*_L64s5 exist (extras), NOT pooled unless --with-extras
SEEDS_L64_CURRENT = [1, 2, 3]    # --mode current seed set
# =========================================================================================================
import argparse, glob, os, re, sys
import numpy as np, pandas as pd
from math import comb, sqrt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["current", "4seed"], required=True)
ap.add_argument("--seeds", action="append", default=[])
ap.add_argument("--with-extras", action="store_true")
ap.add_argument("--csv", default=None)   # default: four_seed_staging/pull_raw[_s1]/csv
ap.add_argument("--episodes", default=os.path.join(RES, "misc_outputs", "EPISODES.tsv"))
ap.add_argument("--paper-tex", default=os.path.join(RES, "four_seed_staging", "ref", "paper_ieee_v14_snapshot_20261009.tex"))
ap.add_argument("--outdir", default=None)
ap.add_argument("--spreading", type=int, choices=[8, 1], default=8, help="1 = score the spreading-1 re-evaluations (S1_*), staged under the original family names in four_seed_staging/pull_raw_s1 (--mode 4seed only)")
A = ap.parse_args()
S1 = A.spreading == 1
if S1 and A.mode != "4seed":
    raise SystemExit("--spreading 1 needs --mode 4seed")
if A.csv is None:
    A.csv = os.path.join(RES, "four_seed_staging", "pull_raw_s1" if S1 else "pull_raw", "csv")
# spreading 1: only the plotted beacon-ON arms were re-evaluated; the beacon-OFF columns (ND*, EPISODES.tsv) are
# spreading-8 data and are left out entirely (no ON-OFF tests).
BCS = ["ON"] if S1 else ["OFF", "ON"]

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
CSV, EPI = A.csv, A.episodes
OUT = A.outdir or os.path.join(RES, "four_seed_scored", f"outages_{SUF}")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261006)
B = 20000

rows = []
for f in sorted(glob.glob(os.path.join(CSV, "OM2G_*.csv"))):
    m = re.search(r"_(ND(?:UB)?)(\d+)_(ENG|L64s\d)_(?:n10_)?seed\d+\.csv$", os.path.basename(f))
    if not m: continue
    if m.group(3) != "ENG" and int(m.group(3)[-1]) not in (SEEDS if m.group(1) == "NDUB" else SEEDS_L64_CURRENT): continue
    d = pd.read_csv(f)
    assert list(d.columns[:3]) == ["seed", "episode_reward", "steps"], f
    assert (d.steps == 250).all(), f
    for _, r in d.iterrows():
        rows.append(dict(beacon="ON" if m.group(1) == "NDUB" else "OFF", chi=int(m.group(2)) / 100,
                         arm="ENG" if m.group(3) == "ENG" else "L64",
                         tseed=0 if m.group(3) == "ENG" else int(m.group(3)[-1]),
                         ep=int(r.seed), reward=float(r.episode_reward)))
D = pd.DataFrame(rows)
# old chi=1.0 from EPISODES.tsv
E = pd.read_csv(EPI, sep="\t", comment="#") if not S1 else pd.DataFrame(columns=["family", "ep_seed", "reward"])
for fam, arm in [("eng_U40", "ENG"), ("FS_L64s1", "L64"), ("FS_L64s2", "L64"), ("FS_L64s3", "L64")]:
    e = E[E.family == fam]
    for _, r in e.iterrows():
        D = pd.concat([D, pd.DataFrame([dict(beacon="OFF", chi=1.0, arm=arm,
              tseed=0 if arm == "ENG" else int(fam[-1]), ep=int(r.ep_seed), reward=float(r.reward))])])
# cross-check local ND CSVs against EPISODES.tsv
for fam in sorted(set(E.family)):
    m = re.fullmatch(r"ND(\d+)_(ENG|L64s[123])", fam)
    if not m: continue
    a = E[E.family == fam].set_index("ep_seed").reward.sort_index()
    arm = "ENG" if m.group(2) == "ENG" else "L64"; ts = 0 if arm == "ENG" else int(m.group(2)[-1])
    b = D[(D.beacon == "OFF") & (D.chi == int(m.group(1)) / 100) & (D.arm == arm) & (D.tseed == ts)].set_index("ep").reward.sort_index()
    assert len(a) == len(b) == 100 and np.allclose(a.values, b.values, rtol=0, atol=1e-3) and (a.index == b.index).all(), fam
D["out"] = (D.reward == 0).astype(int)
D.to_csv(os.path.join(OUT, "episodes_outage_long.csv"), index=False)

def wilson(k, n, z=1.959964):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0, c - h), min(1, c + h)

def boot2(sub):
    seeds = sorted(sub.tseed.unique()); arrs = [sub[sub.tseed == s].out.values for s in seeds]
    k = len(arrs); res = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, k, k)
        res[b] = np.mean(np.concatenate([a[rng.integers(0, len(a), len(a))] for a in (arrs[i] for i in pick)]))
    return np.percentile(res, [2.5, 97.5])

def paired(x, y):   # x, y arrays aligned by ep seed; returns diff, CI, two-sided boot p
    d = x - y; n = len(d); bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean()) if d.mean() != 0 else 1.0
    return d.mean(), np.percentile(bs, [2.5, 97.5]), min(p, 1.0)

def mcnemar(b, c):  # exact two-sided
    n = b + c
    if n == 0: return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n)

summ = []; tests = []
for chi in [0.1, 0.2, 0.4, 0.6, 1.0]:
    rec = dict(chi=chi)
    for bc in BCS:
        for arm in ["ENG", "L64"]:
            s = D[(D.beacon == bc) & (D.chi == chi) & (D.arm == arm)]
            n, k = len(s), int(s.out.sum())
            lo, hi = wilson(k, n)
            rec.update({f"{arm}_{bc}_n": n, f"{arm}_{bc}_k": k, f"{arm}_{bc}_share": k / n,
                        f"{arm}_{bc}_wlo": lo, f"{arm}_{bc}_whi": hi,
                        f"{arm}_{bc}_meankbps": s.reward.mean() / 250})
            if arm == "L64":
                blo, bhi = boot2(s) if k > 0 else (0.0, 0.0)
                rec.update({f"L64_{bc}_blo": blo, f"L64_{bc}_bhi": bhi})
                rec.update({f"L64_{bc}_perseed": "/".join(str(int(s[s.tseed == t].out.sum())) for t in sorted(s.tseed.unique()))})
        # ENG vs L64 paired by ep seed
        eg = D[(D.beacon == bc) & (D.chi == chi) & (D.arm == "ENG")].set_index("ep").out.sort_index()
        lg = D[(D.beacon == bc) & (D.chi == chi) & (D.arm == "L64")].groupby("ep").out.mean().sort_index()
        common = eg.index.intersection(lg.index)
        dm, ci, p = paired(eg.loc[common].values.astype(float), lg.loc[common].values)
        tests.append(dict(chi=chi, beacon=bc, test="ENG - L64 outage share (paired by ep seed)", n_pairs=len(common),
                          diff=dm, lo=ci[0], hi=ci[1], p_boot=p))
    for arm in ([] if S1 else ["ENG", "L64"]):
        on = D[(D.beacon == "ON") & (D.chi == chi) & (D.arm == arm)]
        off = D[(D.beacon == "OFF") & (D.chi == chi) & (D.arm == arm)]
        key = ["tseed", "ep"]
        mm = on.merge(off, on=key, suffixes=("_on", "_off"))
        if arm == "L64":
            x = mm.groupby("ep").out_on.mean().values; y = mm.groupby("ep").out_off.mean().values
        else:
            x = mm.out_on.values.astype(float); y = mm.out_off.values.astype(float)
        dm, ci, p = paired(x, y)
        b = int(((mm.out_on == 1) & (mm.out_off == 0)).sum()); c = int(((mm.out_on == 0) & (mm.out_off == 1)).sum())
        tests.append(dict(chi=chi, beacon="ON-OFF", test=f"{arm}: beacon ON - OFF outage share (paired by ep seed)",
                          n_pairs=len(mm), diff=dm, lo=ci[0], hi=ci[1], p_boot=p,
                          note=f"discordant on-only={b} off-only={c}, exact McNemar p={mcnemar(b, c):.3g}" + ("" if chi < 1 else " (OFF chi=1 = eng_U40/FS_L64 families, same ep seeds 0..99)")))
    summ.append(rec)
S = pd.DataFrame(summ); T = pd.DataFrame(tests)
# ---- completeness: every family 100 ep seeds 0..99 once (ON L64 per target seed) ----
STATUS = []
for chi in [0.1, 0.2, 0.4, 0.6, 1.0]:
    for bc in BCS:
        for ts in [0] + (SEEDS if bc == "ON" else SEEDS_L64_CURRENT):
            arm = "ENG" if ts == 0 else "L64"
            g = D[(D.beacon == bc) & (D.chi == chi) & (D.arm == arm) & (D.tseed == ts)]
            ok = sorted(g.ep) == list(range(100))
            if not ok:
                STATUS.append(f"INCOMPLETE {bc} chi={chi} {arm}{'' if ts == 0 else f' seed {ts}'}: {len(g)} episodes")
S.to_csv(os.path.join(OUT, "outages_summary.csv"), index=False, float_format="%.4f")
T.to_csv(os.path.join(OUT, "outages_tests.csv"), index=False, float_format="%.4f")
pd.set_option("display.width", 250)
print(S.T.to_string()); print(T.to_string())

# ---- figure rows (beacon ON, the plotted values) + cross-check vs the paper source ----
rowsG = {int(round(r.chi * 100)): f"{100 * r.ENG_ON_share:.0f}" for _, r in S.iterrows()}
rowsL = {int(round(r.chi * 100)): f"{100 * r.L64_ON_share:.1f}" for _, r in S.iterrows()}
inc = [x for x in STATUS if x.startswith("INCOMPLETE ON")]
L = [f"% fig:outages, BEACON ON (NDUB<chi>_ENG, NDUB<chi>_L64s{{{','.join(map(str, SEEDS))}}}); 40 users, 1200 m, cluster spawn",
     f"% outage = episode_reward == 0; share pooled over episodes (ENG n=100, L64 n={100 * len(SEEDS)} = {len(SEEDS)} training seeds x 100)",
     f"% scored by results/outages_beacon_oct06/score_outages_4seed.py --mode {SUF}" + (" -- INCOMPLETE: " + "; ".join(inc) if inc else ""),
     "% --- Greedy (replace the rows of the first \\addplot) ---", "    c    f  \\\\"]
L += [f"    {c}  {v} \\\\" for c, v in rowsG.items()]
L += ["% --- L=64 (replace the rows of the second \\addplot) ---", "    c    f  \\\\"]
L += [("%" if inc else " ") + f"   {c}  {v} \\\\" for c, v in rowsL.items()]
open(os.path.join(OUT, "fig_outages_beacon_on_rows.tex"), "w").write("\n".join(L) + "\n")
t = open(A.paper_tex).read()
fig = t[t.index("\\label{fig:separation}"):t.index("\\label{fig:outages}")]
blocks = re.findall(r"table \[x=c, y=f, row sep=\\\\\] \{(.*?)\};", fig, re.S)
PG = {int(c): v for c, v in re.findall(r"^\s*(\d+)\s+([\d.]+)\s*\\\\", blocks[0], re.M)}
PL = {int(c): v for c, v in re.findall(r"^\s*(\d+)\s+([\d.]+)\s*\\\\", blocks[1], re.M)}
REG = []
for nm, P, R in [("Greedy", PG, rowsG), ("L64", PL, rowsL)]:
    for c in sorted(P):
        REG.append(f"{nm:6s} chi={c:>3d} paper {P[c]:>4s} | this run {R[c]:>4s} {'EXACT' if P[c] == R[c] else 'DIFFERS'}")
nm = sum("DIFFERS" in r for r in REG)
if A.mode == "current" and not A.with_extras:
    old = open(os.path.join(HERE, "fig_outages_beacon_on_rows.tex")).read()
    REG.insert(0, "non-comment lines identical to outages_beacon_oct06/fig_outages_beacon_on_rows.tex: "
               + str([l for l in old.splitlines() if not l.startswith("%")] == [l for l in L if not l.startswith("%")]))
REG.insert(0, f"paper ({os.path.basename(A.paper_tex)}) fig:outages: {nm} of {len(PG) + len(PL)} points differ from this run")
txt = "\n".join((["SPREADING 1 (drone TX 6.59004 mW, same checkpoints): every family below is its spreading-1 re-evaluation (S1_<family>, or the verified SPR1 twin), staged under the original family name by pull_from_cluster_s1.sh (four_seed_staging/pull_raw_s1/MANIFEST.tsv). Paper values quoted below are the spreading-8 numbers of ieee v14.", ""] if S1 else []) + ["MODE " + SUF + f": L64 beacon-ON seeds {SEEDS}; OFF L64 seeds {SEEDS_L64_CURRENT} (only ones evaluated beacon-off)", "",
                  "REGRESSION vs the paper:"] + ["  " + r for r in REG] + ["", "Completeness:"] + (["  " + x for x in STATUS] or ["  all families complete (100 ep seeds 0..99)"])
                 + ["", "Per chi (share %, Wilson / two-level bootstrap CI, per-seed outage counts):"]
                 + [f"  chi={r.chi:4.2f} ENG_ON {100*r.ENG_ON_share:5.1f} [{100*r.ENG_ON_wlo:4.1f},{100*r.ENG_ON_whi:5.1f}]  L64_ON {100*r.L64_ON_share:5.2f} n={int(r.L64_ON_n)} boot [{100*r.L64_ON_blo:4.2f},{100*r.L64_ON_bhi:4.2f}] per-seed {r.L64_ON_perseed}  mean kb/s ENG {r.ENG_ON_meankbps:.2f} L64 {r.L64_ON_meankbps:.2f}" for _, r in S.iterrows()]) + "\n"
open(os.path.join(OUT, "outages_table.txt"), "w").write(txt)
print(txt)
