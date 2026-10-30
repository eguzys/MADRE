#!/usr/bin/env python3
r"""fig:separation scorer + the text ratios (best/greedy and best/L512 at 1200 m and 1700 m).

Implements the three estimators used for fig:separation on the shard-log episode lines, pulled read-only into
four_seed_staging/pull_raw[_s1]/episodes_logs.txt:
  m        = pooled mean episode_reward / 250 * 1000 over all episodes of all pooled seeds [bit/s], rounded to 100
  ep, em   = two-level bootstrap: random.Random(0), B=20000, resample training seeds with replacement, then
             episodes within each picked seed; percentile 2.5/97.5 by linear interpolation; rounded to 100
  x greedy = ratio of pooled means; CI two-level on the arm x episode bootstrap of the greedy, B=4000
  x L512   = ratio of pooled means; CI two-level on both sides, B=20000, random.Random(0)
Every interval has its own Random(0) and episodes are taken in ep-seed order.

Families (40 users, chi=1, cluster spawn, beacon OFF, 250 steps, ep seeds 0..99):
  1200 m: greedy eng_U40; L8/L32/L128 seeds 1-3 = ADPT40_<L>s<k>, seed 5 = X5_<L>s5, seed 4 = FS_<L>s4;
          L16 seeds 1-3 = W1ev_br12rtL16s<k>_SPAWNcluster_TAIL (resilient-backbone/om2g_logs), seed 4 = FS_L16s4;
          L64 / L256 / L512 = FS_<L>s<k>.
  1300..1700 m: SEP<s>_<L>s<k>; greedy SEP<s>_ENG, except 1600 = B16_ENG (no SEP1600_ENG exists; score_sep.sh rule).
A family counts only with exactly ep seeds 0..99 (once each) and 250 steps; a rung below its target seed count is
scored but flagged INCOMPLETE and its row is commented out in the pgfplots file.

Usage: score_separation_4seed.py --mode current|4seed [--with-extras] [--seeds L64=1,2,4,5]
"""
# =========================================================================================================
# SEED SETS -- the ONLY place to change which training seeds are pooled (or --seeds L64=1,2,4,5 on the CLI)
SEEDS_4SEED = {"L8": [1, 2, 3, 5], "L16": [1, 2, 3, 4], "L32": [1, 2, 3, 5], "L64": [1, 2, 3, 4],
               "L128": [1, 2, 3, 5], "L256": [1, 2, 3, 4], "L512": [1, 2, 3, 4]}
# Training runs pooled per message size (edit the line above or pass --seeds).
EXTRA_SEEDS = {"L8": [4], "L32": [4], "L64": [5]}   # trained + evaluated, NOT pooled unless --with-extras
# --mode current seed sets: 1200 m L8/L32/L128 = 1,2,3,5; L512 = 2,3 everywhere; everything else 1,2,3
SEEDS_CURRENT_1200 = {"L8": [1, 2, 3, 5], "L16": [1, 2, 3], "L32": [1, 2, 3, 5], "L64": [1, 2, 3],
                      "L128": [1, 2, 3, 5], "L256": [1, 2, 3], "L512": [2, 3]}
SEEDS_CURRENT_SEP = {"L8": [1, 2, 3], "L16": [1, 2, 3], "L32": [1, 2, 3], "L64": [1, 2, 3],
                     "L128": [1, 2, 3], "L256": [1, 2, 3], "L512": [2, 3]}
# =========================================================================================================
import argparse, os, random, re, statistics as st
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["current", "4seed"], required=True)
ap.add_argument("--with-extras", action="store_true")
ap.add_argument("--seeds", action="append", default=[])
ap.add_argument("--staging", default=None)   # default: four_seed_staging/pull_raw[_s1]/episodes_logs.txt
ap.add_argument("--paper-tex", default=os.path.join(RES, "four_seed_staging", "ref", "paper_ieee_v14_snapshot_20261009.tex"))
ap.add_argument("--outdir", default=None)
ap.add_argument("--spreading", type=int, choices=[8, 1], default=8, help="1 = score the spreading-1 re-evaluations (S1_*), staged under the original family names in four_seed_staging/pull_raw_s1 (--mode 4seed only)")
A = ap.parse_args()
S1 = A.spreading == 1
if S1 and A.mode != "4seed":
    raise SystemExit("--spreading 1 needs --mode 4seed")
if A.staging is None:
    A.staging = os.path.join(RES, "four_seed_staging", "pull_raw_s1" if S1 else "pull_raw", "episodes_logs.txt")
SUF = A.mode + ("_extras" if A.with_extras else "") + ("_s1" if S1 else "")
OUT = A.outdir or os.path.join(RES, "four_seed_scored", f"separation_{SUF}")
os.makedirs(OUT, exist_ok=True)
RUNGS = ["L8", "L16", "L32", "L64", "L128", "L256", "L512"]
SEPS = [1200, 1300, 1400, 1500, 1600, 1700]
S4 = {r: list(v) for r, v in SEEDS_4SEED.items()}
for spec in A.seeds:
    r, v = spec.split("="); S4[r] = [int(x) for x in v.split(",")]
if A.with_extras:
    for r, v in EXTRA_SEEDS.items():
        S4[r] += [k for k in v if k not in S4[r]]


def seeds_for(s, r):
    if A.mode == "current":
        return (SEEDS_CURRENT_1200 if s == 1200 else SEEDS_CURRENT_SEP)[r]
    return S4[r]


def fam(s, r, k):
    if s != 1200:
        return f"SEP{s}_{r}s{k}"
    if r in ("L8", "L32", "L128") and k <= 3:
        return f"ADPT40_{r}s{k}"
    if r in ("L8", "L32", "L128") and k == 5:
        return f"X5_{r}s{k}"
    if r == "L16" and k <= 3:
        return f"W1ev_br12rtL16s{k}_SPAWNcluster_TAIL"
    return f"FS_{r}s{k}"


def greedy(s):
    return "eng_U40" if s == 1200 else ("B16_ENG" if s == 1600 else f"SEP{s}_ENG")


# ---------- load (same parsing as ladder_vs_eng.load / arm_vs_arm.load: ep -> reward, last line wins) ----------
G = pd.read_csv(A.staging, sep="\t")
DATA, BAD, DUP = {}, {}, {}
for f, g in G.groupby("fam", sort=False):
    g = g.sort_values(["src", "ep"], kind="stable")
    d = {}
    for ep, rw, stp in zip(g.ep, g.reward, g.steps):
        if ep in d:
            DUP[f] = DUP.get(f, 0) + 1
        d[int(ep)] = float(rw)
        if stp != 250:
            BAD[f] = "steps != 250"
    DATA[f] = dict(sorted(d.items()))


def complete(f):
    return f in DATA and f not in BAD and list(DATA[f]) == list(range(100))


def vals(f, scale=1.0):
    return list(DATA[f].values())


# ---------- the three estimators ----------
def pct(xs, q):   # sep_bars.py
    xs = sorted(xs); i = q * (len(xs) - 1); lo = int(i); return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (i - lo)


def sep_bars(seed_lists, B=20000):
    """sep_bars.py (a) 2-level, values in kbit/s as there; returns m, ep, em in bit/s"""
    seeds = [[v / 250 * 1000 / 1000 for v in sl] for sl in seed_lists]
    allv = [v for sl in seeds for v in sl]
    m = st.mean(allv)
    rng = random.Random(0)
    a = []
    for _ in range(B):
        pick = [seeds[rng.choice(range(len(seeds)))] for _ in seeds]
        a.append(st.mean([rng.choice(p) for p in pick for _ in p]))
    lo, hi = pct(a, 0.025), pct(a, 0.975)
    return m * 1000, (hi - m) * 1000, (m - lo) * 1000


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def ladder_ci(seed_sets, eng, B=4000):
    """ladder_vs_eng.boot"""
    rng = random.Random(0)
    eng_vals = [eng[e] for e in sorted(eng)]
    ratios = []
    for _ in range(B):
        picked = [seed_sets[rng.randrange(len(seed_sets))] for _ in seed_sets]
        num = []
        for s_ in picked:
            eps = list(s_.values())
            num.extend(eps[rng.randrange(len(eps))] for _ in eps)
        den = [eng_vals[rng.randrange(len(eng_vals))] for _ in eng_vals]
        ratios.append(mean(num) / mean(den))
    ratios.sort()
    return ratios[int(0.025 * B)], ratios[int(0.975 * B)]


def arm_vs_arm(num, den, B=20000):
    """arm_vs_arm.py"""
    def resample(sets, rng):
        picked = [sets[rng.randrange(len(sets))] for _ in sets]
        out = []
        for s_ in picked:
            eps = list(s_.values())
            out.extend(eps[rng.randrange(len(eps))] for _ in eps)
        return out
    pt = st.mean(v for s_ in num for v in s_.values()) / st.mean(v for s_ in den for v in s_.values())
    rng = random.Random(0)
    ratios = sorted(st.mean(resample(num, rng)) / st.mean(resample(den, rng)) for _ in range(B))
    return pt, ratios[int(0.025 * B)], ratios[int(0.975 * B)]


sig = lambda lo, hi: "SIG" if lo > 1 or hi < 1 else "ns"

# ---------- score ----------
CELL, MISSING, STATUS = {}, {}, []
for s in SEPS:
    gname = greedy(s)
    assert complete(gname), f"greedy {gname} incomplete"
    for r in RUNGS:
        want = [fam(s, r, k) for k in seeds_for(s, r)]
        got = [f for f in want if complete(f)]
        miss = [f for f in want if f not in got]
        if miss:
            MISSING[(s, r)] = miss
            STATUS += [f"{f}: {len(DATA.get(f, {}))} episodes" + (f" ({BAD[f]})" if f in BAD else "") for f in miss]
        if not got:
            continue
        m, ep, em = sep_bars([vals(f) for f in got])
        CELL[(s, r)] = dict(fams=got, k=len(got), n=sum(len(DATA[f]) for f in got), m=m, ep=ep, em=em,
                            g=mean(vals(gname)) / 250 * 1000)

lines, ratio_lines, REG = [], [], []
# rows of the paper source (cross-check)
t = open(A.paper_tex).read()
fig = t[t.index("Cluster separation $s$"):t.index(r"\label{fig:separation}")]
P = {}
for body, leg in re.findall(r"table \[x=s, y=m(?:, y error plus=ep, y error minus=em)?, row sep=\\\\\] \{(.*?)\};\s*\\addlegendentry\{(.*?)\}", fig, re.S):
    L_ = re.search(r"L=(\d+)", leg)
    P[f"L{L_.group(1)}" if L_ else "Greedy"] = {int(a[0]): tuple(int(x) for x in a[1:] if x) for a in
                                                re.findall(r"^\s*(\d+)\s+(\d+)(?:\s+(\d+)\s+(\d+))?\s*\\\\", body, re.M)}

rows_tex = ["% fig:separation rows, 40 users, chi=1, drones on the clusters, beacon OFF. m = pooled mean [bit/s] rounded to 100;",
            "% ep/em = sep_bars.py two-level bootstrap (training seeds -> episodes), B=20000, rounded to 100.",
            f"% MODE {SUF}: seeds per rung 1200 m | 1300-1700 m: " + "  ".join(f"{r}={','.join(map(str, seeds_for(1200, r)))}|{','.join(map(str, seeds_for(1300, r)))}" for r in RUNGS),
            "% Rows of a rung below its target seed count are COMMENTED OUT (INCOMPLETE).",
            "% scored by results/separation_4seed_oct09/score_separation_4seed.py. Nothing here is in the paper.", "",
            "% --- Greedy ---", "    s     m     \\\\"]
for s in SEPS:
    rows_tex.append(f"    {s}  {round(mean(vals(greedy(s))) / 250 * 1000, -2):.0f}  \\\\")
nmis = nbar = 0
for r in RUNGS:
    rows_tex += ["", f"% --- L={r[1:]} ---", "    s     m      ep    em   \\\\"]
    for s in SEPS:
        c = CELL.get((s, r))
        if not c:
            rows_tex.append(f"%   {s}  PENDING (no complete seed)"); continue
        row = (int(round(c["m"], -2)), int(round(c["ep"], -2)), int(round(c["em"], -2)))
        txt = f"    {s}  {row[0]}   {row[1]}   {row[2]}   \\\\"
        rows_tex.append(("%" + txt[1:] + f"   % INCOMPLETE k={c['k']}/{len(seeds_for(s, r))}") if (s, r) in MISSING else txt)
        tex = P.get(r, {}).get(s)
        if tex:
            same_m = tex[0] == row[0]
            nmis += not same_m
            nbar += same_m and tex[1:] != row[1:]
            REG.append(f"{r:5s} {s} paper m={tex[0]:>6d} ep={tex[1]:>5d} em={tex[2]:>5d} | this run k={c['k']} m={c['m']:8.0f} "
                       f"ep={c['ep']:5.0f} em={c['em']:5.0f}  mean {'EXACT' if same_m else 'DIFFERS'}; bars dev {c['ep'] - tex[1]:+5.0f}/{c['em'] - tex[2]:+5.0f}"
                       + ("  INCOMPLETE" if (s, r) in MISSING else ""))
for s in SEPS:
    g = round(mean(vals(greedy(s))) / 250 * 1000, -2)
    tex = P["Greedy"][s][0]
    nmis += g != tex
    REG.append(f"Greedy {s} paper {tex} | this run {g:.0f} ({greedy(s)}) {'EXACT' if g == tex else 'DIFFERS'}")
REG.insert(0, f"paper ({os.path.basename(A.paper_tex)}) fig:separation: {nmis} plotted means differ from this run; "
              f"{nbar} rows with exact mean have rounded bars that differ (bootstrap noise, see RNG note)")
open(os.path.join(OUT, "fig_separation_rows.tex"), "w").write("\n".join(rows_tex) + "\n")

# per-separation table + ratios
for s in SEPS:
    gname = greedy(s); eng = DATA[gname]
    have = [r for r in RUNGS if (s, r) in CELL]
    best = max(have, key=lambda r: CELL[(s, r)]["m"])
    lines.append(f"=== {s} m  greedy {gname} {mean(list(eng.values())) / 250:.2f} kb/s   best = {best}")
    for r in have:
        c = CELL[(s, r)]
        sets = [DATA[f] for f in c["fams"]]
        x = c["m"] / c["g"]; lo, hi = ladder_ci(sets, eng)
        flag = f"  INCOMPLETE k={c['k']}/{len(seeds_for(s, r))}" if (s, r) in MISSING else ""
        lines.append(f"  {r:5s} k={c['k']} n={c['n']:3d} m={c['m'] / 1000:7.2f} +{c['ep'] / 1000:5.2f}/-{c['em'] / 1000:5.2f} kb/s  "
                     f"x greedy {x:.3f} [{lo:.3f}, {hi:.3f}] {sig(lo, hi):3s}  [{'+'.join(c['fams'])}]{flag}")
    for top in sorted({best, "L64", "L128"} & set(have)):
        if (s, "L512") in CELL and top != "L512":
            pt, lo, hi = arm_vs_arm([DATA[f] for f in CELL[(s, top)]["fams"]], [DATA[f] for f in CELL[(s, "L512")]["fams"]])
            lines.append(f"  {top}/L512 = {pt:.3f} [{lo:.3f}, {hi:.3f}] {sig(lo, hi)}" + ("   <- best" if top == best else ""))
    lines.append("")
    if s in (1200, 1700):
# The text ratios are ratios of the PLOTTED means (rounded to 100 bit/s); the ratio of the unrounded means is
# printed next to each.
        b = CELL[(s, best)]; l5 = CELL[(s, "L512")]
        R = lambda x: round(x, -2)
        gR = R(b["g"])
        if s == 1200:
            ratio_lines.append(f"{s} m: best = {best}: best/greedy plotted {R(b['m']):.0f}/{gR:.0f} = {R(b['m']) / gR:.3f} -> {R(b['m']) / gR:.2f} "
                               f"(paper 2.42; unrounded {b['m'] / b['g']:.4f}); best/L512 plotted {R(b['m']):.0f}/{R(l5['m']):.0f} = "
                               f"{R(b['m']) / R(l5['m']):.3f} -> {R(b['m']) / R(l5['m']):.2f} (paper 1.87; unrounded {b['m'] / l5['m']:.4f})")
        else:
            ratio_lines.append(f"{s} m: best = {best}: best/L512 plotted {R(b['m']):.0f}/{R(l5['m']):.0f} = {R(b['m']) / R(l5['m']):.3f} -> "
                               f"{R(b['m']) / R(l5['m']):.2f} (paper 1.60; unrounded {b['m'] / l5['m']:.4f}); "
                               "rungs whose x-greedy CI contains 1 (paper: L8, L16, L512 'no better than greedy'): "
                               + ", ".join(r for r in RUNGS if (s, r) in CELL and sig(*ladder_ci([DATA[f] for f in CELL[(s, r)]['fams']], DATA[greedy(s)])) == "ns"))
    for r in [x for x in ("L64", "L128") if x != best and (s, x) in CELL]:
        c = CELL[(s, r)]
        ratio_lines.append(f"      ({s} m, {r}: /greedy {c['m'] / c['g']:.3f}, /L512 {c['m'] / CELL[(s, 'L512')]['m']:.3f})")

W = [f"fig:separation + text ratios, MODE {SUF}. kb/s; bars = sep_bars two-level 95%; x greedy CI = ladder_vs_eng; /L512 CI = arm_vs_arm.", "",
     "TEXT RATIOS (abstract l.93, body l.1013, conclusion l.1063):"] + ["  " + x for x in ratio_lines]
if S1:
    W.insert(1, "SPREADING 1 (drone TX 6.59004 mW, same checkpoints): every family below is its spreading-1 re-evaluation (S1_<family>, or the verified SPR1 twin), staged under the original family name by pull_from_cluster_s1.sh (four_seed_staging/pull_raw_s1/MANIFEST.tsv). Paper values quoted below are the spreading-8 numbers of ieee v14.")
W += ["", "REGRESSION vs the paper:"] + ["  " + x for x in REG]
W += ["", "MISSING / INCOMPLETE families (not pooled):"] + (["  " + x for x in STATUS] or ["  none"])
if DUP:
    W += ["", "duplicate ep lines in logs (last kept, as ladder_vs_eng): " + ", ".join(f"{k}:{v}" for k, v in DUP.items())]
W += ["", "Per separation:"] + lines
open(os.path.join(OUT, "separation_table.txt"), "w").write("\n".join(W) + "\n")
pd.DataFrame([dict(sep=s, L=r, k=c["k"], n=c["n"], m=c["m"], ep=c["ep"], em=c["em"], greedy=c["g"], ratio_greedy=c["m"] / c["g"],
                   incomplete=(s, r) in MISSING, fams="+".join(c["fams"])) for (s, r), c in CELL.items()]).to_csv(
    os.path.join(OUT, "separation_summary.csv"), index=False, float_format="%.4f")
print("\n".join(W))
