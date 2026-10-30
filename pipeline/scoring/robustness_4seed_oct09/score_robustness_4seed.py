#!/usr/bin/env python3
r"""Robustness numbers (paragraph after fig:outages), four-seed version:
  uniform start: relative throughput change of the learned policies and of the greedy when the drones start
  uniformly over the area instead of on the clusters, and the lead of L=64 over the greedy (ratio + CI);
  final checkpoint: lead over greedy at the best checkpoint / lead at the final checkpoint.

  World: 40 users, chi=1, 1200 m, 4 clusters, beacon OFF, 250 steps, ep seeds 0..99.
  Uniform spawn: USPAWN_<L>s<k> (cluster-trained policies, DIAG_SPAWN=uniform) vs greedy USPAWN_ENG.
  Cluster-spawn reference (same seeds, same checkpoints): L8/L32/L128 ADPT40_<L>s1-3 (s5: X5_, s4: FS_), L64 FS_L64s<k>;
    greedy eng_U40.
  relative change = 1 - uniform/cluster pooled mean per rung; greedy change = 1 - USPAWN_ENG/eng_U40
  lead = pooled mean / USPAWN_ENG mean, two-level bootstrap B=4000 Random(0)
  final-checkpoint factor = mean(best L128) / mean(CKFIN_L128) over the pooled seeds; CI by two-level resampling.
Episodes are taken in ep-seed order; each interval has its own Random(0).

Usage: score_robustness_4seed.py --mode current|4seed [--with-extras] [--seeds L64=1,2,4,5]
"""
# =========================================================================================================
# SEED SETS -- the ONLY place to change which training seeds are pooled (or --seeds L64=1,2,4,5 on the CLI)
SEEDS_4SEED = {"L8": [1, 2, 3, 5], "L32": [1, 2, 3, 5], "L64": [1, 2, 3, 4], "L128": [1, 2, 3, 5]}
# Training runs pooled per message size (edit the line above or pass --seeds).
EXTRA_SEEDS = {"L8": [4], "L32": [4], "L64": [5]}   # USPAWN/FS of these exist; NOT pooled unless --with-extras
SEEDS_CURRENT = {"L8": [1, 2, 3], "L32": [1, 2, 3], "L64": [1, 2, 3], "L128": [1, 2, 3]}   # --mode current seed sets
# final-checkpoint test is on L128 only (CKFIN_L128s<k>); it uses the L128 seed set above
# EXTENSION: all seven message sizes. Spreading-1 families S1_USPAWN_{L16,L256,L512}s<k> and
# S1_CKFIN_{L8,L16,L32,L64,L256,L512}s<k> (plan: $S/u100diag/c4s/s1/plan_s1_robust_ext.tsv). The sections above
# cover four sizes and the L128 final checkpoint; the extension is appended as separate sections and only when at
# least one of its families is present.
SEEDS_EXT = {"L8": [1, 2, 3, 5], "L16": [1, 2, 3, 4], "L32": [1, 2, 3, 5], "L64": [1, 2, 3, 4], "L128": [1, 2, 3, 5],
             "L256": [1, 2, 3, 4], "L512": [1, 2, 3, 4]}   # the 4-seed sets (same as the separation scorer)
# seeds whose LAST checkpoint IS the best-return one (so CKFIN re-evaluates the same policy as the best family):
# read from the checkpoint dirs (best = pick_best_ckpt)
FINAL_IS_BEST = {("L8", 3): 288000, ("L64", 4): 300000, ("L256", 1): 300000, ("L256", 2): 300000, ("L256", 3): 300000,
                 ("L512", 4): 300000}
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
OUT = A.outdir or os.path.join(RES, "four_seed_scored", f"robustness_{SUF}")
os.makedirs(OUT, exist_ok=True)
RUNGS = ["L8", "L32", "L64", "L128"]
SEEDS = {r: list(v) for r, v in (SEEDS_CURRENT if A.mode == "current" else SEEDS_4SEED).items()}
for spec in A.seeds:
    r, v = spec.split("=")
    if r in SEEDS:
        SEEDS[r] = [int(x) for x in v.split(",")]
if A.with_extras:
    for r, v in EXTRA_SEEDS.items():
        SEEDS[r] += [k for k in v if k not in SEEDS[r]]


def cluster_fam(r, k):
    if r in ("L8", "L32", "L128") and k <= 3:
        return f"ADPT40_{r}s{k}"
    if r in ("L8", "L32", "L128") and k == 5:
        return f"X5_{r}s{k}"
    if r == "L16" and k <= 3:   # extension only (the four section-1 sizes never reach this line)
        return f"W1ev_br12rtL16s{k}_SPAWNcluster_TAIL"
    return f"FS_{r}s{k}"


G = pd.read_csv(A.staging, sep="\t")
DATA, BAD = {}, set()
for f, g in G.groupby("fam", sort=False):
    g = g.sort_values(["src", "ep"], kind="stable")
    d = {}
    for ep, rw, stp in zip(g.ep, g.reward, g.steps):
        d[int(ep)] = float(rw)
        if stp != 250:
            BAD.add(f)
    DATA[f] = dict(sorted(d.items()))
complete = lambda f: f in DATA and f not in BAD and list(DATA[f]) == list(range(100))
mean = lambda x: sum(x) / len(x)


def pooled(fams):
    return mean([v for f in fams for v in DATA[f].values()])


def ladder_ci(seed_sets, eng, B=4000):   # ladder_vs_eng.boot
    rng = random.Random(0)
    ev = [eng[e] for e in sorted(eng)]
    out = []
    for _ in range(B):
        picked = [seed_sets[rng.randrange(len(seed_sets))] for _ in seed_sets]
        num = []
        for s_ in picked:
            eps = list(s_.values()); num.extend(eps[rng.randrange(len(eps))] for _ in eps)
        den = [ev[rng.randrange(len(ev))] for _ in ev]
        out.append(mean(num) / mean(den))
    out.sort()
    return out[int(0.025 * B)], out[int(0.975 * B)]


def arm_vs_arm(num, den, B=20000):   # arm_vs_arm.py
    def rs(sets, rng):
        picked = [sets[rng.randrange(len(sets))] for _ in sets]
        o = []
        for s_ in picked:
            eps = list(s_.values()); o.extend(eps[rng.randrange(len(eps))] for _ in eps)
        return o
    rng = random.Random(0)
    r = sorted(st.mean(rs(num, rng)) / st.mean(rs(den, rng)) for _ in range(B))
    return r[int(0.025 * B)], r[int(0.975 * B)]


sig = lambda lo, hi: "SIG" if lo > 1 or hi < 1 else "ns"
STATUS, W = [], []
for g in ["USPAWN_ENG", "eng_U40"]:
    assert complete(g), g
ue, ce = DATA["USPAWN_ENG"], DATA["eng_U40"]
greedy_loss = 1 - mean(list(ue.values())) / mean(list(ce.values()))

# ---------- 1. uniform spawn ----------
U = {}
for r in RUNGS:
    pairs = [(f"USPAWN_{r}s{k}", cluster_fam(r, k)) for k in SEEDS[r]]
    ok = [(u, c) for u, c in pairs if complete(u) and complete(c)]
    for u, c in pairs:
        for f in (u, c):
            if not complete(f):
                STATUS.append(f"{f}: {len(DATA.get(f, {}))} episodes -> seed not pooled (needs both spawns)")
    if not ok:
        continue
    uf, cf = [u for u, _ in ok], [c for _, c in ok]
    mu, mc = pooled(uf), pooled(cf)
    lo, hi = ladder_ci([DATA[f] for f in uf], ue)
    U[r] = dict(k=len(ok), target=len(SEEDS[r]), mu=mu, mc=mc, xg=mu / mean(list(ue.values())), lo=lo, hi=hi,
                loss=1 - mu / mc, xg_cluster=mc / mean(list(ce.values())), uf=uf, cf=cf)
inc = lambda r: "" if U[r]["k"] == U[r]["target"] else f"  INCOMPLETE k={U[r]['k']}/{U[r]['target']}"
W.append(f"=== Uniform spawn (40 users, chi=1, 1200 m), MODE {SUF}. kb/s = reward/250.")
W.append(f"greedy: cluster eng_U40 {mean(list(ce.values())) / 250:.2f}, uniform USPAWN_ENG {mean(list(ue.values())) / 250:.2f} "
         f"-> loss {100 * greedy_loss:.1f} %  (paper: 'only 5 %')")
for r in RUNGS:
    if r in U:
        x = U[r]
        W.append(f"  {r:5s} k={x['k']} uniform {x['mu'] / 250:7.2f}  x greedy {x['xg']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}] {sig(x['lo'], x['hi'])}"
                 f"  | cluster {x['mc'] / 250:7.2f} (x greedy {x['xg_cluster']:.3f})  loss {100 * x['loss']:5.1f} %{inc(r)}")
losses = [100 * U[r]["loss"] for r in RUNGS if r in U]
W.append(f"  learned-policy loss: {min(losses):.1f} to {max(losses):.1f} %, mean {sum(losses) / len(losses):.1f} %  (paper: 'about 30 %')")
order = lambda key: " > ".join(sorted([r for r in RUNGS if r in U], key=lambda r: -U[r][key]))
W.append(f"  order by mean: cluster {order('mc')}   |   uniform {order('mu')}")
if "L64" in U and "L128" in U:
    lo, hi = arm_vs_arm([DATA[f] for f in U["L64"]["uf"]], [DATA[f] for f in U["L128"]["uf"]])
    W.append(f"  uniform L64/L128 {U['L64']['mu'] / U['L128']['mu']:.3f} [{lo:.3f}, {hi:.3f}] {sig(lo, hi)} (arm_vs_arm)")

# ---------- 2. final checkpoint (L128) ----------
W.append("")
W.append("=== Final vs best checkpoint, L128 (40 users, chi=1, 1200 m, cluster spawn)")
pairs = [(cluster_fam("L128", k), f"CKFIN_L128s{k}") for k in SEEDS["L128"]]
ok = [(b, f) for b, f in pairs if complete(b) and complete(f)]
for b, f in pairs:
    for x in (b, f):
        if not complete(x):
            STATUS.append(f"{x}: {len(DATA.get(x, {}))} episodes -> seed not pooled (needs best and final)")
gm = mean(list(ce.values()))
for b, f in ok:
    W.append(f"  seed {b[-1]}: best {pooled([b]) / gm:.3f} x greedy ({b}) -> final {pooled([f]) / gm:.3f} ({f})")
if ok:
    bf, ff = [b for b, _ in ok], [f for _, f in ok]
    mb, mf = pooled(bf), pooled(ff)
    lob, hib = ladder_ci([DATA[x] for x in bf], ce)
    lof, hif = ladder_ci([DATA[x] for x in ff], ce)
    lor, hir = arm_vs_arm([DATA[x] for x in bf], [DATA[x] for x in ff])
    CK = dict(k=len(ok), best=mb / gm, final=mf / gm, ratio=mb / mf, lo=lor, hi=hir)
    W.append(f"  pooled k={len(ok)}: best {mb / gm:.3f} [{lob:.3f}, {hib:.3f}]  final {mf / gm:.3f} [{lof:.3f}, {hif:.3f}]  "
             f"best/final {mb / mf:.4f} [{lor:.3f}, {hir:.3f}] {sig(lor, hir)}  (paper: 1.07)"
             + ("" if len(ok) == len(SEEDS["L128"]) else f"  INCOMPLETE k={len(ok)}/{len(SEEDS['L128'])}"))
else:
    nan = float("nan")
    CK = dict(k=0, best=nan, final=nan, ratio=nan, lo=nan, hi=nan)
    W.append(f"  no complete best+final pair yet (k=0/{len(SEEDS['L128'])}), L128 final-checkpoint section skipped")

# ---------- 4. EXTENSION: all seven sizes; appended to the output only if any of its families exist ----------
RUNGS7 = ["L8", "L16", "L32", "L64", "L128", "L256", "L512"]
SX = {r: list(v) for r, v in SEEDS_EXT.items()}
for spec in A.seeds:
    r, v = spec.split("=")
    if r in SX:
        SX[r] = [int(x) for x in v.split(",")]
if A.with_extras:
    for r, v in EXTRA_SEEDS.items():
        SX[r] += [k for k in v if k not in SX[r]]
NEWU = [f"USPAWN_{r}s{k}" for r in ("L16", "L256", "L512") for k in SX[r]]
NEWC = [f"CKFIN_{r}s{k}" for r in RUNGS7 if r != "L128" for k in SX[r]]
EXT, EXTSTAT, EXTROWS = [], [], []
if any(f in DATA for f in NEWU + NEWC):
    gu, gc = mean(list(ue.values())), mean(list(ce.values()))
    # 4a. uniform start, every size (the four sizes of section 1 use the same seed sets)
    UX = {}
    for r in RUNGS7:
        pairs = [(f"USPAWN_{r}s{k}", cluster_fam(r, k)) for k in SX[r]]
        ok = [(u, c) for u, c in pairs if complete(u) and complete(c)]
        for u, c in pairs:
            for f in (u, c):
                if not complete(f):
                    EXTSTAT.append(f"{f}: {len(DATA.get(f, {}))} episodes -> seed not pooled (needs both spawns)")
        if not ok:
            continue
        uf, cf = [u for u, _ in ok], [c for _, c in ok]
        mu, mc = pooled(uf), pooled(cf)
        lo, hi = ladder_ci([DATA[f] for f in uf], ue)
        loc, hic = ladder_ci([DATA[f] for f in cf], ce)
        UX[r] = dict(k=len(ok), target=len(SX[r]), mu=mu, mc=mc, xg=mu / gu, lo=lo, hi=hi, xg_cluster=mc / gc,
                     lo_cluster=loc, hi_cluster=hic, loss=1 - mu / mc, uf=uf, cf=cf)
    incx = lambda d, r: "" if d[r]["k"] == d[r]["target"] else f"  INCOMPLETE k={d[r]['k']}/{d[r]['target']}"
    EXT.append("=== EXTENSION (Oct 10): uniform start, ALL message sizes (40 users, chi=1, 1200 m). kb/s = reward/250; "
               "x greedy = ladder_vs_eng two-level bootstrap B=4000 (uniform vs USPAWN_ENG, cluster vs eng_U40)")
    EXT.append(f"greedy: cluster {gc / 250:.2f}, uniform {gu / 250:.2f} -> loss {100 * greedy_loss:.1f} %")
    for r in RUNGS7:
        if r in UX:
            x = UX[r]
            EXT.append(f"  {r:5s} k={x['k']} uniform {x['mu'] / 250:7.2f}  x greedy {x['xg']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}] {sig(x['lo'], x['hi'])}"
                       f"  | cluster {x['mc'] / 250:7.2f} x greedy {x['xg_cluster']:.3f} [{x['lo_cluster']:.3f}, {x['hi_cluster']:.3f}]"
                       f"  loss {100 * x['loss']:5.1f} %{incx(UX, r)}")
        else:
            EXT.append(f"  {r:5s} no complete seed yet")
    if UX:
        lx = [100 * UX[r]["loss"] for r in RUNGS7 if r in UX]
        EXT.append(f"  learned-policy loss over {len(lx)} sizes: {min(lx):.1f} to {max(lx):.1f} %, mean {sum(lx) / len(lx):.1f} %")
        bel = [r for r in RUNGS7 if r in UX and UX[r]["lo"] <= 1]
        EXT.append(f"  sizes whose uniform-start lead over greedy has a CI reaching 1 or below: {', '.join(bel) or 'none'}")
        ordx = lambda key: " > ".join(sorted([r for r in RUNGS7 if r in UX], key=lambda r: -UX[r][key]))
        EXT.append(f"  order by mean: cluster {ordx('mc')}   |   uniform {ordx('mu')}"
                   f"   -> {'SAME order' if ordx('mc') == ordx('mu') else 'order CHANGES'}")
        for r, x in UX.items():
            EXTROWS.append(dict(rung=f"{r}_uniform7", **{k: v for k, v in x.items() if k not in ("uf", "cf")},
                                fams_uniform="+".join(x["uf"]), fams_cluster="+".join(x["cf"])))
    # 4b. final vs best checkpoint, every size (cluster start)
    EXT.append("")
    EXT.append("=== EXTENSION (Oct 10): final vs best checkpoint, ALL message sizes (40 users, chi=1, 1200 m, cluster start). "
               "x greedy vs eng_U40; best/final = ratio of pooled means, CI arm_vs_arm B=20000")
    CX, allb, allf = {}, [], []
    for r in RUNGS7:
        pairs = [(cluster_fam(r, k), f"CKFIN_{r}s{k}", k) for k in SX[r]]
        ok = [(b, f, k) for b, f, k in pairs if complete(b) and complete(f)]
        for b, f, _ in pairs:
            for x in (b, f):
                if not complete(x):
                    EXTSTAT.append(f"{x}: {len(DATA.get(x, {}))} episodes -> seed not pooled (needs best and final)")
        if not ok:
            EXT.append(f"  {r:5s} no complete seed yet")
            continue
        bf, ff = [b for b, _, _ in ok], [f for _, f, _ in ok]
        mb, mf = pooled(bf), pooled(ff)
        lor, hir = arm_vs_arm([DATA[x] for x in bf], [DATA[x] for x in ff])
        same = [f"s{k}" for _, _, k in ok if (r, k) in FINAL_IS_BEST]
        CX[r] = dict(k=len(ok), target=len(SX[r]), best=mb / gc, final=mf / gc, ratio=mb / mf, lo=lor, hi=hir, mb=mb, mf=mf)
        allb += bf; allf += ff
        EXT.append(f"  {r:5s} k={len(ok)} best {mb / gc:.3f} x greedy -> final {mf / gc:.3f}  best/final {mb / mf:.4f} "
                   f"[{lor:.3f}, {hir:.3f}] {sig(lor, hir)}" + (f"  (final ckpt = best ckpt for {','.join(same)})" if same else "")
                   + incx(CX, r))
        EXTROWS.append(dict(rung=f"{r}_ckpt7", k=len(ok), target=len(SX[r]), best=mb / gc, final=mf / gc, ratio=mb / mf,
                            lo=lor, hi=hir, fams_best="+".join(bf), fams_final="+".join(ff), final_is_best="+".join(same)))
    if len(CX) >= 2:
        lor, hir = arm_vs_arm([DATA[x] for x in allb], [DATA[x] for x in allf])
        pr = pooled(allb) / pooled(allf)
        per = [CX[r]["ratio"] for r in CX]
        gm_ = 1.0
        for v in per:
            gm_ *= v
        gm_ **= 1 / len(per)
        EXT.append(f"  POOLED over {len(CX)} sizes ({sum(CX[r]['k'] for r in CX)} seed families): best/final (ratio of pooled means) "
                   f"{pr:.4f} [{lor:.3f}, {hir:.3f}] {sig(lor, hir)};  geometric mean of the per-size ratios {gm_:.4f};  "
                   f"per-size range {min(per):.3f} to {max(per):.3f}")
        EXTROWS.append(dict(rung="ALL_ckpt7", k=sum(CX[r]["k"] for r in CX), ratio=pr, lo=lor, hi=hir, geo_mean_ratio=gm_))
        ordb = " > ".join(sorted(CX, key=lambda r: -CX[r]["mb"]))
        ordf = " > ".join(sorted(CX, key=lambda r: -CX[r]["mf"]))
        EXT.append(f"  order by mean: best {ordb}   |   final {ordf}   -> {'SAME order' if ordb == ordf else 'order CHANGES'}")
    EXT = EXT + ["", "EXTENSION MISSING / INCOMPLETE (not pooled):"] + (["  " + x for x in EXTSTAT] or ["  none"])

# ---------- 3. cross-check vs the paper source ----------
t = open(A.paper_tex).read()
par = t[t.index("The result survives two choices"):t.index(r"\section{Conclusion}")]
P = dict(loss=re.search(r"about \\qty\{(\d+)\}", par).group(1), greedy=re.search(r"only \\qty\{(\d+)\}", par).group(1),
         xg=re.search(r"still carries \\num\{([\d.]+)\}", par).group(1), ci=re.search(r"CI ([\d.]+)--([\d.]+)", par).groups(),
         ck=re.search(r"by a factor of \\num\{([\d.]+)\}", par).group(1))
L64x = U.get("L64", dict(xg=float("nan"), lo=float("nan"), hi=float("nan")))   # the L64 family may be incomplete (S1)
mine = dict(loss=f"{round(sum(losses) / len(losses) / 10) * 10:.0f}", greedy=f"{100 * greedy_loss:.0f}", xg=f"{L64x['xg']:.2f}",
            ci=(f"{L64x['lo']:.2f}", f"{L64x['hi']:.2f}"), ck=f"{CK['ratio']:.2f}")
REG = [f"{k:6s} paper {P[k]} | this run {mine[k]} {'EXACT' if P[k] == mine[k] else 'DIFFERS'}" for k in P]
REG.append("(loss = mean of the four learned losses rounded to 10 %, 'about 30'; ci = ladder_vs_eng interval of L64, "
           "bootstrap; ck = best/final mean ratio)")
W = [f"Robustness paragraph, MODE {SUF}: seeds " + "  ".join(f"{r}={','.join(map(str, v))}" for r, v in SEEDS.items()), "",
     "REGRESSION vs the paper:"] + ["  " + x for x in REG] + ["", "MISSING / INCOMPLETE (not pooled):"] + (["  " + x for x in STATUS] or ["  none"]) + [""] + W
if EXT:   # extension: only when its families exist
    W += [""] + EXT
if S1:
    W.insert(1, "SPREADING 1 (drone TX 6.59004 mW, same checkpoints): every family below is its spreading-1 re-evaluation (S1_<family>, or the verified SPR1 twin), staged under the original family name by pull_from_cluster_s1.sh (four_seed_staging/pull_raw_s1/MANIFEST.tsv). Paper values quoted below are the spreading-8 numbers of ieee v14.")
open(os.path.join(OUT, "robustness_table.txt"), "w").write("\n".join(W) + "\n")
pd.DataFrame([dict(rung=r, **{k: v for k, v in x.items() if k not in ("uf", "cf")}, fams_uniform="+".join(x["uf"]),
                   fams_cluster="+".join(x["cf"])) for r, x in U.items()] + [dict(rung="L128_ckpt", **CK)] + EXTROWS).to_csv(
    os.path.join(OUT, "robustness_summary.csv"), index=False, float_format="%.4f")
print("\n".join(W))
