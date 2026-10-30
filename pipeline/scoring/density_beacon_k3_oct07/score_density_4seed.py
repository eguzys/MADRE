#!/usr/bin/env python3
r"""fig:density (+ \datamadre) scorer, four-seed version. Pools an explicit training-seed set per message size
at every chi. Estimator, RNG seed, B and the order of every RNG draw are those of the earlier single-seed density
scorer, so --mode current reproduces its rows exactly.

=========================================================================================================
SEED SETS -- the ONLY place to change which training seeds are pooled (or use --seeds L64=1,2,4,5 on the CLI)
=========================================================================================================
"""
SEEDS_4SEED = {"L8": [1, 2, 3, 5], "L16": [1, 2, 3, 4], "L32": [1, 2, 3, 5], "L64": [1, 2, 3, 4],
               "L128": [1, 2, 3, 5], "L256": [1, 2, 3, 4], "L512": [1, 2, 3, 4]}
# Training runs pooled per message size (edit the line above or pass --seeds).
EXTRA_SEEDS = {"L8": [4], "L32": [4], "L64": [5]}   # trained, NOT in the target sets: pooled only with --with-extras
# --mode current = the seed sets of the single-seed scorer: seed 1 (L512: seed 2) at chi .05/.1/.6/1,
# seeds 1,2,3 (L512: 2,3) at chi .2/.4.
SEEDS_CURRENT_K1 = {r: ([2] if r == "L512" else [1]) for r in SEEDS_4SEED}
SEEDS_CURRENT_K3 = {r: ([2, 3] if r == "L512" else [1, 2, 3]) for r in SEEDS_4SEED}
# =========================================================================================================
r"""
FAMILY NAMES (100 users, beacon ON, shard_ub.sbatch): PDUB<chi>_<rung> = seed 1, EXCEPT L512 where the
UNSUFFIXED PDUB<chi>_L512 is SEED 2; every other seed is PDUB<chi>_<rung>s<seed> (so L512 seed 1 = PDUB<chi>_L512s1).
A family counts only if it has exactly the ep seeds 0..99, once each, all with steps == 250. A (chi, rung) cell with
fewer complete seeds than its target is scored but flagged INCOMPLETE, and its row in the pgfplots file is commented
out, so a drop-in never contains a partial cell.

INPUTS
  raw/episodes_from_logs.txt, raw/episodes_step4.txt, raw/sacct_PDUB.txt (next to this script)
  --staging results/four_seed_staging/pull_raw/episodes_logs.txt (4seed mode only): every "ep seed=" line of the
      u100diag shard logs (tab separated fam, shard, ep, reward, steps, src). Families already in raw/ are taken from
      raw/ and cross-checked against the staging copy (|diff| <= 1e-3).
  --old-tex: reference figure source with the beacon-off rows of fig:density; the beacon-off figure is
      recomputed first, which also sets the RNG sequence.
  --paper-tex: paper source (read only), used as a cross-check of the written rows against fig:density/\datamadre.

ESTIMATOR: m = mean(arm episode_reward)/250*1000 over all pooled
  episodes; ratio = mean(arm)/mean(greedy ENUB_N100_<chi>); CI = 95% percentile of the ratio under a two-level
  bootstrap of the arm (training seeds, then episodes; k=1: episodes) and an episode bootstrap of the greedy;
  ep, em = (hi - ratio) * mean(greedy), (ratio - lo) * mean(greedy). B = 20000, rng = default_rng(20261007).

Usage:
  score_density_4seed.py --mode current            # regression: must reproduce fig_density_beacon_on_rows.tex
  score_density_4seed.py --mode 4seed [--with-extras] [--seeds L64=1,2,4,5]
"""
import argparse, os, re, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["current", "4seed"], required=True)
ap.add_argument("--with-extras", action="store_true", help="also pool EXTRA_SEEDS (L8 s4, L32 s4, L64 s5)")
ap.add_argument("--seeds", action="append", default=[], help="override one rung, e.g. L64=1,2,4,5 (4seed mode)")
ap.add_argument("--staging", default=None)   # default: four_seed_staging/pull_raw[_s1]/episodes_logs.txt
ap.add_argument("--old-tex", default=os.path.join(RES, "four_seed_staging", "ref", "beaconoff_reference_antonio_v10.tex"))
ap.add_argument("--paper-tex", default=os.path.join(RES, "four_seed_staging", "ref", "paper_ieee_v14_snapshot_20261009.tex"))
ap.add_argument("--outdir", default=None)
ap.add_argument("--spreading", type=int, choices=[8, 1], default=8, help="1 = score the spreading-1 re-evaluations (S1_*), staged under the original family names in four_seed_staging/pull_raw_s1 (--mode 4seed only)")
A = ap.parse_args()
S1 = A.spreading == 1
if S1 and A.mode != "4seed":
    raise SystemExit("--spreading 1 needs --mode 4seed")
if A.staging is None:
    A.staging = os.path.join(RES, "four_seed_staging", "pull_raw_s1" if S1 else "pull_raw", "episodes_logs.txt")
OUT = A.outdir or os.path.join(RES, "four_seed_scored", f"density_{A.mode}" + ("_extras" if A.with_extras else "") + ("_s1" if S1 else ""))
os.makedirs(OUT, exist_ok=True)
RAW = os.path.join(HERE, "raw")
TEX = A.old_tex
B = 20000
rng = np.random.default_rng(20261007)
CHIS = ["05", "10", "20", "40", "60", "100"]
C = {"05": 5, "10": 10, "20": 20, "40": 40, "60": 60, "100": 100}
RUNGS = ["L8", "L16", "L32", "L64", "L128", "L256", "L512"]
EN_OFF = {"05": "EN_N100_sp5", "10": "EN_N100_sp1", "20": "EN_N100_sp2", "40": "EN_N100_sp4",
          "60": "EN_N100_sp6", "100": "EN_N100_spALL"}
EN_ON = {c: f"ENUB_N100_{c}" for c in CHIS}

S4 = {r: list(v) for r, v in SEEDS_4SEED.items()}
for spec in A.seeds:
    r, v = spec.split("=")
    S4[r] = [int(x) for x in v.split(",")]
if A.with_extras:
    for r, v in EXTRA_SEEDS.items():
        S4[r] = S4[r] + [k for k in v if k not in S4[r]]


def target_seeds(c, r):
    if A.mode == "current":
        return (SEEDS_CURRENT_K3 if c in ("20", "40") else SEEDS_CURRENT_K1)[r]
    return S4[r]


def famname(c, r, k):
    base = 2 if r == "L512" else 1
    return f"PDUB{c}_{r}" if k == base else f"PDUB{c}_{r}s{k}"


# ---- the families behind the beacon-off reference figure ----
OLD = {}
for c in CHIS:
    for r in ["L8", "L32", "L64", "L128", "L256", "L512"]:
        OLD[(c, r)] = [f"PD{c}_{r}"]
for r in ["L8", "L32"]:
    OLD[("20", r)] = [f"U6SP2_{r}", f"U6SP2S2_{r}", f"U6SP2S3_{r}"]
OLD[("20", "L128")] = ["U6SP2S2_L128", "U6SP2S3_L128"]
for r in ["L8", "L32", "L128"]:
    OLD[("40", r)] = [f"U6SP4_{r}", f"U6SP4S2_{r}", f"U6SP4S3_{r}"]
OLD[("40", "L64")] = ["U6SP4_L64", "U6SP4S2_L64"]
OLD[("40", "L256")] = ["U6SP4_eL256", "U6SP4S2_eL256", "U6SP4S3_eL256"]
OLD[("40", "L512")] = ["U6SP4S2_eL512", "U6SP4S3_eL512"]
OLD_SAME = {k: v[0] for k, v in OLD.items()}
OLD_SAME[("20", "L128")] = None
OLD_SAME[("40", "L512")] = "U6SP4S2_eL512"

# ---- load episodes ----
cols = ["fam", "shard", "ep", "r", "steps"]
E = pd.concat([pd.read_csv(os.path.join(RAW, fn), sep=" ", names=cols) for fn in ["episodes_from_logs.txt", "episodes_step4.txt"]])
_s4 = set(pd.read_csv(os.path.join(RAW, "episodes_step4.txt"), sep=" ", names=cols).fam)
E = E[~(E.fam.isin(_s4) & E.duplicated(["fam", "ep"], keep="last"))]
assert not E.duplicated(["fam", "ep"]).any()
assert (E.steps == 250).all()
XCHECK = []
if S1:   # beacon-on families (the figure) come ONLY from the spreading-1 staging; raw/ keeps the spreading-8
    # beacon-OFF families (OLD columns + the reproduction step that sets the RNG sequence)
    E = E[~E.fam.str.match(r"^(PDUB\d+_L\d+(s\d)?|ENUB_N100_\d+)$")]
if A.mode == "4seed":
    G = pd.read_csv(A.staging, sep="\t")
    G = G[G.fam.str.match(r"^(PDUB\d+_L\d+(s\d)?|ENUB_N100_\d+)$")]
    G = G.rename(columns={"reward": "r"})
    G = G.drop_duplicates(["fam", "ep"], keep="last")[cols]
    have = set(E.fam)
    for f, g in G[G.fam.isin(have)].groupby("fam"):
        a = E[E.fam == f].set_index("ep").r.sort_index(); b = g.set_index("ep").r.sort_index()
        com = a.index.intersection(b.index)
        d = float(np.abs(a[com].values - b[com].values).max()) if len(com) else float("nan")
        XCHECK.append(f"{f}: raw n={len(a)} staging n={len(b)} common={len(com)} max|diff|={d:.2g} " + ("OK" if len(com) == len(a) == len(b) and d <= 1e-3 else "DIFFERS"))
    E = pd.concat([E, G[~G.fam.isin(have)]])
FAM, BAD = {}, {}
for f, g in E.groupby("fam"):
    FAM[f] = g.set_index("ep").r.sort_index()
    if not (g.steps == 250).all():
        BAD[f] = "steps != 250"


def complete(f):
    return f in FAM and f not in BAD and sorted(FAM[f].index) == list(range(100))


# ---- completeness (current mode also requires the sacct check for the seed-1 families) ----
sacct = open(os.path.join(RAW, "sacct_PDUB.txt")).read().split("===Q")[0]
st = {}
for line in sacct.strip().splitlines():
    p = line.split("|")
    st.setdefault(re.sub(r"_s\d+$", "", p[1].strip()), []).append(p[2].strip())


def ok_family(f):
    if not complete(f):
        return False
    if A.mode == "current" and re.fullmatch(r"PDUB\d+_L\d+", f):
        return st.get(f, []).count("COMPLETED") == 10 and len(st[f]) == 10
    return True


NEW, PENDING, MISSING, TARGET = {}, [], {}, {}
for c in CHIS:
    for r in RUNGS:
        want = [famname(c, r, k) for k in target_seeds(c, r)]
        TARGET[(c, r)] = want
        if r != "L512" and f"PDUB{c}_{r}s1" in FAM:
            raise SystemExit(f"AMBIGUOUS: PDUB{c}_{r}s1 exists next to the unsuffixed seed-1 family")
        if f"PDUB{c}_L512s2" in FAM:
            raise SystemExit(f"AMBIGUOUS: PDUB{c}_L512s2 exists next to the unsuffixed seed-2 family PDUB{c}_L512")
        got = [f for f in want if ok_family(f)]
        miss = [f for f in want if f not in got]
        if miss:
            MISSING[(c, r)] = miss
            for f in miss:
                PENDING.append(f"{f}: {len(FAM.get(f, []))} episodes" + (f" ({BAD[f]})" if f in BAD else "")
                               + (f", sacct {sorted(set(st.get(f, [])))} x{len(st.get(f, []))}" if A.mode == "current" else ""))
        if got:
            NEW[(c, r)] = got
# reference set for the comparison column: current mode -> seed-1 only;
# 4seed mode -> the --mode current seed set (the table shows both values)
NEW1 = {}
for (c, r), fams in NEW.items():
    if A.mode == "current":
        NEW1[(c, r)] = [fams[0]] if fams[0] == famname(c, r, 2 if r == "L512" else 1) else None
    else:
        ref = [famname(c, r, k) for k in (SEEDS_CURRENT_K3 if c in ("20", "40") else SEEDS_CURRENT_K1)[r]]
        NEW1[(c, r)] = ref if all(complete(f) for f in ref) else None
NEW1 = {k: v for k, v in NEW1.items() if v}
pooled_extra = sorted({f for v in NEW.values() for f in v if re.search(r"s\d$", f)})


def bps(f):
    return FAM[f].values / 250 * 1000


def two_level(fams, n_draw=B):
    arrs = [bps(f) for f in fams]
    k = len(arrs)
    if k == 1:
        a = arrs[0]
        return a[rng.integers(0, len(a), (n_draw, len(a)))].mean(1)
    picks = rng.integers(0, k, (n_draw, k))
    tot = np.zeros(n_draw); cnt = np.zeros(n_draw)
    for j in range(k):
        for s in range(k):
            sel = picks[:, j] == s
            a = arrs[s]
            m = a[rng.integers(0, len(a), (sel.sum(), len(a)))].sum(1)
            tot[sel] += m; cnt[sel] += len(a)
    return tot / cnt


def point(fams):
    return np.concatenate([bps(f) for f in fams]).mean()


def ratio_ci(fams, den):
    num_d = two_level(fams)
    d = bps(den)
    den_d = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    r = num_d / den_d
    pt = point(fams) / d.mean()
    lo, hi = np.percentile(r, [2.5, 97.5])
    return pt, lo, hi, d.mean()


def bar(fams, den):
    pt, lo, hi, dm = ratio_ci(fams, den)
    m = pt * dm
    return dict(m=m, ep=(hi - pt) * dm, em=(pt - lo) * dm, ratio=pt, rlo=lo, rhi=hi, k=len(fams),
                n=sum(len(FAM[f]) for f in fams))


def arm_vs_arm(a_fams, b_fams, paired=False):
    if paired:
        a, b = FAM[a_fams[0]], FAM[b_fams[0]]
        com = a.index.intersection(b.index)
        x, y = a[com].values, b[com].values
        idx = rng.integers(0, len(com), (B, len(com)))
        r = x[idx].mean(1) / y[idx].mean(1)
        return x.mean() / y.mean(), *np.percentile(r, [2.5, 97.5]), len(com)
    r = two_level(a_fams) / two_level(b_fams)
    return point(a_fams) / point(b_fams), *np.percentile(r, [2.5, 97.5]), None


sig = lambda lo, hi: "SIG" if lo > 1 or hi < 1 else "ns"


def parse_density(texpath):
    t = open(texpath).read()
    dens = t[t.index(r"\addlegendentry{\acs{MADRE}, $L=64$}"):t.index(r"\label{fig:density}")]
    dens = dens[dens.index(r"\begin{axis}"):]
    P = {}
    for body, leg in re.findall(r"\\addplot \[.*?\]\s*table \[[^\]]*\] \{(.*?)\};\s*\\addlegendentry\{(.*?)\}\n", dens, re.S):
        L = re.search(r"L=(\d+)", leg)
        key = f"L{L.group(1)}" if L else "Greedy"
        P[key] = {int(a[0]): tuple(int(x) for x in a[1:] if x != "")
                  for a in re.findall(r"^\s*(\d+)\s+(\d+)(?:\s+(\d+)\s+(\d+))?\s*\\\\", body, re.M)}
    dm = re.search(r"\\pgfplotstableread\[row sep=\\\\\]\{(.*?)\}\\datamadre", t, re.S).group(1)
    P["L64"] = {int(c): (int(m), int(p), int(q)) for c, m, p, q in re.findall(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\\\\", dm, re.M)}
    return P


# ---------- 1. beacon-off reference figure (also sets the RNG sequence) ----------
PLOT = parse_density(TEX)
repro, OLDB = [], {}
for c in CHIS:
    g = PLOT["Greedy"][C[c]][0]
    gm = bps(EN_OFF[c]).mean()
    repro.append(f"Greedy chi={C[c]:>3d} tex {g:>7d} | {EN_OFF[c]} {gm:9.1f} -> {round(gm, -2):.0f} {'EXACT' if round(gm, -2) == g else 'MISMATCH'}")
    for r in ["L8", "L32", "L64", "L128", "L256", "L512"]:
        b = bar(OLD[(c, r)], EN_OFF[c])
        OLDB[(c, r)] = b
        m, p, q = PLOT[r][C[c]]
        ok = round(b["m"], -2) == m
        repro.append(f"{r:5s} chi={C[c]:>3d} tex m={m:>6d} ep={p:>5d} em={q:>5d} | k={b['k']} n={b['n']:3d} m={b['m']:8.0f} "
                     f"ep={b['ep']:6.0f} em={b['em']:6.0f}  mean {'EXACT' if ok else 'MISMATCH'}; "
                     f"bars dev {b['ep'] - p:+5.0f}/{b['em'] - q:+5.0f}  [{'+'.join(OLD[(c, r)])}]")
        assert ok, repro[-1]

# ---------- 2. beacon-on (THE FIGURE ROWS) ----------
NEWB = {k: bar(v, EN_ON[k[0]]) for k, v in NEW.items()}
NEWB1 = {k: bar(v, EN_ON[k[0]]) for k, v in NEW1.items()}

# ---------- 3. beacon-off vs beacon-on tests ----------
rows = []
for c in CHIS:
    for r in RUNGS:
        o, n = OLDB.get((c, r)), NEWB.get((c, r))
        rec = dict(chi=C[c] / 100, L=r, target_k=len(TARGET[(c, r)]), missing="+".join(MISSING.get((c, r), [])))
        for tag, b in [("old", o), ("new", n)]:
            if b:
                rec.update({f"{tag}_k": b["k"], f"{tag}_n": b["n"], f"{tag}_m": b["m"], f"{tag}_lo": b["m"] - b["em"],
                            f"{tag}_hi": b["m"] + b["ep"], f"{tag}_ratio": b["ratio"], f"{tag}_rlo": b["rlo"], f"{tag}_rhi": b["rhi"]})
        b1 = NEWB1.get((c, r))
        if b1:
            rec.update(s1_k=b1["k"], s1_m=b1["m"], s1_lo=b1["m"] - b1["em"], s1_hi=b1["m"] + b1["ep"])
        if n:
            rec["new_fams"] = "+".join(NEW[(c, r)])
        rec["greedy_old"] = bps(EN_OFF[c]).mean(); rec["greedy_new"] = bps(EN_ON[c]).mean()
        if o and n:
            x, lo, hi, _ = arm_vs_arm(NEW[(c, r)], OLD[(c, r)])
            rec.update(newold=x, newold_lo=lo, newold_hi=hi, newold_sig=sig(lo, hi))
            same = OLD_SAME[(c, r)]
            same_new = famname(c, r, 2 if r == "L512" else 1)
            if same and same_new in NEW[(c, r)]:
                x, lo, hi, nn = arm_vs_arm([same_new], [same], paired=True)
                rec.update(paired_fam=same, paired=x, paired_lo=lo, paired_hi=hi, paired_sig=sig(lo, hi))
        rows.append(rec)
T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, "density_old_vs_new.csv"), index=False, float_format="%.4f")

TAG_REF = "SEED-1 (beacon-on, k=1)" if A.mode == "current" else "CURRENT-PAPER seed set (beacon-on)"
TAG_NEW = "K3 (beacon-on, chi .2/.4 pooled)" if A.mode == "current" else "FOUR-SEED (beacon-on)"
rank_lines = []
for tag, BB, FF in [("OLD (beacon-off)", OLDB, OLD), (TAG_REF, NEWB1, NEW1), (TAG_NEW, NEWB, NEW)]:
    rank_lines.append(f"--- {tag}: rungs by mean, and min(L64,L128) vs every other rung ---")
    for c in CHIS:
        have = [r for r in RUNGS if (c, r) in BB]
        order = sorted(have, key=lambda r: -BB[(c, r)]["m"])
        top2 = set(order[:2]) == {"L64", "L128"} if {"L64", "L128"} <= set(have) else None
        s = f"chi={C[c] / 100:4.2f}: " + " > ".join(f"{r}({BB[(c, r)]['m'] / 1000:.1f}k" + (f",k={BB[(c, r)]['k']}" if BB is NEWB and BB[(c, r)]['k'] < len(TARGET[(c, r)]) else "") + ")" for r in order)
        s += f"   top-two = L64,L128: {top2 if top2 is not None else 'n/a (pending)'}"
        rank_lines.append(s)
        for top in ["L64", "L128"]:
            if (c, top) not in BB:
                continue
            for r in have:
                if r in ("L64", "L128"):
                    continue
                x, lo, hi, _ = arm_vs_arm(FF[(c, top)], FF[(c, r)])
                extra_s = ""
                if len(FF[(c, top)]) == 1 and len(FF[(c, r)]) == 1:
                    xp, plo, phi, nn = arm_vs_arm(FF[(c, top)], FF[(c, r)], paired=True)
                    extra_s = f"  paired {xp:.3f} [{plo:.3f},{phi:.3f}] {sig(plo, phi)}"
                rank_lines.append(f"      {top}/{r:5s} {x:.3f} [{lo:.3f},{hi:.3f}] {sig(lo, hi):3s}{extra_s}")
        if (c, "L64") in BB and (c, "L128") in BB:
            x, lo, hi, _ = arm_vs_arm(FF[(c, "L64")], FF[(c, "L128")])
            rank_lines.append(f"      L64/L128  {x:.3f} [{lo:.3f},{hi:.3f}] {sig(lo, hi)}")
        for top in ["L64", "L128"]:
            if (c, "L16") in BB and (c, top) in BB:
                x, lo, hi, _ = arm_vs_arm(FF[(c, "L16")], FF[(c, top)])
                rank_lines.append(f"      L16/{top:4s} {x:.3f} [{lo:.3f},{hi:.3f}] {sig(lo, hi)}")

# ---------- 4. MADRE (L=64) vs fig:baselines arms, beacon-on ----------
if not S1:
    BL = pd.read_csv(os.path.join(RES, "baselines_beacon_oct07", "episodes_baselines_long.csv"))
    BL = BL[BL.beacon == "ON"]
    g60 = pd.read_csv(os.path.join(RAW, "OM2G_baseline_gtm_N8_n10_DUB100_GTMG_seed60.csv"))
    assert (g60.steps == 250).all() and len(g60) == 10
    BL = pd.concat([BL, pd.DataFrame(dict(beacon="ON", chi=1.0, arm="GTMG", ep=g60.seed, shard=60,
                                          reward=g60.episode_reward, bps=g60.episode_reward / 250 * 1000))])
else:   # spreading 1: the S1_DUB<chi>_<arm> CSVs of the spreading-1 staging
    import glob as _glob
    _b = []
    for _f in sorted(_glob.glob(os.path.join(os.path.dirname(A.staging), "csv", "OM2G_baseline_*_n10_DUB*_seed*.csv"))):
        _m = re.search(r"_DUB(\d+)_(GTMG|GRD|AGG|R1H)_seed(\d+)\.csv$", _f)
        if not _m:
            continue
        _x = pd.read_csv(_f)
        assert (_x.steps == 250).all(), _f
        _b.append(pd.DataFrame(dict(beacon="ON", chi=C[_m.group(1)] / 100, arm=_m.group(2), ep=_x.seed, shard=int(_m.group(3)),
                                    reward=_x.episode_reward, bps=_x.episode_reward / 250 * 1000)))
    BL = pd.concat(_b) if _b else pd.DataFrame(columns=["beacon", "chi", "arm", "ep", "shard", "reward", "bps"])
madre = []
for c in CHIS:
    if f"PDUB{c}_L64" not in FAM or (c, "L64") not in NEW:
        madre.append(f"chi={C[c] / 100:4.2f}: MADRE PENDING ({'PDUB' + c + '_L64'})")
        continue
    M = FAM[f"PDUB{c}_L64"] / 250 * 1000
    line = [f"chi={C[c] / 100:4.2f}: MADRE(seed 1) {M.mean():8.0f}"]
    for arm in ["GTMG", "GRD", "AGG", "R1H"]:
        Y = BL[(BL.chi == C[c] / 100) & (BL.arm == arm)].set_index("ep").bps.sort_index()
        if S1 and len(Y) != 100:   # spreading-1 baseline not complete yet
            line.append(f"/{arm} PENDING (n={len(Y)})"); continue
        assert Y.index.is_unique and len(Y) == 100, (c, arm, len(Y))
        x = M.values; y = Y.values
        r = x[rng.integers(0, 100, (B, 100))].mean(1) / y[rng.integers(0, 100, (B, 100))].mean(1)
        lo, hi = np.percentile(r, [2.5, 97.5])
        com = M.index.intersection(Y.index); idx = rng.integers(0, len(com), (B, len(com)))
        rp = M[com].values[idx].mean(1) / Y[com].values[idx].mean(1)
        plo, phi = np.percentile(rp, [2.5, 97.5])
        line.append(f"/{arm} {x.mean() / y.mean():.3f} [{lo:.3f},{hi:.3f}] {sig(lo, hi)} (paired [{plo:.3f},{phi:.3f}] {sig(plo, phi)})")
    madre.append("  ".join(line))
POOLED_CHIS = ["20", "40"] if A.mode == "current" else [c for c in CHIS if (c, "L64") in NEW and len(NEW[(c, "L64")]) > 1]
for c in POOLED_CHIS:
    for arm in ["GTMG", "GRD"] + ([] if A.mode == "current" else ["AGG", "R1H"]):
        Y = BL[(BL.chi == C[c] / 100) & (BL.arm == arm)].bps.values
        if S1 and len(Y) != 100:
            madre.append(f"POOLED chi={C[c] / 100:4.2f}: /{arm} PENDING (n={len(Y)})"); continue
        num = two_level(NEW[(c, "L64")]); den = Y[rng.integers(0, len(Y), (B, len(Y)))].mean(1)
        lo, hi = np.percentile(num / den, [2.5, 97.5])
        madre.append(f"POOLED chi={C[c] / 100:4.2f}: MADRE {point(NEW[(c, 'L64')]):8.0f} (k={len(NEW[(c, 'L64')])}) /{arm} {point(NEW[(c, 'L64')]) / Y.mean():.3f} [{lo:.3f},{hi:.3f}] {sig(lo, hi)}")

# ---------- 5. pgfplots rows ----------
def seeds_txt(c, r):
    return ",".join(str(k) for k in target_seeds(c, r))


L = ["% fig:density, BEACON ON (PDUB<chi>_<rung>[s<seed>]), 100 users, 1200 m, cluster spawn.",
     "% m = mean/250*1000 [bit/s] rounded to 100; ep/em = ratio-of-means 95% CI vs the beacon-on greedy ENUB_N100_<chi>",
     "% (arm: two-level bootstrap training seeds -> episodes; greedy: episodes) x greedy mean, rounded to 100.",
     f"% MODE {A.mode}{' + extras' if A.with_extras else ''}: seed sets per rung (chi .2/.4 | other chi):",
     "%   " + "  ".join(f"{r}={seeds_txt('20', r)}|{seeds_txt('05', r)}" for r in RUNGS),
     "% Rows whose cell has fewer complete seeds than its target are COMMENTED OUT and tagged INCOMPLETE.",
     "% scored by results/density_beacon_k3_oct07/score_density_4seed.py. Nothing here is in the paper.", ""]
L.append("% --- Greedy (dashed, no bars): ENUB_N100_<chi> ---")
L.append("    c     m        \\\\")
for c in CHIS:
    L.append(f"    {C[c]:<5d} {round(bps(EN_ON[c]).mean(), -2):<9.0f} \\\\")
ROWS = {}
for r in RUNGS:
    L.append("")
    L.append(f"% --- L={r[1:]} ---" + ("   = \\datamadre (preamble; also the MADRE line of fig:baselines, which skips index 3 = c40)" if r == "L64" else ""))
    L.append("    c     m        ep      em     \\\\")
    for c in CHIS:
        if (c, r) in NEWB:
            b = NEWB[(c, r)]
            row = f"    {C[c]:<5d} {round(b['m'], -2):<8.0f} {round(b['ep'], -2):<7.0f} {round(b['em'], -2):<6.0f} \\\\"
            if (c, r) in MISSING:
                L.append("%" + row[1:] + f"   % INCOMPLETE k={b['k']}/{len(TARGET[(c, r)])}, missing {'+'.join(MISSING[(c, r)])}")
            else:
                L.append(row)
                ROWS[(r, C[c])] = (int(round(b['m'], -2)), int(round(b['ep'], -2)), int(round(b['em'], -2)))
        else:
            o = OLDB.get((c, r))
            L.append(f"%   {C[c]:<5d} PENDING (no complete seed among {'+'.join(TARGET[(c, r)])})" + (f"   (old beacon-off row: {round(o['m'], -2):.0f})" if o else ""))
open(os.path.join(OUT, "fig_density_beacon_on_rows.tex"), "w").write("\n".join(L) + "\n")

# ---------- 5b. cross-check: rows vs the reference rows file and vs the paper source ----------
REG = []
if A.mode == "current" and not A.with_extras:
    k3 = open(os.path.join(HERE, "fig_density_beacon_on_rows.tex")).read()
    same = [l for l in k3.splitlines() if not l.startswith("%")] == [l for l in "\n".join(L).splitlines() if not l.startswith("%")]
    REG.append(f"non-comment lines identical to density_beacon_k3_oct07/fig_density_beacon_on_rows.tex: {same}")
P = parse_density(A.paper_tex)
nmis = 0
for r in RUNGS:
    for c in CHIS:
        tex = P.get(r, {}).get(C[c])
        mine = ROWS.get((r, C[c]))
        if tex is None:
            continue
        if mine != tex:
            nmis += 1
        if A.mode == "current" or mine != tex:
            REG.append(f"{r:5s} chi={C[c]:>3d} paper {tex} | this run {mine if mine else 'INCOMPLETE/PENDING'} {'EXACT' if mine == tex else 'DIFFERS'}")
for c in CHIS:
    g = round(bps(EN_ON[c]).mean(), -2)
    tex = P["Greedy"][C[c]][0]
    if g != tex:
        nmis += 1
    REG.append(f"Greedy chi={C[c]:>3d} paper {tex} | this run {g:.0f} {'EXACT' if g == tex else 'DIFFERS'}")
REG.insert(0, f"paper ({os.path.basename(A.paper_tex)}) fig:density + \\datamadre: {nmis} coordinates differ from this run")

# ---------- 6. text table ----------
fmt = lambda m, lo, hi: f"{m / 1000:6.1f} [{lo / 1000:5.1f},{hi / 1000:6.1f}]"
REFCOL = "s1" if A.mode == "current" else "paper"
W = [f"fig:density MODE={A.mode}{' +extras' if A.with_extras else ''}. OLD (beacon-off) vs beacon-on: '{REFCOL}' = "
     + ("seed-1-only mean" if A.mode == "current" else "mean with the CURRENT PAPER seed set") + ", then the scored seed set mean [CI]. kb/s.",
     "mean [95% CI] = ratio-to-greedy CI x greedy mean (the figure's own bars). xG = ratio to greedy at the same chi.",
     "new/old = ratio of means with two-level bootstrap of both sides (independent, conservative);",
     "paired = new family of the SAME training seed as the old one (seed 1; L512 seed 2) vs the old family, 100 shared ep seeds.",
     "k = complete seeds pooled; '!k' marks a cell below its target seed count (INCOMPLETE).", ""]
W.append(f"{'chi':>4s} {'L':5s}| {'k':>1s} {'OLD mean [CI]':22s} {'xG':>5s} | {'NEW mean [CI]':22s} {'xG':>5s} | {'new/old [CI]':24s} | paired new/old-same-seed")
W.append("-" * 130)
for _, x in T.iterrows():
    o = fmt(x.old_m, x.old_lo, x.old_hi) if pd.notna(x.get("old_m")) else "--".center(22)
    ok = f"{int(x.old_k)}" if pd.notna(x.get("old_k")) else " "
    oxg = f"{x.old_ratio:5.2f}" if pd.notna(x.get("old_ratio")) else "     "
    n = fmt(x.new_m, x.new_lo, x.new_hi) if pd.notna(x.get("new_m")) else "PENDING".center(22)
    s1 = f"{x.s1_m / 1000:6.1f}" if pd.notna(x.get("s1_m")) else "      "
    kk = f"{int(x.new_k)}" if pd.notna(x.get("new_k")) else "-"
    if pd.notna(x.get("new_k")) and int(x.new_k) < int(x.target_k):
        kk = "!" + kk
    n = f"{REFCOL} {s1} | k={kk} " + n
    nxg = f"{x.new_ratio:5.2f}" if pd.notna(x.get("new_ratio")) else "     "
    no = f"{x.newold:.3f} [{x.newold_lo:.3f},{x.newold_hi:.3f}] {x.newold_sig}" if pd.notna(x.get("newold")) else ""
    pa = f"{x.paired:.3f} [{x.paired_lo:.3f},{x.paired_hi:.3f}] {x.paired_sig} vs {x.paired_fam}" if pd.notna(x.get("paired")) else ""
    W.append(f"{x.chi:4.2f} {x.L:5s}| {ok} {o} {oxg} | {n} {nxg} | {no:24s} | {pa}")
    if x.L == "L512":
        W.append("")
if S1:
    W.insert(0, "SPREADING 1 (drone TX 6.59004 mW, same checkpoints): every family below is its spreading-1 re-evaluation (S1_<family>, or the verified SPR1 twin), staged under the original family name by pull_from_cluster_s1.sh (four_seed_staging/pull_raw_s1/MANIFEST.tsv). Paper values quoted below are the spreading-8 numbers of ieee v14.")
    W.insert(1, "OLD (beacon-off) columns and the reproduction section are the spreading-8 Sep data (not re-evaluated, not in the paper).")
W += rank_lines + ["", "MADRE (L=64, beacon-on) vs the fig:baselines arms (beacon-on DUB<chi>_*; GTM chi=1 n=100 incl. rerun s60):"] + ["  " + m for m in madre]
W += ["", "MISSING / INCOMPLETE beacon-on families (not pooled):"] + (["  " + p for p in PENDING] or ["  none"])
W += ["", "Suffixed (non-base-seed) families pooled: " + ", ".join(pooled_extra)]
if XCHECK:
    W += ["", "raw/ vs staging cross-check (families present in both):"] + ["  " + x for x in XCHECK if not x.endswith("OK")] \
         + [f"  {sum(x.endswith('OK') for x in XCHECK)} of {len(XCHECK)} families identical (|diff| <= 1e-3)"]
W += ["", "REGRESSION vs the paper:"] + ["  " + r for r in REG]
W += ["", "Reproduction of the OLD beacon-off figure (mean must round EXACTLY; bars to bootstrap noise):"] + ["  " + r for r in repro]
open(os.path.join(OUT, "old_vs_new_table.txt"), "w").write("\n".join(W) + "\n")
print("\n".join(W))

# ---------- 7. PNG ----------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
col = {"L8": "#1b9aaa", "L16": "#2a7f62", "L32": "#2e8b2e", "L64": "#7b3fa0", "L128": "#1f4fd1", "L256": "#d97a00", "L512": "#b22222"}
mk = {"L8": "D", "L16": "o", "L32": "s", "L64": "p", "L128": "o", "L256": "^", "L512": "x"}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for i, (tag, BB, EN) in enumerate([("OLD: beacon OFF", OLDB, EN_OFF), (f"beacon ON, mode {A.mode} (hollow = INCOMPLETE)", NEWB, EN_ON)]):
    ax[i].plot([C[c] for c in CHIS], [bps(EN[c]).mean() for c in CHIS], "--", color="grey", label="Greedy")
    for r in RUNGS:
        pts = [(C[c], BB[(c, r)], (c, r) in MISSING and i == 1) for c in CHIS if (c, r) in BB]
        if not pts:
            continue
        ax[i].errorbar([p[0] for p in pts], [p[1]["m"] for p in pts], yerr=[[p[1]["em"] for p in pts], [p[1]["ep"] for p in pts]],
                       color=col[r], marker=mk[r], ms=5, capsize=2, lw=1.6 if r in ("L64", "L128") else 1.1, label=f"L={r[1:]}")
        inc = [p for p in pts if p[2]]
        if inc:
            ax[i].plot([p[0] for p in inc], [p[1]["m"] for p in inc], "o", mfc="white", mec=col[r], ms=9)
    ax[i].set_title(tag, fontsize=10)
    ax[i].set_xlabel("Active user ratio chi [%]"); ax[i].grid(alpha=.3); ax[i].set_xticks([5, 10, 20, 40, 60, 100])
ax[0].set_ylabel("Throughput [bit/s]"); ax[1].legend(fontsize=8, ncol=2, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_density_old_vs_new.png"), dpi=130)
