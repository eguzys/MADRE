#!/usr/bin/env python3
r"""fig:baselines scorer: beacon-off (D<chi>_*) vs beacon-on (DUB<chi>_*) baseline families, plus the MADRE line.
World: 100 users, 1200 m, cluster spawn, 4 clusters, 8 drones, 250-step episodes, ep seeds 0..99 (10 shards x 10).
Baseline codes -> legend: GTMG = GTM (SUITE_GTM_OBJ=global, uncharged), GRD = Greedy, AGG = Heuristic message (131 B),
R1H = Group relay (relay_1hop, RELAY_MODE=blind, 51 B).  (ENUB_N100_* / EN_N100_* = engineered1hop, the Greedy of
fig:density; used here only as the denominator of the MADRE ratio-CI, as in fig:density.)

ESTIMATOR (check_old_rows() checks it against the reference figure source):
  m = mean(episode_reward) / 250 * 1000   [bit/s]
  e = 1.96 * sd(ddof=1) / sqrt(n)         [symmetric 95% normal CI over the episodes]
MADRE line (\datamadre, shared with fig:density): L=64 density fine-tune br12rtL64s<k>u100sp<N>UB,
  m = mean/250*1000 (rounded to 100 in the paper); bars = ratio-of-means percentile 95% CI vs the engineered baseline
  at the same chi, multiplied by the baseline mean. Several training seeds are pooled with a two-level bootstrap
  (training seeds, then episodes); paired-by-episode tests use the first seed of the set.

Inputs: DUB100_GTMG s60 is read from the completed job of that shard (n=100).
The MADRE training seeds are an explicit set per chi (config below); several CSV dirs can be given;
\datamadre is read from the density rows file of the SAME mode (score_density_4seed.py output), rows tagged
INCOMPLETE there stay commented; outputs go to <scoring root>/four_seed_scored/baselines_<mode>/.
Baseline arms (GTMG/GRD/AGG/R1H/ENG) are deterministic policies, one family each: unaffected by the seed sets.

Usage: score_baselines_4seed.py --mode current|4seed [--seeds 1,2,4,5] [--with-extras]
       [--csv DIR ...] [--old-tex beacon-off tex] [--density-rows FILE]
"""
# =========================================================================================================
# SEED SETS -- MADRE = L64. The ONLY place to change which training seeds are pooled (or --seeds 1,2,4,5).
SEEDS_L64_4SEED = [1, 2, 3, 4]
# L64 training runs pooled (edit the line above or pass --seeds).
EXTRA_SEEDS_L64 = [5]          # trained, NOT in the target set: pooled only with --with-extras (no 100-user s5 exists)
SEEDS_L64_CURRENT = {"05": [1], "10": [1], "20": [1, 2, 3], "40": [1, 2, 3], "60": [1], "100": [1]}   # --mode current seed sets
# =========================================================================================================
import argparse, glob, os, re, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["current", "4seed"], required=True)
ap.add_argument("--seeds", action="append", default=[])
ap.add_argument("--with-extras", action="store_true")
ap.add_argument("--csv", action="append", default=None)
ap.add_argument("--old-tex", default=os.path.join(RES, "four_seed_staging", "ref", "beaconoff_reference_antonio_v10.tex"))
ap.add_argument("--density-rows", default=None)
ap.add_argument("--paper-tex", default=os.path.join(RES, "four_seed_staging", "ref", "paper_ieee_v14_snapshot_20261009.tex"))
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

CSVS = A.csv or ([os.path.join(RES, "four_seed_staging", "pull_raw_s1", "csv")] if S1 else
                 [os.path.join(HERE, "csv"), os.path.join(RES, "four_seed_staging", "pull_raw", "csv")])
TEX = A.old_tex
SUF = A.mode + ("_extras" if A.with_extras else "") + ("_s1" if S1 else "")
OUT = A.outdir or os.path.join(RES, "four_seed_scored", f"baselines_{SUF}")
os.makedirs(OUT, exist_ok=True)
DENSROWS = A.density_rows or os.path.join(RES, "four_seed_scored", f"density_{SUF}", "fig_density_beacon_on_rows.tex")
S64 = _l64_override(A.seeds) or list(SEEDS_L64_4SEED)
if A.with_extras:
    S64 += [k for k in EXTRA_SEEDS_L64 if k not in S64]
SEEDS_L64 = SEEDS_L64_CURRENT if A.mode == "current" else {c: S64 for c in SEEDS_L64_CURRENT}
MADRE_MISSING = {}
rng = np.random.default_rng(20261007)
B = 20000
CHIS = ["05", "10", "20", "40", "60", "100"]
CODES = {"GTMG": "GTM", "GRD": "Greedy", "AGG": "Heuristic message", "R1H": "One-hop relay"}
EN_OFF = {"05": "EN_N100_sp5", "10": "EN_N100_sp1", "20": "EN_N100_sp2", "40": "EN_N100_sp4",
          "60": "EN_N100_sp6", "100": "EN_N100_spALL"}
PROVISIONAL = set()


def load(pattern):
    # first CSV dir that holds the pattern wins (the csv/ dir next to this script first, so --mode current reads those files)
    fs = []
    for CSV in CSVS:
        fs = sorted(glob.glob(os.path.join(CSV, pattern)))
        if fs:
            break
    if not fs:
        return None
    d = []
    for f in fs:
        x = pd.read_csv(f)
        assert list(x.columns[:3]) == ["seed", "episode_reward", "steps"], f
        assert (x.steps == 250).all(), f
        x["shard"] = int(re.search(r"_seed(\d+)\.csv$", f).group(1))
        d.append(x)
    d = pd.concat(d)
    assert d.seed.is_unique, pattern
    return d.set_index("seed").sort_index()


# ---------- load ----------
rows = []
FAM = {}
POOL = {}
for chi in CHIS:
    for bc, pre in ([("ON", "DUB")] if S1 else [("OFF", "D"), ("ON", "DUB")]):
        for code in CODES:
            d = load(f"OM2G_baseline_*_n10_{pre}{chi}_{code}_seed*.csv")
            if d is None:
                continue
            if (bc, chi, code) == ("ON", "100", "GTMG") and not S1:
                assert sorted(set(d.shard)) == list(range(0, 100, 10)) and len(d) == 100, "GTM chi=1 must be complete"
            FAM[(bc, chi, code)] = d
        en = load(f"OM2G_baseline_engineered1hop_N8_n10_{EN_OFF[chi] if bc == 'OFF' else 'ENUB_N100_' + chi}_seed*.csv")
        FAM[(bc, chi, "ENG")] = en
        if bc == "OFF":
            md = load(f"OM2G_trained_br12rtL64s1u100*_PD{chi}_L64_n10_seed*.csv")
            if md is not None:
                FAM[(bc, chi, "MADRE")] = md
                POOL[(bc, chi)] = [md]
            continue
        # beacon ON: one family per training seed in SEEDS_L64[chi]; seed 1 = unsuffixed PDUB<chi>_L64
        got = []
        for k in SEEDS_L64[chi]:
            pat = (f"OM2G_trained_br12rtL64s1u100*_PDUB{chi}_L64_n10_seed*.csv" if k == 1 else
                   f"OM2G_trained_br12rtL64s{k}u100*UB_PDUB{chi}_L64s{k}_n10_seed*.csv")
            x = load(pat)
            if x is not None and sorted(x.index) == list(range(100)):
                got.append((k, x))
            else:
                MADRE_MISSING.setdefault(chi, []).append(f"PDUB{chi}_L64{'' if k == 1 else f's{k}'} ({0 if x is None else len(x)} eps in CSVs)")
        if got:
            # FAM[...MADRE] = the seed-1 family (paired tests use it); if seed 1 is not in the set, the first seed
            FAM[(bc, chi, "MADRE")] = got[0][1]
            POOL[(bc, chi)] = [x for _, x in got]
# spreading 1: a baseline family that is missing or not exactly ep seeds 0..99 is NOT used (INCOMPLETE); MADRE at a
# chi whose greedy ENUB_N100_<chi> is missing is moved to MADRE_MISSING (its bars need that greedy).
S1_INC = {}
if S1:
    for k in list(FAM):
        if k[2] != "MADRE" and (FAM[k] is None or sorted(FAM[k].index) != list(range(100))):
            S1_INC[k] = "missing" if FAM[k] is None else f"n={len(FAM[k])}"
            del FAM[k]
    for chi in CHIS:
        if ("ON", chi, "ENG") not in FAM and ("ON", chi, "MADRE") in FAM:
            del FAM[("ON", chi, "MADRE")]; POOL.pop(("ON", chi), None)
            MADRE_MISSING.setdefault(chi, []).append(f"ENUB_N100_{chi} {S1_INC.get(('ON', chi, 'ENG'), 'missing')}")
for (bc, chi, code), d in FAM.items():
    for s, r in d.iterrows():
        rows.append(dict(beacon=bc, chi=int(chi) / 100, arm=code, ep=int(s), shard=int(r.shard),
                         reward=float(r.episode_reward), bps=float(r.episode_reward) / 250 * 1000))
D = pd.DataFrame(rows)
D.to_csv(os.path.join(OUT, "episodes_baselines_long.csv"), index=False)


def bps(k):
    return FAM[k].episode_reward.values / 250 * 1000


def sym(k):
    x = bps(k)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x)), len(x)


def ratio_ci(num, den, paired=False):
    """ratio of means, percentile 95%; independent resampling (conservative, arm_vs_arm style) or paired by ep."""
    n1, n2 = len(num), len(den)
    if paired:
        idx = rng.integers(0, n1, (B, n1))
        r = num[idx].mean(1) / den[idx].mean(1)
    else:
        r = num[rng.integers(0, n1, (B, n1))].mean(1) / den[rng.integers(0, n2, (B, n2))].mean(1)
    return num.mean() / den.mean(), *np.percentile(r, [2.5, 97.5])


def two_level(arrs):
    """B mean draws: resample training seeds, then episodes within each (k=1: episodes only)."""
    k = len(arrs)
    if k == 1:
        a = arrs[0]
        return a[rng.integers(0, len(a), (B, len(a)))].mean(1)
    picks = rng.integers(0, k, (B, k))
    tot = np.zeros(B); cnt = np.zeros(B)
    for j in range(k):
        for s in range(k):
            sel = picks[:, j] == s
            a = arrs[s]
            tot[sel] += a[rng.integers(0, len(a), (sel.sum(), len(a)))].sum(1); cnt[sel] += len(a)
    return tot / cnt


def marrs(bc, chi):
    return [d.episode_reward.values / 250 * 1000 for d in POOL[(bc, chi)]]


def madre_bar(bc, chi):
    arrs, b = marrs(bc, chi), bps((bc, chi, "ENG"))
    m = np.concatenate(arrs).mean()
    r = two_level(arrs) / b[rng.integers(0, len(b), (B, len(b)))].mean(1)
    lo, hi = np.percentile(r, [2.5, 97.5])
    return m, hi * b.mean() - m, m - lo * b.mean()


# ---------- 1. reference figure (beacon off) ----------
def parse_tex_rows():
    t = open(TEX).read()
    blocks = re.findall(r"\\addplot \[(.*?)\][^{]*?table \[x=c, y=m, y error=e, row sep=\\\\\] \{(.*?)\};\s*\\addlegendentry\{([^\n]*)\}", t, re.S)
    out = {}
    for _, body, leg in blocks:
        pts = re.findall(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s*\\\\", body, re.M)
        out[leg] = {c: (int(m), int(e)) for c, m, e in pts}
    dm = re.search(r"\\pgfplotstableread\[row sep=\\\\\]\{(.*?)\}\\datamadre", t, re.S).group(1)
    out["MADRE"] = {c: (int(m), int(ep), int(em)) for c, m, ep, em in re.findall(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\\\\", dm, re.M)}
    return out


LEG = {"GTMG": r"\acs{GTM}", "GRD": "Greedy", "AGG": "Heuristic message", "R1H": "One-hop relay"}
tex = parse_tex_rows()
check = []
for code, leg in ({} if S1 else LEG).items():
    for c, (m, e) in tex[leg].items():
        chi = f"{int(c):02d}" if int(c) < 100 else "100"
        mm, ee, n = sym(("OFF", chi, code))
        ok = (round(mm) == m) and (round(ee) == e)
        check.append(f"{code:5s} chi={c:>3s} tex m={m:>6d} e={e:>5d} | recomputed m={mm:9.1f} e={ee:7.1f} n={n:3d} {'EXACT' if ok else 'MISMATCH'}")
        assert ok, check[-1]
for c, (m, ep, em) in ({} if S1 else tex["MADRE"]).items():
    chi = f"{int(c):02d}" if int(c) < 100 else "100"
    if ("OFF", chi, "MADRE") in FAM:
        mm, p, q = madre_bar("OFF", chi)
        check.append(f"MADRE chi={c:>3s} tex m={m:>6d} ep={ep:>5d} em={em:>5d} | recomputed m={mm:9.1f} (->{round(mm, -2):.0f}) ep={p:7.0f} em={q:7.0f} "
                     f"{'mean EXACT' if round(mm, -2) == m else 'mean MISMATCH'}, bars to bootstrap noise")
    else:
        check.append(f"MADRE chi={c:>3s} tex m={m} -- source not among PD*_L64 families (chi=0.4 is not plotted in fig:baselines)")

# ---------- 2. summary ----------
summ = []
for chi in CHIS:
    for code in list(CODES) + ["ENG", "MADRE"]:
        rec = dict(chi=f"{int(chi) / 100:.2f}", arm=code, legend={**CODES, "ENG": "engineered (fig:density greedy)", "MADRE": "MADRE L=64"}[code])
        for bc in ["OFF", "ON"]:
            k = (bc, chi, code)
            if k not in FAM:
                rec[f"{bc}_n"] = 0
                continue
            m, e, n = sym(k)
            rec.update({f"{bc}_n": n, f"{bc}_m": m, f"{bc}_e": e})
            if code == "MADRE":
                m, p, q = madre_bar(bc, chi)
                rec.update({f"{bc}_m": m, f"{bc}_n": sum(map(len, marrs(bc, chi))), f"{bc}_k": len(POOL[(bc, chi)]), f"{bc}_ep": p, f"{bc}_em": q})
            rec[f"{bc}_note"] = "PROVISIONAL n=90 (s60 rerun pending)" if k in PROVISIONAL else ("old n=60 (4 shards never written)" if k == ("OFF", "100", "GTMG") else "")
        summ.append(rec)
S = pd.DataFrame(summ)
S.to_csv(os.path.join(OUT, "baselines_summary.csv"), index=False, float_format="%.1f")

# ---------- 3. tests ----------
tests = []
for chi in CHIS:
    for code in CODES:
        a, b = ("ON", chi, code), ("OFF", chi, code)
        if a in FAM and b in FAM:
            x, y = FAM[a].episode_reward, FAM[b].episode_reward
            com = x.index.intersection(y.index)
            dx = (x[com] - y[com]).values / 250 * 1000
            bs = dx[rng.integers(0, len(dx), (B, len(dx)))].mean(1)
            tests.append(dict(chi=int(chi) / 100, test=f"{code} beacon ON - OFF (paired by ep seed) [bit/s]", n=len(com),
                              est=dx.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                              note=f"identical episodes {int((np.abs(dx) < 1e-6).sum())}/{len(com)}" + (" PROVISIONAL" if a in PROVISIONAL else "")))
    for bc in ["OFF", "ON"]:
        if (bc, chi, "MADRE") not in FAM:
            continue
        M = FAM[(bc, chi, "MADRE")].episode_reward
        for code in CODES:
            k = (bc, chi, code)
            if k not in FAM:
                continue
            Y = FAM[k].episode_reward
            arrs = marrs(bc, chi)
            yv = Y.values / 250 * 1000
            rr = two_level(arrs) / yv[rng.integers(0, len(yv), (B, len(yv)))].mean(1)
            r = np.concatenate(arrs).mean() / yv.mean()
            lo, hi = np.percentile(rr, [2.5, 97.5])
            com = M.index.intersection(Y.index)
            rp, plo, phi = ratio_ci(M[com].values, Y[com].values, paired=True)
            sig = "SIG" if lo > 1 or hi < 1 else "ns"
            psig = "SIG" if plo > 1 or phi < 1 else "ns"
            tests.append(dict(chi=int(chi) / 100, test=f"{bc}: MADRE / {code} ratio of means", n=f"{sum(map(len, arrs))}/{len(Y)}" + (f" k={len(arrs)}" if len(arrs) > 1 else ""),
                              est=r, lo=lo, hi=hi,
                              note=f"indep {sig}; paired({len(com)}{', seed 1 only' if len(arrs) > 1 else ''}) {rp:.3f} [{plo:.3f},{phi:.3f}] {psig}" + (" PROVISIONAL" if k in PROVISIONAL else "")))
T = pd.DataFrame(tests)
T.to_csv(os.path.join(OUT, "baselines_tests.csv"), index=False, float_format="%.4f")

# ---------- 4. pgfplots rows ----------
L = ["% fig:baselines, BEACON ON (wave_baselines_ub.sh: DUB<chi>_{GTMG,GRD,AGG,R1H}); 100 users, 1200 m, cluster spawn",
     "% m = mean episode_reward/250*1000 [bit/s], e = 1.96*sd/sqrt(n), n=100 episodes (ep seeds 0..99) -- same estimator",
     "% that reproduces every baseline coordinate of the current (beacon-off) figure exactly.",
     f"% scored by results/baselines_beacon_final_oct07/score_baselines_4seed.py --mode {SUF} (baseline arms identical in every mode;",
     "% GTM chi=1 n=100, s60 = rerun 2298104). MADRE L64 seeds per chi: " + "  ".join(f"{int(c)}:{','.join(map(str, SEEDS_L64[c]))}" for c in CHIS),
     "% Drop-in: replace the rows of each \\addplot below (5 points, as now). A chi=0.4 row is given in a comment",
     "% for each, in case the figure gains the c=40 column (beacon-on data now exist at 0.4; the old run had none).", ""]
for code, leg in LEG.items():
    L.append(f"% --- {leg} ---")
    L.append("    c     m        e      \\\\")
    for chi in CHIS:
        if ("ON", chi, code) not in FAM:   # spreading 1 only (never in spreading-8 mode)
            L.append(f"%   {str(int(chi)):<5s} INCOMPLETE {S1_INC.get(('ON', chi, code), 'missing')}"); continue
        m, e, n = sym(("ON", chi, code))
        c = str(int(chi))
        line = f"    {c:<5s} {round(m):<8d} {round(e):<6d} \\\\"
        if chi == "40":
            line = "%" + line[1:] + "   % optional chi=0.4 point"
        if ("ON", chi, code) in PROVISIONAL:
            line += f"   % PROVISIONAL n={n}"
        L.append(line)
    L.append("")
# \datamadre is shared with fig:density -> take the fig:density rows of the SAME mode verbatim, and check them here.
DK3 = open(DENSROWS).read()
blk = DK3[DK3.index("% --- L=64 ---"):]
blk = blk[:blk.index("% --- L=128 ---")]
DM = {c: (int(m), int(ep), int(em)) for c, m, ep, em in re.findall(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\\\\", blk, re.M)}
if A.mode == "current":
    assert len(DM) == 6, DM
L.append(f"% --- \\datamadre (MADRE, L=64), BEACON ON = the L=64 block of {os.path.relpath(DENSROWS, RES)}")
L.append("% (shared by fig:density and fig:baselines; fig:baselines skips the c=40 row). Seeds per chi as in the header line;")
L.append("% m rounded to 100, ep/em = ratio-CI vs ENUB_N100_<chi> x its mean (two-level bootstrap when k > 1).")
L.append("% A chi whose MADRE seed set is incomplete (here or in the density rows) is COMMENTED OUT and tagged INCOMPLETE.")
L.append("    c     m        ep      em     \\\\")
DMCHECK = []
for chi in CHIS:
    c = str(int(chi))
    have_pool = ("ON", chi) in POOL and ("ON", chi, "ENG") in FAM
    m, p, q = madre_bar("ON", chi) if have_pool else (float("nan"),) * 3
    k = len(POOL[("ON", chi)]) if have_pool else 0
    if c in DM and chi not in MADRE_MISSING:
        m0, p0, q0 = DM[c]
        assert round(m, -2) == m0, (chi, m, m0)
        DMCHECK.append(f"chi={c:>3s} density rows m={m0} ep={p0} em={q0} | recomputed here (k={k}) m={m:.1f} ep={p:.0f} em={q:.0f} (mean EXACT, bars to bootstrap noise)")
        L.append(f"    {c:<5s} {m0:<8d} {p0:<7d} {q0:<6d} \\\\")
    else:
        why = (f"missing here: {'; '.join(MADRE_MISSING.get(chi, []))}" if chi in MADRE_MISSING else "") + ("" if c in DM else " | density rows: INCOMPLETE/PENDING")
        DMCHECK.append(f"chi={c:>3s} INCOMPLETE k={k}/{len(SEEDS_L64[chi])} partial m={m:.0f} -- {why}")
        L.append(f"%   {c:<5s} INCOMPLETE k={k}/{len(SEEDS_L64[chi])} (partial m={m:.0f}) {why}")
open(os.path.join(OUT, "fig_baselines_beacon_on_rows.tex"), "w").write("\n".join(L) + "\n")

# ---------- 4b. cross-check: beacon-on baseline rows and \datamadre vs the paper source ----------
_pt = open(A.paper_tex).read()
PAPER = {}
for _, body, leg in re.findall(r"\\addplot \[(.*?)\][^{]*?table \[x=c, y=m, y error=e, row sep=\\\\\] \{(.*?)\};\s*\\addlegendentry\{([^\n]*)\}", _pt, re.S):
    PAPER[leg] = {c: (int(m), int(e)) for c, m, e in re.findall(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s*\\\\", body, re.M)}
_dm = re.search(r"\\pgfplotstableread\[row sep=\\\\\]\{(.*?)\}\\datamadre", _pt, re.S).group(1)
PAPER["MADRE"] = {c: (int(m), int(ep), int(em)) for c, m, ep, em in re.findall(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\\\\", _dm, re.M)}
REG, nmis = [], 0
for code, leg in LEG.items():
    for c, tex in PAPER.get(leg, {}).items():
        chi = f"{int(c):02d}" if int(c) < 100 else "100"
        if ("ON", chi, code) not in FAM:   # spreading 1 only
            nmis += 1; REG.append(f"{code:5s} chi={c:>3s} paper {tex} | this run INCOMPLETE {S1_INC.get(('ON', chi, code), 'missing')}"); continue
        m, e, n = sym(("ON", chi, code))
        mine = (round(m), round(e)); nmis += mine != tex
        REG.append(f"{code:5s} chi={c:>3s} paper {tex} | this run {mine} {'EXACT' if mine == tex else 'DIFFERS'}")
for c, tex in PAPER["MADRE"].items():
    mine = DM.get(c) if (f"{int(c):02d}" if int(c) < 100 else "100") not in MADRE_MISSING else None
    nmis += mine != tex
    REG.append(f"MADRE chi={c:>3s} paper {tex} | this run {mine if mine else 'INCOMPLETE'} {'EXACT' if mine == tex else 'DIFFERS'}")
REG.insert(0, f"paper ({os.path.basename(A.paper_tex)}) fig:baselines + \\datamadre: {nmis} coordinates differ from this run")
if A.mode == "current" and not A.with_extras:
    _o = open(os.path.join(HERE, "fig_baselines_beacon_on_rows.tex")).read()
    REG.insert(1, "non-comment lines identical to baselines_beacon_final_oct07/fig_baselines_beacon_on_rows.tex: "
               + str([l for l in _o.splitlines() if not l.startswith("%")] == [l for l in L if not l.startswith("%")]))

# ---------- 5. beacon-off vs beacon-on table ----------
W = ["fig:baselines, bit/s (mean +- e, e = 1.96 SE). 100 users, 1200 m, cluster spawn, n=100 unless marked.",
     "OFF = current figure (D<chi>_*, Sep 16-17); ON = beacon rerun (DUB<chi>_*, Oct 6-7). Delta = paired ON-OFF mean [95% CI].",
     "+ = old point was n=60. MADRE ON: training seeds pooled per chi = " + "  ".join(f"{int(c)}:{','.join(map(str, SEEDS_L64[c]))}" for c in CHIS)
     + " (paired tests: first seed only).", ""]
if S1:
    W.insert(0, "SPREADING 1 (drone TX 6.59004 mW, same checkpoints): every family below is its spreading-1 re-evaluation (S1_<family>, or the verified SPR1 twin), staged under the original family name by pull_from_cluster_s1.sh (four_seed_staging/pull_raw_s1/MANIFEST.tsv). Paper values quoted below are the spreading-8 numbers of ieee v14.")
    W.insert(1, "INCOMPLETE spreading-1 baseline families (not used): " + (", ".join(f"{k[2]} chi={k[1]} {v}" for k, v in sorted(S1_INC.items())) or "none"))
W.append(f"{'arm':18s} {'chi':>4s} | {'OFF':>18s} | {'ON':>18s} | {'Delta paired [95% CI]':>28s}  identical eps")
W.append("-" * 112)
for code in list(CODES) + ["MADRE"]:
    for chi in CHIS:
        off, on = ("OFF", chi, code), ("ON", chi, code)
        def f(k):
            if k not in FAM:
                return "pending" if k[0] == "ON" else "--"
            if k[2] == "MADRE":
                m, p, q = madre_bar(k[0], k[1])
                return f"{m:7.0f} +{p:5.0f}/-{q:5.0f}" + ("k3" if len(POOL[(k[0], k[1])]) > 1 else "  ")
            m, e, n = sym(k)
            tag = "*" if k in PROVISIONAL else ("+" if k == ("OFF", "100", "GTMG") else " ")
            return f"{m:8.0f} +-{e:6.0f}{tag}"
        t = T[(T.chi == int(chi) / 100) & (T.test.str.startswith(code + " beacon"))]
        dl = f"{t.est.iloc[0]:+7.0f} [{t.lo.iloc[0]:+7.0f},{t.hi.iloc[0]:+7.0f}]" if len(t) else ""
        idn = t.note.iloc[0] if len(t) else ""
        W.append(f"{CODES.get(code, 'MADRE L=64'):18s} {int(chi) / 100:4.2f} | {f(off):>18s} | {f(on):>18s} | {dl:>28s}  {idn}")
    W.append("")
W.append("MADRE vs baselines (ratio of means; independent two-sided bootstrap = conservative; paired by ep seed in note):")
for _, r in T[T.test.str.contains("MADRE /")].iterrows():
    W.append(f"  chi={r.chi:4.2f} {r.test:34s} n={r.n:8s} {r.est:6.3f} [{r.lo:6.3f}, {r.hi:6.3f}]  {r.note}")
W.append("")
W.append(f"\\datamadre (beacon on) = density rows of mode {SUF}, check:")
W += ["  " + c for c in DMCHECK]
W.append("")
W.append("REGRESSION vs the paper:")
W += ["  " + r for r in REG]
W.append("")
W.append("Reproduction of the OLD beacon-off figure (old tex) from beacon-off data:")
W += ["  " + c for c in check]
open(os.path.join(OUT, "old_vs_new_table.txt"), "w").write("\n".join(W) + "\n")
print("\n".join(W))

# ---------- 6. PNG preview ----------
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    col = {"MADRE": "tab:blue", "GTMG": "green", "GRD": "grey", "AGG": "purple", "R1H": "darkorange"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for i, bc in enumerate(["OFF", "ON"]):
        for code in ["MADRE"] + list(CODES):
            xs, ms, lo, hi = [], [], [], []
            for chi in CHIS:
                k = (bc, chi, code)
                if k not in FAM or (code == "MADRE" and chi == "40" and bc == "OFF"):
                    continue
                if code == "MADRE":
                    m, p, q = madre_bar(bc, chi)
                else:
                    m, p, _ = sym(k); q = p
                xs.append(int(chi)); ms.append(m); hi.append(p); lo.append(q)
            ax[i].errorbar(xs, ms, yerr=[lo, hi], color=col[code], marker="o", ms=4, capsize=2,
                           ls="--" if code == "GRD" else "-", label=CODES.get(code, "MADRE L=64"))
        ax[i].set_title(f"beacon {bc}" + ("  (final; MADRE chi .2/.4 k=3)" if bc == "ON" else ""), fontsize=9)
        ax[i].set_xlabel("chi [%]"); ax[i].grid(alpha=.3)
    ax[0].set_ylabel("throughput [bit/s]"); ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_baselines_old_vs_new.png"), dpi=130)
except Exception as ex:
    print("PNG skipped:", ex)
