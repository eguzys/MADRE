#!/usr/bin/env python3
"""Final 1200 m table. Pairs PER SEED against engineered (the thesis baseline).

Everything here was produced with the byte-identical world string from
final_eval_1200.sh -- MINSEP=1200, MAXNN=1500, SPREAD=25, SPAWN=cluster,
shaping OFF. Do not mix in archived CSVs: baseline CSVs record only
seed,episode_reward,steps and NOT the world they ran in.
"""
import glob, csv, math, statistics as st

def load(pat):
    d = {}
    for f in glob.glob(pat):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                try: d[int(float(r["seed"]))] = float(r["episode_reward"])
                except (KeyError, ValueError): pass
    return d

ARMS = [
    ("engineered",     "om2g_out/OM2G_baseline_engineered1hop_*W1200*.csv"),
    ("hover",          "om2g_out/OM2G_baseline_hover_*W1200*.csv"),
    ("AI talk  (s1)",  "om2g_out/OM2G_trained_G_x1200talk_300000_*.csv"),
    ("AI silent(s1)",  "om2g_out/OM2G_trained_G_x1200sil_*.csv"),
    ("AI talk  (s2)",  "om2g_out/OM2G_trained_G_x1200talks2_*.csv"),
    ("AI silent(s2)",  "om2g_out/OM2G_trained_G_x1200sils2_*.csv"),
]
data = {n: load(p) for n, p in ARMS}
base = data["engineered"]
print(f"\n  1200 m world (MINSEP=1200 MAXNN=1500 SPREAD=25 SPAWN=cluster, unshaped)")
print(f"  {'arm':16s} {'n':>4s} {'mean':>10s} {'se':>8s} {'ratio':>7s} {'t(paired)':>10s} {'95% CI ratio':>20s}")
for name, _ in ARMS:
    d = data[name]
    if not d:
        print(f"  {name:16s} {0:4d}  (not yet)"); continue
    v = list(d.values()); m = st.mean(v)
    se = st.pstdev(v)/math.sqrt(len(v)) if len(v) > 1 else 0.0
    row = f"  {name:16s} {len(v):4d} {m:10,.1f} {se:8,.1f}"
    common = sorted(set(d) & set(base))
    if base and name != "engineered" and len(common) > 2:
        diff = [d[s] - base[s] for s in common]
        md = st.mean(diff); sd = st.stdev(diff)
        sed = sd/math.sqrt(len(diff)); t = md/sed if sed else float("nan")
        bm = st.mean([base[s] for s in common])
        dm = st.mean([d[s] for s in common])   # paired mean, NOT the overall
        lo, hi = (md-1.96*sed)/bm + 1, (md+1.96*sed)/bm + 1
        row += f" {dm/bm:7.2f}x {t:10.2f}   [{lo:.2f}, {hi:.2f}]  paired n={len(common)}"
    print(row)
print()
