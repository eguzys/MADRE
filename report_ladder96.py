#!/usr/bin/env python3
"""1000 m U-curve at MATCHED 96k frames — every rung on an equal training budget.

This is the correct control for "does message size matter": all rungs get the
same number of frames, so differences cannot be explained by one arm having
trained longer. It is NOT the converged optimum -- 96k is undertrained (the
1200 m arm gained +55.7% between 216k and 300k), so the RANKING here may differ
from the 300k ranking. Always report both.
"""
import glob, csv, math, statistics as st
def load(pat):
    d={}
    for f in glob.glob(pat):
        try:
            for r in csv.DictReader(open(f)):
                d[int(float(r["seed"]))]=float(r["episode_reward"])
        except Exception: pass
    return d
LADDER=[("silent (mask_out)",  0,"om2g_out/OM2G_trained_M96_w1000sil_96000_*.csv"),
        ("L2",               107,"om2g_out/OM2G_trained_M96_m1000L2_96000_*.csv"),
        ("L8  (talk)",       131,"om2g_out/OM2G_trained_M96_w1000talk_96000_*.csv"),
        ("L32",              262,"om2g_out/OM2G_trained_M96_m1000L32_96000_*.csv"),
        ("f3  (L56)",        393,"om2g_out/OM2G_trained_M96_m1000f3_96000_*.csv"),
        ("L128",             786,"om2g_out/OM2G_trained_M96_m1000L128_96000_*.csv"),
        ("uncompressed",    3799,"om2g_out/OM2G_trained_M96_u1000talk_96000_*.csv")]
eng=load("om2g_out/OM2G_baseline_engineered1hop_*W1000*.csv")
print("\n  1000 m U-CURVE @ MATCHED 96k frames  (vs engineered 15,636.7)")
print(f"  {'rung':18s} {'on-air B':>9s} {'n':>4s} {'mean':>10s} {'se':>8s} {'ratio':>7s}")
for lab,b,pat in LADDER:
    d=load(pat)
    if not d: print(f"  {lab:18s} {b:9d} {0:4d}   (pending)"); continue
    v=list(d.values()); m=st.mean(v); se=st.pstdev(v)/math.sqrt(len(v)) if len(v)>1 else 0
    c=sorted(set(d)&set(eng)); r=""
    if len(c)>2:
        r=f" {st.mean([d[s] for s in c])/st.mean([eng[s] for s in c]):7.2f}x"
    print(f"  {lab:18s} {b:9d} {len(v):4d} {m:10,.1f} {se:8,.1f}{r}")
print()
