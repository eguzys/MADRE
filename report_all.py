#!/usr/bin/env python3
"""ONE table: every world x every rung, converged, vs its own matched-world engineered.

Tag prefixes are ANCHORED (never a bare *) so training depths cannot mix:
  F_/G_ = 300k headline evals   H_ = 1000 m ladder watcher
  K_    = generic watcher (1200/1465)   M96_ = matched-96k preview (EXCLUDED here)
Baselines are the W1000/W1200/W1465 sets measured in each world's own eval string.
"""
import glob, csv, math, statistics as st
def load(*pats):
    d={}
    for p in pats:
        for f in glob.glob(p):
            try:
                for r in csv.DictReader(open(f)): d[int(float(r["seed"]))]=float(r["episode_reward"])
            except Exception: pass
    return d
RUNGS=[("silent",0),("L2",107),("L8",131),("L32",262),("f3",393),("f4",524),("L128",786),("unc",3799)]
CELL={1000:{"silent":["om2g_out/OM2G_trained_F_w1000sil_*.csv"],
            "L2":["om2g_out/OM2G_trained_H_m1000L2_*.csv","om2g_out/OM2G_trained_K_m1000L2_*.csv"],
            "L8":["om2g_out/OM2G_trained_F_w1000talk_*.csv"],
            "L32":["om2g_out/OM2G_trained_H_m1000L32_*.csv"],
            "f3":["om2g_out/OM2G_trained_H_m1000f3_*.csv"],
            "f4":["om2g_out/OM2G_trained_H_m1000f4_*.csv"],
            "L128":["om2g_out/OM2G_trained_H_m1000L128_*.csv"],
            "unc":["om2g_out/OM2G_trained_H_u1000talk_*.csv"]},
      1200:{"silent":["om2g_out/OM2G_trained_G_x1200sil_*.csv"],
            "L8":["om2g_out/OM2G_trained_G_x1200talk_*.csv"]},
      1465:{"silent":["om2g_out/OM2G_trained_K_x1465sil_*.csv"],
            "L8":["om2g_out/OM2G_trained_K_x1465talk_*.csv"]}}
for r,b in RUNGS:
    for w,pre in ((1200,"m1200"),(1465,"m1465")):
        if r in ("L2","L32","f3","f4","L128"):
            CELL[w][r]=[f"om2g_out/OM2G_trained_K_{pre}{r}_*.csv"]
    CELL[1200].setdefault("unc",["om2g_out/OM2G_trained_K_u1200talk_*.csv"])
    CELL[1465].setdefault("unc",["om2g_out/OM2G_trained_K_u1465talk_*.csv"])
print()
for w in (1000,1200,1465):
    eng=load(f"om2g_out/OM2G_baseline_engineered1hop_*W{w}*.csv")
    hov=load(f"om2g_out/OM2G_baseline_hover_*W{w}*.csv")
    em=st.mean(list(eng.values())) if eng else None
    print(f"  ===== {w} m =====  engineered={em:,.0f} (n={len(eng)})" if em else f"  ===== {w} m =====  engineered: none")
    if hov: print(f"    {'hover':10s} {'':>6s} n={len(hov):3d} {st.mean(list(hov.values())):9,.0f}  {st.mean(list(hov.values()))/em:5.2f}x")
    for rn,b in RUNGS:
        d=load(*CELL[w].get(rn,[])) if CELL[w].get(rn) else {}
        if not d: continue
        c=sorted(set(d)&set(eng))
        if len(c)<3: print(f"    {rn:10s} {b:6d} n={len(d):3d} {st.mean(list(d.values())):9,.0f}"); continue
        dm=st.mean([d[s] for s in c]); bm=st.mean([eng[s] for s in c])
        diff=[d[s]-eng[s] for s in c]; t=st.mean(diff)/(st.stdev(diff)/math.sqrt(len(diff)))
        print(f"    {rn:10s} {b:6d} n={len(c):3d} {dm:9,.0f}  {dm/bm:5.2f}x  t={t:6.2f}")
    print()
