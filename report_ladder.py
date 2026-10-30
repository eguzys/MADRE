#!/usr/bin/env python3
"""Canonical OM2G ladder report. One naming scheme: L<latent>.

See LADDER_NAMES.md. beaconBytes = 4*latent + 56. Globs are ANCHORED on tag
prefix so training depths and campaigns cannot mix (a bare * once produced a
"converged" table full of 96k numbers).
"""
import glob, csv, math, statistics as st

def load(*pats):
    d = {}
    for p in pats:
        for f in glob.glob(p):
            try:
                for r in csv.DictReader(open(f)):
                    d[int(float(r["seed"]))] = float(r["episode_reward"])
            except Exception:
                pass
    return d

# canonical name -> (latent, glob per world). None = not built for that world.
LADDER = [
    ("deaf",  None, {1000: "om2g_out/OM2G_trained_TRB_w1000sil*.csv",
                     1200: "om2g_out/OM2G_trained_HON_x1200sil_*.csv",
                     1465: "om2g_out/OM2G_trained_HON_x1465sil_*.csv"}),
    ("L2",       2, {1000: "om2g_out/OM2G_trained_K_m1000L2_300000_*.csv",
                     1200: "om2g_out/OM2G_trained_K_m1200L2_300000_*.csv",
                     1465: "om2g_out/OM2G_trained_K_m1465L2_*.csv"}),
    ("L8",       8, {1000: "om2g_out/OM2G_trained_MW_w1000talk_*.csv",
                     1200: "om2g_out/OM2G_trained_G_x1200talk_*.csv",
                     1465: "om2g_out/OM2G_trained_K_x1465talk_*.csv"}),
    ("L32",     32, {1000: "om2g_out/OM2G_trained_K_m1000L32_300000_*.csv",
                     1200: "om2g_out/OM2G_trained_K_m1200L32_300000_*.csv"}),
    ("L56",     56, {1000: "om2g_out/OM2G_trained_K_m1000f3_300000_*.csv"}),
    ("L128",   128, {1000: "om2g_out/OM2G_trained_K_m1000L128_*.csv"}),
    ("L680",   680, {1000: "om2g_out/OM2G_trained_K_u1000talk_300000_*.csv"}),
]
BASE = {1000: "W1000", 1200: "W1200", 1465: "W1465"}

for w in (1000, 1200, 1465):
    eng = load(f"om2g_out/OM2G_baseline_engineered1hop_*{BASE[w]}*.csv")
    if not eng:
        print(f"\n  ===== {w} m =====  no baseline"); continue
    em = st.mean(list(eng.values()))
    print(f"\n  ===== {w} m =====   engineered {em:,.0f}  (n={len(eng)})")
    print(f"    {'rung':6s} {'latent':>6s} {'beaconB':>8s} {'n':>4s} {'mean':>9s} {'ratio':>7s} {'t':>7s}")
    for name, lat, gl in LADDER:
        p = gl.get(w)
        if not p:
            continue
        d = load(p)
        b = "-" if lat is None else f"{4*lat+56}"
        if len(d) < 50:
            print(f"    {name:6s} {str(lat or '-'):>6s} {b:>8s} {len(d):4d}   (pending)"); continue
        c = sorted(set(d) & set(eng))
        diff = [d[x]-eng[x] for x in c]
        t = st.mean(diff)/(st.stdev(diff)/math.sqrt(len(diff)))
        print(f"    {name:6s} {str(lat or '-'):>6s} {b:>8s} {len(d):4d} "
              f"{st.mean(list(d.values())):9,.0f} "
              f"{st.mean([d[x] for x in c])/st.mean([eng[x] for x in c]):6.2f}x {t:7.2f}")
print()
