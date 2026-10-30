#!/usr/bin/env python3
"""1000 m U-CURVE (Claim 2): reward as a function of ON-AIR BYTES per message.

The x-axis is airtime actually charged to the radio, NOT latent width -- that is
the whole point of the claim. Bytes verified against OMNeT's own beaconBytes
(= 4*latent + 56); the control charge adds framing on top.

All arms must be scored in the byte-identical 1000 m world
(MINSEP=1000 MAXNN=1250 SPREAD=25 SPAWN=cluster, shaping OFF).
Baseline CSVs do NOT record their world -- only use the W1000-tagged ones
measured tonight.
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

# (label, charged on-air B/msg, glob).  None = not a comms arm.
LADDER = [
    ("engineered",   None, "om2g_out/OM2G_baseline_engineered1hop_*W1000*.csv"),
    ("hover",        None, "om2g_out/OM2G_baseline_hover_*W1000*.csv"),
    ("AI silent",       0, "om2g_out/OM2G_trained_F_w1000sil_*.csv"),
    ("AI L2",         107, "om2g_out/OM2G_trained_H_m1000L2_*.csv"),
    ("AI talk (L8)",  131, "om2g_out/OM2G_trained_F_w1000talk_*.csv"),
    ("AI L32",        262, "om2g_out/OM2G_trained_H_m1000L32_*.csv"),
    ("AI f3 (L56)",   393, "om2g_out/OM2G_trained_H_m1000f3_*.csv"),
    ("AI f4 (L80)",   524, "om2g_out/OM2G_trained_H_m1000f4_*.csv"),
    ("AI L128",       786, "om2g_out/OM2G_trained_H_m1000L128_*.csv"),
    ("AI uncompressed",3799,"om2g_out/OM2G_trained_H_u1000talk_*.csv"),
]
data = {l: load(p) for l, _, p in LADDER}
base = data["engineered"]
print("\n  1000 m U-curve — reward vs ON-AIR BYTES (unshaped, n=100 target)")
print(f"  {'arm':17s} {'bytes':>6s} {'n':>4s} {'mean':>10s} {'se':>8s} {'ratio':>7s} {'t':>7s}")
for label, byts, _ in LADDER:
    d = data[label]
    b = f"{byts:6d}" if byts is not None else "     -"
    if not d:
        print(f"  {label:17s} {b} {0:4d}   (not yet)"); continue
    v = list(d.values()); m = st.mean(v)
    se = st.pstdev(v)/math.sqrt(len(v)) if len(v) > 1 else 0.0
    row = f"  {label:17s} {b} {len(v):4d} {m:10,.1f} {se:8,.1f}"
    common = sorted(set(d) & set(base))
    if base and label != "engineered" and len(common) > 2:
        diff = [d[s]-base[s] for s in common]
        md = st.mean(diff); sed = st.stdev(diff)/math.sqrt(len(diff))
        bm = st.mean([base[s] for s in common]); dm = st.mean([d[s] for s in common])
        row += f" {dm/bm:7.2f}x {md/sed if sed else float('nan'):7.2f}"
    print(row)
print()
