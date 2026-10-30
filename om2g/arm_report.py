#!/usr/bin/env python3
"""arm_report.py — same statistics as ladder_report.py, for a SINGLE arm.

WHY THIS EXISTS: ladder_report.py hardcodes an `L<rung>` in the filename, so it
cannot score arms named `OM2G_trained_br12fixunc_s1_SPAWNcluster` or the
engineered `OM2G_engineered_...`. Rather than loosen that script (its rigid
naming is part of what makes a cross-spawn mix hard), this reuses its load,
pool and two-level bootstrap verbatim and only changes how the filename is
built.

Same guarantees: arm and baseline default to the SAME suffix; a mismatch must be
asked for and prints a warning. Two-level bootstrap over training seeds AND
episodes for arm and baseline both. Both metrics reported.
"""
import argparse, csv, glob, os, random, statistics as st

def load(pattern, key):
    d = {}
    for f in sorted(glob.glob(pattern)):
        try:
            for r in csv.DictReader(open(f)):
                if key in r and r[key] not in ('', None):
                    d[int(r['seed'])] = float(r[key])
        except Exception:
            pass
    return d

def pool(g):
    return [v for d in g for v in d.values()]

def boot(a, b, B, rng):
    out = []
    for _ in range(B):
        x, y = [], []
        for _ in range(len(a)):
            d = rng.choice(a); ks = list(d)
            x += [d[k] for k in rng.choices(ks, k=len(ks))]
        for _ in range(len(b)):
            d = rng.choice(b); ks = list(d)
            y += [d[k] for k in rng.choices(ks, k=len(ks))]
        out.append(st.mean(x) / st.mean(y))
    out.sort()
    return out[int(.025 * B)], out[int(.975 * B)]

p = argparse.ArgumentParser()
p.add_argument('--arm', required=True, help='e.g. br12fixunc')
p.add_argument('--arm-prefix', default='OM2G_trained', help='OM2G_trained or OM2G_engineered')
p.add_argument('--seeds', default='1,2,3,4,5')
p.add_argument('--baseline', required=True)
p.add_argument('--baseline-seeds', default='1,2,3,5')
p.add_argument('--baseline-prefix', default='OM2G_trained')
p.add_argument('--suffix', required=True)
p.add_argument('--baseline-suffix', default=None)
p.add_argument('--arm-glob', default=None,
               help=('COMMA-SEPARATED globs for the ARM files, ONE PER TRAINING '
                     'SEED, overriding --arm/--seeds/--suffix. One glob per seed '
                     'is mandatory: load() keys by EPISODE seed, so pooling '
                     'several training seeds into a single glob silently '
                     'overwrites them and collapses n to one seed. '
                     '--suffix. Use for names that do not fit '
                     'PREFIX_TAG_s<seed>SUFFIX, e.g. the engineered files whose '
                     'cell tag already carries its seed. The suffix guard cannot '
                     'inspect a raw glob, so state the spawn in your notes.'))
p.add_argument('--baseline-glob', default=None,
               help='explicit glob for the BASELINE files. Same caveat.')
p.add_argument('--boot', type=int, default=8000)
p.add_argument('--outdir', default='om2g_out')
a = p.parse_args()
rng = random.Random(20260909)
bsuf = a.baseline_suffix if a.baseline_suffix is not None else a.suffix
if a.arm_glob or a.baseline_glob:
    print('!! RAW GLOB in use - the suffix guard cannot verify spawn. Confirm by hand.')
    print()
if a.baseline_suffix is not None and a.baseline_suffix != a.suffix:
    print(f'!! arm suffix {a.suffix!r} vs baseline suffix {bsuf!r} - DELIBERATE MISMATCH.')
    print('!! Both MUST denote the same spawn.')
    print()

bseeds = [s for s in a.baseline_seeds.split(',') if s]
seeds  = [s for s in a.seeds.split(',') if s]

for key, label in (('episode_reward','EPISODE MEAN'), ('tail125_mean_step','CONVERGED tail125')):
    if a.baseline_glob:
        base = [load(f'{a.outdir}/{gl}', key) for gl in a.baseline_glob.split(',') if gl]
    else:
        base = [load(f'{a.outdir}/{a.baseline_prefix}_{a.baseline}_s{s}{bsuf}_n*_seed*.csv', key) for s in bseeds]
    base = [d for d in base if d]
    if not base:
        print(f'=== {label} === NO BASELINE DATA for {a.baseline}{bsuf}'); continue
    bm = st.mean(pool(base))
    print(f'=== {label} ===  baseline {a.baseline}{bsuf}  k={len(base)} n={len(pool(base))} mean={bm:.5g}')
    if a.arm_glob:
        g = [load(f'{a.outdir}/{gl}', key) for gl in a.arm_glob.split(',') if gl]
    else:
        g = [load(f'{a.outdir}/{a.arm_prefix}_{a.arm}_s{s}{a.suffix}_n*_seed*.csv', key) for s in seeds]
    g = [d for d in g if d]
    if not g:
        print(f'  {a.arm:<16s} no data'); print(); continue
    m = st.mean(pool(g)); lo, hi = boot(g, base, a.boot, rng)
    sig = 'SIG' if lo > 1 or hi < 1 else 'ns'
    print(f'  {a.arm:<16s} k={len(g)} n={len(pool(g)):3d}  {m:10.5g}  ratio {m/bm:.3f}  [{lo:.3f}, {hi:.3f}]  {sig}')
    print()
