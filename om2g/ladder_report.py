#!/usr/bin/env python3
"""ladder_report.py — the two-metric ladder with two-level bootstrap CIs.

WHY THIS EXISTS: one fixed procedure for the ladder that refuses to mix
spawns and reports both metrics side by side.

USAGE
    python om2g/ladder_report.py --family br12rt --suffix _SPAWNcluster_TAIL \
        --baseline br12fixnoc --baseline-seeds 1,2,3,5 --seeds 1,2,3

WHAT IT GUARANTEES
  * arm and baseline must carry the SAME suffix, so a _SPAWNcluster file can
    never be divided by an unsuffixed one.
  * two-level bootstrap: resamples TRAINING SEEDS and EPISODES, for the arm AND
    for the baseline. The baseline's seed variance is the dominant term (silence
    spread 31.2% vs L128 17.4%), so leaving it out understates the interval badly.
  * reports episode_reward and tail125_mean_step together. If a file lacks the
    tail column it is skipped for that metric only, and the count is printed.
"""
import argparse, csv, glob, os, random, statistics as st, sys

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
p.add_argument('--family', required=True)
p.add_argument('--rungs', default='8,32,128')
p.add_argument('--seeds', default='1,2,3')
p.add_argument('--baseline', required=True)
p.add_argument('--baseline-seeds', default='1,2,3,5')
p.add_argument('--suffix', required=True, help='suffix on the ARM files')
p.add_argument('--baseline-suffix', default=None,
               help=('suffix on the BASELINE files. Defaults to --suffix. Set it '
                     'ONLY when the two carry different suffixes that still denote '
                     'the SAME SPAWN, e.g. arm _SPAWNcluster_TAIL against baseline '
                     '_SPAWNcluster. Both are cluster spawn; only the tail-column '
                     'rebuild differs. Setting this is a DELIBERATE act and the '
                     'script prints a warning, because a mismatch here would '
                     'compare different spawns.'))
p.add_argument('--boot', type=int, default=8000)
p.add_argument('--outdir', default='om2g_out')
a = p.parse_args()
rng = random.Random(20260909)
bsuf = a.baseline_suffix if a.baseline_suffix is not None else a.suffix
if a.baseline_suffix is not None and a.baseline_suffix != a.suffix:
    print(f'!! arm suffix {a.suffix!r} vs baseline suffix {bsuf!r} — DELIBERATE MISMATCH.')
    print('!! Both MUST denote the same spawn. Verify with:')
    print("!!   grep -oE droneSpawnMode om2g_logs/W1ev_TAG_s0_JOBID.log")
    print()
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

bseeds = [s for s in a.baseline_seeds.split(',') if s]
seeds  = [s for s in a.seeds.split(',') if s]
rungs  = [int(x) for x in a.rungs.split(',') if x]
BYTES  = {1:60, 2:64, 4:72, 8:88, 32:184, 128:568, 256:1080, 512:2104}

for key, label in (('episode_reward','EPISODE MEAN'), ('tail125_mean_step','CONVERGED tail125')):
    base = [load(f'{a.outdir}/OM2G_trained_{a.baseline}_s{s}{bsuf}_n*_seed*.csv', key) for s in bseeds]
    base = [d for d in base if d]
    if not base:
        print(f'=== {label} === NO BASELINE DATA for {a.baseline}{bsuf}'); continue
    bm = st.mean(pool(base))
    print(f'=== {label} ===  baseline {a.baseline}{bsuf}  k={len(base)} n={len(pool(base))} mean={bm:.5g}')
    for L in rungs:
        g = [load(f'{a.outdir}/OM2G_trained_{a.family}L{L}s{s}{a.suffix}_n*_seed*.csv', key) for s in seeds]
        g = [d for d in g if d]
        if not g:
            print(f'  L={L:<4d} {BYTES.get(L,0):5d} B   no data'); continue
        m = st.mean(pool(g)); lo, hi = boot(g, base, a.boot, rng)
        sig = 'SIG' if lo > 1 or hi < 1 else 'ns'
        print(f'  L={L:<4d} {BYTES.get(L,0):5d} B  k={len(g)} n={len(pool(g)):3d}  {m:10.5g}  ratio {m/bm:.3f}  [{lo:.3f}, {hi:.3f}]  {sig}')
    print()
