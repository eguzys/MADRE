import glob, csv, os, re, yaml, statistics as st
# Map: slurm log (has model[+N]+u in its name) -> results dir (from "Results written to:")
SC = '/cluster/scratch/eguzys'
# New per-N x per-user learned grid: le_{st,ds,ffn}_N{N}_u{u}_{jobid}.out (no mode field -- implicitly learned)
SWEEP_RE = re.compile(r'le_(st|ds|ffn)_N(\d+)_u(\d+)_(\d+)\.out$')
# Legacy frozen-vs-learned u-sweep (banked N4 curves): eval_{arch}_{mode}_u{u}_{jobid}.out, all trained at N=4
LEGACY_RE = re.compile(r'eval_(st|ds|ffn)_(learned|frozen)_u(\d+)_(\d+)\.out$')
LEGACY_N = 4

rows = []
for log in glob.glob(f'{SC}/le_*.out') + glob.glob(f'{SC}/eval_*_u*.out'):
    base = os.path.basename(log)
    m = SWEEP_RE.search(base)
    if m:
        arch, N, u, jid = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        mode = 'learned'
    else:
        m = LEGACY_RE.search(base)
        if not m: continue
        arch, mode, u, jid = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        N = LEGACY_N
    txt = open(log, errors='ignore').read()
    d = re.search(r'Results written to:\s*(\S+)', txt)
    if not d: continue
    csv_p = os.path.join(d.group(1), 'per_episode.csv')
    if not os.path.exists(csv_p): continue
    r = [float(x['episode_reward_total']) for x in csv.DictReader(open(csv_p))]
    if not r: continue
    r.sort()
    med = r[len(r)//2] if len(r)%2 else (r[len(r)//2-1]+r[len(r)//2])/2
    rows.append((f"{arch}_{mode}", N, u, len(r), round(med,2), round(st.mean(r),2),
                 sum(1 for x in r if x==0), jid))
rows.sort(key=lambda t:(t[0], t[1], t[2]))
print(f"{'model':<12}{'N':>4}{'u':>4}{'n':>5}{'median':>11}{'mean':>11}{'zeros':>7}{'job':>10}")
for t in rows: print(f"{t[0]:<12}{t[1]:>4}{t[2]:>4}{t[3]:>5}{t[4]:>11}{t[5]:>11}{t[6]:>7}{t[7]:>10}")
print(f"\n{len(rows)} cells")
