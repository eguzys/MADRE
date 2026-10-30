import glob, csv, os, re, yaml, statistics as st
# Map: slurm log (has model+u in its name) -> results dir (from "Results written to:")
rows = []
for log in glob.glob('/cluster/scratch/eguzys/eval_*_u*.out'):
    m = re.search(r'eval_(st|ds|ffn)_(learned|frozen)_u(\d+)_(\d+)\.out$', os.path.basename(log))
    if not m: continue
    arch, mode, u, jid = m.group(1), m.group(2), int(m.group(3)), m.group(4)
    txt = open(log, errors='ignore').read()
    d = re.search(r'Results written to:\s*(\S+)', txt)
    if not d: continue
    csv_p = os.path.join(d.group(1), 'per_episode.csv')
    if not os.path.exists(csv_p): continue
    r = [float(x['episode_reward_total']) for x in csv.DictReader(open(csv_p))]
    if not r: continue
    r.sort()
    med = r[len(r)//2] if len(r)%2 else (r[len(r)//2-1]+r[len(r)//2])/2
    rows.append((f"{arch}_{mode}", u, len(r), round(med,2), round(st.mean(r),2),
                 sum(1 for x in r if x==0), jid))
rows.sort(key=lambda t:(t[0], t[1]))
print(f"{'model':<12}{'u':>4}{'n':>5}{'median':>11}{'mean':>11}{'zeros':>7}{'job':>10}")
for t in rows: print(f"{t[0]:<12}{t[1]:>4}{t[2]:>5}{t[3]:>11}{t[4]:>11}{t[5]:>7}{t[6]:>10}")
print(f"\n{len(rows)} cells of 73")
