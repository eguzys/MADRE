#!/usr/bin/env python3
"""Spreading-1 evaluation planner. Read-only on all existing data.
For every family a paper figure reads (the families of results/*_4seed.py at the target seed sets) derive the exact
spec of its spreading-1 twin S1_<orig> from the ORIGINAL family's own slurm header + shard logs (+ c4s ledgers):
cell, users, explicit checkpoint (must be unique over all original shard logs), separation/spawn override, sender
probability, beacon, baseline policy. Also decides which existing SPR1_* family is an exact equivalent (same ckpt,
same world string up to the drone tx power, same charge line, 100 eps) and can be reused instead.
Writes <outdir>/plan_s1.tsv, <outdir>/reuse_s1.tsv, <outdir>/plan_s1_problems.txt.
Usage: python3 plan_s1.py <outdir>
"""
import glob, os, re, sys

S = "/usr/itetnas04/data-scratch-01/eguzys/data/u100diag"
RB = "/usr/itetnas04/data-scratch-01/eguzys/data/resilient-backbone"
W = S + "/c4s"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
SEEDS = {"L8": [1, 2, 3, 5], "L16": [1, 2, 3, 4], "L32": [1, 2, 3, 5], "L64": [1, 2, 3, 4],
         "L128": [1, 2, 3, 5], "L256": [1, 2, 3, 4], "L512": [1, 2, 3, 4]}
RUNGS = list(SEEDS)
CHI40 = ["10", "20", "40", "60", "100"]
CHI100 = ["05", "10", "20", "40", "60", "100"]
P = {"05": "0.05", "10": "0.1", "20": "0.2", "40": "0.4", "60": "0.6", "100": "1.0"}
problems = []


def fam1200(r, k):
    if r in ("L8", "L32", "L128") and k <= 3:
        return f"ADPT40_{r}s{k}"
    if r in ("L8", "L32", "L128") and k == 5:
        return f"X5_{r}s{k}"
    if r == "L16" and k <= 3:
        return f"W1ev_br12rtL16s{k}_SPAWNcluster_TAIL"
    return f"FS_{r}s{k}"


def pdub(c, r, k):
    base = 2 if r == "L512" else 1
    return f"PDUB{c}_{r}" if k == base else f"PDUB{c}_{r}s{k}"


def shard_logs(name, logdir):
    if name.startswith("W1ev_"):
# om2g_logs holds two runs per W1ev L8/L32/L128 shard: an explore-spawn run (job ids 22454xx) and the
# cluster-spawn run (22461xx) that ADPT40_* reformatted. Only the cluster-spawn logs are used.
        fs = glob.glob(f"{RB}/om2g_logs/{name}_s*_*.log")
        return sorted(f for f in fs if re.search(r"_s\d+_\d+\.log$", f)
                      and 'droneSpawnMode=\\"cluster\\"' in open(f, errors="ignore").read())
    fs = glob.glob(f"{logdir}/{name}_s*.log")
    return sorted(f for f in fs if re.fullmatch(re.escape(name) + r"_s\d+\.log", os.path.basename(f)))


def header(name, slurmdir):
    fs = glob.glob(f"{slurmdir}/{name}_s0_*.out")
    fs = [f for f in fs if re.fullmatch(re.escape(name) + r"_s0_\d+\.out", os.path.basename(f))]
    if not fs:
        return None
    f = max(fs, key=lambda x: int(re.search(r"_(\d+)\.out$", x).group(1)))
    for line in open(f, errors="ignore"):
        if line.startswith("=== "):
            return line.strip()
    return None


def world_fp(logf):
    """normalised world string + charge line of one shard log (drone tx power token masked)"""
    w = post = None
    for line in open(logf, errors="ignore"):
        if w is None and "OMNET_EXTRA_ARGS =" in line:
            w = line.split("OMNET_EXTRA_ARGS =", 1)[1].strip()
        if post is None and (line.startswith("[post]") or line.startswith("[charge]")):
            post = line.strip()
        if w and post:
            break
    if w:
        w = re.sub(r"(host\[\*\]\.wlan\[\*\]\.radio\.transmitter\.power=)[0-9.]+mW", r"\1<P>", w)
    return w, post


def ledger_rows(fn):
    rows = []
    if os.path.exists(fn):
        for line in open(fn):
            rows.append(line.rstrip("\n").split("\t"))
    return rows


LPDUB = {}
for r in ledger_rows(W + "/ledger_pdub.tsv"):
    LPDUB.setdefault(r[1], set()).add((r[5], r[6]))   # name -> {(cell, ckpt)}

plan = []   # dicts


def learned(orig, ub=False, chi=None, minsep=None, spawn=None):
    src = orig
    if orig.startswith("ADPT40_"):   # reformatted copies of the W1ev TAIL evals (no banner): take the source evals
        src = "W1ev_br12rt" + orig.split("_", 1)[1] + "_SPAWNcluster_TAIL"
    logs = shard_logs(src, S + "/logs")
    cks, cell, users, eng = set(), None, None, False
    for f in logs:
        for line in open(f, errors="ignore"):
            m = re.match(r"\s+ckpt=(\S+)", line)
            if m:
                cks.add(m.group(1))
            if "out=" in line and "OM2G_engineered_" in line:
                eng = True
    if src.startswith("W1ev_"):
        cell = re.match(r"W1ev_(br12\w+?)_SPAWN", src).group(1); users = "40"
    else:
        h = header(src, S + "/slurm")
        if h:
            m = re.search(r"args: (\S+) (\d+) \d+ \d+ (\S+) (\S+) SENDERPROB=(\S*)", h)
            if m:
                cell, users = m.group(1), m.group(2)
                if chi and m.group(5) != P[chi]:
                    problems.append(f"{orig}: header SENDERPROB={m.group(5)} != {P[chi]}")
    if not cell and logs:   # no slurm header (e.g. PDUB40_L32s2): banner "tag=<cell>_<name>" of the shard log
        for line in open(logs[0], errors="ignore"):
            m = re.search(r"OM2G EVAL .* tag=(\S+)_" + re.escape(src) + r"\s", line)
            if m:
                cell = m.group(1); users = "100" if src.startswith("PDUB") else "40"
                problems.append(f"NOTE {orig}: no slurm header, cell {cell} / users {users} from the log banner"); break
    if orig in LPDUB and not logs:   # submitted (ledger) but no log yet
        (cell, ck), = LPDUB[orig]; cks = {ck}; users = "100"
    elif orig in LPDUB:
        lc = {c for _, c in LPDUB[orig]}
        if lc != cks:
            problems.append(f"{orig}: ledger ckpt {lc} != log ckpt {cks}")
    if len(cks) != 1 or not cell:
        problems.append(f"{orig}: cannot derive spec (ckpts={cks}, cell={cell}, logs={len(logs)})"); return
    ck = cks.pop()
    if not os.path.isfile(f"{RB}/src/{ck}") or os.path.getsize(f"{RB}/src/{ck}") == 0:
        problems.append(f"{orig}: ckpt missing on disk {ck}"); return
    if src.startswith("W1ev_") is False and cell and not re.search(r"(L\d+)s(\d)", cell):
        problems.append(f"{orig}: odd cell {cell}")
    exp = ["ALL", "DIAG_SPREADING=1"]
    if minsep:
        exp.append(f"DIAG_MINSEP={minsep}")
    if spawn:
        exp.append(f"DIAG_SPAWN={spawn}")
    if eng:   # B16_ENG: engineered policy through the shard path, payload-matched 67 B charge
        exp += ["OM2G_EVAL_POLICY=engineered", "DIAG_POST=export OM2G_CONTROL_ON_AIR_BYTES=67"]
    if ub:
        exp += [f"DIAG_SENDERPROB={P[chi]}", "UB"]
    big = re.search(r"L512", cell) or (users == "100" and re.search(r"L256", cell))
    fp = world_fp(logs[0]) if logs else (None, None)
    plan.append(dict(s1="S1_" + orig, orig=orig, kind="learned", sb=S + ("/shard_spr_ub.sbatch" if ub else "/shard_spr.sbatch"),
                     a1=cell, users=users, ck=ck, exp=",".join(exp), mem="32G" if big else "-", time="-",
                     logdir=S + "/logs", split="std", fpw=fp[0] or "-", fpp=fp[1] or "-"))


def baseline(orig, policy, users, sb, logdir, slurmdir, exp_extra, time="-", mem="-", split="std", chi=None):
    h = header(orig, slurmdir)
    logs = shard_logs(orig, logdir)
    if not h or not logs:
        problems.append(f"{orig}: no header/logs ({h is not None}, {len(logs)})"); return
    m = re.search(r"(\S+) users=(\d+) ss=\d+ ep=\d+ name=(\S+) SENDERPROB=(\S*)", h)
    if not m or m.group(1) != policy or m.group(2) != users or m.group(3) != orig:
        problems.append(f"{orig}: header mismatch: {h}")
    if chi and m and m.group(4) != P[chi]:
        problems.append(f"{orig}: SENDERPROB {m.group(4)} != {P[chi]}")
    fp = world_fp(logs[0])
    plan.append(dict(s1="S1_" + orig, orig=orig, kind="baseline", sb=sb, a1=policy, users=users, ck="-",
                     exp=",".join(["ALL", "DIAG_SPREADING=1"] + exp_extra), mem=mem, time=time, logdir=logdir,
                     split=split, fpw=fp[0] or "-", fpp=fp[1] or "-"))


F = S + "/suite"
# ---------- P0: 40 users ----------
for r in RUNGS:
    for k in SEEDS[r]:
        learned(fam1200(r, k))
        for d in (1300, 1400, 1500, 1600, 1700):
            learned(f"SEP{d}_{r}s{k}", minsep=d)
baseline("eng_U40", "engineered", "40", S + "/baseline_shard_cl_spr.sbatch", S + "/logs", S + "/slurm", ["DIAG_CLUSTERS=4"], mem="14G")
for d in (1300, 1400, 1500, 1700):
    baseline(f"SEP{d}_ENG", "engineered", "40", S + "/baseline_shard_cl_spr.sbatch", S + "/logs", S + "/slurm",
             ["DIAG_CLUSTERS=4", f"DIAG_MINSEP={d}"], mem="14G")
learned("B16_ENG")
for r in ("L8", "L32", "L64", "L128"):
    for k in SEEDS[r]:
        learned(f"USPAWN_{r}s{k}", spawn="uniform")
baseline("USPAWN_ENG", "engineered", "40", S + "/baseline_shard_cl_spr.sbatch", S + "/logs", S + "/slurm",
         ["DIAG_CLUSTERS=4", "DIAG_SPAWN=uniform"], mem="14G")
for k in SEEDS["L128"]:
    learned(f"CKFIN_L128s{k}")
for c in CHI40:
    for k in SEEDS["L64"]:
        learned(f"NDUB{c}_L64s{k}", ub=True, chi=c)
    baseline(f"NDUB{c}_ENG", "engineered", "40", S + "/baseline_shard_cl_ub_spr.sbatch", S + "/logs", S + "/slurm",
             ["DIAG_CLUSTERS=4", f"DIAG_SENDERPROB={P[c]}", "UB"], chi=c)
    baseline(f"NDUB{c}_GTMG", "gtm", "40", F + "/suite_shard2_ub_spr.sbatch", F + "/logs", F + "/slurm",
             [f"DIAG_SENDERPROB={P[c]}", "UB", "SUITE_GTM_OBJ=global"], time="16:00:00",
             split="gtm64" if c == "100" else "std", chi=c)
    baseline(f"NDUB{c}_AGG", "aggregate", "40", F + "/suite_shard2_ub_spr.sbatch", F + "/logs", F + "/slurm",
             [f"DIAG_SENDERPROB={P[c]}", "UB"], chi=c)
    baseline(f"NDUB{c}_R1H", "relay_1hop", "40", F + "/relay2_shard_ub_spr.sbatch", F + "/logs", F + "/slurm",
             [f"DIAG_SENDERPROB={P[c]}", "UB"], chi=c)
# ---------- P1: 100 users ----------
have = {re.sub(r"_s\d+\.log$", "", os.path.basename(f)) for f in glob.glob(S + "/logs/PDUB*_s*.log")
        if re.search(r"_s\d+\.log$", f)} | set(LPDUB)
pd_missing = []
for c in CHI100:
    for r in RUNGS:
        for k in SEEDS[r]:
            n = pdub(c, r, k)
            if n in have:
                learned(n, ub=True, chi=c)
            else:
                pd_missing.append(n)
for c in CHI100:
    for code, pol, sb, extra, t in [("GTMG", "gtm", "suite_shard2_ub_spr.sbatch", ["SUITE_GTM_OBJ=global"], "1-12:00:00"),
                                    ("GRD", "greedy", "suite_shard2_ub_spr.sbatch", [], "-"),
                                    ("AGG", "aggregate", "suite_shard2_ub_spr.sbatch", [], "-"),
                                    ("R1H", "relay_1hop", "relay2_shard_ub_spr.sbatch", [], "-")]:
        baseline(f"DUB{c}_{code}", pol, "100", F + "/" + sb, F + "/logs", F + "/slurm",
                 [f"DIAG_SENDERPROB={P[c]}", "UB"] + extra, time=t, chi=c)
    baseline(f"ENUB_N100_{c}", "engineered", "100", S + "/baseline_shard_cl_ub_spr.sbatch", S + "/logs", S + "/slurm",
             ["DIAG_CLUSTERS=4", f"DIAG_SENDERPROB={P[c]}", "UB"], chi=c)

# ---------- reuse of the existing SPR1 families (1200 m, 40 users, beacon off) ----------
REUSE = {}
for r, ks in [("L8", [1, 2, 3, 5]), ("L64", [1, 2, 3, 4]), ("L128", [1, 2, 3, 5]), ("L512", [1, 2, 3, 4])]:
    for k in ks:
        REUSE[fam1200(r, k)] = f"SPR1_{r}s{k}"
REUSE["eng_U40"] = "SPR1_ENG"
reuse_rows = []
byorig = {p["orig"]: p for p in plan}
for orig, spr in REUSE.items():
    p = byorig.get(orig)
    logs = shard_logs(spr, S + "/logs")
    eps = set()
    cks = set()
    for f in logs:
        for line in open(f, errors="ignore"):
            m = re.match(r"\s+ep seed=\s*(\d+)\s+reward=\s*\S+\s+steps=\s*250", line)
            if m:
                eps.add(int(m.group(1)))
            m = re.match(r"\s+ckpt=(\S+)", line)
            if m:
                cks.add(m.group(1))
    why = []
    if p is None:
        why.append("orig not planned")
    else:
        if p["kind"] == "learned" and cks != {p["ck"]}:
            why.append(f"ckpt {cks} != {p['ck']}")
        w, post = world_fp(logs[0]) if logs else (None, None)
        if w != p["fpw"]:
            why.append("world string differs")
        if post != p["fpp"]:
            why.append(f"charge line differs: {post} vs {p['fpp']}")
        if not all(("6.59004mW" in open(f, errors="ignore").read()) for f in logs):
            why.append("not all logs at 6.59004mW")
    if eps != set(range(100)):
        why.append(f"episodes {len(eps)}/100")
    ok = not why
    reuse_rows.append((orig, spr, "REUSE" if ok else "RUN_S1", "; ".join(why) or "same ckpt, same world (tx power masked), same charge line, eps 0..99, all logs 6.59004mW"))
    if ok and p is not None:
        p["reuse"] = spr

with open(OUT + "/plan_s1.tsv", "w") as o:
    cols = ["s1", "orig", "kind", "sb", "a1", "users", "ck", "exp", "mem", "time", "logdir", "split", "reuse", "fpp", "fpw"]
    o.write("\t".join(cols) + "\n")
    for p in plan:
        p.setdefault("reuse", "-")
        o.write("\t".join(str(p[c]) for c in cols) + "\n")
with open(OUT + "/reuse_s1.tsv", "w") as o:
    o.write("orig\tspr1\tdecision\twhy\n")
    for row in reuse_rows:
        o.write("\t".join(row) + "\n")
with open(OUT + "/plan_s1_problems.txt", "w") as o:
    o.write("\n".join(problems) + "\n")
    o.write(f"\nPDUB target families not yet existing (left to the c4s loop when their fine-tune finishes): {len(pd_missing)}\n")
    o.write(" ".join(pd_missing) + "\n")
nrun = sum(1 for p in plan if p["reuse"] == "-")
print(f"planned {len(plan)} families, {nrun} to run, {len(plan) - nrun} reused; problems {len(problems)}; PDUB not yet existing {len(pd_missing)}")
