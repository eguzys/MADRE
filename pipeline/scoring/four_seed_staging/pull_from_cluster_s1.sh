# Read-only extraction for the spreading-1 scorers; writes a tar.gz to stdout, nothing permanent on the
# cluster (temp dir under u100diag/c4s/scorer_night/, removed). Run via rescore_4seed_s1.sh.
# Every spreading-1 family is staged under its base family name, so the 4-seed scorers run unchanged
# on it with --spreading 1:
#   S1_<orig>  (plan_s1.tsv and ledger_s1.tsv)                       -> <orig>
#   SPR1_<...> where reuse_s1.tsv says REUSE (identical settings)      -> <orig>
# A shard log is staged only if its world string sets the drone tx power to 6.59004 mW (host[*]) and never to
# 52.7203 mW; anything else is listed in MANIFEST.tsv as REJECTED. CSVs are copied with the family name replaced.
S=/usr/itetnas04/data-scratch-01/eguzys/data
mkdir -p $S/u100diag/c4s/scorer_night; T=$(mktemp -d $S/u100diag/c4s/scorer_night/pull_s1_XXXX)
python3 - "$T" <<'PY' 1>&2
import os, re, sys, shutil, glob
T = sys.argv[1]
S = "/usr/itetnas04/data-scratch-01/eguzys/data"
U = S + "/u100diag"; W = U + "/c4s"
MAP = {}   # src family -> (orig family, logdir)
for line in open(W + "/s1/plan_s1.tsv").read().splitlines()[1:]:
    c = line.split("\t")
    MAP[c[0]] = (c[1], c[10])
for line in open(W + "/s1/reuse_s1.tsv").read().splitlines()[1:]:
    orig, spr, dec = line.split("\t")[:3]
    if dec == "REUSE":
        MAP[spr] = (orig, U + "/logs")
if os.path.exists(W + "/ledger_s1.tsv"):
    for line in open(W + "/ledger_s1.tsv"):
        c = line.rstrip("\n").split("\t")
        if c[1].startswith("S1_") and c[1] not in MAP:
            MAP[c[1]] = (c[1][3:], c[12])
EP = re.compile(r"^\s+ep seed=\s*(\d+)\s+reward=\s*([-\d.]+)\s+steps=\s*(\d+)", re.M)
HOSTP = re.compile(r"host\[\*\]\.wlan\[\*\]\.radio\.transmitter\.power=([0-9.]+)mW")
ACC = set()   # (src, shard) whose log passed the power check; only their CSVs are staged
man = ["src\torig\tlogs_ok\tlogs_rejected\tepisodes\tcsv_files\trejected_logs"]
n = 0
with open(os.path.join(T, "episodes_logs.txt"), "w") as o:
    o.write("fam\tshard\tep\treward\tsteps\tsrc\n")
    for src, (orig, ld) in sorted(MAP.items()):
        ok = rej = ne = 0; bad = []
        for f in sorted(glob.glob(f"{ld}/{src}_s*.log")):
            m = re.fullmatch(re.escape(src) + r"_s(\d+)\.log", os.path.basename(f))
            if not m:
                continue
            txt = open(f, errors="ignore").read()
            pw = set(HOSTP.findall(txt))
            if pw != {"6.59004"}:
                rej += 1; bad.append(f"{os.path.basename(f)}:{'/'.join(sorted(pw)) or 'no-power-line'}"); continue
            ok += 1; ACC.add((src, int(m.group(1))))
            for e in EP.finditer(txt):
                o.write(f"{orig}\t{m.group(1)}\t{e.group(1)}\t{e.group(2)}\t{e.group(3)}\t{os.path.basename(f)}\n"); n += 1; ne += 1
        MAP[src] = (orig, ld, ok, rej, ne, bad)
print("episode lines", n)
os.makedirs(os.path.join(T, "csv"))
CRX = re.compile(r"_((?:S1|SPR1)_\w+?)(?:_at\d+)?_(?:n10_)?seed(\d+)\.csv$")
k = 0; ncsv = {}
for d in (U + "/out", U + "/suite/out"):
    for fn in os.listdir(d):
        if not fn.startswith("OM2G_"):
            continue
        m = CRX.search(fn)
        if not m or m.group(1) not in MAP or (m.group(1), int(m.group(2))) not in ACC:
            continue
        src = m.group(1); orig = MAP[src][0]
        new = fn[:m.start(1)] + orig + fn[m.end(1):]
        shutil.copy2(os.path.join(d, fn), os.path.join(T, "csv", new)); k += 1; ncsv[src] = ncsv.get(src, 0) + 1
print("csv files", k)
for src, v in sorted(MAP.items()):
    orig, ld, ok, rej, ne, bad = v
    man.append(f"{src}\t{orig}\t{ok}\t{rej}\t{ne}\t{ncsv.get(src, 0)}\t{' '.join(bad)}")
open(os.path.join(T, "MANIFEST.tsv"), "w").write("\n".join(man) + "\n")
print("families", len(MAP), "with rejected logs", sum(1 for v in MAP.values() if v[3]))
PY
date -u +%FT%TZ > $T/PULLED_AT.txt
tar -C $T -czf - .
rm -rf "$T"
