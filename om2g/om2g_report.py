#!/usr/bin/env python3
"""OM2G final report — one command, markdown out.

⚠️ OM2G TRACK. Rewards are per-STEP in kbps and are NEVER comparable to an
OMNET-track number (Mbps). See code/om2g/OM2G_README.md.

Reads whatever is present in --dir and prints a markdown block:
  OM2G_baseline_<policy>_N5_n*.csv      hover / random / greedy / aggregate
  OM2G_trained_<tag>_n*.csv             trained policies
  telemetry_OM2G_<arm>_st_s<seed>.csv   control-plane / peer-availability

Everything is compared PAIRED PER SEED (project rule: an unpaired mean hides
the seed variance that dominates this task), and every difference carries its
standard error. The hover->greedy gap is reported with its own error bar
because at n=40 it is NOT significant (t=+1.56) and a gap-normalised score
against it would be a ratio with a denominator consistent with zero.
"""
from __future__ import annotations
import argparse, csv, glob, math, os, re, statistics as st, collections

def load(path):
    rows = list(csv.DictReader(open(path)))
    return {int(r["seed"]): float(r["episode_reward"]) / float(r.get("steps", 250) or 250)
            for r in rows if r.get("episode_reward")}

def paired(a, b):
    ks = sorted(set(a) & set(b))
    if not ks: return None
    d = [a[k] - b[k] for k in ks]
    m = st.mean(d); se = st.pstdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    return m, se, sum(1 for x in d if x > 0), len(d)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    a = ap.parse_args()
    D = a.dir

    base = {}
    for p in glob.glob(os.path.join(D, "OM2G_baseline_*_N5_n*.csv")):
        m = re.search(r"OM2G_baseline_(\w+?)_N5_n(\d+)", os.path.basename(p))
        if m: base[m.group(1)] = load(p)

    trained = {}
    for p in glob.glob(os.path.join(D, "OM2G_trained_*_n*.csv")):
        m = re.search(r"OM2G_trained_(.+)_n(\d+)\.csv", os.path.basename(p))
        if m and not m.group(1).startswith("SMOKE"):
            trained[m.group(1)] = load(p)

    print("### Baselines and trained policies — reward per STEP (kbps)\n")
    print("| policy | n | mean | median | sd |")
    print("|---|---|---|---|---|")
    for name in ("greedy", "aggregate", "hover", "random"):
        if name in base and base[name]:
            v = list(base[name].values())
            print(f"| {name} | {len(v)} | {st.mean(v):.1f} | {st.median(v):.1f} | {st.pstdev(v):.1f} |")
    for name in sorted(trained):
        v = list(trained[name].values())
        if v:
            print(f"| **{name}** | {len(v)} | **{st.mean(v):.1f}** | {st.median(v):.1f} | {st.pstdev(v):.1f} |")

    if "hover" in base and "greedy" in base:
        g = paired(base["greedy"], base["hover"])
        if g:
            print(f"\n**hover → greedy gap = {g[0]:+.1f} ± {g[1]:.1f} kbps/step "
                  f"(t = {g[0]/g[1]:+.2f}, {g[2]}/{g[3]} seeds).** "
                  "Not significant at n=40 — quote the error bar with any "
                  "gap-normalised score.\n")

    if trained and "hover" in base:
        print("### Trained vs baselines, PAIRED per seed (mean ± SE, seeds won)\n")
        print("| arm | vs hover | vs random | vs greedy |")
        print("|---|---|---|---|")
        for name in sorted(trained):
            cells = []
            for ref in ("hover", "random", "greedy"):
                r = paired(trained[name], base.get(ref, {}))
                cells.append(f"{r[0]:+.0f} ± {r[1]:.0f} (t={r[0]/r[1]:+.2f}, {r[2]}/{r[3]})"
                             if r and r[1] == r[1] else "—")
            print(f"| {name} | " + " | ".join(cells) + " |")

        comp = [k for k in trained if "uncompressed" not in k and "compressed" in k]
        unc  = [k for k in trained if "uncompressed" in k]
        if comp and unc:
            print("\n### ⭐ THE COMPARISON — compressed vs uncompressed, paired per seed\n")
            print("| compressed | uncompressed | Δ (mean ± SE) | t | seeds won |")
            print("|---|---|---|---|---|")
            for c in sorted(comp):
                for u in sorted(unc):
                    if c.split("compressed")[-1] != u.split("uncompressed")[-1]:
                        continue
                    r = paired(trained[c], trained[u])
                    if r:
                        print(f"| {c} | {u} | {r[0]:+.1f} ± {r[1]:.1f} | "
                              f"{r[0]/r[1]:+.2f} | {r[2]}/{r[3]} |")

    # ⚠️ Each training run has 10 collector worker processes and they ALL
    # append to the same telemetry csv, so header rows appear interleaved
    # mid-file (and a row could in principle be torn). Rows that do not parse
    # as numbers are dropped rather than trusted.
    def _num(r, k):
        try: return float(r[k])
        except (TypeError, ValueError, KeyError): return None
    tel = collections.defaultdict(list)
    for p in glob.glob(os.path.join(D, "telemetry_OM2G_*.csv")):
        for r in csv.DictReader(open(p)):
            if _num(r, "ctrl_share_of_channel") is not None:
                tel[r["arm"]].append(r)
    if tel:
        print("\n### Network telemetry (live, from the training runs)\n")
        print("| arm | episodes | control kbps | % of channel | peer available | peer fresh | channel busy |")
        print("|---|---|---|---|---|---|---|")
        for arm in ("compressed", "uncompressed"):
            R = tel.get(arm)
            if not R: continue
            m = lambda k: st.mean(x for x in (_num(r,k) for r in R) if x is not None)
            print(f"| {arm} | {len(R)} | {m('ctrl_bps_tx')/1000:.2f} | "
                  f"{m('ctrl_share_of_channel')*100:.2f}% | {m('peer_available_frac')*100:.1f}% | "
                  f"{m('peer_fresh_frac')*100:.1f}% | {m('channel_busy')*100:.1f}% |")

if __name__ == "__main__":
    main()
