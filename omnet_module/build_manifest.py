#!/usr/bin/env python3
"""Build a machine-readable provenance manifest of every training run.

WHY THIS EXISTS
---------------
Directory names encode the config, but inconsistently --
`omnet_st_n5_enclearned_v2_N5_L2000_sprd350_pX05_staging_odim2_st` repeats the
arch (st ... _st) and the cardinality (n5 ... N5), while the ST reference is
identified only by a hash in the RUN dir, not the experiment dir.

Rather than rename live run directories (which would break the checkpoint
resolvers), this derives ONE
CSV row per run with the fields actually needed to interpret a number:

    track, encoder_mode, arch, n_drones, out_dim, train_seed, tag,
    frames, run_dir, experiment_dir, ckpt_count, mtime, encoder_sha

`encoder_sha` is only meaningful for encoder_mode=frozen and is read from the
run's own log when available -- it is the field that proves every frozen arm
shared one encoder.

USAGE (on arton)
    python omnet_module/build_manifest.py \
        --root $SCRATCH/resilient-backbone/src/out/benchmarl_omnet \
        --logs $SCRATCH/frozen_enc/logs \
        --out  $SCRATCH/run_manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import time

# Experiment dirs look like:
#   omnet_<arch>_n<N>_enc<mode>_v2_N<N>_L2000_sprd350_pX05_staging[_<tag>]
EXP_RX = re.compile(
    r"^omnet_(?P<arch>[a-z]+)_n(?P<n1>\d+)_enc(?P<enc>frozen|learned)"
    r"_v2_N(?P<n2>\d+)_L(?P<L>\d+)_sprd(?P<sprd>\d+)_pX(?P<px>\d+)"
    r"_(?P<spawn>\w+?)(?:_(?P<tag>.+))?$")

# Tags that encode an out_dim sweep arm, e.g. odim2_st / odim16_deepsets
ODIM_RX = re.compile(r"odim(?P<odim>\d+)")
# Frozen tags carry the training seed: frozen_st_ts69 / learned_ffn_ts71
TSEED_RX = re.compile(r"ts(?P<ts>\d+)")


def max_frames(run_dir):
    ck = os.path.join(run_dir, "checkpoints")
    best, n = 0, 0
    try:
        for f in os.listdir(ck):
            m = re.match(r"^checkpoint_(\d+)\.pt$", f)
            if m:
                n += 1
                best = max(best, int(m.group(1)))
    except OSError:
        pass
    return best, n


def encoder_shas(logdir):
    """run-dir-name -> encoder sha, scraped from training logs.

    The frozen arms print `[frozen-enc] user encoder sha256=...` at import.
    Mapping it back to a run requires the tag, which the same log prints as
    `## ARCH=<a> TSEED=<s> tag=<t>`.
    """
    out = {}
    for p in glob.glob(os.path.join(logdir, "*rl_*.log")):
        tag = sha = None
        try:
            with open(p, errors="replace") as f:
                for line in f:
                    if tag is None and "tag=" in line and "ARCH=" in line:
                        m = re.search(r"tag=(\S+)", line)
                        if m:
                            tag = m.group(1)
                    if sha is None and "[frozen-enc]" in line:
                        m = re.search(r"sha256=(\w+)", line)
                        if m:
                            sha = m.group(1)
                    if tag and sha:
                        break
        except OSError:
            continue
        if tag:
            out[tag] = sha or ""
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="benchmarl_omnet dir holding the experiment dirs")
    ap.add_argument("--logs", default=None,
                    help="dir of training logs, for the encoder sha column")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-frames", type=int, default=0,
                    help="omit runs below this frame count (0 = keep all, "
                         "including partials -- a partial is NOT a result but "
                         "IS provenance)")
    a = ap.parse_args()

    shas = encoder_shas(a.logs) if a.logs else {}
    rows = []
    for exp in sorted(os.listdir(a.root)):
        expp = os.path.join(a.root, exp)
        if not os.path.isdir(expp):
            continue
        m = EXP_RX.match(exp)
        if not m:
            rows.append({"track": "OMNET", "arch": "?", "encoder_mode": "?",
                         "n_drones": "", "out_dim": "", "train_seed": "",
                         "tag": "", "frames": "", "run_dir": "",
                         "experiment_dir": exp, "ckpt_count": "",
                         "mtime": "", "encoder_sha": "",
                         "note": "UNPARSED experiment dir name"})
            continue
        g = m.groupdict()
        tag = g.get("tag") or ""
        odim = ODIM_RX.search(tag)
        for run in sorted(glob.glob(os.path.join(expp, "*/"))):
            fr, nck = max_frames(run)
            if fr < a.min_frames:
                continue
            ts = TSEED_RX.search(tag)
            # n1 and n2 both encode N; when they disagree, record both.
            note = "" if g["n1"] == g["n2"] else f"N MISMATCH n1={g['n1']} n2={g['n2']}"
            rows.append({
                "track": "OMNET",
                "encoder_mode": g["enc"],
                "arch": g["arch"],
                "n_drones": g["n2"],
                "out_dim": odim.group("odim") if odim else "8",
                "train_seed": ts.group("ts") if ts else "69",
                "tag": tag,
                "frames": fr,
                "run_dir": os.path.basename(run.rstrip("/")),
                "experiment_dir": exp,
                "ckpt_count": nck,
                "mtime": time.strftime("%Y-%m-%d %H:%M",
                                       time.localtime(os.path.getmtime(run))),
                "encoder_sha": shas.get(tag, ""),
                "note": note,
            })

    cols = ["track", "encoder_mode", "arch", "n_drones", "out_dim",
            "train_seed", "tag", "frames", "ckpt_count", "mtime",
            "encoder_sha", "experiment_dir", "run_dir", "note"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"wrote {len(rows)} rows -> {a.out}")

    # Consistency checks worth failing loudly on.
    frozen = [r for r in rows if r.get("encoder_mode") == "frozen"
              and r.get("encoder_sha")]
    if frozen:
        uniq = sorted({r["encoder_sha"] for r in frozen})
        print(f"\nfrozen runs with a recorded encoder sha: {len(frozen)}")
        print(f"distinct encoder shas: {len(uniq)} -> {uniq}")
        if len(uniq) > 1:
            print("!! MORE THAN ONE ENCODER ACROSS FROZEN ARMS -- the shared-"
                  "encoder control is VIOLATED and those arms are not "
                  "comparable.")
    bad = [r for r in rows if r.get("note")]
    if bad:
        print(f"\n{len(bad)} row(s) with notes:")
        for r in bad[:10]:
            print(f"  {r.get('experiment_dir','?')}: {r['note']}")


if __name__ == "__main__":
    main()
