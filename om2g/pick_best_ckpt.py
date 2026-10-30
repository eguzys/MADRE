#!/usr/bin/env python3
"""Pick the checkpoint at PEAK TRAINING RETURN, not the last one.

╔══════════════════════════════════════════════════════════════════════════╗
║  OM2G TRACK.  Best-checkpoint rule used for every evaluation.            ║
╚══════════════════════════════════════════════════════════════════════════╝

WHY. Measured across five cells: training return peaks at ~150-190k frames
and then DRIFTS DOWN (m1000L32 .58@186k -> .47@300k; w1000talk .55 -> .47;
m1000L128 .57 -> .51). Scoring the LAST checkpoint would therefore score
every arm post-peak and penalise a slow-converging small-L rung twice -- once
for converging late, once for drifting past its peak. The rule "score the best
checkpoint by TRAINING return" applies to every arm of both codec modes; it
selects on the training metric, never on eval scores.

Usage:
    pick_best_ckpt.py <run_dir>            # print best checkpoint path
    pick_best_ckpt.py <run_dir> --table    # + the full curve
<run_dir> may be the save_folder tag dir (src/out/OM2G/<TAG>) or the
experiment dir inside it; the newest experiment dir is used if several.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys


def _experiment_dir(run_dir: str) -> str:
    if glob.glob(os.path.join(run_dir, "checkpoints", "checkpoint_*.pt")):
        return run_dir
    subs = [d for d in glob.glob(os.path.join(run_dir, "*/"))
            if glob.glob(os.path.join(d, "checkpoints", "checkpoint_*.pt"))]
    if not subs:
        raise SystemExit(
            f"no checkpoints under {run_dir} (or its subdirs) -- nothing to "
            f"pick. An empty result must FAIL, not pass silently.")
    return max(subs, key=os.path.getmtime)


def _curve_wandb(exp_dir: str):
    """(total_frames, episode_reward_mean) per ITERATION, from the offline
    wandb datastore -- 6k-frame resolution, vs the json logger's
    evaluation_interval (150k), which is far too coarse to catch a peak at
    ~150-190k. Returns [] when no wandb log exists."""
    fs = sorted(glob.glob(os.path.join(
        exp_dir, "wandb", "offline-run-*", "*.wandb")), key=os.path.getmtime)
    if not fs:
        return []
    try:
        from wandb.sdk.internal.datastore import DataStore
        from wandb.proto import wandb_internal_pb2 as pb
    except Exception:
        return []
    pts = []
    ds = DataStore()
    ds.open_for_scan(fs[-1])
    while True:
        try:
            data = ds.scan_data()
        except Exception:
            break
        if data is None:
            break
        rec = pb.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            continue
        if not rec.HasField("history"):
            continue
        kv = {}
        for it in rec.history.item:
            key = it.key or "/".join(it.nested_key)
            kv[key] = it.value_json
        try:
            frames = int(float(kv["counters/total_frames"]))
            ret = float(kv["collection/reward/episode_reward_mean"])
        except (KeyError, ValueError):
            continue
        pts.append((frames, ret))
    return sorted(pts)


def _smooth(pts, w=3):
    """Centered moving average (window w). Picking the argmax of a raw
    per-iteration curve rewards one lucky batch; w=3 is used for every run."""
    if len(pts) < w:
        return pts
    out = []
    for i in range(len(pts)):
        lo_i = max(0, i - w // 2)
        hi_i = min(len(pts), i + w // 2 + 1)
        out.append((pts[i][0],
                    sum(r for _, r in pts[lo_i:hi_i]) / (hi_i - lo_i)))
    return out


def _curve(exp_dir: str):
    """(step_count, mean_return) points from the benchmarl json logger."""
    js = sorted(glob.glob(os.path.join(exp_dir, "*.json")),
                key=os.path.getmtime)
    if not js:
        raise SystemExit(f"no training-curve json in {exp_dir}; cannot pick "
                         f"a best checkpoint without the curve.")
    d = json.load(open(js[-1]))
    # d[env][task][algo][seed_k] = {"step_K": {"step_count": .., "return":[..]}}
    try:
        while isinstance(d, dict) and d and not any(
                re.match(r"step_\d+$", k) for k in d):
            d = next(iter(d.values()))
    except StopIteration:
        raise SystemExit(f"unrecognised curve json structure in {js[-1]}")
    if not isinstance(d, dict):
        raise SystemExit(f"unrecognised curve json structure in {js[-1]}")
    pts = []
    for k, v in d.items():
        m = re.match(r"step_(\d+)$", k)
        if not m or not isinstance(v, dict):
            continue
        ret = v.get("return") or v.get("drone_return")
        if ret is None or "step_count" not in v:
            continue
        vals = ret if isinstance(ret, list) else [ret]
        pts.append((int(v["step_count"]), sum(vals) / len(vals)))
    if not pts:
        raise SystemExit(f"curve json {js[-1]} had no step_K/return points.")
    return sorted(pts)


def main():
    run_dir = sys.argv[1]
    table = "--table" in sys.argv
    exp_dir = _experiment_dir(run_dir)
    pts = _curve_wandb(exp_dir)
    src = "wandb per-iteration"
    if not pts:
        pts = _curve(exp_dir)
        src = "json logger (COARSE: evaluation_interval only)"
    pts = _smooth(pts)
    best_step, best_ret = max(pts, key=lambda p: p[1])
    cks = {}
    for p in glob.glob(os.path.join(exp_dir, "checkpoints",
                                    "checkpoint_*.pt")):
        m = re.search(r"checkpoint_(\d+)\.pt$", p)
        if m:
            cks[int(m.group(1))] = p
    if not cks:
        raise SystemExit(f"no checkpoint files in {exp_dir}/checkpoints")
    pick_frames = min(cks, key=lambda f: abs(f - best_step))
    if table:
        for s, r in pts:
            mark = "  <-- PEAK" if s == best_step else ""
            ck = "  [ckpt]" if s in cks else ""
            print(f"  {s:>9d}  {r:+.4f}{ck}{mark}", file=sys.stderr)
        last_ret = pts[-1][1]
        print(f"# source: {src}; smoothed w=3", file=sys.stderr)
        print(f"# peak {best_ret:+.4f} @ {best_step} frames; last "
              f"{last_ret:+.4f} @ {pts[-1][0]}; picked checkpoint_"
              f"{pick_frames} (|d|={abs(pick_frames - best_step)})",
              file=sys.stderr)
    print(cks[pick_frames])


if __name__ == "__main__":
    main()
