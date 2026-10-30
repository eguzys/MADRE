#!/usr/bin/env python3
"""OM2G regime probe — the gate that must pass BEFORE any training is launched.

╔══════════════════════════════════════════════════════════════════════════╗
║  OM2G TRACK (IEEE 802.15.4 narrowband).  NOT the OMNET track.            ║
╚══════════════════════════════════════════════════════════════════════════╝

Runs the simulator with NO learned model (scripted hover / random actions) and
measures the three things the OM2G design depends on:

  1. CONNECTIVITY REGIME.  Mean number of drone-drone neighbours each drone
     actually hears.  The target is the multi-hop middle: not a full mesh
     (every drone hears every other -> relaying is free -> coordination costs
     nothing), not fragmentation (nobody hears anybody -> chaos).  At N=5 the
     full-mesh ceiling is 4, so the target band is ~1.5-3.0.

  2. STATE-MESSAGE DELIVERY.  What fraction of broadcast state messages are
     actually received and reassembled, per arm.  This is what makes "the
     message traverses the network" a measured claim: the uncompressed arm's
     15-fragment datagram must survive all 15 fragments.

  3. CONTROL-PLANE SHARE.  Measured channel-busy fraction and control-plane
     bytes/s, versus the analytic prediction in om2g_track.py.

Usage:
    export OM2G_TRACK=1
    python om2g_regime_probe.py --arm compressed --episodes 3 --steps 60
    python om2g_regime_probe.py --arm uncompressed --episodes 3 --steps 60
    python om2g_regime_probe.py --arm nocontrol   --episodes 3 --steps 60
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import zmq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "omnet_module"))

import om2g_track as T

T.require_om2g("om2g_regime_probe.py")

OM2G_SMOKE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "omnet_smoke")
OM2G_INET_DIR = os.environ.get(
    "OMNET_INET_DIR", "/home/emilis/Documents/thesis/inet4.6")
OM2G_OMNETPP_DIR = os.environ.get(
    "OMNETPP_DIR", "/home/emilis/Documents/thesis/omnetpp-6.4.0")

ARM_CONFIG = {
    "compressed":   "OM2G_COMPRESSED",
    "uncompressed": "OM2G_UNCOMPRESSED",
    "nocontrol":    "OM2G_NOCONTROL",
}


def _launch(endpoint: str, config: str, seed: int, steps: int, logpath: str):
    inet_src = os.path.join(OM2G_INET_DIR, "src")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(filter(None, [
        inet_src, os.path.join(OM2G_OMNETPP_DIR, "lib"),
        env.get("LD_LIBRARY_PATH", "")]))
    cmd = [
        "./out/clang-release/omnet_smoke", "-u", "Cmdenv",
        "-c", config, "-n", f".:{inet_src}", "om2g.ini",
        f"--*.rlBridge.episodeSeed={seed}",
        f"--*.rlBridge.maxSteps={steps}",
        f'--*.rlBridge.pythonEndpoint="{endpoint}"',
        f"--sim-time-limit={steps + 10}s",
    ]
    fh = open(logpath, "w")
    return subprocess.Popen(cmd, cwd=OM2G_SMOKE_DIR, env=env,
                            stdout=fh, stderr=subprocess.STDOUT,
                            process_group=0), fh


def run_episode(arm: str, seed: int, steps: int, policy: str, port: int):
    """One episode. Returns a dict of per-step measurement arrays."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 240_000)
    actual = sock.bind_to_random_port("tcp://127.0.0.1") if port == 0 else (
        sock.bind(f"tcp://127.0.0.1:{port}") or port)
    endpoint = f"tcp://127.0.0.1:{actual}"

    logfd, logpath = tempfile.mkstemp(prefix=f"om2g_probe_{arm}_", suffix=".log")
    os.close(logfd)
    proc, fh = _launch(endpoint, ARM_CONFIG[arm], seed, steps, logpath)

    rng = np.random.default_rng(seed)
    rec = {k: [] for k in ("nbr", "busy", "ctrl_tx", "ctrl_rx", "thr",
                           "beacon_tx", "beacon_rx", "msg_age", "fresh",
                           "spread")}
    payload_bytes = None
    try:
        while True:
            obs = json.loads(sock.recv().decode("utf-8"))
            if obs.get("done"):
                sock.send_string(json.dumps({"type": "ack"}))
                break
            n = len(obs.get("positions", []))
            step = obs.get("step", 0)

            payload_bytes = obs.get("beacon_payload_bytes", payload_bytes)

            # (1) neighbour count: peers whose beacon arrived within the last
            # 3 sync intervals. Uses the REAL radio, not a geometric proxy.
            age = np.asarray(obs.get("beacon_rx_age", []), dtype=float)
            if age.size:
                fresh = (age >= 0.0) & (age <= 3.0 * T.OM2G_SYNC_INTERVAL_S)
                np.fill_diagonal(fresh, False)          # never count yourself
                rec["nbr"].append(float(fresh.sum(axis=1).mean()))
                seen = age[(age >= 0.0)]
                rec["msg_age"].append(float(seen.mean()) if seen.size else np.nan)
                rec["fresh"].append(float(fresh.mean()))

            # swarm spread: mean pairwise drone distance. The neighbour count
            # is read next to this -- drones spawn bunched in the
            # staging area, so an early-episode mesh says nothing about the
            # regime a dispersed policy actually operates in.
            P = np.asarray(obs.get("positions", []), dtype=float)
            if P.shape[0] > 1:
                dd = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
                rec["spread"].append(float(dd[np.triu_indices(len(P), 1)].mean()))
            rec["busy"].append(float(np.mean(obs.get("channel_busy", [0.0]))))
            rec["ctrl_tx"].append(float(np.sum(obs.get("ctrl_bytes_tx", [0.0]))))
            rec["ctrl_rx"].append(float(np.sum(obs.get("ctrl_bytes_rx", [0.0]))))
            rec["thr"].append(float(np.sum(obs.get("throughput", [0.0]))))
            rec["beacon_tx"].append(float(np.sum(obs.get("beacon_tx_count", [0]))))
            rec["beacon_rx"].append(float(np.sum(obs.get("beacon_rx_count", [0]))))

            # ⚠️ ACTIONS MUST BE SCALED BY STEP_M HERE.
            # RLControlledMobility reads the action as metres per sync interval
            # and maxSpeed is only a CEILING.  OMNeTEnv._step does
            # `dx * STEP_M` before sending; this probe talks to the simulator
            # directly and therefore has to do the same, otherwise every drone
            # moves 1 m/step, stays in the staging area, and the neighbour
            # count reads as a full mesh regardless of the radio range.
            if policy == "hover":
                acts = [[0.0, 0.0] for _ in range(n)]
            elif policy == "spread":
                # Deterministic dispersal: push each drone away from the swarm
                # centroid at full step length.  Approximates the placement a
                # trained or greedy policy converges to, which is the geometry
                # the regime question is actually about -- the staging spawn is
                # not.
                P = np.asarray(obs.get("positions", []), dtype=float)
                c = P.mean(axis=0)
                v = P - c
                nrm = np.linalg.norm(v, axis=1, keepdims=True)
                v = np.where(nrm > 1e-6, v / np.maximum(nrm, 1e-6),
                             rng.normal(size=v.shape))
                acts = (v * T.OM2G_STEP_M).tolist()
            else:
                acts = (rng.uniform(-1, 1, size=(n, 2)) * T.OM2G_STEP_M).tolist()
            sock.send_string(json.dumps(
                {"type": "action", "actions": acts, "reward": 0.0}))
    finally:
        try:
            proc.terminate(); proc.wait(timeout=10)
        except Exception:
            proc.kill()
        fh.close()
        sock.close(); ctx.term()

    rec["_log"] = logpath
    rec["_payload_bytes"] = payload_bytes
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="compressed", choices=sorted(ARM_CONFIG))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--policy", default="random", choices=("hover", "random", "spread"))
    ap.add_argument("--warmup", type=int, default=5,
                    help="steps to discard while beacons/associations settle")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    print(f"OM2G regime probe   arm={args.arm}  policy={args.policy}  "
          f"{args.episodes} x {args.steps} steps", flush=True)
    print(f"  analytic prediction: payload "
          f"{T.om2g_message_bytes(args.arm if args.arm != 'nocontrol' else 'compressed')} B, "
          f"{T.om2g_frames_per_message(args.arm) if args.arm in ('compressed','uncompressed') else 0} frame(s), "
          f"control share "
          f"{T.om2g_control_plane_share(args.arm)*100:.2f}%"
          if args.arm in ("compressed", "uncompressed") else
          "  analytic prediction: control plane silent", flush=True)

    agg = {}
    for ep in range(args.episodes):
        t0 = time.time()
        r = run_episode(args.arm, seed=1000 + ep, steps=args.steps,
                        policy=args.policy, port=0)
        w = args.warmup
        for k, v in r.items():
            if k.startswith("_"):
                continue
            arr = np.asarray(v[w:], dtype=float)
            agg.setdefault(k, []).append(arr)
        print(f"  ep{ep} seed={1000+ep}  {time.time()-t0:.1f}s  "
              f"payload={r['_payload_bytes']} B  log={r['_log']}", flush=True)

    def stat(k):
        a = np.concatenate(agg[k]) if k in agg and agg[k] else np.array([np.nan])
        a = a[np.isfinite(a)]
        return (float(a.mean()), float(a.std())) if a.size else (float("nan"),) * 2

    nbr_m, nbr_s = stat("nbr")
    busy_m, _ = stat("busy")
    ctx_m, _ = stat("ctrl_tx")
    crx_m, _ = stat("ctrl_rx")
    thr_m, _ = stat("thr")
    btx_m, _ = stat("beacon_tx")
    brx_m, _ = stat("beacon_rx")
    age_m, _ = stat("msg_age")

    n = T.OM2G_NUM_DRONES
    expected_rx = btx_m * (n - 1) if btx_m else float("nan")
    deliv = brx_m / expected_rx if expected_rx else float("nan")

    print()
    print("─" * 68)
    print(f"  mean drone-drone neighbours      {nbr_m:6.2f} +- {nbr_s:.2f}"
          f"   (of {n-1} possible)")
    print(f"  state-message delivery ratio     {deliv*100:6.1f} %"
          f"   ({brx_m:.1f} rx / {expected_rx:.1f} expected per step)")
    print(f"  mean age of a known peer msg     {age_m:6.2f} s")
    print(f"  measured channel-busy fraction   {busy_m*100:6.2f} %")
    print(f"  control-plane offered            {ctx_m*8/1000:6.2f} kbps "
          f"({ctx_m*8/T.OM2G_BITRATE_BPS*100:.2f}% of 250 kbps, payload only)")
    print(f"  control-plane delivered          {crx_m*8/1000:6.2f} kbps")
    print(f"  data-plane throughput at drones  {thr_m*8/1000:6.2f} kbps")
    if agg.get("nbr"):
        m = min(len(a) for a in agg["nbr"])
        tr = np.mean(np.stack([a[:m] for a in agg["nbr"]]), axis=0)
        sp = (np.mean(np.stack([a[:m] for a in agg["spread"]]), axis=0)
              if agg.get("spread") else np.zeros_like(tr))
        q = max(1, len(tr) // 5)
        print("  neighbours / swarm spread over the episode:")
        for k in range(0, len(tr) - q + 1, q):
            print(f"    steps {k+1:>4}-{k+q:<4}   nbrs {tr[k:k+q].mean():4.2f}"
                  f"   spread {sp[k:k+q].mean():7.1f} m")
    print("─" * 68)

    ok_regime = 1.5 <= nbr_m <= 3.0
    print(f"  GATE 1 multi-hop regime (1.5 <= nbrs <= 3.0)   "
          f"{'PASS' if ok_regime else 'FAIL'}")
    if not ok_regime:
        print("        -> adjust TX power or arena ONCE, re-probe, then stop. "
              "Do not sweep before this passes.")

    if args.json_out:
        out = {"track": "OM2G", "arm": args.arm, "policy": args.policy,
               "episodes": args.episodes, "steps": args.steps,
               "neighbours_mean": nbr_m, "neighbours_std": nbr_s,
               "delivery_ratio": deliv, "msg_age_s": age_m,
               "channel_busy": busy_m, "ctrl_bps_offered": ctx_m * 8,
               "ctrl_bps_delivered": crx_m * 8, "data_bps": thr_m * 8,
               "gate_multihop": bool(ok_regime)}
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  wrote {args.json_out}")


if __name__ == "__main__":
    main()
