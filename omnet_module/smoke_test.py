#!/usr/bin/env python3
"""
Smoke test for RLBridge ↔ Python integration.

Binds a ZMQ REP socket on tcp://127.0.0.1:5555. For each obs message from
OMNeT++, computes a max-flow reward from drone + user positions and Bonn
demand matrix, prints a summary, then sends back a random action per drone.

Run this FIRST, then start OMNeT++.

Stops cleanly when OMNeT++ sends done=true.
"""
import argparse
import json
import sys
import zmq

try:
    from max_flow_reward import compute_position_oracle_reward as compute_reward
    _have_reward = True
except ImportError:
    _have_reward = False
    print("[py] WARNING: max_flow_reward not found — reward will be 0.0", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--num-drones", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://127.0.0.1:{args.port}")
    print(f"[py] bound REP socket on tcp://127.0.0.1:{args.port}", flush=True)
    print(f"[py] waiting for first obs from OMNeT++...", flush=True)

    step = 0
    active_broadcasts_log = []        # for reproducibility check
    prev_global_speaker   = None      # for multicast-delivery check

    while True:
        msg  = sock.recv()
        obs  = json.loads(msg.decode("utf-8"))
        done = obs.get("done", False)

        # ── Max-flow reward ────────────────────────────────────────────────────
        reward = 0.0
        if _have_reward and not done:
            try:
                reward = compute_reward(obs)["reward"]
            except Exception as e:
                print(f"[py] reward error: {e}", flush=True)

        # ── Bonn talk-group smoke checks (Aschenbruck IPCCC 2007 + LCN 2006) ───
        user_area         = obs.get("user_area", [])
        active_broadcasts = obs.get("active_broadcasts", [])
        talk_group_heads  = obs.get("talk_group_heads", [])
        goodput_now       = obs.get("user_goodput", [])
        N_users_now       = len(obs.get("user_positions", []))

        # Record for reproducibility.
        active_broadcasts_log.append(active_broadcasts)

        if not done:
            # 1. ROLE FIELDS GONE
            assert "user_roles" not in obs, \
                f"BONN CHECK 1 FAIL: user_roles still present in obs at step {step}"

            # 2. NEW FIELDS PRESENT AND SHAPED CORRECTLY
            assert "user_area" in obs, \
                f"BONN CHECK 2 FAIL: user_area missing at step {step}"
            assert "active_broadcasts" in obs, \
                f"BONN CHECK 2 FAIL: active_broadcasts missing at step {step}"
            assert "talk_group_heads" in obs, \
                f"BONN CHECK 2 FAIL: talk_group_heads missing at step {step}"
            if N_users_now > 0:
                assert len(user_area) == N_users_now, \
                    f"BONN CHECK 2 FAIL: user_area len={len(user_area)} != N_users={N_users_now}"
                assert all(a in (0, 1, 2, 3) for a in user_area), \
                    f"BONN CHECK 2 FAIL: user_area contains value outside {{0,1,2,3}}: {user_area}"
            assert isinstance(active_broadcasts, list), \
                f"BONN CHECK 2 FAIL: active_broadcasts not a list"
            assert len(talk_group_heads) == 5, \
                f"BONN CHECK 2 FAIL: talk_group_heads length={len(talk_group_heads)}, expected 5"

            # 3. MULTICAST DELIVERY: if global group (tg=4) was broadcasting last tick,
            #    verify >= N_users-1 users received packets this tick.
            if prev_global_speaker is not None and N_users_now > 1:
                receivers = sum(1 for i, g in enumerate(goodput_now) if g > 0 and i != prev_global_speaker)
                assert receivers >= N_users_now - 1, (
                    f"BONN CHECK 3 FAIL: global broadcast at step {step-1} "
                    f"(speaker={prev_global_speaker}): only {receivers}/{N_users_now-1} "
                    f"non-speaker users received packets at step {step}"
                )

        # Update global-speaker state for next tick.
        prev_global_speaker = None
        for ab in active_broadcasts:
            if isinstance(ab, dict) and ab.get("talk_group_id") == 4:
                prev_global_speaker = ab.get("speaker_user_id")
                break

        # 5. NO OLD CODE (checked once at step 0 against C++ source files only).
        if step == 0 and not done:
            import glob, subprocess as _sp
            cpp_files = (
                glob.glob('/home/emilis/Documents/thesis/code/omnet_smoke/**/*.cc', recursive=True)
                + glob.glob('/home/emilis/Documents/thesis/code/omnet_smoke/**/*.h', recursive=True)
                + glob.glob('/home/emilis/Documents/thesis/code/omnet_smoke/**/*.ned', recursive=True)
            )
            for pat in ('commander', 'section_leader', 'field_responder', 'user_role'):
                for fpath in cpp_files:
                    r = _sp.run(['grep', '-l', pat, fpath], capture_output=True)
                    assert r.returncode != 0, \
                        f"BONN CHECK 5 FAIL: '{pat}' found in {fpath}"
            print("[py] BONN CHECK 5 (NO OLD CODE): PASS", flush=True)

            # 6. CITATION SANITY: print expected constants so a human can verify.
            expected = {
                "headInitiatorProb": 0.46, "meshProb": 0.136,
                "transHeadToHead": 0.142, "transMemberToMember": 0.190,
                "callContinuationProb": 0.7,
                "callCHT_meanlog": 0.4438584, "callCHT_sdlog": 0.7766512,
                "callIdle_shape": 1.530391, "callIdle_rate": 1.606532,
                "convIdle_meanlog": 1.390844, "convIdle_sdlog": 1.628825,
                "convIdle_shift": 3.0, "melpePayloadBytes": 21,
                "melpeInterval_ms": 67.5,
            }
            print(f"[py] BONN CHECK 6 (CITATION SANITY): expected constants = {expected}", flush=True)

        # ── Log summary ────────────────────────────────────────────────────────
        positions   = obs.get("positions", [])
        user_pos    = obs.get("user_positions", [])
        if step == 0:
            print(f"[POS] num_incidents-tell + first 8 user positions:")
            for _i,_p in enumerate(user_pos[:8]):
                print(f"[POS]   user[{_i}] = ({_p[0]:.0f}, {_p[1]:.0f})")
        user_roles  = obs.get("user_roles", [])
        ccs_ids     = obs.get("ccs_incident_ids", [])
        throughput  = obs.get("throughput", [])
        drone_fwd   = obs.get("drone_fwd", [])
        drone_up    = obs.get("drone_up", [])
        goodput     = obs.get("user_goodput", [])
        user_up     = obs.get("user_up", [])
        assoc_cts   = obs.get("drone_user_counts", [])
        num_inc     = obs.get("num_incidents", 0)

        thr_str  = ", ".join(f"{t:.0f}" for t in throughput) if throughput else "[]"
        fwd_str  = ", ".join(f"{t:.0f}" for t in drone_fwd) if drone_fwd else "[]"
        up_str   = ", ".join(f"{t:.0f}" for t in drone_up) if drone_up else "[]"
        good_str = (", ".join(f"{g:.0f}" for g in goodput[:4])
                    + ("..." if len(goodput) > 4 else "")) if goodput else "[]"
        assoc_str = str(assoc_cts) if assoc_cts else "[]"

        print(
            f"[py] step={obs.get('step'):3d} done={done} "
            f"ndrones={len(positions)} nusers={len(user_pos)} "
            f"num_incidents={num_inc} "
            f"reward={reward:.3f} "
            f"drone_thr=[{thr_str}] B/s "
            f"drone_fwd=[{fwd_str}] B/s "
            f"drone_up=[{up_str}] B/s "
            f"user_goodput(0..3)=[{good_str}] B/s "
            f"assoc_counts={assoc_str}",
            flush=True,
        )

        # On first real step: print episode layout (incidents, CCS counts).
        if obs.get("step") == 0 and not done and user_roles:
            role_counts = {}
            ccs_per_inc: dict[int, int] = {}
            for r, cid in zip(user_roles, ccs_ids):
                role_counts[r] = role_counts.get(r, 0) + 1
                if r == "ccs" and cid >= 0:
                    ccs_per_inc[cid] = ccs_per_inc.get(cid, 0) + 1
            print(f"[py]   Layout: {num_inc} incident(s), roles={role_counts}", flush=True)
            for inc, cnt in sorted(ccs_per_inc.items()):
                print(f"[py]   CCS zone {inc}: {cnt} staff", flush=True)
            ccs_centers = obs.get("ccs_zone_centers", [])
            print(f"[py]   CCS zone centers: {ccs_centers}", flush=True)

        if done:
            try:
                sock.send_string(json.dumps({"type": "ack"}))
            except Exception:
                pass
            print(f"[py] received done, exiting after step {step}", flush=True)
            break

        # ── Action: random velocity for each drone ─────────────────────────────
        actions = [[random.uniform(-1, 1), random.uniform(-1, 1)]
                   for _ in range(args.num_drones)]
        reply = {"type": "action", "actions": actions, "reward": reward}
        sock.send_string(json.dumps(reply))
        step += 1

    # 4. REPRODUCIBILITY: persist active_broadcasts log; compare on second run with same seed.
    import os, pickle as _pkl
    if args.seed != 0:
        ab_log_path = f"/tmp/smoke_ab_log_seed{args.seed}.pkl"
        if os.path.exists(ab_log_path):
            with open(ab_log_path, 'rb') as f:
                prev_log = _pkl.load(f)
            assert prev_log == active_broadcasts_log, (
                f"BONN CHECK 4 FAIL (REPRODUCIBILITY): seed={args.seed} "
                f"active_broadcasts differ between runs\n"
                f"  prev[0]={prev_log[0] if prev_log else '[]'}\n"
                f"  curr[0]={active_broadcasts_log[0] if active_broadcasts_log else '[]'}"
            )
            os.remove(ab_log_path)
            print(f"[py] BONN CHECK 4 (REPRODUCIBILITY): seed={args.seed} PASS — runs identical", flush=True)
        else:
            with open(ab_log_path, 'wb') as f:
                _pkl.dump(active_broadcasts_log, f)
            print(f"[py] BONN CHECK 4 (REPRODUCIBILITY): seed={args.seed} — first run saved, "
                  "run again with same seed to compare", flush=True)

    sock.close()
    ctx.term()


if __name__ == "__main__":
    main()
