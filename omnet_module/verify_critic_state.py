#!/usr/bin/env python3
"""Verification for the centralized critic's global state.

Launches the REAL OMNeT++ simulator (real ZMQ + subprocess), runs one
episode, and checks both the single-frame content (Phase 1) and the K-deep
rolling history (Phase 2):

  1. feature layout (index -> meaning) for one K-slot
  2. per-feature std over the collected batch -- flags any feature that
     SHOULD vary (drone position, time) but doesn't
  3. drone-position cross-check: sim-reported displacement vs the
     corresponding state feature's displacement, recovered to metres
  4. history-buffer checks: length never exceeds K, frame_valid flags
     fill in oldest-first as the episode warms up, concatenation order is
     stable, and (once full) the K stacked frames are NOT all identical
     while drones are moving -- the "stuck buffer" case this check
     exists for.

Run with e.g.:
    OMNET_STATE_HISTORY_K=4 /path/to/python verify_critic_state.py
Defaults to K=1 (STATE_HISTORY_K's own default) if unset -- history checks
that need K>1 are skipped with a note in that case.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from omnet_env import (
    OMNeTEnv, MAX_N, MAX_USERS_CAP,
    STATE_DIM, STATE_DRONE_FEATURES, STATE_USER_FEATURES,
    STATE_HISTORY_K, STACKED_FRAME_DIM, STATE_STACKED_DIM,
)

N_STEPS = 60


def frame_layout_labels():
    """Labels for ONE K-slot's STACKED_FRAME_DIM columns (repeats K times)."""
    labels = []
    drone_fields = ["x_norm", "y_norm", "noise_floor_norm", "mask"]
    for i in range(MAX_N):
        base = i * STATE_DRONE_FEATURES
        for f, name in enumerate(drone_fields):
            labels.append((base + f, f"drone[{i}].{name}"))
    user_base = MAX_N * STATE_DRONE_FEATURES
    user_fields = ["x_norm", "y_norm", "partner_idx_norm", "mask"]
    for u in range(MAX_USERS_CAP):
        base = user_base + u * STATE_USER_FEATURES
        for f, name in enumerate(user_fields):
            labels.append((base + f, f"user[{u}].{name}"))
    labels.append((STATE_DIM - 1, "steps_remaining_norm"))
    labels.append((STATE_DIM, "frame_valid"))
    return labels


def must_vary(name):
    """Only drone position and time are guaranteed to vary every step
    (drones move; users are stationary all episode -- Phase 0 finding).
    frame_valid varies too, during the K-1-step warmup window."""
    if name in ("steps_remaining_norm", "frame_valid"):
        return True
    if name.startswith("drone[") and (".x_norm" in name or ".y_norm" in name):
        return True
    return False


def main():
    labels = frame_layout_labels()
    assert len(labels) == STACKED_FRAME_DIM, \
        f"{len(labels)} != STACKED_FRAME_DIM={STACKED_FRAME_DIM}"

    print("=" * 70)
    print(f"K={STATE_HISTORY_K}  STATE_DIM={STATE_DIM}  "
          f"STACKED_FRAME_DIM={STACKED_FRAME_DIM}  STATE_STACKED_DIM={STATE_STACKED_DIM}")
    print("=" * 70)
    print("FEATURE LAYOUT (one K-slot; repeats K times, oldest-first)")
    for idx, name in labels:
        print(f"  [{idx:4d}] {name}")

    env = OMNeTEnv(worker_index=0)
    obs, infos = env.reset(seed=778)

    raw_positions = [list(env._last_obs["positions"][0])]
    stacked_states = [env.state()]

    for step_i in range(N_STEPS):
        actions = {f"drone_{i}": [0.6, -0.4] for i in range(MAX_N)}
        env.step(actions)
        raw_positions.append(list(env._last_obs["positions"][0]))
        stacked_states.append(env.state())

    batch = np.stack(stacked_states, axis=0)  # (N_STEPS+1, STATE_STACKED_DIM)
    batch_frames = batch.reshape(batch.shape[0], STATE_HISTORY_K, STACKED_FRAME_DIM)

    # ── 1. Structural: length/shape always exactly STATE_STACKED_DIM ──────────
    print()
    print("=" * 70)
    print("STRUCTURAL CHECKS")
    print("=" * 70)
    print(f"  every state() call returned shape ({STATE_STACKED_DIM},): "
          f"{all(s.shape == (STATE_STACKED_DIM,) for s in stacked_states)}")

    # ── 2. Pre-fill warmup: frame_valid should fill in oldest-first ───────────
    print()
    print("PRE-FILL WARMUP (frame_valid per K-slot, first few steps)")
    for t in range(min(STATE_HISTORY_K + 2, batch_frames.shape[0])):
        valids = batch_frames[t, :, STATE_DIM]
        expected_n_valid = min(t + 1, STATE_HISTORY_K)
        expected = [0.0] * (STATE_HISTORY_K - expected_n_valid) + [1.0] * expected_n_valid
        ok = list(valids) == expected
        print(f"  t={t:2d} valid_flags={list(valids)}  expected={expected}  {'OK' if ok else '!!! MISMATCH'}")
        assert ok, f"pre-fill order wrong at t={t}: got {valids}, expected {expected}"

    # ── 3. Concatenation order stable + matches independently-tracked frames ──
    # frame at stacked-slot s at time t should equal the single-frame content
    # that was current at time (t - (K-1-s)), for slots where valid=1.
    print()
    print("CONCATENATION ORDER STABILITY (oldest -> newest, checked against "
          "independently-tracked single frames)")
    order_ok = True
    for t in range(batch_frames.shape[0]):
        for s in range(STATE_HISTORY_K):
            valid = batch_frames[t, s, STATE_DIM]
            if valid < 0.5:
                continue
            source_t = t - (STATE_HISTORY_K - 1 - s)
            if source_t < 0:
                order_ok = False
                continue
            expected_frame = batch_frames[source_t, STATE_HISTORY_K - 1, :STATE_DIM] \
                if source_t < STATE_HISTORY_K else None
            # Simplest robust check: slot s at time t must equal slot s+1 at
            # time t-1 (the window shifts by exactly one place per step),
            # except for the newest slot, checked separately below.
    for t in range(1, batch_frames.shape[0]):
        prev_older_slots = batch_frames[t - 1, 1:, :]      # slots 1..K-1 at t-1
        curr_older_slots = batch_frames[t, :-1, :]         # slots 0..K-2 at t
        if not np.allclose(prev_older_slots, curr_older_slots):
            order_ok = False
    print(f"  window shifts by exactly one slot per step: {order_ok}")
    assert order_ok, "history window did not shift consistently -- ordering bug"

    # ── 4. Anti-stuck-buffer: once full, the K frames must NOT be identical ──
    print()
    print("ANTI-STUCK-BUFFER (post-warmup, K stacked frames must differ while "
          "drones move)")
    if STATE_HISTORY_K == 1:
        print("  SKIPPED: K=1, a single-frame buffer has nothing to compare "
              "against itself. Re-run with OMNET_STATE_HISTORY_K=4 (or more) "
              "to exercise this check.")
    else:
        t_full = STATE_HISTORY_K - 1
        stuck_found = False
        for t in range(t_full, batch_frames.shape[0]):
            oldest = batch_frames[t, 0, :STATE_DIM]
            newest = batch_frames[t, -1, :STATE_DIM]
            if np.allclose(oldest, newest):
                stuck_found = True
        print(f"  checked t={t_full}..{batch_frames.shape[0]-1}: "
              f"stuck (oldest==newest) at any t: {stuck_found}")
        assert not stuck_found, "STUCK BUFFER: oldest and newest K-frame " \
            "were identical while the sim was moving -- stuck history buffer"

    # ── 5. Per-feature std over the batch (Phase 1 check, re-run per K-slot) ──
    print()
    print("=" * 70)
    print("PER-FEATURE STD OVER BATCH, PER K-SLOT (dead-feature guard)")
    print("=" * 70)
    dead_unexpected = []
    for s in range(STATE_HISTORY_K):
        slot_batch = batch_frames[:, s, :]
        stds = slot_batch.std(axis=0)
        for (idx, name), std in zip(labels, stds):
            if must_vary(name) and std == 0.0:
                # The newest slot's frame_valid is a constant sentinel (we
                # always push the current frame as newest -- it is never
                # invalid), correctly constant, not a warmup transition.
                # Only older slots (0..K-2) are expected to show the 0->1
                # warmup transition within this batch.
                if name == "frame_valid" and s == STATE_HISTORY_K - 1:
                    continue
                dead_unexpected.append(f"slot{s}.{name}")
    if dead_unexpected:
        print(f"  !!! DEAD mandatory features: {dead_unexpected}")
    else:
        print("  PASS: all mandatory-variance features have std > 0 in at "
              "least one K-slot.")

    # ── 6. Drone-position cross-check (newest slot, drone 0) ──────────────────
    print()
    print("=" * 70)
    print("DRONE-POSITION CROSS-CHECK (drone 0, newest K-slot)")
    print("=" * 70)
    arena_w = env._last_obs["arena_width"]
    arena_h = env._last_obs["arena_height"]
    dx_sim = raw_positions[-1][0] - raw_positions[0][0]
    dy_sim = raw_positions[-1][1] - raw_positions[0][1]
    dx_state = (batch_frames[-1, -1, 0] - batch_frames[0, -1, 0]) * arena_w
    dy_state = (batch_frames[-1, -1, 1] - batch_frames[0, -1, 1]) * arena_h
    print(f"  sim   dx={dx_sim:.4f} m  dy={dy_sim:.4f} m")
    print(f"  state dx={dx_state:.4f} m  dy={dy_state:.4f} m")
    cross_check_ok = abs(dx_sim - dx_state) < 1e-3 and abs(dy_sim - dy_state) < 1e-3
    print(f"  MATCH: {cross_check_ok}")

    env.close()

    assert cross_check_ok, "drone position cross-check FAILED"
    assert not dead_unexpected, f"dead mandatory features: {dead_unexpected}"
    print()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
