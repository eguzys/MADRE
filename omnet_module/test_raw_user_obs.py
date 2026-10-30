#!/usr/bin/env python3
"""
Synthetic-obs test for build_raw_user_tokens() / OMNeTEnv._build_obs().

No OMNeT++ subprocess required — constructs obs dicts shaped exactly like
RLBridge::buildObservationJson()'s real output (positions, user_positions,
noise_floor) and checks the per-user raw token builder against the schema
in omnet_env.py's "Per-user raw observation schema" comment block.

Checks:
  1. Shape: (MAX_N, MAX_USERS_CAP, OBS_FEATURES).
  2. Sensed sets genuinely differ across observing drones (not god-view):
     a near user/far user pair sensed oppositely by two drones placed at
     opposite ends of the pair.
  3. Unsensed users AND structural padding (u >= actual numUsers) both come
     back as an all-zero row (placeholder, present=0).
  4. id_tag (columns 6-9) is unsortable: sorting sensed rows by any tag
     column does not recover ascending user-index order.
  5. MAX_USERS_CAP overflow raises rather than silently truncating.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omnet_env import (
    MAX_N,
    MAX_USERS_CAP,
    OBS_FEATURES,
    build_raw_user_tokens,
    _compute_rssi_dbm,
)

_failures = []


def check(label, condition, detail=""):
    tag = "[PASS]" if condition else "[FAIL]"
    print(f"{tag} {label}" + (f"  ({detail})" if detail else ""), flush=True)
    if not condition:
        _failures.append(label)
    return condition


AMBIENT_NOISE_DBM = -100.0  # matches RLBridge.cc kAmbient


def make_obs(drone_positions, user_positions, noise_floor=None):
    if noise_floor is None:
        noise_floor = [AMBIENT_NOISE_DBM] * len(drone_positions)
    return {
        "positions": drone_positions,
        "user_positions": user_positions,
        "noise_floor": noise_floor,
    }


def main():
    print("=" * 60)
    print("build_raw_user_tokens() synthetic test")
    print("=" * 60)

    # ── 1 & 2: shape + genuinely different sensed sets per drone ─────────────
    print("\n[1/2] Shape + per-drone sensed-set divergence ...")

    # Drone 0 sits right next to user 0 (near, clearly sensed) and far from
    # user 1. Drone 1 is placed the other way around. With ambient noise
    # floor (-100 dBm) and SENSE_MARGIN_DB=0.0, the 0 dB SNR crossover is at
    # ~2083 m (see omnet_env.py comment) — use distances well on either side
    # of that so the test isn't sensitive to exact threshold tuning.
    near_d = 50.0     # well inside sensed range
    far_d = 4000.0    # well outside sensed range

    drone_positions = [
        [0.0, 0.0],                 # drone 0: near user 0, far from user 1
        [near_d + far_d, 0.0],      # drone 1: near user 1, far from user 0
        [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
    ]
    user_positions = [
        [near_d, 0.0],              # user 0: near drone 0
        [near_d + far_d - near_d, 0.0],  # user 1: near drone 1 (= far_d from drone 0)
    ]
    obs = make_obs(drone_positions, user_positions)

    episode_seed = 12345
    tokens = build_raw_user_tokens(obs, episode_seed)

    check("tokens.shape == (MAX_N, MAX_USERS_CAP, OBS_FEATURES)",
          tokens.shape == (MAX_N, MAX_USERS_CAP, OBS_FEATURES),
          f"got {tokens.shape}")

    d0_sees_u0 = tokens[0, 0, 11] == 1.0
    d0_sees_u1 = tokens[0, 1, 11] == 1.0
    d1_sees_u0 = tokens[1, 0, 11] == 1.0
    d1_sees_u1 = tokens[1, 1, 11] == 1.0

    check("drone 0 senses near user 0", d0_sees_u0)
    check("drone 0 does NOT sense far user 1", not d0_sees_u1)
    check("drone 1 senses near user 1", d1_sees_u1)
    check("drone 1 does NOT sense far user 0", not d1_sees_u0)

    check("sensed sets differ across observing drones (not god-view)",
          (d0_sees_u0, d0_sees_u1) != (d1_sees_u0, d1_sees_u1),
          f"drone0={(d0_sees_u0, d0_sees_u1)}  drone1={(d1_sees_u0, d1_sees_u1)}")

    # ── 3: unsensed + structural padding -> all-zero placeholder rows ───────
    print("\n[3] Unsensed/padding rows are all-zero placeholders ...")

    unsensed_row = tokens[0, 1]  # user 1, far from drone 0
    check("unsensed row is all-zero", bool(np.all(unsensed_row == 0.0)),
          f"row={unsensed_row}")

    pad_row = tokens[0, len(user_positions)]  # slot beyond actual numUsers
    check("structural-padding row is all-zero", bool(np.all(pad_row == 0.0)),
          f"row={pad_row}")

    sensed_row = tokens[0, 0]
    check("sensed row is NOT all-zero", not bool(np.all(sensed_row == 0.0)))
    check("sensed row present bit == 1.0", sensed_row[11] == 1.0)
    check("sensed row id_valid bit == 1.0", sensed_row[10] == 1.0)

    # ── 4: id_tag is unsortable ─────────────────────────────────────────────
    print("\n[4] id_tag unsortability ...")

    # All users close to drone 0 so every row is sensed, isolating id_tag.
    n_probe_users = 20
    probe_user_positions = [[10.0 + i, 0.0] for i in range(n_probe_users)]
    probe_drone_positions = [[0.0, 0.0]] + [[0.0, 0.0]] * (MAX_N - 1)
    probe_obs = make_obs(probe_drone_positions, probe_user_positions)
    probe_tokens = build_raw_user_tokens(probe_obs, episode_seed=777)

    sensed_rows = probe_tokens[0, :n_probe_users]
    check("all probe users sensed (isolates id_tag check)",
          bool(np.all(sensed_rows[:, 11] == 1.0)))

    id_tag_block = sensed_rows[:, 6:10]  # (n_probe_users, 4)
    any_monotonic = False
    for col in range(id_tag_block.shape[1]):
        order = np.argsort(id_tag_block[:, col])
        if np.array_equal(order, np.arange(n_probe_users)) or \
           np.array_equal(order, np.arange(n_probe_users)[::-1]):
            any_monotonic = True
    check("no id_tag column sorts into (or against) user-index order",
          not any_monotonic)

    # Sorting the WHOLE token set by an id_tag column must not recover the
    # original user-index order either.
    sort_col = 6
    sorted_by_tag = np.argsort(sensed_rows[:, sort_col])
    check("sorting rows by id_tag does not recover ascending index order",
          not np.array_equal(sorted_by_tag, np.arange(n_probe_users)),
          f"sorted_order={sorted_by_tag}")

    # Determinism: same (episode_seed, user_idx) -> same tag, every call
    # WITHIN an episode — this is the "stable within an episode" guarantee.
    probe_tokens_again = build_raw_user_tokens(probe_obs, episode_seed=777)
    check("id_tag stable within an episode (deterministic across repeated "
          "calls with same episode_seed)",
          bool(np.array_equal(probe_tokens[0, :, 6:10], probe_tokens_again[0, :, 6:10])))

    # Different episode -> different tag (arbitrary-per-episode, not fixed).
    probe_tokens_diff_ep = build_raw_user_tokens(probe_obs, episode_seed=778)
    check("id_tag differs across different episode_seed",
          not bool(np.array_equal(probe_tokens[0, :, 6:10], probe_tokens_diff_ep[0, :, 6:10])))

    # ── 5: MAX_USERS_CAP overflow raises rather than silently truncating ────
    print("\n[5] MAX_USERS_CAP overflow guard ...")
    overflow_user_positions = [[float(i), 0.0] for i in range(MAX_USERS_CAP + 1)]
    overflow_obs = make_obs(drone_positions, overflow_user_positions)
    raised = False
    try:
        build_raw_user_tokens(overflow_obs, episode_seed=1)
    except RuntimeError:
        raised = True
    check("RuntimeError raised when numUsers > MAX_USERS_CAP", raised)

    # ── sanity: RSSI replication is a sane deterministic dBm number ─────────
    print("\n[sanity] RSSI replication ...")
    rssi_1m = _compute_rssi_dbm(1.0)
    check("rssi at 1m is finite and > -50 dBm (near-field, sanity bound)",
          np.isfinite(rssi_1m) and rssi_1m > -50.0, f"rssi_1m={rssi_1m:.2f} dBm")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if _failures:
        print(f"FAILED ({len(_failures)} checks):")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
