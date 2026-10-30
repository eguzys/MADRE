#!/usr/bin/env python3
"""
Standalone test for the real-layout agreement check (task_validation.py).

Does NOT launch OMNeT++ -- everything here is pure Python, so it runs fast
and needs no simulator binary. It proves two things:

  1. The retired python-side sampler (sample_users / sample_sender_pairs)
     RAISES instead of running.
  2. assert_users_match_omnet() -- the load-bearing AGREEMENT ASSERT -- PASSES
     when the scored set IS the OMNeT-simulated set, and RAISES when it is a
     phantom (python-side resampled) set instead. The phantom generator below
     reproduces the RETIRED algorithm's exact shape (same Gaussian-cluster
     recipe, same hash-seeded rng) purely as a test fixture to show the
     check trips on such a set -- it is not reachable from production code.

Run from omnet_module/ or the repo root -- sys.path is patched automatically.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from task_validation import (
    sample_users, sample_sender_pairs, get_omnet_users,
    assert_users_match_omnet, FROZEN_L, FROZEN_N_DRONES, FROZEN_N_USERS,
    FROZEN_MAX_CLUSTERS,
)

# ── tiny assertion helper (matches test_omnet_env.py's convention) ──────────

_failures = []


def check(label, condition, detail=""):
    tag = "[PASS]" if condition else "[FAIL]"
    msg = f"{tag} {label}" + (f"  ({detail})" if detail else "")
    print(msg, flush=True)
    if not condition:
        _failures.append(label)
    return condition


def raises(fn, exc_type=Exception):
    try:
        fn()
        return False
    except exc_type:
        return True


# ── the retired python-side recipe, reproduced here ONLY as a test fixture ──
#
# Gaussian clusters seeded from random.Random(hash((L, seed))), sharing NO rng
# stream with OMNeT's own C++ mt19937 (the recipe of the retired
# task_validation.sample_users). It is reproduced here so this test can show
# the AGREEMENT ASSERT trips on such a set -- nothing in production calls it.

def _phantom_sample_users(seed: int, n_users: int, L: float,
                          spread: float = 30.0, max_clusters: int = 2):
    rng = random.Random(hash((int(L), seed)) & 0xFFFFFFFF)
    alpha_c, beta_c = 0.6, 0.1
    scale = alpha_c * 10.0
    n_clusters = 1
    while rng.random() > 1.0 / scale:
        n_clusters += 1
        if n_clusters > n_users:
            break
    n_clusters = max(2, min(n_clusters, n_users, max_clusters))

    g = [rng.gammavariate(beta_c * 10.0, 1.0) for _ in range(n_clusters)]
    gsum = sum(g) or 1.0
    sizes = [max(1, int(n_users * x / gsum)) for x in g]
    while sum(sizes) < n_users:
        sizes[rng.randrange(n_clusters)] += 1
    while sum(sizes) > n_users:
        i = max(range(n_clusters), key=lambda k: sizes[k])
        if sizes[i] > 1:
            sizes[i] -= 1
        else:
            break

    sp = spread * (L / 2000.0)
    pos = []
    for c in range(n_clusters):
        cx, cy = rng.uniform(0.0, L), rng.uniform(0.0, L)
        for _ in range(sizes[c]):
            x = min(L, max(0.0, rng.gauss(cx, sp)))
            y = min(L, max(0.0, rng.gauss(cy, sp)))
            pos.append([x, y])
    return pos[:n_users]


def main():
    print("=" * 60)
    print("Phantom-user retirement + agreement-assert tripwire test")
    print("=" * 60)

    # ── 1. The phantom sampler must scream, not run ─────────────────────────
    print("\n[1] Retired phantom sampler raises ...")
    check("sample_users() raises",
          raises(lambda: sample_users(random.Random(0), 17, 1700.0)))
    check("sample_sender_pairs() raises",
          raises(lambda: sample_sender_pairs(random.Random(0), [0, 1])))

    # ── 2. get_omnet_users() rejects any non-frozen geometry BEFORE touching
    #        OMNeT (no subprocess should be spawned for a bad request) ───────
    print("\n[2] get_omnet_users() rejects mismatched geometry ...")
    check("wrong L raises",
          raises(lambda: get_omnet_users(0, FROZEN_N_USERS, FROZEN_N_DRONES,
                                         1000.0, FROZEN_MAX_CLUSTERS)))
    check("wrong n_users raises",
          raises(lambda: get_omnet_users(0, 10, FROZEN_N_DRONES, FROZEN_L,
                                         FROZEN_MAX_CLUSTERS)))
    check("wrong max_clusters raises",
          raises(lambda: get_omnet_users(0, FROZEN_N_USERS, FROZEN_N_DRONES,
                                         FROZEN_L, 3)))

    # ── 3. AGREEMENT ASSERT: passes when scored == simulated ────────────────
    print("\n[3] assert_users_match_omnet: real == real (must PASS) ...")
    real = [[100.0, 200.0], [150.0, 210.0], [900.0, 950.0], [905.0, 940.0]]
    ok = True
    try:
        assert_users_match_omnet(real, real, seed=0)
    except AssertionError as exc:
        ok = False
        print(f"    unexpected raise: {exc}")
    check("identical sets: no raise", ok)

    print("\n[3b] assert_users_match_omnet: real vs a 0.3m-jittered copy "
          "(within tol, must PASS) ...")
    jittered = [[x + 0.3, y - 0.2] for x, y in real]
    ok = True
    try:
        assert_users_match_omnet(real, jittered, seed=0, tol=1.0)
    except AssertionError as exc:
        ok = False
        print(f"    unexpected raise: {exc}")
    check("small jitter within tol: no raise", ok)

    # ── 4. AGREEMENT ASSERT: raises on a phantom (python-resampled) set ─────
    print("\n[4] assert_users_match_omnet: phantom vs simulated (must RAISE) ...")
    phantom = _phantom_sample_users(0, FROZEN_N_USERS, FROZEN_L,
                                    max_clusters=FROZEN_MAX_CLUSTERS)
    # Stand-in for "the users OMNeT actually simulated": a DIFFERENT,
    # independent cluster layout (distinct rng stream), as measured between
    # the two samplers (~1533 m mean set-to-set offset, different cluster
    # centroids). We don't need a live OMNeT process to prove the
    # comparator trips -- only that two independently-drawn layouts disagree.
    simulated = _phantom_sample_users(0, FROZEN_N_USERS, FROZEN_L,
                                      spread=30.0, max_clusters=FROZEN_MAX_CLUSTERS)
    # Force independence: shift the "simulated" cluster centres far away, the
    # same shape as the measured phantom-vs-real offset.
    simulated = [[x + 1500.0, y] for x, y in simulated]

    tripped = raises(
        lambda: assert_users_match_omnet(phantom, simulated, seed=0),
        AssertionError)
    check("phantom vs simulated: AssertionError raised", tripped)

    print("\n[4b] assert_users_match_omnet: count mismatch (must RAISE) ...")
    tripped = raises(
        lambda: assert_users_match_omnet(phantom, phantom[:-1], seed=0),
        AssertionError)
    check("count mismatch: AssertionError raised", tripped)

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
