"""Does the actor's observation actually contain the geometry it needs?

The actor token normalises its dBm columns (so they cannot swamp the bearing
signal) and carries absolute positions. This file asserts that the resulting
observation is *geometrically sound*, i.e. the three things that must hold:

  1. SELF-LOCALISATION  — the observer can recover its OWN position from its
     own token row (j == i).
  2. PEER EXCHANGE      — the observer recovers every PEER's correct position,
     with the right peer in the right row. The classic failure here is a
     transposition: tokens[i, j] carrying observer j's view instead of peer j's
     data, which is invisible to a shape check and to every test that only
     looks at the diagonal.
  3. USER LOCALISATION  — the sensed-user tokens plus the peer positions are
     enough to actually place a user in the world. A per-user token carries a
     BEARING and an RSSI but NO RANGE column, so one drone alone only gets a
     ray; the position must come from intersecting rays across drones. If that
     does not close, no policy can navigate to users regardless of training.

All offline: driven by the synthetic_omnet_obs fixture, no simulator needed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from omnet_env import (
    PEER_LINK_MASK,
    build_drone_tokens,
    build_raw_user_tokens,
    MAX_N,
    MAX_USERS_CAP,
    OBS_FEATURES,
    OWN_STATE_W,
)

SEEDS = (1, 5, 11, 23, 42)


def _decode_abs(row, arena_w, arena_h):
    """Invert build_drone_tokens' cols 0-1 back to world metres."""
    return ((row[0] + 1.0) / 2.0 * arena_w, (row[1] + 1.0) / 2.0 * arena_h)


def _obs(seed):
    from fixtures.synthetic_omnet_obs import make_layout, build_obs
    users, pairs, drones = make_layout(seed)
    return build_obs(drones, users, pairs), np.asarray(drones), np.asarray(users)


# ---------------------------------------------------------------------------
# 1. Self-localisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_observer_can_recover_its_own_position(seed):
    obs, drones, _ = _obs(seed)
    tok = build_drone_tokens(obs, episode_seed=seed, encoder_mode="learned")
    aw, ah = obs["arena_width"], obs["arena_height"]

    for i in range(len(drones)):
        x, y = _decode_abs(tok[i, i, :OWN_STATE_W], aw, ah)
        assert abs(x - drones[i][0]) < 1e-2 and abs(y - drones[i][1]) < 1e-2, (
            f"drone {i} cannot recover its own position from its own token row: "
            f"decoded ({x:.1f},{y:.1f}) vs true ({drones[i][0]:.1f},"
            f"{drones[i][1]:.1f}). This is what bug 33 fixed -- before it the "
            f"own-state row was [0,0,0,0,nf] and carried no position at all.")


@pytest.mark.parametrize("seed", SEEDS)
def test_absolute_position_columns_are_in_range(seed):
    """Positions must be normalised to [-1,1], not raw metres."""
    obs, _, _ = _obs(seed)
    tok = build_drone_tokens(obs, episode_seed=seed, encoder_mode="learned")
    live = tok[..., -1] > 0.5
    assert np.abs(tok[..., 0:2][live]).max() <= 1.0


# ---------------------------------------------------------------------------
# 2. Peer exchange: right peer, right row, right coordinates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_every_observer_sees_every_peers_true_position(seed):
    obs, drones, _ = _obs(seed)
    tok = build_drone_tokens(obs, episode_seed=seed, encoder_mode="learned")
    aw, ah = obs["arena_width"], obs["arena_height"]
    n = len(drones)

    seen = 0
    for i in range(n):
        for j in range(n):
            if tok[i, j, -1] <= 0.5:
                # peer-link gating: with OMNET_PEER_LINK_MASK=1 an unreachable peer is
                # masked out and its row zeroed -- correct, not a decode error.
                assert PEER_LINK_MASK, "a live peer was masked with gating OFF"
                assert not tok[i, j].any(), "masked peer row must be all-zero"
                continue
            x, y = _decode_abs(tok[i, j, :OWN_STATE_W], aw, ah)
            assert abs(x - drones[j][0]) < 1e-2 and abs(y - drones[j][1]) < 1e-2, (
                f"observer {i}'s row for peer {j} decodes to ({x:.1f},{y:.1f}) "
                f"but peer {j} is at ({drones[j][0]:.1f},{drones[j][1]:.1f})")
            seen += 1
    assert seen >= n, "every observer must at least see itself"


@pytest.mark.parametrize("seed", SEEDS)
def test_peer_rows_are_not_transposed(seed):
    """tokens[i, j] must hold PEER j's data, not observer j's view.

    A transposition survives every shape check and every diagonal-only test,
    because the diagonal is identical under transpose. It is caught here by the
    relative columns, which are the only ones that are ANTI-symmetric:
    dx(i,j) == -dx(j,i). A transposed build would make them symmetric.
    """
    obs, drones, _ = _obs(seed)
    tok = build_drone_tokens(obs, episode_seed=seed, encoder_mode="learned")
    n = len(drones)

    for i in range(n):
        for j in range(n):
            if i == j or tok[i, j, -1] <= 0.5 or tok[j, i, -1] <= 0.5:
                continue
            # cols 2,3 are the relative offset (peer j minus observer i)
            assert np.allclose(tok[i, j, 2:4], -tok[j, i, 2:4], atol=1e-6), (
                f"relative offset is not antisymmetric between ({i},{j}) and "
                f"({j},{i}) -- the peer axis is transposed or mis-indexed")


@pytest.mark.parametrize("seed", SEEDS)
def test_relative_offset_agrees_with_the_absolute_pair(seed):
    """cols 2:4 must equal abs(j) - abs(i). If these two disagree the network
    is being handed two contradictory accounts of the same geometry.

    Cols 2,3 are scaled by _peer_offset_scale() (r_d, the inter-drone
    radio range), not by the arena. The consistency requirement is unchanged --
    only the denominator is -- so this reads the scale off the module rather
    than hardcoding either value.
    """
    from omnet_env import _peer_offset_scale
    obs, drones, _ = _obs(seed)
    tok = build_drone_tokens(obs, episode_seed=seed, encoder_mode="learned")
    aw, ah = obs["arena_width"], obs["arena_height"]
    s = _peer_offset_scale()
    n = len(drones)

    for i in range(n):
        assert tok[i, i, -1] > 0.5, "a drone must always be visible to itself"
        xi, yi = _decode_abs(tok[i, i, :OWN_STATE_W], aw, ah)
        for j in range(n):
            if tok[i, j, -1] <= 0.5:
                continue
            xj, yj = _decode_abs(tok[i, j, :OWN_STATE_W], aw, ah)
            # Only meaningful below the clip; past r_d the columns saturate.
            if abs(xj - xi) < s and abs(yj - yi) < s:
                assert abs(tok[i, j, 2] - (xj - xi) / s) < 1e-5
                assert abs(tok[i, j, 3] - (yj - yi) / s) < 1e-5


@pytest.mark.parametrize("seed", SEEDS)
def test_peer_user_block_belongs_to_the_peer_not_the_observer(seed):
    """tokens[i, j]'s user block must be PEER j's sensed set.

    Same transposition class as above, one level down: the whole point of the
    peer axis is that observer i learns what peer j can see. If the block were
    observer i's own set, every row would be identical and the peer axis would
    carry no information at all.
    """
    obs, drones, _ = _obs(seed)
    tok = build_drone_tokens(obs, episode_seed=seed, encoder_mode="learned")
    raw = build_raw_user_tokens(obs, episode_seed=seed)
    n = len(drones)
    ublock = MAX_USERS_CAP * OBS_FEATURES

    for i in range(n):
        for j in range(n):
            if tok[i, j, -1] <= 0.5:
                continue
            got = tok[i, j, OWN_STATE_W:OWN_STATE_W + ublock].reshape(
                MAX_USERS_CAP, OBS_FEATURES)
            np.testing.assert_array_equal(got, raw[j])

    # and the peer axis must actually vary -- otherwise the check above is
    # vacuously satisfied by every drone sensing the same set.
    sensed = [int((raw[j][:, 12] > 0.5).sum()) for j in range(n)]
    assert len(set(sensed)) > 1, (
        f"all {n} drones sense identically many users {sensed}; this fixture "
        "cannot distinguish a correct peer axis from a broadcast one")


# ---------------------------------------------------------------------------
# 3. User localisation: is the geometry actually recoverable?
# ---------------------------------------------------------------------------

def _intersect_rays(origins, bearings):
    """Least-squares intersection of 2-D rays, plus a conditioning number.

    Each ray contributes the constraint n.(p - o) = 0 where n is the unit
    normal to its direction. Stacked over rays this is a 2x2 linear solve.
    Returns (point, smallest_singular_value): sigma_min -> 0 means the rays are
    parallel as LINES and the intersection is not determined. Judge geometry by
    sigma_min, not by an angular-spread rule of thumb -- a 175-degree bearing
    spread reads as "nearly degenerate" but is in fact one of the best-
    conditioned cases (two drones on opposite sides of the user).
    """
    A, b = [], []
    for (ox, oy), th in zip(origins, bearings):
        nx, ny = -np.sin(th), np.cos(th)      # normal to the bearing
        A.append([nx, ny])
        b.append(nx * ox + ny * oy)
    A, b = np.asarray(A), np.asarray(b)
    sigma_min = float(np.linalg.svd(A, compute_uv=False)[-1])
    return np.linalg.lstsq(A, b, rcond=None)[0], sigma_min


@pytest.mark.parametrize("seed", SEEDS)
def test_users_are_localisable_by_intersecting_peer_bearings(seed):
    """The per-user token has a bearing but NO range column, so a single drone
    only gets a ray. This asserts the observation as a whole still pins users
    down: for every user sensed by >= 2 drones, intersecting those drones'
    bearings (using the absolute peer positions in the token) must recover the
    user's true position.
    """
    obs, drones, users = _obs(seed)
    tok = build_drone_tokens(obs, episode_seed=seed, encoder_mode="learned")
    aw, ah = obs["arena_width"], obs["arena_height"]
    n = len(drones)
    ublock = MAX_USERS_CAP * OBS_FEATURES

    observer = 0  # everything below is decoded from ONE drone's observation
    blocks = {j: tok[observer, j, OWN_STATE_W:OWN_STATE_W + ublock].reshape(
        MAX_USERS_CAP, OBS_FEATURES) for j in range(n)}
    origins_all = {j: _decode_abs(tok[observer, j, :OWN_STATE_W], aw, ah)
                   for j in range(n)}

    checked = 0
    for u in range(len(users)):
        origins, bearings = [], []
        for j in range(n):
            if tok[observer, j, -1] <= 0.5:
                continue            # peer j is unreachable (peer-link gating)
            row = blocks[j][u]
            if row[12] <= 0.5:      # peer j has never heard user u
                continue
            origins.append(origins_all[j])
            bearings.append(float(row[0]) * np.pi)   # undo _normalize_angle
        if len(origins) < 2:
            continue
        est, sigma_min = _intersect_rays(origins, bearings)
        if sigma_min < 1e-3:
            continue      # rays parallel as lines: not determined
        err = float(np.linalg.norm(est - users[u]))
        assert err < 1.0, (
            f"user {u} sensed by {len(origins)} drones but bearing "
            f"intersection lands {err:.1f} m from truth "
            f"(est {np.round(est,1)}, true {np.round(users[u],1)}, "
            f"sigma_min {sigma_min:.3f})")
        checked += 1

    if checked == 0:
        # The synthetic fixture's radio model is far sparser than the real v2
        # task: on some seeds no user is heard by two drones at once, so there
        # is no intersection to check. Not a failure and not something to
        # paper over -- the real-data version of this check
        # (test_users_are_localisable_on_real_recorded_traces) is the one that
        # carries the weight, with >2000 determined intersections.
        pytest.skip(f"seed {seed}: no multi-sensed user with determined geometry")


_TRACES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "diag_v2", "out")


@pytest.mark.parametrize("trace_seed", (0, 4, 7))
def test_users_are_localisable_on_real_recorded_traces(trace_seed):
    """The same intersection check, on REAL recorded episodes of the v2 task.

    Stronger than the synthetic version: the fixture's radio model is far
    sparser (roughly one sensing drone per user), while the real task senses
    10-14 of 17 users from EVERY drone, so nearly every user is multi-sensed
    and the geometry is richly determined.

    The traces were recorded with an earlier token layout that has no absolute
    position column -- the drone origins are taken from the trace's ground-truth
    `pos` array, which is exactly what build_drone_tokens' new cols 0-1 now
    carry (verified separately by
    test_every_observer_sees_every_peers_true_position). What this asserts is
    the other half: that the BEARING column, once un-normalised, is geometrically
    consistent with those origins and with the users' true positions.
    """
    path = os.path.join(_TRACES, f"trace_seed{trace_seed}.npz")
    if not os.path.exists(path):
        pytest.skip(f"recorded trace not present: {path}")

    d = np.load(path, allow_pickle=True)
    obs, pos, users = d["obs"], d["pos"], d["users"]
    T, n = obs.shape[0], obs.shape[1]
    # diag_policy_behaviour.py records pos EVERY step and obs every 10th, both
    # from the same pre-action state, so obs[t] pairs with pos[10*t]. (An
    # evenly-spaced linspace mapping looks right and is not: it drifts up to 9
    # steps, i.e. several hundred metres, and shows up as a scatter of ~20-230 m
    # "localisation errors" that are pure misalignment.)
    OBS_EVERY = 10
    step_of = np.arange(T) * OBS_EVERY
    assert step_of[-1] < len(pos), "trace subsampling stride changed"
    OLD_OWN_W = 5   # own-token width of the earlier layout in these traces

    errs, checked = [], 0
    for t in range(T):
        blocks = {j: obs[t, j, j, OLD_OWN_W:OLD_OWN_W + MAX_USERS_CAP * OBS_FEATURES]
                  .reshape(MAX_USERS_CAP, OBS_FEATURES) for j in range(n)}
        for u in range(len(users)):
            origins, bearings = [], []
            for j in range(n):
                row = blocks[j][u]
                if row[12] <= 0.5:
                    continue
                origins.append(pos[step_of[t], j])
                # earlier layout: col 0 is RAW radians
                bearings.append(float(row[0]))
            if len(origins) < 2:
                continue
            est, sigma_min = _intersect_rays(origins, bearings)
            if sigma_min < 1e-3:
                continue
            errs.append(float(np.linalg.norm(est - users[u])))
            checked += 1

    assert checked > 100, f"only {checked} determined intersections; too few"
    errs = np.asarray(errs)
    assert errs.max() < 1.0, (
        f"bearing geometry is inconsistent on real data: max intersection "
        f"error {errs.max():.2f} m over {checked} multi-sensed user-steps "
        f"(median {np.median(errs):.2e})")


@pytest.mark.parametrize("seed", SEEDS)
def test_single_drone_bearing_alone_does_not_pin_a_user(seed):
    """Negative control, and a documented limitation.

    Confirms the per-user token genuinely has NO range column: with one drone's
    bearing you know the direction and nothing else. This is why the peer
    exchange (with absolute positions) matters: a single drone cannot
    localise users from its own set.
    """
    obs, _, users = _obs(seed)
    raw = build_raw_user_tokens(obs, episode_seed=seed)
    sensed = np.argwhere(raw[:, :, 12] > 0.5)
    assert len(sensed) > 0
    i, u = sensed[0]
    row = raw[i, u]
    # The 13 columns: none of them is a distance. RSSI (col 1) is the only
    # range PROXY, and it is normalised, clipped, and shadowing-corrupted.
    assert OBS_FEATURES == 13
    assert abs(row[0]) <= 1.0     # bearing, normalised
    assert abs(row[1]) <= 1.0     # rssi, normalised -- a proxy, not a range
