"""Per-drone credit assignment via difference rewards.

With ONE global scalar broadcast to every drone and a shared critic, every
drone receives the same advantage and no gradient asks drone i to behave
differently from drone j (measured: cosine 0.956-0.996 between the five
drones' mean action directions, and a swarm that does not spread). The
difference reward gives each drone its own marginal contribution.

These tests cover the reward arithmetic and the leave-one-out helper without
starting a simulator; the live end-to-end shape check lives in
omnet_module/test_omnet_env.py.
"""
from __future__ import annotations

import pytest

import max_flow_reward as mfr
import omnet_env
from max_flow_reward import compute_oracle_max_flow_reward as R
from omnet_env import _leave_one_out_obs


@pytest.fixture
def obs(synthetic_omnet_obs):
    return synthetic_omnet_obs(seed=3)


# ---------------------------------------------------------------------------
# The leave-one-out helper
# ---------------------------------------------------------------------------

def test_leave_one_out_drops_exactly_one_drone(obs):
    nd = len(obs["positions"])
    for i in range(nd):
        o = _leave_one_out_obs(obs, i)
        assert len(o["positions"]) == nd - 1
        assert obs["positions"][i] not in o["positions"] or \
            obs["positions"].count(obs["positions"][i]) > 1
        assert len(o["noise_floor"]) == len(obs["noise_floor"]) - 1


def test_leave_one_out_does_not_mutate_the_caller(obs):
    before = [list(p) for p in obs["positions"]]
    _leave_one_out_obs(obs, 0)
    assert [list(p) for p in obs["positions"]] == before, (
        "the counterfactual mutated the observation the env is still using")


def test_leave_one_out_leaves_user_indexed_fields_alone(obs):
    o = _leave_one_out_obs(obs, 1)
    assert o["user_positions"] == obs["user_positions"]
    for key in ("sender_pairs", "user_area"):
        if key in obs:
            assert o[key] == obs[key], f"{key} is user-indexed and must survive"


# ---------------------------------------------------------------------------
# The reward itself
# ---------------------------------------------------------------------------

def test_removing_a_drone_never_increases_the_team_reward(obs):
    """D_i >= 0: an extra relay cannot make the max-flow objective worse, so a
    difference reward must never be negative. A negative D_i would actively
    teach a drone to leave."""
    full = R(obs)
    nd = len(obs["positions"])
    for i in range(nd):
        assert R(_leave_one_out_obs(obs, i)) <= full + 1e-6, (
            f"dropping drone {i} RAISED the team reward — the reward is not "
            f"monotone in the drone set and D_i would be negative")


def test_difference_rewards_differ_between_drones(obs):
    """The property the difference reward exists for: drones that contribute differently must
    receive different rewards."""
    full = R(obs)
    d = [full - R(_leave_one_out_obs(obs, i)) for i in range(len(obs["positions"]))]
    assert len(set(round(x, 6) for x in d)) > 1, (
        f"every drone got the same difference reward {d} — no per-drone signal")


# NOT TESTED, deliberately: that sum_i D_i != R. Difference rewards are an
# alignment of the team objective, not a decomposition of it, so the two are
# unrelated in general -- but they are not guaranteed to DIFFER either. Measured
# live against the simulator, sum_i D_i equalled R to display precision on 2 of
# the first 3 steps (732.6 vs 732.6, 680.0 vs 680.0) simply because the drones
# happened to be covering disjoint users, and differed on the next (713.5 vs
# 708.9). An assertion either way would be pinning a coincidence.


# ---------------------------------------------------------------------------
# The env wiring, without a simulator
# ---------------------------------------------------------------------------

class _FakeEnv:
    """Just enough of OMNeTEnv to exercise _per_agent_rewards."""
    possible_agents = [f"drone_{i}" for i in range(omnet_env.MAX_N)]
    _per_agent_rewards = omnet_env.OMNeTEnv._per_agent_rewards


def test_per_agent_rewards_difference_mode(obs, monkeypatch):
    monkeypatch.setattr(omnet_env, "DIFFERENCE_REWARD", True)
    full = R(obs)
    out = _FakeEnv()._per_agent_rewards(obs, full)
    assert set(out) == set(_FakeEnv.possible_agents), "must key ALL possible_agents"
    nd = len(obs["positions"])
    for i in range(nd):
        expected = full - R(_leave_one_out_obs(obs, i))
        # rel=1e-6, not exact equality: R and R(all \ i) come from a warm-started
        # simplex whose tolerance is ~1e-9 relative on R, and D_i = R - R(\i) is
        # a difference of two O(R) numbers, so the relative error is amplified by
        # R/D_i (~4x here). Whether a given call is warm depends on what else
        # touched the LP cache first, so a tighter bound passes alone and fails
        # in the full suite.
        assert out[f"drone_{i}"] == pytest.approx(expected, rel=1e-6)
    for i in range(nd, omnet_env.MAX_N):
        assert out[f"drone_{i}"] == 0.0, "an absent drone's difference reward is 0"


def test_per_agent_rewards_global_mode_is_the_old_behaviour(obs, monkeypatch):
    monkeypatch.setattr(omnet_env, "DIFFERENCE_REWARD", False)
    out = _FakeEnv()._per_agent_rewards(obs, 12.5)
    assert out == {a: 12.5 for a in _FakeEnv.possible_agents}


def test_single_drone_gets_the_whole_reward(obs, monkeypatch):
    """With one drone there is no counterfactual: D_0 == R, not R - R(empty)
    computed on a degenerate graph."""
    monkeypatch.setattr(omnet_env, "DIFFERENCE_REWARD", True)
    one = dict(obs)
    one["positions"] = obs["positions"][:1]
    one["noise_floor"] = obs["noise_floor"][:1]
    full = R(one)
    out = _FakeEnv()._per_agent_rewards(one, full)
    assert out["drone_0"] == pytest.approx(full)


# ---------------------------------------------------------------------------
# The LP cache change that makes this affordable
# ---------------------------------------------------------------------------

def test_lp_cache_is_multi_slot_and_bounded(obs):
    """All N+1 variants must be able to sit in the cache at once, or every
    leave-one-out call pays a full structure rebuild (measured 6.7 ms vs
    1.5 ms). The bound must still hold under a cardinality sweep."""
    mfr._ORACLE_LP_CACHE.clear()
    nd = len(obs["positions"])
    R(obs)
    for i in range(nd):
        R(_leave_one_out_obs(obs, i))
    assert len(mfr._ORACLE_LP_CACHE) <= mfr._ORACLE_LP_CACHE_MAXSLOTS

    for n in range(2, nd + 1):
        o = dict(obs)
        o["positions"] = obs["positions"][:n]
        o["noise_floor"] = obs["noise_floor"][:n]
        R(o)
    assert len(mfr._ORACLE_LP_CACHE) <= mfr._ORACLE_LP_CACHE_MAXSLOTS, (
        "the LRU bound leaked — each slot holds a HiGHS instance")


def test_cache_tag_never_changes_the_value(obs):
    """cache_tag is a warm-start hint only. If it ever influences the returned
    reward, the difference reward silently depends on which drone index it was
    computed under."""
    nd = len(obs["positions"])
    for i in range(nd):
        o = _leave_one_out_obs(obs, i)
        tagged = R(o, cache_tag=i)
        untagged = R(o, cache_tag=None)
        assert tagged == pytest.approx(untagged, rel=1e-6)


def test_each_leave_one_out_gets_its_own_cache_slot(obs):
    """Without per-drone slots every variant collides on one key and shares a
    HiGHS basis left by a DIFFERENT drone's removal — measured 38.0 ms/step vs
    11.6 ms. This pins the slot separation, not the timing."""
    mfr._ORACLE_LP_CACHE.clear()
    nd = len(obs["positions"])
    R(obs)
    for i in range(nd):
        R(_leave_one_out_obs(obs, i), cache_tag=i)
    assert len(mfr._ORACLE_LP_CACHE) == nd + 1, (
        f"expected 1 full + {nd} tagged slots, got "
        f"{len(mfr._ORACLE_LP_CACHE)} — the leave-one-out LPs are sharing a "
        f"warm-start basis again")
    assert mfr._ORACLE_LP_CACHE_MAXSLOTS > nd + 1, (
        "MAXSLOTS must exceed the working set or the LRU evicts a slot that "
        "the same step is about to reuse")


def test_cached_and_uncached_rewards_agree(obs):
    """Warm-start must not change the value. (Simplex tolerance only; the
    original single-slot cache had the same contract.)"""
    nd = len(obs["positions"])
    variants = [obs] + [_leave_one_out_obs(obs, i) for i in range(nd)]
    mfr._ORACLE_LP_CACHE.clear()
    cold = [R(v) for v in variants]
    warm = [R(v) for v in variants]
    for c, w in zip(cold, warm):
        assert c == pytest.approx(w, rel=1e-7, abs=1e-6)
