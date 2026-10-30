#!/usr/bin/env python3
"""
Standalone smoke test for OMNeTEnv.

OMNeT++ is launched BY THE ENV (not manually). Run from omnet_module/ or
from the repo root — sys.path is patched automatically.

Checks:
  1. OMNeTEnv construction.
  2. reset() returns obs for all MAX_N agents, each shape (MAX_N, DRONE_TOKEN).
  3. 5 random steps: obs keyed by live agents, rewards keyed by all possible.
  4. Second reset() launches a NEW process with a different episodeSeed.
  5. close() kills subprocess and closes socket cleanly.
"""
import os
import sys
import random
import numpy as np

# Make the module importable regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omnet_env import OMNeTEnv, MAX_N, DRONE_TOKEN

# Derived, NOT hardcoded: the drone token width has changed (18 -> 26 when the
# event latent was added), so indices are computed from DRONE_TOKEN. mask is
# contractually the LAST column; self_flag sits four
# columns before the tail block (self_flag, relay_thru, channel_busy,
# time_since_rx, mask).
MASK_COL = DRONE_TOKEN - 1
SELF_FLAG_COL = DRONE_TOKEN - 5


# ── tiny assertion helper ─────────────────────────────────────────────────────

_failures = []

def check(label, condition, detail=""):
    tag = "[PASS]" if condition else "[FAIL]"
    msg = f"{tag} {label}" + (f"  ({detail})" if detail else "")
    print(msg, flush=True)
    if not condition:
        _failures.append(label)
    return condition


# ── test body ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("OMNeTEnv smoke test")
    print("=" * 60)

    # ── 1. Construct ──────────────────────────────────────────────────────────
    print("\n[1] Constructing OMNeTEnv ...")
    env = OMNeTEnv(worker_index=0)
    check("construction", env is not None)
    check("possible_agents length", len(env.possible_agents) == MAX_N,
          f"got {len(env.possible_agents)}")

    # ── 2. reset() ────────────────────────────────────────────────────────────
    print("\n[2] First reset() ...")
    obs, infos = env.reset()
    print(f"    obs keys: {sorted(obs.keys())}", flush=True)

    check("obs has MAX_N keys", len(obs) == MAX_N, f"got {len(obs)}")
    check("infos has MAX_N keys", len(infos) == MAX_N, f"got {len(infos)}")

    for k, v in obs.items():
        arr = np.asarray(v)
        check(f"obs[{k}] shape == (MAX_N, DRONE_TOKEN)",
              arr.shape == (MAX_N, DRONE_TOKEN),
              f"got {arr.shape}")

    first_seed = env._episode_counter
    first_user_pos = list(env._last_obs.get("user_positions", []))
    print(f"    episodeSeed={first_seed}  user_positions[:2]={first_user_pos[:2]}",
          flush=True)

    # ── 3. 5 random steps ─────────────────────────────────────────────────────
    print("\n[3] 5 random steps ...")
    all_pa = set(env.possible_agents)
    self_flag_ok = True
    antisym_ok = True
    mask_ok = True
    any_live_event_latent = False
    reward_nonzero_ok = False
    user_latent_live_ok = False

    for step_i in range(5):
        actions = {
            f"drone_{i}": [random.uniform(-1, 1), random.uniform(-1, 1)]
            for i in range(MAX_N)
        }
        obs_s, rewards_s, terms_s, truncs_s, infos_s = env.step(actions)

        live_keys = {f"drone_{i}" for i in range(len(env.agents))}
        n_live = len(env.agents)
        obs_keys = set(obs_s.keys())
        rew_keys = set(rewards_s.keys())

        check(f"step {step_i}: obs keyed by live agents only",
              obs_keys == live_keys,
              f"obs={sorted(obs_keys)}  live={sorted(live_keys)}")

        check(f"step {step_i}: rewards keyed by ALL possible_agents",
              rew_keys == all_pa,
              f"got {sorted(rew_keys)}")

        rvals = list(rewards_s.values())
        # A uniformly-ZERO reward passes the identical-scalar check below
        # while carrying no training signal at all (e.g. with highspy
        # missing, every other check here passes against reward=0.0). The oracle
        # reward is a deterministic function of geometry and is non-zero at
        # spawn on every seed probed (valley_probe, 57.7-191.4), so an
        # all-zero episode means a broken solver or a broken obs, not a hard
        # world.
        if any(float(v) != 0.0 for v in rvals):
            reward_nonzero_ok = True
        # Under difference rewards the per-drone values are
        # D_i = R(all) - R(all \ i) and are SUPPOSED to differ, so they are
        # not asserted identical. The team reward is carried in infos.
        if omnet_env.DIFFERENCE_REWARD:
            check(f"step {step_i}: infos carry a single global_reward",
                  len({float(iv["global_reward"]) for iv in infos_s.values()}) == 1,
                  f"infos={infos_s}")
            check(f"step {step_i}: difference rewards are not all identical",
                  len(set(rvals)) > 1,
                  f"every drone got the same D_i ({rvals}) -- either the "
                  f"counterfactual is not being computed or every drone is "
                  f"perfectly interchangeable")
        else:
            check(f"step {step_i}: all rewards identical scalar",
                  len(set(rvals)) == 1,
                  f"values={rvals}")

        # self_flag (SELF_FLAG_COL): 1.0 only on the diagonal (own row within own
        # obs), 0.0 everywhere else — including dead peer rows.
        for i in range(n_live):
            arr = np.asarray(obs_s[f"drone_{i}"])
            if arr[i, SELF_FLAG_COL] != 1.0:
                self_flag_ok = False
            others = np.delete(np.arange(MAX_N), i)
            if np.any(arr[others, SELF_FLAG_COL] != 0.0):
                self_flag_ok = False

        # dx/dy (cols 0,1): antisymmetric across observer pairs —
        # obs["drone_i"][j] is peer j as seen by i, obs["drone_j"][i] is
        # peer i as seen by j; the displacement vector must flip sign.
        for i in range(n_live):
            arr_i = np.asarray(obs_s[f"drone_{i}"])
            for j in range(n_live):
                arr_j = np.asarray(obs_s[f"drone_{j}"])
                if not np.allclose(arr_i[j, 0:2], -arr_j[i, 0:2], atol=1e-4):
                    antisym_ok = False

        # mask (MASK_COL, MUST be last — ST/DeepSets models hardcode
        # x[..., -1] as the mask column): 1.0 for live peer drones, 0.0 for
        # dead ones.
        for i in range(n_live):
            arr = np.asarray(obs_s[f"drone_{i}"])
            if n_live > 0 and np.any(arr[:n_live, MASK_COL] != 1.0):
                mask_ok = False
            if n_live < MAX_N and np.any(arr[n_live:, MASK_COL] != 0.0):
                mask_ok = False

        # cols 5:13 = the 8-dim user_latent from the REAL frozen encoder
        # (FrozenOMNeTUserEncoder, h128_o8). Invariant flipped from the old
        # stub tripwire: a drone that senses >=1 user (its own token set has a
        # non-zero row) must produce a NON-zero latent; a drone that senses
        # nobody produces a zero latent. All-zero everywhere would mean the
        # stub is still wired in (regression).
        any_live_latent = False
        for arr in obs_s.values():
            arr = np.asarray(arr)
            if np.any(arr[:, 5:13] != 0.0):
                any_live_latent = True
        # at the frozen arena sets are rarely empty (frac_empty ~0), so across a few steps at least one drone should have a live set.
        user_latent_live_ok = any_live_latent

        # cols 13:21 = the 8-dim event_latent from FrozenOMNeTEventEncoder
        # (h64_o8, job 7666758), fed the CUT event set (cut_event_tokens).
        # Same tripwire logic as the user latent above: all-zero everywhere
        # means either the encoder is not wired, the cut filter emptied every
        # set, or RLBridge stopped emitting events -- all silent failures that
        # every other check in this file would pass through. The encoder maps
        # an empty set to to_latent.bias (non-zero), so an all-zero latent
        # cannot happen through a legitimate path.
        for arr in obs_s.values():
            arr = np.asarray(arr)
            if np.any(arr[:, 13:21] != 0.0):
                any_live_event_latent = True

        if terms_s and any(terms_s.values()):
            print(f"    Episode terminated at step {step_i}, stopping loop.",
                  flush=True)
            break

    check("reward is NON-ZERO on at least one step — a flat-zero reward "
          "passes every other check in this file while training on nothing "
          "(missing LP solver, broken obs)", reward_nonzero_ok)
    check("event_latent (cols 13:21) is LIVE (non-zero) — real "
      "FrozenOMNeTEventEncoder wired in over the CUT event set",
      any_live_event_latent)
    check(f"self_flag (col {SELF_FLAG_COL}) is 1.0 on the diagonal only", self_flag_ok)
    check("dx/dy (cols 0,1) are antisymmetric across observer pairs",
          antisym_ok)
    check(f"mask (col {MASK_COL}, last) is 1.0 for live drones, 0.0 for dead ones", mask_ok)
    check("user_latent (cols 5:13) is LIVE (non-zero for a drone that senses "
          ">=1 user) — real FrozenOMNeTUserEncoder wired in, not the stub",
          user_latent_live_ok)

    # ── 4. Second reset() — must be a NEW process with a different seed ───────
    print("\n[4] Second reset() (relaunch check) ...")
    obs2, _ = env.reset()
    second_seed = env._episode_counter
    second_user_pos = list(env._last_obs.get("user_positions", []))

    check("episodeSeed incremented", second_seed == first_seed + 1,
          f"first={first_seed}  second={second_seed}")

    check("new obs dict has MAX_N keys", len(obs2) == MAX_N,
          f"got {len(obs2)}")

    # Seeds differ ⟹ different OMNeT++ process was spawned
    check("new process spawned (seed changed)", second_seed != first_seed,
          f"first_seed={first_seed}  second_seed={second_seed}")

    print(f"    episodeSeed={second_seed}  user_positions[:2]={second_user_pos[:2]}",
          flush=True)

    # ── 5. close() ────────────────────────────────────────────────────────────
    print("\n[5] close() ...")
    try:
        env.close()
        check("close() no exception", True)
    except Exception as exc:
        check("close() no exception", False, str(exc))

    # ── Summary ───────────────────────────────────────────────────────────────
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
