#!/usr/bin/env python3
"""Collect per-drone sensed user-set tokens from the OMNeT env, to train a
frozen encoder analogous to PADRE's deepsets_h64_o8.pt.

Mirrors ../src/collect_user_sets.py (the PADRE collector) in CLI shape,
npz output schema, episode-order-preserving write, and stats printout --
but drives OMNeTEnv instead of PADREParallelEnv.

Drives OMNeTEnv with a random (default) or zero action policy -- no
trained policy is needed here, same as PADRE's greedy/random collection.
The per-agent observation OMNeTEnv.step()/reset() return is now the
DRONE-token tensor (MAX_N, DRONE_TOKEN) -- NOT the user-token set (see
build_drone_tokens's column-layout comment in omnet_env.py). This
collector wants the USER tokens instead (the encoder's training data),
so each step it calls build_raw_user_tokens(obs_json, episode_seed)
directly on the env's raw obs JSON, which returns a
(MAX_N, MAX_USERS_CAP, OBS_FEATURES) array with all-zero rows for users
a drone doesn't sense (see that function's column-layout comment), and
the LAST column ("present") is exactly the sensed/real mask. So there
is no separate compaction/padding step here -- only extraction, plus
sanity asserts that the shapes and the present-column invariant
actually hold.

The OMNeT env runs one episode per subprocess launch and relaunches on
reset() (see OMNeTEnv.reset/_launch_omnet). One OMNeTEnv instance is
reused across --episodes; reset() handles killing the old subprocess and
spawning a fresh one each time, so this script just calls reset()/step()
in a loop and close()s once at the end.

Output: an .npz with
  tokens       : (M, MAX_USERS_CAP, OBS_FEATURES)     float32
  masks        : (M, MAX_USERS_CAP)                   float32  -- present column
  event_tokens : (M, MAX_EVENTS_CAP, EVENT_FEATURES)  float32  -- per-event set,
                 same (episode, step, drone) rows as tokens/masks (see
                 OMNeTEnv._attach_event_tokens / build_raw_event_tokens)
  event_mask   : (M, MAX_EVENTS_CAP)                  float32  -- present column
plus printed summary stats so you can eyeball the throughput (col 4) and
id_tag (cols 6-9) spread before freezing the encoder, plus the analogous
per-column event stats (events-per-set and non-empty fraction included,
since the raw event set is empty in >50% of drone-steps).

Usage:
  python collect_omnet_user_sets.py --episodes 300 --out omnet_user_sets.npz
  python collect_omnet_user_sets.py --episodes 5 --policy zero --out smoke.npz
"""

import argparse
import os
import sys

import numpy as np

# Make the module importable regardless of CWD (matches test_omnet_env.py).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Bind the collection world to the TRAINING world, BEFORE importing
#    omnet_env ────────────────────────────────────────────────────────────────
# Same role as evaluate_omnet.py's _apply_eval_config: without apply_config,
# $OMNET_EXTRA_ARGS is unset and the simulator uses omnetpp.ini's geometry
# (arena 1700, userSpread eff 25.5 m, droneSpawnMode "cluster", maxSpeed 20),
# while the v2/v3 task is arena 2000, spread 350, "staging", step_m 60.
#
# The frozen encoder must be fitted on user-token statistics from the world
# the policies train in, so the collection world is bound explicitly. apply_config sets $OMNET_STEP_M / $OMNET_NUM_DRONES which omnet_env
# reads ONCE at import time -- hence the import ordering below, which must not
# be "tidied".
def _collect_task_config(n_drones: int):
    """Byte-identical to evaluate_omnet.py's _eval_task_config, which is itself
    byte-identical to launch_v2_learnable.sh. Kept as a copy to avoid an import
    cycle; the other two copies are authoritative."""
    from config_knobs import Config
    return Config(
        n_drones=n_drones, L=2000.0, max_clusters=2, p_cross=0.5,
        user_spread_eff=350.0, step_m=60.0,
        drone_tx_mW=200.0, user_tx_mW=10.0,
        ep_steps=250, n_users=17,
        drone_spawn_mode="staging",
        jammer_activate=999999,   # > ep_steps => jammer never fires
        label="collect_frozen_enc",
    )


def _apply_collect_config(n_drones: int) -> dict:
    from config_knobs import apply_config
    applied = apply_config(_collect_task_config(n_drones))
    # No difference reward: this collector never reads a reward, and D_i costs
    # an N-way counterfactual LP per step (35 ms vs 1.5 ms).
    os.environ["OMNET_DIFFERENCE_REWARD"] = "0"
    if not applied.get("extra_args") or not os.environ.get("OMNET_EXTRA_ARGS"):
        raise SystemExit(
            "FATAL: apply_config produced no geometry. Refusing to collect an "
            "encoder dataset from a world that would silently fall back to the "
            "omnetpp.ini defaults.")
    return applied


# Parse --num-drones EARLY and bind the world, because the omnet_env import
# below reads $OMNET_NUM_DRONES / $OMNET_STEP_M at import time.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--num-drones", type=int, default=5, dest="num_drones")
_known, _ = _pre.parse_known_args()
_APPLIED = _apply_collect_config(_known.num_drones)

from signal_baselines import greedy_velocities as _greedy_velocities
from omnet_env import (
    MAX_EVENTS_CAP,
    MAX_USERS_CAP,
    EVENT_FEATURES,
    OBS_FEATURES,
    OMNeTEnv,
    build_raw_user_tokens,
)

# Column names for the raw-token layout (see omnet_env.py's
# build_raw_user_tokens column-layout comment). Only used for the stats
# printout; falls back to generic "colN" names if OBS_FEATURES is ever
# bumped, so the printout degrades gracefully instead of mislabeling.
# The list must have OBS_FEATURES entries in the order of
# build_raw_user_tokens()'s row[] assignments, which is authoritative.
_COL_NAMES = [
    "angle", "rssi", "rssi_valid", "interference", "noise_floor",
    "throughput", "thru_valid",
    "id_tag_0", "id_tag_1", "id_tag_2", "id_tag_3", "id_valid", "present",
]

# Column names for the raw per-event token layout (see omnet_env.py's
# build_raw_event_tokens column-layout comment). Same fallback behavior as
# _COL_NAMES above if EVENT_FEATURES is ever bumped.
_EVENT_COL_NAMES = [
    "snir", "snir_valid", "per", "ber", "size", "is_drop", "drop_retry",
    "drop_queue", "drop_corrupt", "drop_notforus", "from_drone", "present",
]


def _col_name(idx, width, names=_COL_NAMES):
    return names[idx] if width == len(names) else f"col{idx}"


def _sample_actions(agents, policy, rng, env=None):
    """Action dict for one step.

    `greedy` serves the FROZEN-ENCODER dataset: the encoder is
    trained on whatever states these actions produce, and the design decision
    is to train it on states an ORACLE visits (well-placed swarms) rather than
    a random walk. `hover` is here for the
    coverage study -- the degenerate end of the state distribution.

    greedy_velocities needs the RAW simulator obs (positions, user_positions,
    noise_floor), NOT the encoded drone-token tensor, so it takes env._last_obs
    -- identical to evaluate_omnet.py's _select_actions. It reads only fields
    the policy's own observation also carries and never touches the reward.
    """
    if policy == "zero" or policy == "hover":
        return {a: np.zeros(2, dtype=np.float32) for a in agents}
    if policy == "greedy":
        if env is None:
            raise RuntimeError("policy=greedy needs the env for its raw obs")
        vels = _greedy_velocities(env._last_obs)
        if len(vels) < len(agents):
            raise RuntimeError(
                f"greedy_velocities returned {len(vels)} velocities for "
                f"{len(agents)} live drones")
        return {a: np.asarray(vels[i], dtype=np.float32)
                for i, a in enumerate(agents)}
    return {a: rng.uniform(-1.0, 1.0, size=2).astype(np.float32) for a in agents}


def _dump_obs(obs_json, episode_seed, num_agents, tokens_all, masks_all,
              event_tokens_all, event_mask_all):
    """Extract + validate every live drone's USER and EVENT token sets from
    one obs_json.

    num_agents is the count of currently-live drones (indices 0..num_agents-1,
    matching OMNeTEnv's possible_agents/agents naming) -- only those rows of
    build_raw_user_tokens's (MAX_N, ...) output are collected, mirroring what
    the old per-agent-obs-dict loop iterated over. The event tokens/mask come
    straight off obs_json (populated by OMNeTEnv._attach_event_tokens), so
    both sets are read from the SAME obs_json here -- row m of
    event_tokens_all/event_mask_all is guaranteed to line up with row m of
    tokens_all/masks_all (same episode, step, drone).
    """
    user_tokens = build_raw_user_tokens(obs_json, episode_seed)
    event_tokens = obs_json["event_tokens"]
    event_mask = obs_json["event_mask"]
    for i in range(num_agents):
        agent = f"drone_{i}"
        tok = np.asarray(user_tokens[i], dtype=np.float32)
        cap, width = tok.shape
        # Correctness gate: token width comes from the obs itself, never a
        # hardcoded literal -- but must still match the env's declared
        # OBS_FEATURES, or the column-layout comment (and _COL_NAMES above)
        # needs updating.
        assert width == OBS_FEATURES, (
            f"{agent}: obs width {width} != OMNeTEnv.OBS_FEATURES "
            f"{OBS_FEATURES} -- schema changed, re-check the column layout"
        )
        assert cap == MAX_USERS_CAP, (
            f"{agent}: obs cap {cap} != OMNeTEnv.MAX_USERS_CAP {MAX_USERS_CAP}"
        )
        present = tok[:, -1]
        # Correctness gate: masks must align with the present column. Per
        # build_raw_user_tokens, present==0 rows are fully zeroed (angle,
        # noise_ctx included) -- so any nonzero value outside the present
        # rows means the mask and the token content have drifted apart.
        unsensed_rows = tok[present == 0.0]
        assert not unsensed_rows.any(), (
            f"{agent}: found nonzero features in a present==0 row -- "
            "mask does not align with the present column"
        )
        tokens_all.append(tok)
        masks_all.append(present.copy())

        ev_tok = np.asarray(event_tokens[i], dtype=np.float32)
        ev_cap, ev_width = ev_tok.shape
        assert ev_width == EVENT_FEATURES, (
            f"{agent}: event obs width {ev_width} != OMNeTEnv.EVENT_FEATURES "
            f"{EVENT_FEATURES} -- schema changed, re-check the column layout"
        )
        assert ev_cap == MAX_EVENTS_CAP, (
            f"{agent}: event obs cap {ev_cap} != OMNeTEnv.MAX_EVENTS_CAP "
            f"{MAX_EVENTS_CAP}"
        )
        ev_present = ev_tok[:, -1]
        ev_mask_row = np.asarray(event_mask[i], dtype=np.float32)
        # Correctness gate: the separately-populated event_mask array must
        # agree with the present column (col 11) baked into the token itself
        # -- these come from two independent writes in
        # build_raw_event_tokens(), so drift here would mean the mask and
        # the token content disagree about which rows are real events.
        assert np.array_equal(ev_mask_row, ev_present), (
            f"{agent}: event_mask does not agree with the present column"
        )
        # Same correctness gate as the user path: present==0 rows must be
        # fully zeroed.
        ev_unsensed_rows = ev_tok[ev_present == 0.0]
        assert not ev_unsensed_rows.any(), (
            f"{agent}: found nonzero features in a present==0 event row -- "
            "mask does not align with the present column"
        )
        event_tokens_all.append(ev_tok)
        event_mask_all.append(ev_present.copy())


def collect(episodes, out_path, policy="random", seed=0):
    rng = np.random.default_rng(seed)

    tokens_all = []
    masks_all = []
    event_tokens_all = []
    event_mask_all = []
    # Per-row episode index. train_final_encoder.py splits train/val on this
    # when present and otherwise falls back to --ep-rows arithmetic that
    # assumes a constant number of rows per episode, which does not hold when
    # a drone dies mid-episode. The index keeps the split episode-aligned.
    episode_id_all = []

    # encoder_mode="learned": build_drone_tokens refuses "frozen" with the
    # current token normalisation (the shipped deepsets_h128_o8.pt was fitted
    # on raw dBm), and that is the encoder this dataset is collected to replace.
    # The mode is irrelevant to what we collect: _dump_obs calls
    # build_raw_user_tokens on the raw obs json directly and never reads the
    # pooled per-drone tensor.
    env = OMNeTEnv(worker_index=0, encoder_mode="learned")
    try:
        for ep in range(episodes):
            # Seed EACH reset uniquely from the base seed + episode index, so
            # parallel array tasks (each passing a strided --seed) collect
            # DISTINCT episode layouts instead of all replaying the same worlds
            # off OMNeTEnv's internal counter (§8.6a).
            # base `seed` also drives the action RNG above; layouts get seed+ep.
            obs, _infos = env.reset(seed=seed + ep)
            _before = len(tokens_all)
            _dump_obs(env._last_obs, env._episode_counter, len(obs),
                      tokens_all, masks_all, event_tokens_all, event_mask_all)
            episode_id_all.extend([seed + ep] * (len(tokens_all) - _before))

            while True:
                live_agents = list(obs.keys())
                if not live_agents:
                    break
                actions = _sample_actions(live_agents, policy, rng, env)
                obs, _rewards, terms, truncs, _infos = env.step(actions)
                _before = len(tokens_all)
                _dump_obs(env._last_obs, env._episode_counter, len(obs),
                          tokens_all, masks_all, event_tokens_all,
                          event_mask_all)
                episode_id_all.extend([seed + ep] * (len(tokens_all) - _before))
                done = (any(terms.values()) if terms else False) or \
                       (any(truncs.values()) if truncs else False)
                if done:
                    break

            if (ep + 1) % 10 == 0:
                print(f"  episode {ep+1}/{episodes} done, "
                      f"{len(tokens_all)} samples so far")
    finally:
        env.close()

    # Episode-order preserving write: tokens_all/masks_all (and the parallel
    # event_tokens_all/event_mask_all) were appended in strict
    # (episode, step, drone) order above and never reordered or shuffled, so
    # a held-out tail slice (e.g. tokens[-N:]) stays an episode-aligned split, exactly
    # like collect_user_sets.py -- and row m of the event arrays lines up
    # with row m of the user arrays.
    tokens = np.stack(tokens_all, axis=0)  # (M, MAX_USERS_CAP, OBS_FEATURES)
    masks = np.stack(masks_all, axis=0)    # (M, MAX_USERS_CAP)
    event_tokens = np.stack(event_tokens_all, axis=0)  # (M, MAX_EVENTS_CAP, EVENT_FEATURES)
    event_mask = np.stack(event_mask_all, axis=0)      # (M, MAX_EVENTS_CAP)
    episode_id = np.asarray(episode_id_all, dtype=np.int64)   # (M,)
    assert episode_id.shape[0] == tokens.shape[0], (
        f"episode_id has {episode_id.shape[0]} rows but tokens has "
        f"{tokens.shape[0]} -- the per-row bookkeeping drifted")

    np.savez(out_path, tokens=tokens, masks=masks,
             event_tokens=event_tokens, event_mask=event_mask,
             episode_id=episode_id, policy=np.array(policy))

    # ---- summary stats: what the frozen encoder will train on ----
    print("\n=== collected dataset ===")
    print(f"samples (M):        {tokens.shape[0]}")
    print(f"users per set (U):  {tokens.shape[1]}")
    print(f"feature cols:       {tokens.shape[2]}")
    sensed = masks.sum(axis=1)
    print(f"sensed users/set:   mean={sensed.mean():.2f} "
          f"min={int(sensed.min())} max={int(sensed.max())}")
    print(f"fraction of non-empty sets: {(sensed > 0).mean():.2%}")

    m = masks.astype(bool)
    width = tokens.shape[2]
    print(f"\nper-column stats over SENSED tokens ({width} cols):")
    for c in range(width):
        vals = tokens[:, :, c][m]
        name = _col_name(c, width)
        if vals.size == 0:
            print(f"  {name:10s} (no sensed tokens)")
            continue
        print(f"  {name:10s} min={vals.min():+.3f} "
              f"max={vals.max():+.3f} mean={vals.mean():+.3f} "
              f"std={vals.std():.3f}")

    print("\n=== collected event dataset ===")
    print(f"samples (M):        {event_tokens.shape[0]}")
    print(f"events per set (E): {event_tokens.shape[1]}")
    print(f"feature cols:       {event_tokens.shape[2]}")
    ev_present = event_mask.sum(axis=1)
    print(f"events/set:         mean={ev_present.mean():.2f} "
          f"min={int(ev_present.min())} max={int(ev_present.max())}")
    print(f"fraction of non-empty event sets: {(ev_present > 0).mean():.2%}")

    ev_m = event_mask.astype(bool)
    ev_width = event_tokens.shape[2]
    print(f"\nper-column stats over SENSED event tokens ({ev_width} cols):")
    for c in range(ev_width):
        vals = event_tokens[:, :, c][ev_m]
        name = _col_name(c, ev_width, names=_EVENT_COL_NAMES)
        if vals.size == 0:
            print(f"  {name:14s} (no sensed events)")
            continue
        print(f"  {name:14s} min={vals.min():+.3f} "
              f"max={vals.max():+.3f} mean={vals.mean():+.3f} "
              f"std={vals.std():.3f}")

    # Realised std per column over SENSED rows. Every column is already
    # bounded to [-1,1] upstream, but BOUNDED IS NOT SCALE-MATCHED:
    # a plain MSE reconstruction spends capacity in proportion to realised
    # variance, so a constant column (rssi_valid/id_valid/present are 1.0 on
    # every sensed row) contributes nothing and a hash column (id_tag) is
    # pure noise the encoder cannot and should not fit. Printed here so the
    # encoder's loss weighting is a decision, not an accident.
    print("\n=== realised per-column std over SENSED user tokens ===")
    stds = []
    for c in range(width):
        vals = tokens[:, :, c][m]
        stds.append(float(vals.std()) if vals.size else 0.0)
    smax = max(stds) if stds else 1.0
    for c, sdv in enumerate(stds):
        bar = "#" * int(round(40 * sdv / smax)) if smax > 0 else ""
        print(f"  {_col_name(c, width):12s} std={sdv:6.4f}  {bar}")
    print(f"\nsaved -> {out_path}  (policy={policy}, "
          f"episodes={len(np.unique(episode_id))})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--out", type=str, default="omnet_user_sets.npz")
    ap.add_argument("--policy",
                     choices=["random", "zero", "greedy", "hover"],
                     default="random",
                     help="action policy driving OMNeTEnv; no trained "
                          "policy needed, this only shapes drone positions "
                          "(and therefore which users get sensed)")
    ap.add_argument("--num-drones", type=int, default=5, dest="num_drones",
                     help="N for the collection world. Bound BEFORE the "
                          "omnet_env import by the pre-parser above; declared "
                          "here so --help lists it.")
    ap.add_argument("--seed", type=int, default=0,
                     help="seed for the action-sampling RNG (random policy)")
    args = ap.parse_args()
    print(f"[config] collection world bound via config_knobs.apply_config: "
          f"N={_APPLIED['n_drones']} step_m={_APPLIED['step_m']} "
          f"spread_base={_APPLIED['user_spread_base']:.1f} "
          f"r_u={_APPLIED['r_u']:.1f} r_d={_APPLIED['r_d']:.1f}", flush=True)
    print(f"[config] policy={args.policy} episodes={args.episodes}", flush=True)
    collect(args.episodes, args.out, args.policy, args.seed)


if __name__ == "__main__":
    main()
