#!/usr/bin/env python3
"""Collect per-drone sensed user-set tokens from the OM2G WORLD, to train
the arm's frozen wire codec.

╔══════════════════════════════════════════════════════════════════════════╗
║  OM2G TRACK.  omnet_module/collect_omnet_user_sets.py is the OMNET       ║
║  collector and BINDS ITSELF to the OMNET v2/v3 world (L=2000, 17 users)  ║
║  -- a codec fitted on that distribution would not match the world it is  ║
║  deployed in here.                                                       ║
║  This one inherits the OM2G world from the environment, the same way     ║
║  smoke.sh / the launchers do.                                            ║
╚══════════════════════════════════════════════════════════════════════════╝

Usage (mirrors smoke.sh's env preamble; nocontrol = cheapest arm, and the
user-token distribution does not depend on the arm -- sensing is local):

    export OM2G_TRACK=1
    export OMNET_APPTAINER_SIF=... OMNET_HOST_CODE=... OMNET_CONTAINER_CODE=...
    eval "$(python om2g/om2g_config.py --arm nocontrol --shell)"
    python om2g/om2g_collect_codec_data.py --episodes 40 --out codec_data/om2g_sets.npz

Then per ladder rung L:

    python src/encoder_experiments/train_frozen_omnet_encoder.py \
        --npz 'codec_data/om2g_sets*.npz' --output L --out codecs/om2g_codec_oL.pt

Output npz (trainer-compatible: `tokens`, `masks`, plus provenance):
    tokens : (M, MAX_USERS_CAP, OBS_FEATURES) float32
    masks  : (M, MAX_USERS_CAP)               float32  (the `present` column)
    meta   : (M, 3) int32  -- (episode, step, drone), for the within-episode
             variation analysis (how much does a drone's sensed set actually
             change while users stand still?)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "omnet_module"))

import om2g_track as T  # noqa: E402

T.require_om2g("om2g_collect_codec_data.py")

if not os.environ.get("OMNET_EXTRA_ARGS"):
    raise SystemExit(
        "FATAL: $OMNET_EXTRA_ARGS is unset -- the OM2G world was not bound. "
        "Run  eval \"$(python om2g/om2g_config.py --arm nocontrol --shell)\" "
        "first (see this script's docstring); otherwise the simulator falls "
        "back to omnetpp.ini's geometry and the codec is fitted on a world "
        "no OM2G policy ever flies in.")

from om2g_env import OM2GEnv  # noqa: E402
from omnet_env import (  # noqa: E402
    MAX_USERS_CAP, OBS_FEATURES, build_raw_user_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--steps-per-episode", type=int, default=0,
                    help="0 = run each episode to its natural end")
    ap.add_argument("--every", type=int, default=1,
                    help="record every k-th step")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    # episode_id must be GLOBALLY unique across shards (load_shards refuses
    # collisions): offset by the seed block, so shard --seed 1 gets
    # 100000.., disjoint from shard 0's 0..N-1.
    ep_base = a.seed * 100000
    env = OM2GEnv(worker_index=0, encoder_mode="learned")
    toks, masks, meta = [], [], []
    for ep in range(a.episodes):
        env.reset(seed=a.seed * 100003 + ep)
        step = 0
        while env.agents:
            tk = build_raw_user_tokens(env._last_obs, env._episode_counter)
            if step % a.every == 0:
                n_live = len(env._last_obs.get("positions", []))
                for i in range(n_live):
                    toks.append(tk[i])
                    masks.append(tk[i, :, 12])          # `present`
                    meta.append((ep_base + ep, step, i))
            acts = {ag: rng.uniform(-1, 1, 2).astype(np.float32)
                    for ag in env.agents}
            env.step(acts)
            step += 1
            if a.steps_per_episode and step >= a.steps_per_episode:
                break
        print(f"episode {ep + 1}/{a.episodes}: {len(toks)} rows so far",
              flush=True)
    env.close()

    X = np.asarray(toks, dtype=np.float32)
    M = np.asarray(masks, dtype=np.float32)
    E = np.asarray(meta, dtype=np.int32)
    assert X.shape[1:] == (MAX_USERS_CAP, OBS_FEATURES), X.shape
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(
        a.out, tokens=X, masks=M, meta=E,
        episode_id=E[:, 0].astype(np.int64),      # trainer's split key
        policy="random", track="OM2G",
        obs_features=OBS_FEATURES, max_users_cap=MAX_USERS_CAP,
        partner_tag=int(bool(os.environ.get("OM2G_USER_PARTNER_TAG"))))
    occ = float(M.sum(1).mean())
    print(f"wrote {a.out}: {X.shape[0]} sets, mean sensed users "
          f"{occ:.1f}/{MAX_USERS_CAP}, features={OBS_FEATURES}", flush=True)


if __name__ == "__main__":
    main()
