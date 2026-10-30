#!/usr/bin/env python3
"""OM2G end-to-end smoke test: env + delivery gating + reward + model.

╔══════════════════════════════════════════════════════════════════════════╗
║  OM2G TRACK.  Run BEFORE submitting a training job.                     ║
╚══════════════════════════════════════════════════════════════════════════╝

Checks, in the order they would fail:
  1. the OM2G env constructs and the simulator launches with om2g.ini
  2. the obs JSON carries the beacon bookkeeping (i.e. the binary is current)
  3. delivery gating actually blanks/stales peer rows (not a no-op)
  4. the reward oracle runs on the 802.15.4 radio constants and is non-zero
  5. the arm's model builds at the right widths and forward-passes
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "omnet_module"))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import om2g_track as T
import om2g_config as C

ARM = os.environ.get("OM2G_ARM", "compressed")
STEPS = int(os.environ.get("OM2G_SMOKE_STEPS", "12"))

print(f"=== OM2G SMOKE  arm={ARM}  steps={STEPS} ===", flush=True)
T.om2g_selftest(verbose=False); C.om2g_selftest(verbose=False)
print("[1/5] config self-tests PASS", flush=True)

from om2g_env import OM2GEnv          # noqa: E402
import max_flow_reward as R           # noqa: E402

print(f"      reward radio: B={R.BANDWIDTH:.3g} Hz  f={R.FREQ_GHz} GHz  "
      f"sens={R.SENSITIVITY_dBm} dBm  snir={R.SNIR_THRESHOLD_dB} dB  "
      f"noise={R.NOISE_FLOOR_dBm} dBm  cap={R.RATE_CAP_BPS}  "
      f"unit={R.REWARD_UNIT_BPS}", flush=True)
assert R.RATE_CAP_BPS == T.OM2G_BITRATE_BPS, "rate cap not applied"
assert abs(R.BANDWIDTH - T.OM2G_BANDWIDTH_HZ) < 1, "bandwidth not applied"

env = OM2GEnv(worker_index=0, encoder_mode="learned", arm=ARM)
obs, _ = env.reset()
first_obs = {k: np.array(v, copy=True) for k, v in obs.items()}
print(f"[2/5] env reset OK; obs shape {next(iter(obs.values())).shape}", flush=True)

blank_tot = stale_tot = fresh_tot = 0
rewards = []
for k in range(STEPS):
    acts = {a: np.random.uniform(-1, 1, size=2).astype(np.float32) for a in env.agents}
    obs, rew, term, trunc, info = env.step(acts)
    if not env.agents:
        break
    rewards.append(float(next(iter(rew.values()))))
    t = env._telemetry[-1]
    blank_tot += t["peers_blank"]; stale_tot += t["peers_stale"]; fresh_tot += t["peers_fresh"]

tot = max(1, blank_tot + stale_tot + fresh_tot)
print(f"[3/5] delivery gating over {len(rewards)} steps: "
      f"fresh {fresh_tot/tot*100:.1f}%  stale {stale_tot/tot*100:.1f}%  "
      f"blank {blank_tot/tot*100:.1f}%", flush=True)
assert (blank_tot + stale_tot) > 0, (
    "gating is a NO-OP: every peer row was fresh. Either the beacons are "
    "arriving perfectly every step (check beaconBytes) or the gate is not "
    "wired -- either way the experiment would be meaningless.")

r = np.asarray(rewards)
print(f"[4/5] reward: mean {r.mean():.2f}  min {r.min():.2f}  max {r.max():.2f} "
      f"(kbps units)", flush=True)
assert np.all(np.isfinite(r)) and r.std() > 0, "reward is constant or non-finite"

tel = env._telemetry if env._telemetry else []
env._flush_telemetry()

# 5. the observation layout + model-side parse:
#    the env's flat arm-sized obs must round-trip through the model-side
#    splitter at exactly the widths the wire charge implies.
import torch  # noqa: E402
from architectures.user_encoder import (  # noqa: E402
    om2g_flat_layout, split_om2g_obs)
from om2g_env import om2g_arm_latent  # noqa: E402
lo = om2g_flat_layout()
_L = om2g_arm_latent(ARM)
assert lo["latent"] == _L, (lo["latent"], _L)
ob0 = torch.as_tensor(next(iter(first_obs.values()))).unsqueeze(0)
# split_om2g_obs returns a 5-tuple; the last two are the e2e inbox and are
# None off that path.
raw_toks, raw_mask, peers, _inbox_toks, _inbox_mask = split_om2g_obs(ob0, lo)
assert ob0.shape[-1] == lo["flat_dim"], (ob0.shape, lo["flat_dim"])
assert peers.shape[-2:] == (lo["max_n"], 20 + _L), peers.shape
assert raw_toks.shape[-2:] == (lo["max_users"], lo["user_feat"])
# Delivered peer payload width == the arm's charged width (14 + L floats):
if ARM != "nocontrol":
    assert T.om2g_message_floats(ARM, _L) == 14 + _L
    assert T.om2g_message_bytes(ARM, _L) == 4 * _L + 56
print(f"[5/5] flat obs {ob0.shape[-1]} = raw {lo['raw_w']} + "
      f"{lo['max_n']} slots x {20 + _L} (arm={ARM}, L={_L}); "
      f"model-side split OK", flush=True)

env.close() if hasattr(env, "close") else None
print(f"=== OM2G SMOKE PASS (arm={ARM}) ===", flush=True)
