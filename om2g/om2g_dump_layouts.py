#!/usr/bin/env python3
"""Dump REAL episode layouts (staging drone spawn + user positions + pairs).

⚠️ OM2G TRACK. Used to tune the reward's demand/taper against the geometry the
simulator actually produces rather than a hand-made layout (the real staging
spawn already sits near the user clusters).
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "omnet_module"))
import om2g_track as T
T.require_om2g("om2g_dump_layouts.py")
from om2g_env import OM2GEnv

n = int(os.environ.get("OM2G_DUMP_EPISODES", "10"))
out = []
env = OM2GEnv(worker_index=0, encoder_mode="learned",
              arm=os.environ.get("OM2G_ARM", "compressed"))
for i in range(n):
    env.reset()
    o = env._last_obs
    out.append(dict(positions=o["positions"], user_positions=o["user_positions"],
                    sender_pairs=o.get("sender_pairs", []),
                    noise_floor=o.get("noise_floor", [])))
    print(f"  episode {i}: {len(o['positions'])} drones, "
          f"{len(o['user_positions'])} users", flush=True)
p = os.environ.get("OM2G_DUMP_OUT", "om2g_out/OM2G_layouts.json")
json.dump(out, open(p, "w"))
print(f"wrote {p}", flush=True)
