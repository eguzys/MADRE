"""Offline synthetic obs_json generator for OMNeT-side tests.

Builds schema-correct obs_json dicts for omnet_env.build_drone_tokens /
build_raw_user_tokens -- radio quantities (rssi, interference, noise floor)
come from the SAME Friis/log-distance model max_flow_reward.py uses to score
the training reward, so a synthetic obs is physically consistent with a real
one, not just shape-compatible. Entirely offline: no OMNeT++/INET subprocess
anywhere in this module.

Paths are resolved relative to THIS file (not hardcoded to one machine),
mirroring omnet_env.py's own _REPO resolution. Setting OMNET_APPTAINER_SIF
(any truthy value) makes omnet_env.py skip its host-side INET directory
check (see omnet_env.py's `if not os.environ.get("OMNET_APPTAINER_SIF")`
guard) -- the OMNeT binary is never launched by anything importing this
module, only pure functions of obs_json are exercised.
"""
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))                 # .../src/tests/fixtures
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))   # .../code

os.environ.setdefault("OMNET_APPTAINER_SIF", os.path.join(_REPO, "omnet.sif"))
os.environ.setdefault("OMNET_SMOKE_DIR", os.path.join(_REPO, "omnet_smoke"))
sys.path.insert(0, os.path.join(_REPO, "omnet_module"))
sys.path.insert(0, os.path.join(_REPO, "src"))

import max_flow_reward as M  # noqa: E402

L = 1700.0        # arena side, metres (matches omnetpp.ini's default arena)
NU = 17           # numUsers (omnetpp.ini default)
MAXSTEPS = 1000.0

FSPL1 = 20 * math.log10(4 * math.pi / (3e8 / (M.FREQ_GHz * 1e9)))


def rx_dbm(d: float, tx_mW: float) -> float:
    """Free-space path loss received power, same model max_flow_reward uses."""
    d = max(d, 1.0)
    return 10 * math.log10(tx_mW) - (FSPL1 + 10 * M.ALPHA * math.log10(d))


def make_layout(seed, nd: int = 5, sigma: float = 150.0, spawn_sigma: float = 250.0):
    """Deterministic-per-seed drone/user layout + sender_pairs.

    Two "camps" (random centers) so some drone-user pairs are close and some
    far, exercising both rssi_valid=1 and rssi_valid=0 rows.
    """
    r = np.random.default_rng(seed)
    centers = r.uniform(0.15 * L, 0.85 * L, size=(2, 2))
    users = np.array([np.clip(centers[u % 2] + r.normal(0, sigma, 2), 0, L) for u in range(NU)])
    pairs = [[u, int(r.choice([v for v in range(NU) if v != u]))] for u in range(NU)]
    drones = np.array([np.clip(centers[i % 2] + r.normal(0, spawn_sigma, 2), 0, L) for i in range(nd)])
    return users, pairs, drones


def build_obs(drones, users, pairs, step: int = 0) -> dict:
    """Build one schema-correct obs_json, as RLBridge would emit it."""
    nd, nu = len(drones), len(users)
    nf = [M.NOISE_FLOOR_dBm] * nd

    # user->drone rssi (10 mW), valid iff it clears sensitivity+SNIR (a real
    # frame arrived) -- same gate build_raw_user_tokens expects.
    urssi, uvalid = [], []
    for i in range(nd):
        row_r, row_v = [], []
        for u in range(nu):
            d = float(np.linalg.norm(users[u] - drones[i]))
            p = rx_dbm(d, M.USER_TX_mW)
            ok = (p >= M.SENSITIVITY_dBm) and (p - nf[i] >= M.SNIR_THRESHOLD_dB)
            row_r.append(p if ok else 0.0)
            row_v.append(bool(ok))
        urssi.append(row_r)
        uvalid.append(row_v)

    prssi, pvalid = [], []
    for i in range(nd):
        row_r, row_v = [], []
        for j in range(nd):
            if i == j:
                row_r.append(0.0)
                row_v.append(False)
                continue
            d = float(np.linalg.norm(drones[j] - drones[i]))
            p = rx_dbm(d, M.DRONE_TX_mW)
            ok = (p >= M.SENSITIVITY_dBm) and (p - nf[i] >= M.SNIR_THRESHOLD_dB)
            row_r.append(p if ok else 0.0)
            row_v.append(bool(ok))
        prssi.append(row_r)
        pvalid.append(row_v)

    # association: nearest drone (by rssi) that can hear the user
    u2d = []
    for u in range(nu):
        best, bi = -1e9, -1
        for i in range(nd):
            if uvalid[i][u] and urssi[i][u] > best:
                best, bi = urssi[i][u], i
        u2d.append(bi)

    goodput = [3000.0 if u2d[u] >= 0 else 0.0 for u in range(nu)]

    interference = []
    for i in range(nd):
        lin = 0.0
        for u in range(nu):
            lin += 10 ** (rx_dbm(float(np.linalg.norm(users[u] - drones[i])), M.USER_TX_mW) / 10)
        for j in range(nd):
            if j != i:
                lin += 10 ** (rx_dbm(float(np.linalg.norm(drones[j] - drones[i])), M.DRONE_TX_mW) / 10)
        interference.append(10 * math.log10(max(lin, 1e-15)))

    served = [sum(1 for u in range(nu) if u2d[u] == i) for i in range(nd)]

    events = []
    for i in range(nd):
        ev = []
        for _ in range(min(15, 2 + 3 * served[i])):
            ev.append([25.0, 1.0, 0.01, 1e-5, 120.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        events.append(ev)

    return {
        "positions": [[float(p[0]), float(p[1])] for p in drones],
        "user_positions": [[float(p[0]), float(p[1])] for p in users],
        "noise_floor": nf,
        "interference": interference,
        "user_to_drone": u2d,
        "user_goodput": goodput,
        "user_rssi": urssi,
        "user_rssi_valid": uvalid,
        "peer_rssi": prssi,
        "peer_rssi_valid": pvalid,
        "drone_fwd": [float(3000 * served[i]) for i in range(nd)],
        "channel_busy": [min(1.0, 0.05 * served[i]) for i in range(nd)],
        "time_since_rx": [0.1 if served[i] else float(step) for i in range(nd)],
        "events": events,
        "sender_pairs": pairs,
        "arena_width": L,
        "arena_height": L,
        "max_steps": MAXSTEPS,
        "step": step,
        "done": False,
    }
