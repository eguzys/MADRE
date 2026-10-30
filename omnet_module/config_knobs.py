#!/usr/bin/env python3
"""
config_knobs.py — the ONE function that sets every knob on BOTH sides of the
OMNeT <-> Python-oracle boundary, for the Tier-A/B/C configuration search.

WHY THIS EXISTS
----------------
Three knobs in this codebase have TWO copies that must move together:

  drone speed  ->  OMNeT `**.host[*].mobility.maxSpeed={v}mps`  (live knob;
                    the "DEAD" comment at omnetpp.ini:102 covers only the
                    initialX/Y block, NOT maxSpeed)
                    AND  greedy_oracle.STEP_M (the probes' own step size)
  drone TX     ->  OMNeT `--*.host[*].wlan[*].radio.transmitter.power={x}mW`
                    AND  max_flow_reward.DRONE_TX_mW
  user  TX     ->  OMNeT `--*.user[*].wlan[*].radio.transmitter.power={y}mW`
                    AND  max_flow_reward.USER_TX_mW

If the OMNeT-side override changes but the Python-oracle copy does not (or
vice versa), the reward computed by the probes disagrees with what the
simulator would actually deliver (see task_validation.py's `spawn_staging`
docstring).

A FOURTH, less obvious site also drifts if TX power changes: task_validation
imports DRONE_TX_mW/USER_TX_mW by VALUE at its own import time and computes
R_U/R_D from them ONCE (task_validation.py:102-103). Those two module
attributes are then read by probe_scripted_relay's clustering/relay-spacing
logic and by every sweep's void/reach computation (`tv.R_U`). Mutating
max_flow_reward's copies after task_validation has already imported does
NOT update tv.R_U/tv.R_D — the same synchronisation issue described above,
one level further removed. apply_config() below patches
all four sites from one call.

ONE function, ONE call, before anything is scored: apply_config(cfg).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


@dataclass
class Config:
    """Every knob this search varies. Layout knobs determine what OMNeT
    actually simulates (they change the user layout / boot); the three
    dual-site knobs (drone_tx_mW, user_tx_mW, speed_mps) determine the radio
    and kinematics both sides must agree on. `label` is free text for
    filenames/reports; it is not read by apply_config."""
    n_drones: int
    L: float                      # arena side, metres (square)
    max_clusters: int
    p_cross: float = 0.5          # ini default
    user_spread_eff: float = 25.5  # metres, EFFECTIVE (post arena-scaling) spread
                                   # ini default: base=30.0 at L=1700 -> eff=25.5
    # Metres a drone moves per RL step at |action| == 1: omnet_env.STEP_M
    # scales the unit action before it is sent, and
    # --**.host[*].mobility.maxSpeed={step_m}mps is the isotropic ceiling that
    # renormalises diagonal moves back to step_m.
    step_m: float = 20.0          # == greedy_oracle.STEP_M == omnet_env.STEP_M
    drone_tx_mW: float = 200.0
    user_tx_mW: float = 10.0
    ep_steps: int = 1000
    n_users: int = 17
    # "cluster" deals drones round-robin onto the user-cluster CENTROIDS, i.e.
    # it hands the policy a near-optimal placement at t=0 -- measured over 144
    # cells, ZERO had reward(spawn)/reward(best) <= 0.6, so nothing is learnable
    # under it. "staging" puts them all on the largest cluster and is the
    # default here for that reason. "ini" = legacy fixed initialX/Y.
    drone_spawn_mode: str = "staging"
    # Step at which the jammer freezes and starts emitting. Larger than
    # ep_steps == jammer never fires. Off by default: it deletes ~85% of the
    # reward on some seeds and nothing on others, with zero gradient either
    # side of the switch, and it moves at 20 m/step so it cannot be evaded.
    jammer_activate: int = 999999
    label: str = ""


def user_spread_base(target_eff: float, L: float) -> float:
    """Invert RLBridge.cc:415's `sp = userSpread_ * (arenaWidth_/2000.0)`
    (arenaWidth_ == our `L`, since sweep_arena.py already overrides both to
    the same value): base = target_eff * 2000.0 / L. Pass the RESULT as
    `--*.rlBridge.userSpread=`, never target_eff itself, or the effective
    spread at this L will not be what was asked for."""
    return target_eff * 2000.0 / L


def radii_for(user_tx_mW: float, drone_tx_mW: float) -> tuple[float, float]:
    """(r_u, r_d) for arbitrary TX powers, via task_validation's OWN
    closed-form radio inversion (_max_range_m) -- NOT a second radio model.
    task_validation.R_U/R_D are cached at import time for the FROZEN 10/200
    mW; for any other TX power those two module attributes are stale (see
    module docstring) and must not be read. This helper is what re-derives
    the correct pair for a config's own TX powers, and what apply_config
    uses to patch tv.R_U/tv.R_D in place.

    Imports task_validation lazily (not at config_knobs module load) so
    setting env vars via apply_config BEFORE the first task_validation
    import (the ordering task_validation.py's own docstring insists on)
    still works for callers that import config_knobs first.
    """
    import task_validation as tv
    return tv._max_range_m(user_tx_mW), tv._max_range_m(drone_tx_mW)


def build_omnet_extra_args(cfg: Config) -> str:
    """The full $OMNET_EXTRA_ARGS string for this config -- layout knobs
    (arena/clusters/pCross/userSpread) AND the OMNeT-side half of the three
    dual-site knobs (TX power, speed), plus maxSteps. Appended LAST by
    omnet_env.py's _launch_omnet, so this always wins over the ini.
    """
    base = user_spread_base(cfg.user_spread_eff, cfg.L)
    return (
        f"--*.rlBridge.arenaWidth={cfg.L} "
        f"--*.rlBridge.arenaHeight={cfg.L} "
        f"--**.mobility.constraintAreaMaxX={cfg.L}m "
        f"--**.mobility.constraintAreaMaxY={cfg.L}m "
        f"--*.rlBridge.maxClusters={cfg.max_clusters} "
        f"--*.rlBridge.pCross={cfg.p_cross} "
        f"--*.rlBridge.userSpread={base} "
        f"--*.host[*].wlan[*].radio.transmitter.power="
        # EVAL PATH twin of om2g_config: control-plane spreading raises the
        # simulator's drone TX only. cfg.drone_tx_mW still feeds the LP.
        f'{cfg.drone_tx_mW * (10.0 ** (10.0 * __import__("math").log10(max(1.0, float(os.environ.get("OM2G_CTRL_SPREADING", "1.0")))) / 10.0)):.6g}mW '
        f"--*.user[*].wlan[*].radio.transmitter.power={cfg.user_tx_mW}mW "
        f"--**.host[*].mobility.maxSpeed={cfg.step_m}mps "
        f"--*.numUsers={cfg.n_users} "
        f"--*.jammer.mobility.activate={cfg.jammer_activate} "
        # droneSpawnMode is a STRING parameter. The quotes must reach OMNeT, or
        # it parses the bare word as an expression and dies at network setup
        # with "has no parameter named 'staging'". This string is consumed by
        # omnet_env._launch_omnet via shlex.split(), which STRIPS ordinary
        # quotes -- hence the backslashes. Export it from a shell in SINGLE
        # quotes so bash leaves them alone.
        f'--*.rlBridge.droneSpawnMode=\\"{cfg.drone_spawn_mode}\\" '
        # $OM2G_SYNC_INTERVAL_S: control-plane send rate. THIS BUILDER IS THE
        # EVAL PATH; om2g_config.om2g_extra_args is the training path, and this
        # module's own docstring requires BOTH to emit every world parameter,
        # so the two paths always build the same world.
        # Charged from the same constant (om2g_track divides every control-plane
        # charge by it), so sending faster is billed proportionally and
        # repetition can never be free.
        f"--*.host[*].app[0].beaconInterval="
        # ARM-CONDITIONAL, as in om2g_config.om2g_extra_args: the nocontrol
        # arm gets beaconInterval=3600s (it never transmits within an episode),
        # every other arm the sync interval. A command-line option overrides
        # the ini, so the value is passed explicitly on this EVAL path too.
        f'{3600.0 if os.environ.get("OM2G_ARM") == "nocontrol" else float(os.environ.get("OM2G_SYNC_INTERVAL_S", "1.0"))}s '
        f"--*.rlBridge.maxSteps={cfg.ep_steps}"
    )


def apply_python_side(cfg: Config) -> tuple[float, float]:
    """Set the Python-side halves of the three dual-site knobs, PLUS
    task_validation's derived R_U/R_D (see module docstring's "fourth site").
    Returns the freshly-computed (r_u, r_d) so the caller can assert/report
    them without a second lookup.

    Sets BOTH:
      (a) the module attributes directly (max_flow_reward.DRONE_TX_mW/
          USER_TX_mW, greedy_oracle.STEP_M) -- takes effect immediately for
          any IN-PROCESS caller (Tier A/B: one Python process, one or more
          OMNeT subprocesses it launches and scores).
      (b) $OMNET_DRONE_TX_mW / $OMNET_USER_TX_mW / $OMNET_STEP_M -- these do
          nothing to THIS process (already covered by (a)) but are what a
          Stage-2 training run's `python run.py` SUBPROCESS reads at ITS OWN
          import of max_flow_reward.py / greedy_oracle.py, since in-process
          attribute mutation cannot cross a process boundary. Without (b),
          a Stage-2 job would drift the exact way this module's own
          docstring warns against, just one hop further downstream.

    Safe to call whether or not task_validation has been imported yet: if
    not, only max_flow_reward / greedy_oracle are patched here, and
    task_validation will compute its OWN R_U/R_D correctly at its own import
    time (which reads the SAME max_flow_reward globals this just set) as
    long as this function ran first -- see the module ordering note in
    task_validation.py / sweep_arena.py. If task_validation is ALREADY
    imported (e.g. by a caller that imported it before calling this), its
    module-level R_U/R_D are patched explicitly here too, since a stale
    import-time value would otherwise survive untouched.
    """
    import sys
    import max_flow_reward as mfr
    import greedy_oracle as go
    import signal_baselines as sb

    mfr.DRONE_TX_mW = float(cfg.drone_tx_mW)
    mfr.USER_TX_mW = float(cfg.user_tx_mW)
    go.STEP_M = float(cfg.step_m)
    # signal_baselines is the THIRD site of the step_m knob. It is the
    # no-reward greedy the eval harness actually runs, so its probe distance
    # must match the configured step.
    sb.STEP_M = float(cfg.step_m)

    os.environ["OMNET_DRONE_TX_mW"] = str(float(cfg.drone_tx_mW))
    os.environ["OMNET_USER_TX_mW"] = str(float(cfg.user_tx_mW))
    os.environ["OMNET_STEP_M"] = str(float(cfg.step_m))

    r_u, r_d = radii_for(cfg.user_tx_mW, cfg.drone_tx_mW)

    tv = sys.modules.get("task_validation")
    if tv is not None:
        tv.R_U = r_u
        tv.R_D = r_d

    return r_u, r_d


def apply_config(cfg: Config, *, set_num_drones_env: bool = True) -> dict:
    """THE single entry point. Call this BEFORE the first task_validation
    import (or immediately after, if task_validation is already imported --
    apply_python_side handles both). Sets:
      - os.environ["OMNET_NUM_DRONES"]   (read once at omnet_env.py import time
                                           -- see its own "Never silently
                                           falls back" comment; set this before
                                           omnet_env.py is ever imported)
      - os.environ["OMNET_EXTRA_ARGS"]   (read fresh per-launch; safe any time
                                           before the NEXT _launch_omnet call)
      - max_flow_reward.DRONE_TX_mW / USER_TX_mW
      - greedy_oracle.STEP_M
      - task_validation.R_U / R_D (if already imported)

    Returns a dict of everything that was set, for logging / the consistency
    test (test_config_knobs_dual_site.py).
    """
    extra_args = build_omnet_extra_args(cfg)
    # MERGE, never replace. This builder knows only the knobs Config models;
    # the OM2G track adds its own (userSpawnMode, dumbbellD, maxNeighbourSep,
    # the beacon sizing) by exporting $OMNET_EXTRA_ARGS from om2g_config before
    # this process starts, and a plain assignment would drop all of them. Keep
    # any flag the caller set that this builder does not emit; ours still wins
    # wherever both define the same flag.
    _flag = lambda tok: tok.split("=", 1)[0]
    mine = {_flag(t) for t in extra_args.split() if t.startswith("--")}
    inherited = [t for t in os.environ.get("OMNET_EXTRA_ARGS", "").split()
                 if t.startswith("--") and _flag(t) not in mine]
    if inherited:
        extra_args = " ".join(inherited) + " " + extra_args
    if set_num_drones_env:
        os.environ["OMNET_NUM_DRONES"] = str(cfg.n_drones)
    os.environ["OMNET_EXTRA_ARGS"] = extra_args
    r_u, r_d = apply_python_side(cfg)
    return {
        "extra_args": extra_args,
        "n_drones": cfg.n_drones,
        "drone_tx_mW": cfg.drone_tx_mW,
        "user_tx_mW": cfg.user_tx_mW,
        "step_m": cfg.step_m,
        "user_spread_base": user_spread_base(cfg.user_spread_eff, cfg.L),
        "r_u": r_u,
        "r_d": r_d,
    }


# ── Consistency check (used by test_config_knobs_dual_site.py) ─────────────

_MW_RE = re.compile(r"radio\.transmitter\.power=([0-9.eE+-]+)mW")
_MPS_RE = re.compile(r"mobility\.maxSpeed=([0-9.eE+-]+)mps")


def extract_dual_site_values_from_extra_args(extra_args: str) -> dict:
    """Pull the OMNeT-side numeric values for the three dual-site knobs back
    out of an $OMNET_EXTRA_ARGS string, by regex on the exact flag shapes
    build_omnet_extra_args emits. Two power flags exist (host=drone,
    user=user); distinguished by which side of the `power=` match the
    `host[*]`/`user[*]` selector sits on.
    """
    out: dict[str, float] = {}
    for m in re.finditer(r"--\*\.(host|user)\[\*\]\.wlan\[\*\]\.radio\.transmitter\.power=([0-9.eE+-]+)mW", extra_args):
        who, val = m.group(1), float(m.group(2))
        out["drone_tx_mW" if who == "host" else "user_tx_mW"] = val
    m = _MPS_RE.search(extra_args)
    if m:
        out["step_m"] = float(m.group(1))
    return out


def assert_dual_site_consistent(cfg: Config) -> dict:
    """Call apply_config(cfg), then verify every dual-site knob agrees
    between the OMNET_EXTRA_ARGS string that will reach the simulator and
    the Python-oracle module globals that will score it. Raises
    AssertionError with a full dump on any mismatch. Returns the applied
    dict (see apply_config) on success -- this is the load-bearing function
    behind smoke test 4 / test_config_knobs_dual_site.py.
    """
    import max_flow_reward as mfr
    import greedy_oracle as go

    applied = apply_config(cfg)
    from_args = extract_dual_site_values_from_extra_args(applied["extra_args"])

    env_names = {"drone_tx_mW": "OMNET_DRONE_TX_mW", "user_tx_mW": "OMNET_USER_TX_mW",
                "step_m": "OMNET_STEP_M"}
    checks = {
        "drone_tx_mW": (from_args.get("drone_tx_mW"), mfr.DRONE_TX_mW, cfg.drone_tx_mW),
        "user_tx_mW": (from_args.get("user_tx_mW"), mfr.USER_TX_mW, cfg.user_tx_mW),
        "step_m": (from_args.get("step_m"), go.STEP_M, cfg.step_m),
    }
    mismatches = []
    for name, (from_omnet, from_python, from_cfg) in checks.items():
        if from_omnet is None:
            mismatches.append(f"{name}: could not find a value in OMNET_EXTRA_ARGS at all")
            continue
        from_env = float(os.environ.get(env_names[name], "nan"))
        if not (abs(from_omnet - from_python) < 1e-9 and abs(from_python - from_cfg) < 1e-9
               and abs(from_env - from_cfg) < 1e-9):
            mismatches.append(
                f"{name}: OMNeT-side={from_omnet}  python-oracle-side={from_python}  "
                f"${env_names[name]}={from_env}  cfg={from_cfg}  -- DUAL-SITE DESYNC"
            )
    if mismatches:
        raise AssertionError(
            "config_knobs dual-site consistency check FAILED:\n  "
            + "\n  ".join(mismatches)
            + f"\nOMNET_EXTRA_ARGS={applied['extra_args']!r}"
        )
    return applied
