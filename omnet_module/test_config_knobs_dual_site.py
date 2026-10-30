#!/usr/bin/env python3
"""
test_config_knobs_dual_site.py — SMOKE TEST 4 (see the task brief): asserts
config_knobs.apply_config leaves the OMNeT-bound $OMNET_EXTRA_ARGS string and
the Python-oracle module constants (max_flow_reward.DRONE_TX_mW/USER_TX_mW,
greedy_oracle.STEP_M) IN AGREEMENT, for several configs including ones that
deviate from the frozen 200mW/10mW/20mps defaults -- the case that actually
exercises the dual-site hazard (at the frozen defaults everything already
"agrees" trivially, since nothing changed).

Also checks the "fourth site" named in config_knobs.py's docstring:
task_validation.R_U/R_D track a NON-default TX power once task_validation is
imported.

No OMNeT, no LP solve -- pure Python, runs in well under a second.

Usage:
    python3 test_config_knobs_dual_site.py
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_knobs import Config, assert_dual_site_consistent, radii_for  # noqa: E402


def _check(name: str, cond: bool, detail: str) -> bool:
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}: {detail}")
    return cond


def main() -> None:
    print("=" * 78)
    print("config_knobs dual-site consistency -- SMOKE TEST 4")
    print("=" * 78)
    all_ok = True

    # ── Case 1: the frozen default config ------------------------------------
    cfg_default = Config(n_drones=5, L=1700.0, max_clusters=2)
    applied = assert_dual_site_consistent(cfg_default)
    all_ok &= _check(
        "default config consistent",
        True,  # assert_dual_site_consistent already raised if not
        f"r_u={applied['r_u']:.1f}m r_d={applied['r_d']:.1f}m "
        f"extra_args={applied['extra_args']!r}",
    )
    all_ok &= _check(
        "default radii match documented facts (r_u~157.8, r_d~523.1)",
        abs(applied["r_u"] - 157.8) < 1.0 and abs(applied["r_d"] - 523.1) < 1.0,
        f"r_u={applied['r_u']:.2f}  r_d={applied['r_d']:.2f}",
    )

    # ── Case 2: a config that DEVIATES on all three dual-site knobs ----------
    # This is the case that actually exercises the hazard: if apply_config
    # forgot to patch one of the three sites, this is where it would show.
    cfg_deviant = Config(n_drones=4, L=1200.0, max_clusters=3, p_cross=0.1,
                         user_spread_eff=60.0, step_m=40.0,
                         drone_tx_mW=500.0, user_tx_mW=50.0, ep_steps=1000)
    applied2 = assert_dual_site_consistent(cfg_deviant)
    import max_flow_reward as mfr
    import greedy_oracle as go
    all_ok &= _check(
        "deviant config: max_flow_reward.DRONE_TX_mW matches cfg",
        mfr.DRONE_TX_mW == 500.0, f"mfr.DRONE_TX_mW={mfr.DRONE_TX_mW}")
    all_ok &= _check(
        "deviant config: max_flow_reward.USER_TX_mW matches cfg",
        mfr.USER_TX_mW == 50.0, f"mfr.USER_TX_mW={mfr.USER_TX_mW}")
    all_ok &= _check(
        "deviant config: greedy_oracle.STEP_M matches cfg",
        go.STEP_M == 40.0, f"go.STEP_M={go.STEP_M}")
    all_ok &= _check(
        "deviant config: OMNET_EXTRA_ARGS carries the SAME three numbers",
        "power=500.0mW" in os.environ["OMNET_EXTRA_ARGS"]
        and "power=50.0mW" in os.environ["OMNET_EXTRA_ARGS"]
        and "maxSpeed=40.0mps" in os.environ["OMNET_EXTRA_ARGS"],
        os.environ["OMNET_EXTRA_ARGS"],
    )
    r_u2, r_d2 = radii_for(50.0, 500.0)
    all_ok &= _check(
        "deviant config: r_u/r_d actually differ from the frozen defaults",
        abs(r_u2 - applied["r_u"]) > 1.0 and abs(r_d2 - applied["r_d"]) > 1.0,
        f"default (r_u,r_d)=({applied['r_u']:.1f},{applied['r_d']:.1f})  "
        f"deviant (r_u,r_d)=({r_u2:.1f},{r_d2:.1f})",
    )

    # ── Case 2b: a FRESH SUBPROCESS (the Stage-2 training case) picks up the
    # env-var side of the CURRENT (deviant) config without any in-process
    # mutation -- this is the case in-process attribute patching alone
    # cannot cover, and MUST run before anything below re-applies a
    # different config and overwrites the env vars again.
    import subprocess
    out = subprocess.run(
        [sys.executable, "-c",
         "import max_flow_reward as mfr, greedy_oracle as go; "
         "print(mfr.DRONE_TX_mW, mfr.USER_TX_mW, go.STEP_M)"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=dict(os.environ), capture_output=True, text=True, timeout=30,
    )
    sub_vals = out.stdout.strip().split()
    all_ok &= _check(
        "fresh subprocess inherits dual-site knobs via env vars (Stage-2 case)",
        out.returncode == 0 and sub_vals == ["500.0", "50.0", "40.0"],
        f"subprocess stdout={out.stdout!r} stderr={out.stderr[-300:]!r}",
    )

    # ── Case 3: the "fourth site" -- task_validation.R_U/R_D track it too ----
    # Import task_validation HERE (after both apply_config calls above), the
    # order a Tier-B sweep script would actually use if it deviates from the
    # frozen defaults before task_validation is first imported.
    import task_validation as tv
    all_ok &= _check(
        "task_validation.R_U matches the deviant config's r_u (not the frozen default)",
        math.isclose(tv.R_U, r_u2, rel_tol=1e-6),
        f"tv.R_U={tv.R_U:.2f}  expected={r_u2:.2f}",
    )
    all_ok &= _check(
        "task_validation.R_D matches the deviant config's r_d (not the frozen default)",
        math.isclose(tv.R_D, r_d2, rel_tol=1e-6),
        f"tv.R_D={tv.R_D:.2f}  expected={r_d2:.2f}",
    )

    # ── Case 4: re-applying the default config re-syncs everything back -----
    applied3 = assert_dual_site_consistent(cfg_default)
    all_ok &= _check(
        "re-applying the default config resyncs task_validation.R_U/R_D too",
        math.isclose(tv.R_U, applied3["r_u"], rel_tol=1e-6)
        and math.isclose(tv.R_D, applied3["r_d"], rel_tol=1e-6),
        f"tv.R_U={tv.R_U:.2f}  tv.R_D={tv.R_D:.2f}",
    )

    # ── Case 5: user_spread_base inverts correctly ---------------------------
    from config_knobs import user_spread_base
    base = user_spread_base(25.5, 1700.0)
    eff_back = base * (1700.0 / 2000.0)
    all_ok &= _check(
        "user_spread_base inverts RLBridge.cc:415's sp = base * (L/2000)",
        abs(eff_back - 25.5) < 1e-9,
        f"base={base:.4f} -> eff={eff_back:.4f} (target 25.5)",
    )

    print("=" * 78)
    print("ALL PASS" if all_ok else "ONE OR MORE CHECKS FAILED")
    print("=" * 78)
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
