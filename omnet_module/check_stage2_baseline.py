#!/usr/bin/env python3
"""
check_stage2_baseline.py — Tier C's pass/fail check: does this run's eval
episode return CROSS and STAY above the config's own measured P0v baseline
(never "above zero", never P2/P1 -- see the Tier-C brief in the task that
added this file). Called by stage2_train.sbatch after training finishes.

WHERE THE NUMBER COMES FROM: BenchMARL's default loggers are [csv, wandb]
(benchmarl/conf/experiment/base_experiment.yaml); the csv logger
(torchrl.record.loggers.csv.CSVExperiment) writes one HEADERLESS
"<step>,<value>" file per scalar at
    <save_folder>/<exp_name>/<exp_name>/scalars/<metric_name>.csv
Verified against real runs in this repo's src/outputs/ tree: the eval-return
metric is named "eval_reward_episode_reward_mean" for a single reward group,
or "eval_<group>_reward_episode_reward_mean" (e.g. "eval_drone_reward_...")
if the task defines a named reward group. This script globs for BOTH shapes
and picks whichever one exists -- if your task/model config uses a different
group name, override --metric-glob explicitly rather than trusting the
default silently.

"Cross AND STAY above": we require the mean of the LAST --window eval points
to exceed the baseline, not a single lucky evaluation -- one high point could
be noise (evaluation_episodes defaults to only 2 in conf/experiment/omnet.yaml,
by design, to bound the ~35h eval-time cost noted there; Stage 2 runs inherit
that unless the launcher is edited to raise it, which is a known caveat, see
stage2_train.sbatch's header).

Usage:
    python3 check_stage2_baseline.py --save-dir out/.../omnet_st_n4_encfrozen_stage2_x \\
        --p0v-baseline 68.3 [--window 3] [--metric-glob "eval*reward*episode_reward_mean.csv"]

Exit code 0 = PASS (crossed and stayed above baseline), 1 = FAIL/inconclusive.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys


def _read_csv_last_n(path: str, n: int) -> list[float]:
    """CSVExperiment.add_scalar writes headerless "step,value" rows. Returns
    the last n VALUES (not steps), in chronological order."""
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    vals = []
    for ln in lines[-n:]:
        parts = ln.split(",")
        if len(parts) != 2:
            continue
        try:
            vals.append(float(parts[1]))
        except ValueError:
            continue
    return vals


def find_eval_reward_csv(save_dir: str, metric_glob: str) -> str | None:
    matches = sorted(glob.glob(os.path.join(save_dir, "**", "scalars", metric_glob),
                               recursive=True))
    return matches[-1] if matches else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save-dir", required=True, help="the training run's experiment.save_folder")
    ap.add_argument("--p0v-baseline", type=float, required=True,
                    help="THIS config's own measured P0v (sweep_config.py's median_P0v). "
                         "Baked into the generated stage2_train.sbatch by gen_stage2_job.py "
                         "-- never re-derived here.")
    ap.add_argument("--window", type=int, default=3,
                    help="require the mean of the last N eval points to exceed the "
                         "baseline (\"cross AND STAY above\", not one lucky evaluation).")
    ap.add_argument("--metric-glob", type=str,
                    default="eval*reward*episode_reward_mean.csv",
                    help="filename pattern under <save_dir>/**/scalars/ to read the "
                         "eval episode return from. See module docstring if this "
                         "does not match your task's reward-group naming.")
    ap.add_argument("--reward-scale", type=float, default=1e-4,
                    help="REWARD_SCALE applied by the RewardScaling transform in "
                         "src/benchmarl_padre/omnet_common.py. The logged metric is "
                         "the SCALED return; the baseline is in RAW oracle units, so "
                         "the logged value must be divided by this to compare. Keep "
                         "in lockstep with omnet_common.REWARD_SCALE.")
    ap.add_argument("--ep-steps", type=int, default=1000,
                    help="Episode length. The logged metric is a per-EPISODE return "
                         "while --p0v-baseline is a per-STEP number, so the logged "
                         "value must also be divided by this. Keep in lockstep with "
                         "*.rlBridge.maxSteps / the task's ep_length.")
    a = ap.parse_args()

    if not os.path.isdir(a.save_dir):
        print(f"FAIL (inconclusive): save_dir does not exist: {a.save_dir}")
        sys.exit(1)

    csv_path = find_eval_reward_csv(a.save_dir, a.metric_glob)
    if csv_path is None:
        print(f"FAIL (inconclusive): no file matching {a.metric_glob!r} under "
             f"{a.save_dir}/**/scalars/ -- see module docstring: adjust "
             f"--metric-glob if this task's reward group is named differently.")
        sys.exit(1)

    vals = _read_csv_last_n(csv_path, a.window)
    if not vals:
        print(f"FAIL (inconclusive): {csv_path} has no parseable rows.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # UNIT CONVERSION
    #
    # `mean_last` and --p0v-baseline are different quantities:
    #   * the logged metric is a per-EPISODE return, SCALED by
    #     omnet_common.REWARD_SCALE = 1e-4 (a RewardScaling transform on the
    #     collector, so every logged reward is 1e-4 x the raw oracle value);
    #   * --p0v-baseline is sweep_config's median_P0v, a RAW per-STEP number.
    # For a 1000-step episode the two differ by 1e4/1000 = 10x, so the logged
    # value is converted to raw per-step units before the comparison
    # (e.g. 11.12 logged -> 111.2 raw/step).
    # ------------------------------------------------------------------
    mean_last_raw_per_step = (mean_last := sum(vals) / len(vals)) / a.reward_scale / a.ep_steps
    passed = mean_last_raw_per_step > a.p0v_baseline
    print(f"metric file: {csv_path}")
    print(f"last {len(vals)} eval point(s) (scaled per-episode): {vals}")
    if len(vals) < a.window:
        print(f"WARNING: asked for the last {a.window} eval points but the metric "
              f"file only has {len(vals)}. If that is 1, the run's "
              f"evaluation_interval is >= its max_n_frames and the ONLY point is "
              f"the pre-training eval at frame 0 -- see conf/experiment/omnet.yaml. "
              f"Grading an untrained network is not a verdict.")
    print(f"mean of last {len(vals)}: {mean_last:.4f} scaled/episode "
          f"-> {mean_last_raw_per_step:.2f} raw/step "
          f"(/{a.reward_scale:g} /{a.ep_steps} steps)")
    print(f"P0v baseline: {a.p0v_baseline:.2f} raw/step")
    print(f"==> {'PASS' if passed else 'FAIL'}: eval return "
         f"{'stayed above' if passed else 'did NOT stay above'} the config's own P0v baseline")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
