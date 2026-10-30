"""Metric reducer registry for PADRE evaluation.

HOW TO ADD A METRIC
-------------------
1. Write a reducer function with signature:
       def my_metric(steps: list[dict]) -> tuple[str, float]
   `steps` is the per-step log for one episode (list of dicts, keys below).
   Return (metric_name, scalar_value).

2. Append the function to REGISTRY at the bottom of this file.

That is the entire change required.  The runner collects the per-step logs;
`compute_episode_metrics` iterates REGISTRY; aggregation in evaluate.py
iterates the returned dicts.

Per-step log dict keys
----------------------
  step          int    env step index (0-based)
  reward        float  scalar joint reward for this step
  throughput    float  network throughput in bits/s (PADRE only; absent for
                        OMNeT episodes -- see runner._run_single_episode)
  max_throughput float theoretical maximum throughput in bits/s, sum of
                        offered loads (PADRE only; absent for OMNeT)
  terminated    bool
  truncated     bool
"""
from __future__ import annotations
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Reducer functions
# ---------------------------------------------------------------------------

def episode_reward_total(steps: list[dict]) -> tuple[str, float]:
    return "episode_reward_total", float(np.sum([s["reward"] for s in steps]))


def mean_throughput_mbps(steps: list[dict]) -> tuple[str, float]:
    """Mean of PADRE's network.getThroughput() over the episode.

    Returns NaN when 'throughput' is absent (OMNeT episodes -- there is no
    equivalent single-network API; see runner._run_single_episode), same
    convention as fraction_committed_steps below.
    """
    tps = [s["throughput"] / 1e6 for s in steps if "throughput" in s]
    return "mean_throughput_mbps", float(np.mean(tps)) if tps else float("nan")


def mean_utilization(steps: list[dict]) -> tuple[str, float]:
    """Mean of throughput / max_throughput over the episode.

    Returns NaN when 'max_throughput' is absent (OMNeT episodes) rather than
    a fabricated 0.0 -- see mean_throughput_mbps.
    """
    utils = []
    for s in steps:
        if "max_throughput" not in s:
            continue
        mx = s["max_throughput"]
        utils.append(s["throughput"] / mx if mx > 1e-9 else 0.0)
    return "mean_utilization", float(np.mean(utils)) if utils else float("nan")


def episode_length(steps: list[dict]) -> tuple[str, float]:
    return "episode_length", float(len(steps))


def episode_outcome(steps: list[dict]) -> tuple[str, float]:
    """1.0 = terminated (goal reached), 0.0 = truncated (time limit)."""
    if steps and steps[-1]["terminated"]:
        return "episode_outcome", 1.0
    return "episode_outcome", 0.0


def fraction_committed_steps(steps: list[dict]) -> tuple[str, float]:
    """Fraction of steps where the heuristic reported a committed decision.

    Returns NaN when 'committed' is absent (trained / random policies don't set it).
    """
    committed_vals = [s["committed"] for s in steps if "committed" in s]
    if not committed_vals:
        return "fraction_committed_steps", float("nan")
    return "fraction_committed_steps", float(np.mean(committed_vals))


# ---------------------------------------------------------------------------
# Registry — add new reducers here
# ---------------------------------------------------------------------------

REGISTRY: list[Callable[[list[dict]], tuple[str, float]]] = [
    episode_reward_total,
    mean_throughput_mbps,
    mean_utilization,
    episode_length,
    episode_outcome,
    fraction_committed_steps,
]


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def compute_episode_metrics(steps: list[dict]) -> dict[str, float]:
    """Apply all REGISTRY reducers to one episode's per-step log."""
    return dict(fn(steps) for fn in REGISTRY)


# Keys added to the per-episode dict that are metadata, not metrics
_METADATA_KEYS = {"seed"}


def aggregate(per_episode: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    """Compute mean ± std over all seeded episodes for each REGISTRY metric.

    Returns {metric_name: (mean, std)}.  Metadata keys (seed, etc.) are excluded.
    """
    if not per_episode:
        return {}
    all_keys = [k for k in per_episode[0] if k not in _METADATA_KEYS]
    result = {}
    for key in all_keys:
        vals = np.array([ep[key] for ep in per_episode], dtype=float)
        result[key] = (float(np.mean(vals)), float(np.std(vals)))
    return result
