"""THE STOP CHECK. Training-free.

High-volume data adds nothing if a simple average captures everything the
decision needs. This asks whether the traffic process carries more than its
average, before any training run is spent.

The decision a drone makes is "which cluster will be busy over the next ~30 s"
(its travel time). Four predictors of that quantity:

  oracle    the true future                       -- upper bound
  pattern   age-aware: each active call's residual life under the log-normal,
            i.e. it USES per-call age                                   <- the message
  average   mean rate over the last 25 s          -- a rate-average latent
  static    the episode mean                      -- knows nothing

GATE:
    skill(pattern) - skill(average) >= 15% of skill(average)
where skill = reduction in RMSE against the static predictor. If the average is
nearly as good as the pattern, our data fails criterion (b) exactly as spectrum
sampling does, and we stop and say so rather than spend a training run.
"""
from __future__ import annotations

import math
import statistics
import sys

sys.path.insert(0, "om2g")
from traffic_process import TrafficProcess, CALL_MEDIAN_S, CALL_SIGMA, PKT_RATE_HZ

HORIZON_S = 30.0      # a drone's travel time between clusters
WINDOW_S = 25.0       # what the "average" latent gets to see


def _residual_frac(age_s: float, horizon_s: float) -> float:
    """P(a log-normal call of current age `age` is still up, averaged over the
    next `horizon`). This is the quantity an average CANNOT represent: for a
    heavy tail it RISES with age."""
    mu = math.log(CALL_MEDIAN_S)
    def surv(x):
        if x <= 0:
            return 1.0
        z = (math.log(x) - mu) / CALL_SIGMA
        return 0.5 * math.erfc(z / math.sqrt(2.0))
    s_age = surv(age_s)
    if s_age <= 1e-9:
        return 0.0
    n = 10
    return sum(surv(age_s + horizon_s * k / n) / s_age for k in range(n)) / n


def run(seeds=range(20), n_users=40, n_clusters=4, horizon_ep=250.0, verbose=True):
    err = {k: [] for k in ("pattern", "average", "static")}
    for seed in seeds:
        uc = [i % n_clusters for i in range(n_users)]
        tp = TrafficProcess(seed=seed, n_users=n_users, user_cluster=uc, horizon_s=horizon_ep)
        ep_mean = {c: statistics.mean(tp.cluster_rate(t, c) for t in range(0, 250, 10))
                   for c in range(n_clusters)}
        for t in range(40, 200, 10):
            for c in range(n_clusters):
                truth = tp.future_cluster_rate(float(t), c, HORIZON_S)
                # pattern-aware: sum residual-life-weighted active calls
                # Pattern-aware = the two things a packet stream reveals that a
                # rate average cannot separate: how long existing calls will
                # persist (age-dependent under a heavy tail), AND how fast new
                # ones are arriving right now (whether this cluster is hot).
                act = [call for call in tp.active(float(t)) if uc[call.user] == c]
                persist = PKT_RATE_HZ * sum(
                    _residual_frac(call.age_at(float(t)), HORIZON_S) for call in act)
                recent = sum(1 for call in tp.calls
                             if uc[call.user] == c
                             and float(t) - WINDOW_S <= call.start_s < float(t))
                arr_hz = recent / WINDOW_S
                # new calls arriving in the horizon, each up ~half the horizon
                arrivals = PKT_RATE_HZ * arr_hz * HORIZON_S * 0.5
                pat = persist + arrivals
                # average over the trailing window
                avg = statistics.mean(tp.cluster_rate(float(t) - k, c)
                                      for k in range(int(WINDOW_S)))
                err["pattern"].append((pat - truth) ** 2)
                err["average"].append((avg - truth) ** 2)
                err["static"].append((ep_mean[c] - truth) ** 2)
    rmse = {k: math.sqrt(statistics.mean(v)) for k, v in err.items()}
    skill = {k: 1.0 - rmse[k] / rmse["static"] for k in ("pattern", "average")}
    gain = (skill["pattern"] - skill["average"]) / skill["average"] if skill["average"] > 0 else float("inf")
    if verbose:
        print("predicting cluster packet-rate over the next %.0f s\n" % HORIZON_S)
        for k in ("static", "average", "pattern"):
            s = skill.get(k)
            print("  %-8s RMSE %8.1f   skill %s" % (k, rmse[k], "  --" if s is None else "%+.3f" % s))
        print("\n  pattern beats average by %.1f%% of its skill" % (100 * gain))
        print("  GATE: >= 15%%  ->  %s" % ("PASS" if gain >= 0.15 else "FAIL"))
    return gain, rmse, skill


if __name__ == "__main__":
    run()
