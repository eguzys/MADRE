"""DOES TRANSMITTING HELP AT ALL? (vs sending nothing)

stop_check answered a different question -- whether the data compresses well
(pattern latent vs average latent). This asks the one that matters: does a drone
that HEARS OTHER DRONES' summaries place itself better than one that only knows
its own patch?

The key asymmetry: a drone hears the traffic of users it
can decode, i.e. its OWN cluster. It cannot hear another cluster's traffic at
all -- that is 1465 m away, far outside the 158 m user radio. So the only way to
learn "cluster 3 just went hot" is for a drone sitting there to say so.

Arms, all using the SAME pattern-aware predictor, differing ONLY in what they
can see:
  oracle   true future rates of every cluster        -- upper bound
  talking  predicted rates of every cluster          -- peers reported theirs
  silent   predicted rate of its OWN cluster only; for the others it must fall
           back on the prior (the episode mean), which is all it can know
  static   nobody moves

Scoring: assign drones to clusters to maximise served demand, where a cluster
with `n` drones serves min(rate, n * RADIO_CAP_PKT). Then score that assignment
against the TRUE future rates. Capacity saturation is what makes the assignment
matter -- without it every drone would just sit on the biggest cluster.
"""
from __future__ import annotations

import math
import statistics
import sys

sys.path.insert(0, "om2g")
from traffic_process import TrafficProcess, PKT_RATE_HZ
from stop_check import _residual_frac, HORIZON_S, WINDOW_S

RADIO_CAP_PKT = float(__import__("os").environ.get("OM2G_RADIO_CAP_PKT", 260.0))


def _predict(tp, uc, t, c):
    act = [call for call in tp.active(float(t)) if uc[call.user] == c]
    persist = PKT_RATE_HZ * sum(_residual_frac(call.age_at(float(t)), HORIZON_S) for call in act)
    recent = sum(1 for call in tp.calls if uc[call.user] == c
                 and float(t) - WINDOW_S <= call.start_s < float(t))
    return persist + PKT_RATE_HZ * (recent / WINDOW_S) * HORIZON_S * 0.5


def _assign(rates, nd, ncl):
    """Greedy: each drone in turn to the cluster with the most UNSERVED demand."""
    served = [0.0] * ncl
    counts = [0] * ncl
    for _ in range(nd):
        best = max(range(ncl), key=lambda c: max(0.0, rates[c] - served[c]))
        counts[best] += 1
        served[best] = min(rates[best], counts[best] * RADIO_CAP_PKT)
    return counts


def _score(counts, truth, ncl):
    return sum(min(truth[c], counts[c] * RADIO_CAP_PKT) for c in range(ncl))


def run(seeds=range(30), n_users=40, ncl=4, nd=8, verbose=True):
    tot = {k: [] for k in ("oracle", "talking", "silent", "static")}
    for seed in seeds:
        uc = [i % ncl for i in range(n_users)]
        tp = TrafficProcess(seed=seed, n_users=n_users, user_cluster=uc, horizon_s=250.0)
        ep_mean = {c: statistics.mean(tp.cluster_rate(t, c) for t in range(0, 250, 10)) for c in range(ncl)}
        for t in range(40, 200, 20):
            truth = [tp.future_cluster_rate(float(t), c, HORIZON_S) for c in range(ncl)]
            pred = [_predict(tp, uc, t, c) for c in range(ncl)]
            # one drone per cluster is resident and reports; a SILENT swarm has
            # each drone knowing only its own cluster, prior elsewhere.
            # SILENT: each drone decides ALONE. It knows its own
            # cluster's pattern and nothing else, so it cannot compute a
            # swarm-wide assignment -- doing so would assume every drone sees
            # the same thing and agrees, which is precisely what a message buys.
            # Its only rational rule: hold station if its own cluster looks at
            # least as busy as the prior expectation elsewhere, else move to the
            # best cluster it can name -- which, knowing nothing, is the prior.
            homes = [c % ncl for c in range(nd)]
            counts_sil = [0] * ncl
            for d in range(nd):
                h = homes[d]
                best_other = max((c for c in range(ncl) if c != h),
                                 key=lambda c: ep_mean[c])
                counts_sil[h if pred[h] >= ep_mean[best_other] else best_other] += 1
            sil_scores = [_score(counts_sil, truth, ncl)]
            tot["oracle"].append(_score(_assign(truth, nd, ncl), truth, ncl))
            tot["talking"].append(_score(_assign(pred, nd, ncl), truth, ncl))
            tot["silent"].append(statistics.mean(sil_scores))
            tot["static"].append(_score([nd // ncl] * ncl, truth, ncl))
    m = {k: statistics.mean(v) for k, v in tot.items()}
    if verbose:
        print("served demand (packets/s) under each information regime\n")
        for k in ("static", "silent", "talking", "oracle"):
            print("  %-9s %8.1f" % (k, m[k]))
        gap = m["talking"] - m["silent"]
        head = m["oracle"] - m["silent"]
        print("\n  talking - silent = %+.1f  (%.1f%% of the oracle's headroom)"
              % (gap, 100 * gap / head if head > 0 else float("nan")))
        print("  VERDICT: %s" % ("transmitting HELPS" if gap > 0 else "transmitting does NOT help"))
    return m


if __name__ == "__main__":
    run()
