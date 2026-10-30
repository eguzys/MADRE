"""Per-conversation traffic, generated once and consumed by BOTH the reward and
the observation builder.

WHY THIS EXISTS
A payload of 40 users x 13 features whose useful content fits in ~8 floats
makes compression trivially lossless. This module provides data that is
expensive to send and lossy to summarise.

Design criteria, and how each is met:
  (1) locally sensed   -- a drone sees packet statistics only for users whose
                          frames it can decode (`user_rssi_valid`), same gate the
                          user block already uses.
  (2) high volume      -- K=32 recent packet events x F=4 features per heard
                          user; ~10 heard users is ~5.1 kB per exchange against
                          a 32 B latent, a ~160x ratio that is real, not staged.
  (3) decision-relevant-- what a drone wants is the cluster's rate over the next
                          15-30 steps (its travel time), which is a function of
                          the active-call SET and their AGES.
  (4) right timescale  -- travel 15-30 s vs call median 60 s vs epoch ~100 s.
                          Talk-spurt dynamics (~1 s) are deliberately NOT made
                          decision-relevant (too fast to act on).

THE LOAD-BEARING CHOICE: LOG-NORMAL DURATIONS
Call lengths are heavy-tailed, so residual life INCREASES with call age. The
sufficient statistic therefore includes each call's age, which any rate average
discards. With exponential durations the memorylessness property makes the
average very nearly sufficient and the compression question is trivial -- so the
distribution here is not a cosmetic realism choice, it is what makes the
question well-posed. `stop_check.py` measures that this is true before any
training run is spent.

Determinism: everything derives from (episode_seed, step). The reward and the
observation therefore see bit-identical traffic, and offline replay stays exact.
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field

# ── the process, in SECONDS (one step == one second at OM2G's 1 Hz decisions) ──
CALL_MEDIAN_S = float(os.environ.get("OM2G_CALL_MEDIAN_S", 60.0))
CALL_SIGMA = float(os.environ.get("OM2G_CALL_SIGMA", 1.0))     # log-normal shape
EPOCH_MEAN_S = float(os.environ.get("OM2G_EPOCH_MEAN_S", 100.0))
HOT_MULT = float(os.environ.get("OM2G_HOT_MULT", 4.0))         # hot-cluster arrival boost
BASE_ARRIVAL_HZ = float(os.environ.get("OM2G_ARRIVAL_HZ", 0.02))  # per user, cold
PKT_RATE_HZ = float(os.environ.get("OM2G_PKT_RATE_HZ", 14.8))  # MELPE voice frames/s


@dataclass
class Call:
    user: int
    partner: int
    start_s: float
    dur_s: float

    def active_at(self, t: float) -> bool:
        return self.start_s <= t < self.start_s + self.dur_s

    def age_at(self, t: float) -> float:
        return max(0.0, t - self.start_s)


@dataclass
class TrafficProcess:
    """Calls for one episode. Built once, queried per step."""
    seed: int
    n_users: int
    user_cluster: list          # user -> cluster id
    horizon_s: float
    calls: list = field(default_factory=list)

    def __post_init__(self):
        rng = random.Random((self.seed * 2654435761) & 0xFFFFFFFF)
        n_cl = max(1, (max(self.user_cluster) + 1) if self.user_cluster else 1)

        # Per-cluster hot/cold epochs: a semi-Markov chain over which cluster is
        # hot. This is the DEMAND CHURN -- remembered load goes WRONG, which is
        # what makes a delivered update worth re-delivering.
        epochs = []
        t = 0.0
        while t < self.horizon_s:
            dur = rng.expovariate(1.0 / EPOCH_MEAN_S)
            epochs.append((t, min(t + dur, self.horizon_s), rng.randrange(n_cl)))
            t += dur
        self.epochs = epochs

        def hot_at(ts):
            for a, b, c in epochs:
                if a <= ts < b:
                    return c
            return epochs[-1][2] if epochs else 0

        # Poisson call arrivals per user, intensity modulated by whether that
        # user's cluster is currently hot.
        for u in range(self.n_users):
            cu = self.user_cluster[u] if u < len(self.user_cluster) else 0
            ts = 0.0
            while True:
                lam = BASE_ARRIVAL_HZ * (HOT_MULT if hot_at(ts) == cu else 1.0)
                ts += rng.expovariate(lam)
                if ts >= self.horizon_s:
                    break
                partner = rng.randrange(self.n_users)
                if partner == u:
                    continue
                # log-normal: median CALL_MEDIAN_S, heavy right tail
                dur = math.exp(math.log(CALL_MEDIAN_S) + CALL_SIGMA * rng.gauss(0.0, 1.0))
                self.calls.append(Call(u, partner, ts, dur))

    # ── queries ──────────────────────────────────────────────────────────────
    def active(self, t: float):
        return [c for c in self.calls if c.active_at(t)]

    def cluster_rate(self, t: float, cluster: int) -> float:
        """Packets/s originating in `cluster` at time t. The reward's demand."""
        return PKT_RATE_HZ * sum(
            1 for c in self.active(t)
            if c.user < len(self.user_cluster) and self.user_cluster[c.user] == cluster)

    def future_cluster_rate(self, t: float, cluster: int, horizon_s: float) -> float:
        """Mean rate over [t, t+horizon]: what a drone deciding to fly wants to
        know, and what a rate AVERAGE over the past cannot supply."""
        n = max(1, int(horizon_s))
        return sum(self.cluster_rate(t + k, cluster) for k in range(n)) / n

    def user_events(self, t: float, user: int, k: int = 32):
        """The last k packet events this user emitted, as (dt, size, dir, age).

        This is the HIGH-VOLUME local observation. A drone sees it only for
        users it can actually decode.
        """
        out = []
        for c in self.calls:
            if c.user != user:
                continue
            if c.start_s > t:
                continue
            end = min(t, c.start_s + c.dur_s)
            n_pkt = int((end - c.start_s) * PKT_RATE_HZ)
            for i in range(max(0, n_pkt - k), n_pkt):
                ts = c.start_s + i / PKT_RATE_HZ
                out.append((t - ts, 1.0, 1.0, ts - c.start_s))
        out.sort(key=lambda e: e[0])
        return out[:k]
