# Position-Oracle Max-Flow Reward

## What was implemented

A new reward function `compute_position_oracle_reward` was added to
`omnet_module/max_flow_reward.py`.  It implements the *position-oracle*
weighted multi-commodity max-flow reward described in the task spec.

### Core idea

The reward is a deterministic function of drone and user **positions only**.
Given positions, it builds a capacity graph, runs a weighted multi-commodity
max-flow LP, and returns the weighted sum of delivered flows:

```
Reward = Σ_k  w_k · τ_k
```

where:
- `τ_k` = flow (bps) delivered to commodity `k` by the LP
- `w_k = W_CROSS` if users `i` and `j` are in **different** areas
- `w_k = W_LOCAL` if users `i` and `j` are in the **same** area

### LP formulation

```
maximize   Σ_k  w_k · τ_k

subject to:
    Σ_k f_{k,i,j}  ≤  c_{i,j}                     ∀ edges      (capacity)
    Σ_i f_{k,i,n}  =  Σ_j f_{k,n,j}               ∀ k, transit n  (conservation)
    Σ_j f_{k,src_k,j} - Σ_i f_{k,i,src_k} = τ_k   ∀ k          (source balance)
    Σ_i f_{k,i,dst_k} - Σ_j f_{k,dst_k,j} = τ_k   ∀ k          (sink balance)
    τ_k  ≤  DEMAND_PER_COMMODITY                    ∀ k          (hungry cap)
    f_{k,i,j} ≥ 0,  τ_k ≥ 0
```

Solver: PuLP + CBC.

### What was NOT changed

The legacy `compute_weighted_maxmin_flow` / `compute_max_flow_reward` functions
(max-min fair LP, Kelly 1998 weights) are **left intact** for backward
compatibility.

---

## Design decisions

### Commodities from structure, not live calls

Every ordered pair `(user_i, user_j)` with `i ≠ j` is a commodity,
derived from `obs["user_area"]` — a structural field assigned at episode
start by the Bonn talk-group model.  `active_broadcasts` is deliberately
ignored.  This ensures the reward is smooth and independent of bursty
traffic patterns (a key property of the position-oracle approach).

### Hungry demands (never saturate)

`DEMAND_PER_COMMODITY = 1e9` bps (1 Gbps per commodity).  The maximum
achievable per-link Shannon capacity with `BANDWIDTH = 20 MHz` is ≈ 500 Mbps
(at 1 m distance, 200 mW TX).  With N users there are N(N−1) commodities each
wanting 1 Gbps; the total demand always exceeds network capacity, so τ_k
values are always limited by capacity constraints, never by the demand cap.
The reward has a gradient everywhere a relay chain can be improved.

### W_CROSS / W_LOCAL as config parameters

```python
W_CROSS: float = 10.0   # module-level; pass as w_cross= to override
W_LOCAL: float = 1.0
DEMAND_PER_COMMODITY: float = 1e9
```

All three are module-level constants **and** keyword arguments to
`compute_position_oracle_reward(obs, w_cross=…, w_local=…, demand=…)`.
A training sweep can patch them at import time or pass them per-call.

### Capacity function

The existing `_link_capacity_bps` function is used unchanged.  It implements
the Friis + log-distance path loss model (α = 2.5, BANDWIDTH = 20 MHz).

**Cliff note:** `_link_capacity_bps` returns exactly
0.0 when `SNR < −10 dB`.  This is a hard cutoff — not a smooth decay — which
creates zero-gradient dead zones if relay nodes sit just outside radio range.
The smoke test emits a `logging.WARNING` about this on first call.  Whether
this causes training problems depends on the arena geometry; if learning
stalls, consider smoothing the cutoff (e.g., replace the hard threshold with
a sigmoid-weighted decay) or lowering the −10 dB floor.

---

## Smoke test results

Run: `python max_flow_reward.py`

| Check | Result | Detail |
|-------|--------|--------|
| 1 Determinism | **PASS** | Same positions → same reward (1 203 293 929.20 both calls) |
| 2 Live-call independence | **PASS** | `active_broadcasts` ignored; reward unchanged |
| 3 Position sensitivity | **PASS** | Bridge drone at x=1000 → 1.20B bps; at x=800 → 1.24B bps |
| 4 No saturation | **PASS** | 3-drone chain → 1.20B; 5-drone chain → 1.88B |
| 5 Cross-area weighting | **PASS** | reward_cross / reward_local = 10.000 (= W_CROSS/W_LOCAL) |
| 6 Capacity cliff | **WARN** | Hard zero at SNR < −10 dB; see above |

All hard checks passed on first run.

---

## Function API

```python
from max_flow_reward import (
    compute_position_oracle_reward,   # full result dict
    compute_oracle_max_flow_reward,   # scalar wrapper
    W_CROSS, W_LOCAL, DEMAND_PER_COMMODITY,
)

obs = {
    "positions":      [[x, y], ...],   # drone positions (m)
    "user_positions": [[x, y], ...],   # user positions  (m)
    "user_area":      [0, 1, 0, ...],  # area ID per user (0-3); missing → all 0
}

reward = compute_oracle_max_flow_reward(obs)                      # float
result = compute_position_oracle_reward(obs, w_cross=10.0, w_local=1.0, demand=1e9)
# result keys: reward, per_commodity_tau, per_commodity_flow,
#              weights, num_commodities, num_cross, num_local
```

The `user_area` field comes from the Bonn talk-group model in OMNeT++
(`obs["user_area"]`).  If the field is absent the function falls back to
treating all users as area 0 (fully local, no cross-area pairs).

---

## Files changed

| File | Change |
|------|--------|
| `omnet_module/max_flow_reward.py` | Added `W_CROSS`, `W_LOCAL`, `DEMAND_PER_COMMODITY` constants; added `_build_oracle_commodities`, `compute_position_oracle_reward`, `compute_oracle_max_flow_reward`; replaced legacy smoke test with full 6-check suite |
| `omnet_module/REWARD.md` | This file |
