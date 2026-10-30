# LP Structure Cache in `compute_position_oracle_reward`

## What changed

A single-slot module-level cache (`_ORACLE_LP_CACHE`) was added to
`max_flow_reward.py` to skip rebuilding the LP constraint matrices on steps
where the network topology is unchanged.

The only modified function is `compute_position_oracle_reward` in
`max_flow_reward.py`. No LP math, solver calls, column-index formulas, or
solution extraction code was touched.

---

## How the cache works

### Cache key

```python
cache_key = (nd, nu, frozenset(edges.keys()), tuple(user_area))
```

- `nd`, `nu` — drone and user counts (determine variable blocks)
- `frozenset(edges.keys())` — the set of directed links in the graph; capacities
  are deliberately excluded
- `tuple(user_area)` — controls commodity weights via `_build_oracle_commodities`

Together these capture every structural degree of freedom in A\_ub and A\_eq.
Capacities (`edges[e]`) and objective weights live only in `b_ub` and `c_obj`
and are excluded because they change every step but do not affect matrix shape.

### Cache hit (structure unchanged)

Reuse from cache: `A_ub`, `A_eq`, `edge_list`, `edge_to_col`, `N`, `E`, `K`,
`row_i`, `b_eq`, `commodities_raw`.

Rebuild cheaply:
- `b_ub = np.array([edges[e] for e in edge_list])` — new capacities, indexed
  via the *cached* `edge_list` to preserve column ordering
- `c_obj` — from the cached `commodities_raw` (weights may vary with w\_cross /
  w\_local parameters)

### Cache miss (structure changed or first call)

Full current code path runs unchanged: build adjacency helpers, COO-triplet
loops, `csc_matrix`, then store all structural objects in the cache.

### Why use cached `edge_list` for `b_ub`

`edges` is a plain Python dict. On a cache miss the original call uses
`list(edges.keys())` to set the column ordering. On a cache hit, `edges.keys()`
could return keys in a different order (CPython dict order is insertion-order,
but `_build_edges` inserts edges in geometry-dependent order that could vary if
floating-point distance comparisons are borderline). Using the cached
`edge_list` guarantees `b_ub[e_idx]` matches column `e_idx` in A\_ub exactly.

---

## What is cached vs. what is rebuilt every call

| Object | Cached? | Reason |
|---|---|---|
| `A_ub` | yes | purely structural (all values 1.0) |
| `A_eq` | yes | purely structural (values ±1.0) |
| `b_eq` | yes | all zeros, shape fixed by structure |
| `edge_list` | yes | determines column ordering |
| `edge_to_col` | yes | index lookup derived from edge\_list |
| `N`, `E`, `K`, `row_i` | yes | shape scalars |
| `commodities_raw` | yes | used to rebuild c\_obj |
| `b_ub` | no | carries live edge capacities |
| `c_obj` | no | carries live commodity weights |
| `bounds` | no | cheap to rebuild from K, E, demand |

---

## Correctness test results

Run: one greedy episode (OMNeT++ sim), then offline replay.

```
Total steps collected:  101
Edge-change steps:      8 (8.0% of steps)
Steps compared:         58 (all 8 edge-change + 50 random cache-hit samples)
Parity result:          ALL MATCH (rel tol < 1e-6)
```

The critical assertion — that steps where the edge set *changes* (cache
invalidation) still produce identical rewards — passed on all 8 such steps.
Compared against the CBC backup (`max_flow_reward.py.cbc_bak`) as ground truth.

---

## Benchmark results

200 calls on a fixed mid-episode obs (5 drones, small topology):

```
Cached:     519 ms/call
Non-cached: 570 ms/call
Speedup:    1.1x
```

The modest speedup on this small episode reflects that the HiGHS LP solver
dominates at ~519 ms/call. The matrix build (COO loops + `csc_matrix`) accounts
for ~51 ms — a meaningful saving that scales super-linearly with problem size
(K grows as O(nu²), matrix fill as O(K·E)), so larger topologies benefit more.

---

## Files changed

- `omnet_module/max_flow_reward.py` — added `_ORACLE_LP_CACHE` module-level
  dict and restructured the LP formulation section of
  `compute_position_oracle_reward` into hit/miss branches. All other code
  (solve block, solution extraction, `_build_edges`, `_build_oracle_commodities`,
  smoke tests, `compute_oracle_max_flow_reward`) is unchanged.
