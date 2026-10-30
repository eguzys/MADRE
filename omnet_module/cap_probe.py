"""SS8.4 -- is the greedy step cap binding?

probe_greedy caps hill-climbing at 60 steps. That number was chosen for SPEED and
never checked for convergence. P1 (greedy) is the DENOMINATOR of criterion C3
(P2 >= 2 x P1) -- the number the arena gets frozen on.

  If greedy has NOT converged at 60 steps, P1 is under-measured and every P2/P1
  ratio is overestimated by the same factor.

Reuses task_validation's own layout sampling and scoring, so this measures the
SAME greedy the sweep will use.
"""
import argparse, random, time
import numpy as np
import task_validation as tv

# ── RETIRED ────────────────────────────────────────────────────────────────
# This probe calls task_validation.sample_users(), the python-side user
# sampler, which now raises: user positions come from the simulator through
# task_validation.get_omnet_users(). The probe's result (the greedy step cap
# does not bind) is recorded in OMNET.md §8.5. Rewire it to get_omnet_users()
# before any re-run; until then it exits immediately.
raise SystemExit(
    "cap_probe.py is RETIRED: it used the phantom sampler (see OMNET.md §8.5, "
    "result banked/CLOSED). Rewire to task_validation.get_omnet_users() before "
    "any re-run.")
# ───────────────────────────────────────────────────────────────────────────

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=8)
p.add_argument("--arena", type=float, default=1400.0)
p.add_argument("--max-clusters", type=int, default=3)
p.add_argument("--n-drones", type=int, default=5)
p.add_argument("--n-users", type=int, default=17)
p.add_argument("--caps", type=int, nargs="+", default=[30, 60, 120, 250, 500])
a = p.parse_args()

caps, ref = sorted(a.caps), max(a.caps)
rows = []

print(f"greedy step cap probe | L={a.arena:.0f} max_clusters={a.max_clusters} "
      f"n={a.seeds} seeds | R_U={tv.R_U:.0f}m R_D={tv.R_D:.0f}m\n")

for s in range(a.seeds):
    # identical layout draw to run_cell, so this IS the sweep's greedy
    rng = random.Random(hash((int(a.arena), s)) & 0xFFFFFFFF)
    users, areas = tv.sample_users(rng, a.n_users, a.arena,
                                   max_clusters=a.max_clusters)
    start = tv.spawn_staging(rng, a.n_drones, users, areas)
    obs = {"positions": [q[:] for q in start],
           "user_positions": users, "user_area": areas}

    sc = {}
    for cap in caps:
        t = time.perf_counter()
        pos = tv.probe_greedy(obs, a.n_drones, a.arena, steps=cap)
        sc[cap] = (tv.score(pos, users, areas), time.perf_counter() - t)
    rows.append(sc)
    line = "  ".join(f"{c}:{sc[c][0]:7.0f}" for c in caps)
    print(f"seed {s:2d}  {line}", flush=True)

print(f"\n{'cap':>5} {'mean score':>11} {'median':>10} {'frac of '+str(ref):>12} {'sec/seed':>9}")
print("-" * 52)
for cap in caps:
    v  = np.array([r[cap][0] for r in rows])
    tm = np.array([r[cap][1] for r in rows])
    rf = np.array([r[ref][0] for r in rows])
    frac = np.mean(v / np.maximum(rf, 1e-9))
    print(f"{cap:>5} {v.mean():>11.1f} {np.median(v):>10.1f} {frac:>12.3f} {tm.mean():>9.2f}")

f60 = np.mean([r[60][0] / max(r[ref][0], 1e-9) for r in rows]) if 60 in caps else None
print()
if f60 is None:
    print("(60 not in caps -- no verdict)")
elif f60 >= 0.98:
    print(f">> CONVERGED. 60 steps reaches {f60:.1%} of the {ref}-step score.")
    print(f"   The cap is NOT binding. P1 is fully measured. SS8.4 CLOSED -- sweep at 60.")
elif f60 >= 0.90:
    print(f">> MARGINAL. 60 steps reaches {f60:.1%}. P1 under-measured by ~{1-f60:.0%},")
    print(f"   so every P2/P1 is overestimated by roughly that much. Raise the cap.")
else:
    print(f">> BINDING. 60 steps reaches only {f60:.1%} of converged greedy.")
    print(f"   P1 is BADLY under-measured; every P2/P1 ratio is overestimated by")
    print(f"   that factor. Do NOT sweep at 60. Raise the cap and re-derive C3.")
