#!/usr/bin/env python3
"""
harvest_sweep.py — read every per-cell JSON sweep_arena.py wrote into a
directory and print one comparison table across cells.

sweep_arena.py is one-cell-per-invocation (see its module docstring), so a
24-cell grid (sweep_arena.sbatch) lands as 24 independent
sweep_N{N}_L{L}_mc{MC}.json files. This script does no computation of its
own -- every number here was already computed and written by sweep_arena.py;
this just tabulates.

Usage:
    python3 harvest_sweep.py /cluster/scratch/eguzys/arena_sweep
    python3 harvest_sweep.py /cluster/scratch/eguzys/arena_sweep --csv out.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def _load_cells(directory: str) -> list[dict]:
    cells = []
    for path in sorted(glob.glob(os.path.join(directory, "sweep_N*.json"))):
        with open(path) as f:
            st = json.load(f)
        st["_path"] = path
        cells.append(st)
    return cells


def _fmt(v, width, prec=None):
    if v is None:
        s = "-"
    elif isinstance(v, bool):
        s = "PASS" if v else "FAIL"
    elif isinstance(v, float) and prec is not None:
        s = f"{v:.{prec}f}"
    else:
        s = str(v)
    return s.rjust(width)


def print_table(cells: list[dict]) -> None:
    cells = sorted(cells, key=lambda c: (c.get("N", 0), c.get("L", 0), c.get("max_clusters", 0)))

    header = ["N", "L", "mc", "P0", "P0v", "P1", "P2", "P2/P1",
             "R1", "R2", "x-rwd", "zero%", "PASS"]
    widths = [3, 6, 3, 8, 8, 8, 8, 6, 5, 5, 6, 6, 6]
    print("  ".join(h.rjust(w) for h, w in zip(header, widths)))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))

    for st in cells:
        row = [
            _fmt(st.get("N"), widths[0]),
            _fmt(st.get("L"), widths[1]),
            _fmt(st.get("max_clusters"), widths[2]),
            _fmt(st.get("median_P0"), widths[3], 1),
            _fmt(st.get("median_P0v"), widths[4], 1),
            _fmt(st.get("median_P1"), widths[5], 1),
            _fmt(st.get("median_P2"), widths[6], 1),
            _fmt(st.get("P2_over_P1"), widths[7], 2),
            _fmt(st.get("R1_reachable"), widths[8]),
            _fmt(st.get("R2_prize_reachable"), widths[9]),
            _fmt(st.get("median_frac_cross_reward"), widths[10], 2),
            _fmt(100.0 * st.get("median_frac_zero_steps_P0v", 0.0), widths[11], 0),
            _fmt(st.get("overall_PASS"), widths[12]),
        ]
        print("  ".join(row))

    n_pass = sum(1 for c in cells if c.get("overall_PASS"))
    print()
    print(f"{n_pass}/{len(cells)} cells PASS (all gates: C1-C6/E1-E4 as applicable, "
         f"plus R1_reachable, R2_prize_reachable).")
    passing = [c for c in cells if c.get("overall_PASS")]
    if passing:
        best = max(passing, key=lambda c: c["L"])
        print(f"Largest passing L overall: {best['L']:.0f} m "
             f"(N={best['N']}, max_clusters={best['max_clusters']}).")


def _write_csv(cells: list[dict], path: str) -> None:
    import csv
    fields = ["N", "L", "max_clusters", "median_P0", "median_P0v", "median_P1",
             "median_P2", "P2_over_P1", "R1_reachable", "R2_prize_reachable",
             "median_frac_cross_reward", "median_frac_zero_steps_P0v",
             "median_frac_zero_steps_P1", "infeasible_rate", "median_relays",
             "mean_coverage", "overall_PASS", "_path"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for c in sorted(cells, key=lambda c: (c.get("N", 0), c.get("L", 0), c.get("max_clusters", 0))):
            w.writerow(c)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", help="directory containing sweep_N*.json files")
    ap.add_argument("--csv", type=str, default="", help="also write a flat CSV here")
    a = ap.parse_args()

    cells = _load_cells(a.directory)
    if not cells:
        raise SystemExit(f"no sweep_N*.json files found in {a.directory}")
    print(f"Loaded {len(cells)} cell(s) from {a.directory}\n")
    print_table(cells)

    if a.csv:
        _write_csv(cells, a.csv)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
