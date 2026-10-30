#!/usr/bin/env python3
"""
harvest_configs.py — read every per-config JSON sweep_config.py wrote (Tier
B) into a directory, rank by how many gates pass then by margin, and print
an explicit "TOP 5 FOR STAGE 2" block.

Sibling to harvest_sweep.py (which harvests sweep_arena.py's N/L/max_clusters
grid and its narrower C/E+R1/R2 gate set). This harvests sweep_config.py's
JSON, which additionally has B2/B3/B4 (gate 4, signal-vs-noise, is the whole
reason this file exists as a bigger table than harvest_sweep.py's).

IF NOTHING PASSES ALL GATES, THIS SCRIPT SAYS SO PLAINLY and names the single
gate that eliminates the most configs (the binding constraint) -- it never
fabricates a winner by relaxing the printed ranking criteria.

Usage:
    python3 harvest_configs.py /path/to/tier_b_outputs
    python3 harvest_configs.py /path/to/tier_b_outputs --glob "cfg_*.json" --csv ranked.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os


GATE_KEYS = [
    # from tv.evaluate / tv.evaluate_tier2's own "criteria" dict, whichever
    # tier this config's N actually used -- flattened here as top-level keys.
    "R1_reachable", "R2_prize_reachable",
    "B2_nontrivial", "B3_gradient_rich", "B4_signal_over_noise",
]


def _load_configs(directory: str, pattern: str) -> list[dict]:
    cells = []
    for path in sorted(glob.glob(os.path.join(directory, pattern))):
        with open(path) as f:
            st = json.load(f)
        st["_path"] = path
        # Flatten the tier-specific "criteria" dict (C1-C6 or E1-E4) into the
        # same top-level namespace as R1/R2/B2/B3/B4, so ranking can treat
        # every gate uniformly regardless of which tier produced this JSON.
        for k, v in st.get("criteria", {}).items():
            st[k] = v
        cells.append(st)
    return cells


def _all_gate_keys(cells: list[dict]) -> list[str]:
    """Union of every gate name actually present, in a stable order: the
    tier-specific criteria first (in first-seen order), then R1/R2/B2/B3/B4."""
    seen = []
    for c in cells:
        for k in c.get("criteria", {}).keys():
            if k not in seen:
                seen.append(k)
    for k in GATE_KEYS:
        if k not in seen:
            seen.append(k)
    return seen


def _n_gates_passed(cell: dict, gate_keys: list[str]) -> int:
    return sum(1 for k in gate_keys if bool(cell.get(k, False)))


def _margin_score(cell: dict) -> float:
    """Ranking tiebreak among configs with the SAME n_gates_passed: bigger is
    better on every term, each roughly normalised so no single term
    dominates by unit accident.
        snr_ratio                      (gate 4's own margin, target >=1)
        P1_over_P0v                    (non-triviality margin, target >=1.3)
        diffusive_reach - void_traverse (reachability margin, target >=0)
    """
    snr = cell.get("snr_ratio", 0.0)
    p1_p0v = cell.get("P1_over_P0v", 0.0)
    reach_margin = cell.get("diffusive_reach_m", 0.0) - cell.get("median_void_traverse_m", 0.0)
    # reach_margin is in metres (O(100s)); divide down so it does not swamp
    # the two O(1) ratios in the sum -- this is a display/ranking heuristic,
    # not a scored criterion, so an exact normalisation constant is not
    # load-bearing the way a gate threshold would be.
    return snr + p1_p0v + reach_margin / 200.0


def print_table(cells: list[dict], gate_keys: list[str]) -> list[dict]:
    ranked = sorted(
        cells,
        key=lambda c: (_n_gates_passed(c, gate_keys), _margin_score(c)),
        reverse=True,
    )

    header = ["N", "L", "mc", "pX", "sprd", "stp", "P0v", "P1", "P2",
             "P1/P0v", "snr", "gates", "PASS"]
    widths = [3, 6, 3, 4, 5, 4, 7, 7, 7, 6, 6, 5, 6]
    print("  ".join(h.rjust(w) for h, w in zip(header, widths)))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))

    for c in ranked:
        n_pass = _n_gates_passed(c, gate_keys)
        row = [
            str(c.get("N", "-")).rjust(widths[0]),
            f"{c.get('L', 0):.0f}".rjust(widths[1]),
            str(c.get("max_clusters", "-")).rjust(widths[2]),
            f"{c.get('p_cross', 0):.1f}".rjust(widths[3]),
            f"{c.get('user_spread_eff', 0):.0f}".rjust(widths[4]),
            f"{c.get('step_m', 0):.0f}".rjust(widths[5]),
            f"{c.get('median_P0v', 0):.1f}".rjust(widths[6]),
            f"{c.get('median_P1', 0):.1f}".rjust(widths[7]),
            f"{c.get('median_P2', 0):.1f}".rjust(widths[8]),
            f"{c.get('P1_over_P0v', 0):.2f}".rjust(widths[9]),
            f"{c.get('snr_ratio', 0):.2f}".rjust(widths[10]),
            f"{n_pass}/{len(gate_keys)}".rjust(widths[11]),
            ("PASS" if c.get("overall_PASS") else "FAIL").rjust(widths[12]),
        ]
        print("  ".join(row))
    return ranked


def print_binding_constraint(cells: list[dict], gate_keys: list[str]) -> None:
    n_total = len(cells)
    fail_counts = {g: sum(1 for c in cells if not bool(c.get(g, False))) for g in gate_keys}
    print("\nFailure counts by gate (a config can fail more than one):")
    for g, cnt in sorted(fail_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {g}: {cnt}/{n_total} fail ({cnt / max(n_total, 1):.0%})")
    if fail_counts:
        binding = max(fail_counts, key=fail_counts.get)
        print(f"BINDING CONSTRAINT (eliminates the most configs): {binding}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", help="directory containing Tier-B JSON files")
    ap.add_argument("--glob", type=str, default="*.json",
                    help="filename pattern to load (default: every *.json in the dir).")
    ap.add_argument("--csv", type=str, default="", help="also write the ranked table here")
    ap.add_argument("--top", type=int, default=5, help="how many to print in the "
                    "'TOP N FOR STAGE 2' block.")
    a = ap.parse_args()

    cells = _load_configs(a.directory, a.glob)
    if not cells:
        raise SystemExit(f"no files matching {a.glob!r} found in {a.directory}")
    print(f"Loaded {len(cells)} config(s) from {a.directory} (pattern {a.glob!r})\n")

    gate_keys = _all_gate_keys(cells)
    ranked = print_table(cells, gate_keys)

    n_full_pass = sum(1 for c in cells if c.get("overall_PASS"))
    print(f"\n{n_full_pass}/{len(cells)} configs PASS every gate.")

    print_binding_constraint(cells, gate_keys)

    print("\n" + "=" * 78)
    if n_full_pass == 0:
        print("NO CONFIG PASSES EVERY GATE. Do not force a winner -- see the binding")
        print("constraint above. The ranking below is by (gates passed, margin) among")
        print("configs that are CLOSEST, not configs that qualify for Stage 2.")
    print(f"TOP {min(a.top, len(ranked))} (closest to passing, or the actual Stage-2 "
         f"shortlist if n_full_pass > 0):")
    print("=" * 78)
    for c in ranked[: a.top]:
        n_pass = _n_gates_passed(c, gate_keys)
        print(f"  N={c.get('N')} L={c.get('L', 0):.0f} mc={c.get('max_clusters')} "
             f"pCross={c.get('p_cross')} spread_eff={c.get('user_spread_eff')} "
             f"step_m={c.get('step_m')} userTX={c.get('user_tx_mW')} "
             f"droneTX={c.get('drone_tx_mW')}  "
             f"gates={n_pass}/{len(gate_keys)}  overall={'PASS' if c.get('overall_PASS') else 'FAIL'}  "
             f"snr_ratio={c.get('snr_ratio', 0):.2f}  P1/P0v={c.get('P1_over_P0v', 0):.2f}  "
             f"file={c.get('_path')}")

    if a.csv:
        import csv as _csv
        fields = (["N", "L", "max_clusters", "p_cross", "user_spread_eff", "step_m",
                   "user_tx_mW", "drone_tx_mW", "median_P0v", "median_P1", "median_P2",
                   "P1_over_P0v", "P2_over_P1", "snr_ratio", "policy_std", "seed_std",
                   "overall_PASS", "_path"] + gate_keys)
        with open(a.csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for c in ranked:
                w.writerow(c)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
