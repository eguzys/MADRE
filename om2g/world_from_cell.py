#!/usr/bin/env python3
"""Emit the WORLD env of a training cell, as shell exports.

`arch_out/cells/<tag>.sh` is what the training actually used, so it is the
only source of truth for the world; this script reads it instead of assembling
the world by hand.

Rules:
  * LAST export wins, exactly as bash would (cells set shaping non-zero, then
    zero it further down; taking the first would re-enable shaping).
  * Run-scoped and arm-scoped vars are dropped -- they identify a cell, not a
    world, and copying them would make every baseline inherit one arm's
    ablation.
  * OM2G_SPAWN is forced to `cluster`: cells train with `explore` (exploring
    starts) and the cell file's own comment says eval MUST use cluster.

Usage:  eval "$(python3 world_from_cell.py z3sil_s1)"
"""
import os
import re
import sys

CELL_DIR = "arch_out/cells"
# Identify a run or an arm, not a world.
DROP = {
    "OM2G_RUN_TAG", "OM2G_TMPDIR", "OM2G_EXTRA_OVERRIDES",
    "OM2G_MSG_ABLATE", "OM2G_TAG_SWAP", "OM2G_USER_PARTNER_TAG",
    "OM2G_PARTNER_TAG_MODE", "OMNET_ALLOW_CPU",
    # Shaping is TRAINING-ONLY: evaluate_omnet refuses to score with it on.
    "OM2G_SHAPE_EDGE", "OM2G_SHAPE_FIEDLER", "OM2G_SHAPE_VORONOI",
    "OM2G_SHAPE_SPREAD", "OM2G_SHAPE_GAMMA",
}

def main():
    if len(sys.argv) < 2:
        sys.exit("usage: world_from_cell.py <cell-tag>")
    path = os.path.join(CELL_DIR, sys.argv[1] + ".sh")
    if not os.path.exists(path):
        sys.exit(f"no such cell: {path}")
    pat = re.compile(r'^\s*export\s+((?:OM2G|OMNET)_[A-Z0-9_]+)=(.*)$')
    world = {}
    for line in open(path):
        m = pat.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if key in DROP:
            continue
        world[key] = val          # last wins, like bash
    world["OM2G_SPAWN"] = "cluster"
    world.setdefault("OM2G_TRACK", "1")
    for k in sorted(world):
        print(f"export {k}={world[k]}")

if __name__ == "__main__":
    main()
