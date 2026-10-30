#!/usr/bin/env bash
# Pull a full BenchMARL run dir (config.pkl + checkpoints) from cluster.
# Always pulls the WHOLE run dir so config.pkl rides along — never a bare .pt.
# Usage: ./pull_canary.sh <run_path_under_out/benchmarl>
#   e.g. ./pull_canary.sh st_probe/normalizedmappo_static_stmodel__ada0e8a8_26_05_28-01_09_34
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: $0 <run_path under out/benchmarl, e.g. st_probe/normalizedmappo_...>"
  exit 1
fi

RUN="$1"
CLUSTER="eguzys@student-cluster.inf.ethz.ch"
REMOTE_BASE="/work/projects/eguzys/code/src/out/benchmarl"
LOCAL_DEST="src/out/cluster_full/"

mkdir -p "$LOCAL_DEST"
rsync -avz --progress "${CLUSTER}:${REMOTE_BASE}/${RUN}" "$LOCAL_DEST"

echo "Pulled ${RUN} -> ${LOCAL_DEST} (config.pkl + checkpoints, canonical layout)."
