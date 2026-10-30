#!/bin/bash
# Run the four-seed figure scorers on the spreading-1 evaluations.
#   bash pipeline/scoring/four_seed_staging/rescore_4seed_s1.sh            # pull (pull_from_cluster_s1.sh) + score
#   NOPULL=1 bash pipeline/scoring/four_seed_staging/rescore_4seed_s1.sh   # score the existing pull_raw_s1 only
# Outputs: <scoring root>/four_seed_scored/<figure>_4seed_s1/.
# Staging: four_seed_staging/pull_raw_s1/ -- every S1_<family> (and each SPR1 family with identical settings) under
# its base family name, only shard logs whose drone tx power is 6.59004 mW (MANIFEST.tsv lists rejected logs).
# A scorer that stops on an incomplete family is reported and the others run.
# Extra args go to every scorer (e.g. --with-extras, --seeds L64=1,2,4,5). Needs ssh access to the cluster.
set -uo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
ST=$R/four_seed_staging
if [ -z "${NOPULL:-}" ]; then
  tmp=$(mktemp -d "$ST/pull_s1_new_XXXX")
  ssh -4 -o BatchMode=yes -o ServerAliveInterval=30 eguzys@tik42x.ethz.ch 'bash -s' < "$ST/pull_from_cluster_s1.sh" | tar -C "$tmp" -xzf - || { echo "pull failed"; exit 1; }
  test -s "$tmp/episodes_logs.txt" || { echo "empty pull"; exit 1; }
  rm -rf "$ST/pull_raw_s1.prev"; [ -d "$ST/pull_raw_s1" ] && mv "$ST/pull_raw_s1" "$ST/pull_raw_s1.prev"
  mv "$tmp" "$ST/pull_raw_s1"; echo "pulled at $(cat "$ST/pull_raw_s1/PULLED_AT.txt")"
  awk -F'\t' 'NR>1 && $4>0' "$ST/pull_raw_s1/MANIFEST.tsv" | sed 's/^/REJECTED LOGS: /'
fi
cd "$R"
run() { local n=$1; shift; if CUDA_VISIBLE_DEVICES="" python3 "$@" --mode 4seed --spreading 1 $EXTRA > "$ST/pull_raw_s1/log_$n.txt" 2>&1; then echo "OK   $n"; else echo "STOP $n: $(tail -1 "$ST/pull_raw_s1/log_$n.txt")"; fi; }
EXTRA="$*"
run density    density_beacon_k3_oct07/score_density_4seed.py   # must run before baselines (\datamadre rows)
run baselines  baselines_beacon_final_oct07/score_baselines_4seed.py
run outages    outages_beacon_oct06/score_outages_4seed.py
run outage_baselines outages_baselines_oct08/score_outage_baselines_4seed.py
run separation separation_4seed_oct09/score_separation_4seed.py
run robustness robustness_4seed_oct09/score_robustness_4seed.py
echo "outputs: $R/four_seed_scored/*_4seed_s1/  (logs: $ST/pull_raw_s1/log_*.txt)"
