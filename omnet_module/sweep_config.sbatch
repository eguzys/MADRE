#!/bin/bash
#SBATCH --job-name=config_sweep
#SBATCH --array=0-499
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/eguzys/config_sweep/logs/cfg_%A_%a.out
#
# TIER B, sharded: one array task per Tier-A survivor (screen_configs.py's
# --screen --out survivors.json), NOT a hand-picked grid -- sweep_config.py
# (unlike its sibling sweep_arena.py) takes every knob as a CLI arg, so the
# search space here is exactly whatever Tier A's analytic pre-screen already
# narrowed down to. --array's upper bound MUST match
# min(--top passed to screen_configs.py, len(survivors)) - 1; screen_configs.py
# prints both numbers, so set --array from THAT run's own stdout, not from
# this file's default (500 is just screen_configs.py's own --top default).
#
# Re-run only the cells that failed/are missing:
#   sbatch --array=<comma-separated task ids> omnet_module/sweep_config.sbatch
#
# House pattern (OMNeT-on-Euler precedent) copied from omnet_module/
# eval_baselines.sh / sweep_arena.sbatch -- NOT the GPU src/eval_*.sbatch
# jobs, which never launch OMNeT directly and set none of these vars.
set -uo pipefail

source /cluster/scratch/eguzys/pisa_env/bin/activate
export OMNET_APPTAINER_SIF=/cluster/scratch/eguzys/omnet.sif
export OMNET_HOST_CODE=/cluster/scratch/eguzys/code_git
export OMNET_INET_DIR=/opt/inet4.6
export OMNETPP_DIR=/opt/omnetpp-6.4.0

SURVIVORS_JSON="${SURVIVORS_JSON:-/cluster/scratch/eguzys/config_sweep/survivors.json}"
SEEDS="${SEEDS:-40}"
OUT_DIR=/cluster/scratch/eguzys/config_sweep
mkdir -p "$OUT_DIR/logs"

cd "$OMNET_HOST_CODE/omnet_module"

# Pull this task's ONE config (by array index) out of the Tier-A survivors
# JSON and print it as shell-safe `KEY=VALUE` lines -- avoids a second,
# hand-maintained copy of the config schema inside this bash file (the same
# "one copy of the logic" discipline as sweep_lib.py / config_knobs.py).
eval "$(python3 - "$SURVIVORS_JSON" "$SLURM_ARRAY_TASK_ID" <<'PYEOF'
import json, sys
path, idx = sys.argv[1], int(sys.argv[2])
with open(path) as f:
    d = json.load(f)
survivors = d["survivors"]
if idx >= len(survivors):
    print(f"echo 'FATAL: array index {idx} >= {len(survivors)} survivors in {path}'; exit 1")
    sys.exit(0)
c = survivors[idx]
for k in ("n_drones", "L", "max_clusters", "p_cross", "user_spread_eff",
          "step_m", "user_tx_mW", "drone_tx_mW", "ep_steps"):
    print(f"CFG_{k.upper()}={c[k]!r}")
PYEOF
)"

CSV="$OUT_DIR/cfg_${SLURM_ARRAY_TASK_ID}_N${CFG_N_DRONES}_L${CFG_L%.*}_mc${CFG_MAX_CLUSTERS}.csv"
JSON="$OUT_DIR/cfg_${SLURM_ARRAY_TASK_ID}_N${CFG_N_DRONES}_L${CFG_L%.*}_mc${CFG_MAX_CLUSTERS}.json"

echo "=== task=$SLURM_ARRAY_TASK_ID  N=$CFG_N_DRONES L=$CFG_L mc=$CFG_MAX_CLUSTERS pCross=$CFG_P_CROSS spread=$CFG_USER_SPREAD_EFF step_m=$CFG_STEP_M userTX=$CFG_USER_TX_MW droneTX=$CFG_DRONE_TX_MW ==="
echo "=== csv=$CSV  json=$JSON ==="

python sweep_config.py \
  --n-drones "$CFG_N_DRONES" --arena "$CFG_L" --max-clusters "$CFG_MAX_CLUSTERS" \
  --p-cross "$CFG_P_CROSS" --user-spread-eff "$CFG_USER_SPREAD_EFF" \
  --step-m "$CFG_STEP_M" --user-tx-mw "$CFG_USER_TX_MW" --drone-tx-mw "$CFG_DRONE_TX_MW" \
  --ep-steps "$CFG_EP_STEPS" --seeds "$SEEDS" --out "$CSV" --json "$JSON" --verbose
RC=$?
if [ $RC -ne 0 ]; then echo "=== FAILED rc=$RC ==="; exit $RC; fi
echo "=== OK $(wc -l < "$CSV") csv lines ==="
