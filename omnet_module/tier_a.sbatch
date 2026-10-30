#!/bin/bash
#SBATCH --job-name=tier_a
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/cluster/scratch/eguzys/config_sweep/logs/tier_a_%j.out
set -uo pipefail
source /cluster/scratch/eguzys/pisa_env/bin/activate
cd /cluster/scratch/eguzys/code_git
python omnet_module/screen_configs.py --screen \
  --n-samples 20000 --top 500 \
  --out /cluster/scratch/eguzys/config_sweep/survivors.json \
  --csv /cluster/scratch/eguzys/config_sweep/survivors.csv
echo "=== TIER A DONE rc=$? ==="
