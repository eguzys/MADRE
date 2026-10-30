#!/bin/bash
#SBATCH --job-name=learnenc
#SBATCH --gpus=5060ti:1
#SBATCH --account=projects
#SBATCH --time=24:00:00
#SBATCH --output=/work/scratch/eguzys/learnenc_%j.out
set -uo pipefail
ARCH="${1:?arch required: st|deepsets|ffn}"
source /work/scratch/eguzys/miniconda3/etc/profile.d/conda.sh
conda activate /work/scratch/eguzys/padre_env
cd /work/projects/eguzys/code/src
export WANDB_MODE=offline
SAVE=out/learnenc/${ARCH}
mkdir -p "$SAVE"
ALGO="padre_mappo_norm_ent0.03"
EXPECT_ENT="${EXPECT_ENT:-0.03}"
ALGO_YAML="conf/algorithm/${ALGO}.yaml"
ACTUAL_ENT=$(ENT_GUARD_ALGO_YAML="$ALGO_YAML" ENT_GUARD_EXPECT="$EXPECT_ENT" python -c '
import os, sys, yaml
path = os.environ["ENT_GUARD_ALGO_YAML"]
expect = float(os.environ["ENT_GUARD_EXPECT"])
if not os.path.isfile(path):
    print(f"FATAL: no such algorithm yaml: {path}", file=sys.stderr); sys.exit(1)
d = yaml.safe_load(open(path))
actual = d.get("entropy_coef")
if actual is None:
    print(f"FATAL: entropy_coef missing from {path}", file=sys.stderr); sys.exit(1)
actual = float(actual)
print(actual)
sys.exit(0 if abs(actual - expect) < 1e-9 else 1)
')
RC=$?
if [ $RC -ne 0 ]; then
  echo "FATAL: entropy guard failed: ${ALGO_YAML} has entropy_coef=${ACTUAL_ENT:-<unreadable, see stderr above>}, expected $EXPECT_ENT. Refusing to train -- this is the 22h x5 bug."
  exit 1
fi
echo "entropy guard OK: ${ALGO} -> ${ACTUAL_ENT}"

echo "=== arch=$ARCH encoder_mode=learned $(date) ==="
python run.py task=padre/raw_learnenc algorithm=${ALGO} \
  model=padre_${ARCH} model.encoder_mode=learned \
  experiment.max_n_frames=3000000 \
  experiment.save_folder="$SAVE"
RC=$?
[ $RC -ne 0 ] && { echo "=== FAILED rc=$RC ==="; exit $RC; }
echo "=== DONE $(date) ==="
