#!/bin/bash
#SBATCH --job-name=learnenc
#SBATCH --gpus=5060ti:1
#SBATCH --account=projects
#SBATCH --time=48:00:00
#SBATCH --output=/work/scratch/eguzys/learnenc_%j.out
set -uo pipefail
ARCH="${1:?arch required: st|deepsets|ffn}"
# frozen mode exists for ONE purpose: padre_ffn is a DIFFERENT model from
# padre_mlp, and padre_mlp produced the banked frozen FFN numbers. So
# learned-FFN vs banked-FFN crosses two implementations, not one variable.
# padre_ffn+frozen at N=4 is the bridge -- it must reproduce padre_mlp's
# result before any learned-FFN number can be read as an encoder effect.
MODE="${2:-learned}"
case "$MODE" in
  learned) TASK=padre/raw_learnenc ;;
  frozen)  TASK=padre/raw ;;
  *) echo "FATAL: mode must be learned|frozen, got '$MODE'"; exit 1 ;;
esac
source /work/projects/eguzys/miniconda3/etc/profile.d/conda.sh
conda activate /work/scratch/eguzys/padre_env
cd /work/projects/eguzys/code/src
export WANDB_MODE=offline
SAVE=out/learnenc/${ARCH}_${MODE}
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

echo "=== arch=$ARCH encoder_mode=$MODE task=$TASK $(date) ==="
python run.py task=$TASK algorithm=${ALGO} \
  model=padre_${ARCH} model.encoder_mode=$MODE \
  experiment.max_n_frames=3000000 \
  experiment.save_folder="$SAVE"
RC=$?
[ $RC -ne 0 ] && { echo "=== FAILED rc=$RC ==="; exit $RC; }
echo "=== DONE $(date) ==="
