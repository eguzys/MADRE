#!/bin/bash
#SBATCH --job-name=eval_ffn_raw
#SBATCH --gpus=5060ti:1
#SBATCH --account=projects
#SBATCH --time=02:00:00
#SBATCH --output=/work/projects/eguzys/code/src/eval_ffn_raw_%j.out
set -euo pipefail
source ~/.bashrc
conda activate /work/scratch/eguzys/padre_env
cd /work/projects/eguzys/code/src
CKPT=out/benchmarl_raw/ffn_raw_varU/normalizedmappo_static_mlp__e553ef82_26_07_07-16_28_46/checkpoints/checkpoint_3000000.pt
python -m evaluation.evaluate "$CKPT" --policy trained --condition sinr --numb-of-nodes 4 -n 100 --no-render
