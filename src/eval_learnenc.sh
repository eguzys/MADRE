#!/bin/bash
#SBATCH --job-name=eval_learnenc
#SBATCH --gpus=5060ti:1
#SBATCH --account=projects
#SBATCH --time=04:00:00
#SBATCH --output=/work/scratch/eguzys/eval_learnenc_%j.out

source /work/scratch/eguzys/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate /work/scratch/eguzys/padre_env
cd /work/projects/eguzys/code/src

for c in \
  out/learnenc/ffn_frozen/*/checkpoints/checkpoint_3000000.pt \
  out/learnenc/ffn_learned/*/checkpoints/checkpoint_3000000.pt \
  out/learnenc/deepsets_learned/*/checkpoints/checkpoint_3000000.pt \
  out/learnenc/st_learned/*/checkpoints/checkpoint_3000000.pt ; do
  echo "=== $(echo $c | cut -d/ -f3)"
  python -u evaluation/evaluate.py "$c" --policy trained --condition sinr -n 20
done
