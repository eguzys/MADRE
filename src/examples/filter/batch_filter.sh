#!/bin/bash
#SBATCH --job-name="filter"
#SBATCH --partition=epyc2,bdw
#SBATCH --time=72:00:00

# Your code below this line
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate MANETRel_env
python src/examples/filter/example_filter.py
