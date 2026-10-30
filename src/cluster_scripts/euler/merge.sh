#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --job-name=merge
#SBATCH --output=/cluster/scratch/eguzys/merge_%j.log
set -e
apptainer exec --bind /cluster/scratch/eguzys:/out \
  /cluster/scratch/eguzys/omnet.sif \
  python -c "
import numpy as np, glob
fs = sorted(glob.glob('/out/omnet_sets_[0-9]*.npz'))
print('files:', len(fs), flush=True)
T = np.concatenate([np.load(f)['tokens'] for f in fs])
M = np.concatenate([np.load(f)['masks'] for f in fs])
print('MERGED', T.shape, M.shape, flush=True)
np.savez('/out/omnet_sets_merged.npz', tokens=T, masks=M)
print('saved', flush=True)
"
echo DONE
