#!/bin/bash
source /work/projects/eguzys/miniconda3/etc/profile.d/conda.sh
conda activate /work/scratch/eguzys/padre_env
python -c "import torch; print(torch.cuda.get_device_name(0)); x=torch.randn(1000,1000).cuda(); print((x@x).sum().item())"
