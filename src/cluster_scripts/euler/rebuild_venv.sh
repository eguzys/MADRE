#!/bin/bash
source /etc/profile.d/modules.sh 2>/dev/null || true
module load stack/2024-06 python/3.11.6 2>/dev/null || module load python/3.11.9 2>/dev/null || module load python 2>/dev/null
which python3 || { echo "NO PYTHON"; exit 1; }
python3 -V
python3 -m venv /cluster/scratch/eguzys/pisa_env || exit 1
source /cluster/scratch/eguzys/pisa_env/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.12.1+cpu torchvision==0.27.1
grep -viE "^(torch|torchvision|hoptorch)==" ~/requirements-euler.txt > /tmp/req_rest.txt
python -m pip install -r /tmp/req_rest.txt
python -c "import torch,tensordict,torchrl,benchmarl,typing_extensions; from scipy.stats import wilcoxon; print('VERIFY OK', torch.__version__)"
