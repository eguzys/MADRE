#!/bin/bash
. /cluster/scratch/eguzys/pisa_env/bin/activate
cd /cluster/scratch/eguzys/resilient-backbone/src
export PADRE_EVAL_USERS=20
python evaluation/evaluate.py "$1" --policy trained --condition sinr --numb-of-nodes 4 -n 100
