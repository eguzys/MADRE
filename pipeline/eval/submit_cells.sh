#!/bin/bash
# submit_cells.sh <name> <cell-tag> <users> <ckpt-rel-to-src|best> [senderProb]   -> 10 shards x 10 episodes
export PATH=/usr/sepp/bin:/usr/bin:/bin:$PATH; export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
S=/usr/itetnas04/data-scratch-01/eguzys/data/u100diag
NAME=$1; TAG=$2; USERS=$3; CK=$4; SP=${5:-}
EXP="ALL"; [ -n "$SP" ] && EXP="ALL,DIAG_SENDERPROB=$SP"
for ss in 0 10 20 30 40 50 60 70 80 90; do
  j=$(sbatch --parsable --export=$EXP --job-name=${NAME}_s$ss $S/shard.sbatch $TAG $USERS $ss 10 $CK $NAME | tail -1)
  scontrol update JobId=$j Partition=disco.med,cpu.normal,gpu.normal >/dev/null 2>&1
  echo "$NAME s$ss -> $j"
done
