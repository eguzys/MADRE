#!/bin/bash
# n=100 final eval for the 1200 m world. Faithful port of final_eval_1000.sh:
# ONLY MINSEP (1000->1200) and MAXNN (1250->1500) differ, matching what the
# x1200* cells were trained with. score_ckpt.sh is not used here because it
# hardcodes the 1465 m world. Both seeds are included, since the talk-vs-silent
# comparison is a training-condition comparison and needs more than one run.
RB="${OM2G_SCRATCH:-/usr/itetnas04/data-scratch-01/eguzys/data}/resilient-backbone"
export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
W="OM2G_DRONES=8,OMNET_NUM_DRONES=8,OM2G_USER_SPAWN=archipelago,OM2G_CLUSTERS=4,OM2G_ARENA=3800,OM2G_SPREAD=25,OM2G_PCROSS=0.9,OM2G_MINSEP=1200,OM2G_MAXNN=1500,OM2G_MARGIN=300,OM2G_SPAWN=cluster,OM2G_USER_PARTNER_TAG=1,OM2G_PARTNER_TAG_MODE=group,OM2G_CTRL_SPREADING=8,OM2G_PHYSICAL_METRIC=1,OM2G_PEER_MEMORY=1"
S="OM2G_SHAPE_FIEDLER=0,OM2G_SHAPE_VORONOI=0,OM2G_SHAPE_EDGE=0,OM2G_SHAPE_SPREAD=0,OM2G_SHAPE_KERNEL=exp,OM2G_SHAPE_VOR_KERNEL=exp"
for c in x1200talk x1200sil x1200talks2 x1200sils2; do
  while squeue -u eguzys -h -n "$c" -o "%t" 2>/dev/null | grep -q .; do sleep 120; done
  N=$(find $RB/src/out/OM2G -path "*_${c}/*" -name "checkpoint_*.pt" -printf "%f\n" 2>/dev/null | sed 's/checkpoint_//;s/.pt//' | sort -n | tail -1)
  CK=$(find $RB/src/out/OM2G -path "*_${c}/*" -name "checkpoint_${N}.pt" | head -1)
  [ -z "$CK" ] && { echo "$(date +%H:%M) SKIP $c (no checkpoint)" >> $RB/final_eval_1200.log; continue; }
  for SS in 0 10 20 30 40 50 60 70 80 90; do
    sbatch --parsable --export=ALL,$W,$S,OM2G_RUN_TAG=G${c}${SS} \
      $RB/om2g/om2g_eval_arton.sh compressed "${CK#$RB/src/}" G_${c}_${N}_${SS} 10 $SS >/dev/null 2>&1
  done
  echo "$(date +%H:%M) fired n=100 eval for $c at ${N} frames" >> $RB/final_eval_1200.log
done
echo "$(date +%H:%M) ALL 1200 m evals dispatched" >> $RB/final_eval_1200.log
