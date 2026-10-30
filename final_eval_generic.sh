#!/bin/bash
# Auto-score any cell once it leaves the queue. The eval config is DERIVED from
# the cell's own script:
#   * OM2G_LATENT_DIM/OMNET_USER_ENC_OUT_DIM must match the checkpoint's encoder
#     or the model is built with to_latent[8,64] and the load fails.
#   * the ARM (compressed/uncompressed) is part of the model shape, not a label.
# World (MINSEP/MAXNN) is read from the cell too, so a 1465 cell is never scored
# in the 1200 world.
RB="${OM2G_SCRATCH:-/usr/itetnas04/data-scratch-01/eguzys/data}/resilient-backbone"
export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
LOG=$RB/final_eval_generic.log
for c in "$@"; do
  S=$RB/arch_out/cells/$c.sh
  [ -f "$S" ] || { echo "$(date +%H:%M) SKIP $c (no cell script)" >>$LOG; continue; }
  while squeue -u eguzys -h -n "$c" -o "%t" 2>/dev/null | grep -q .; do sleep 120; done
  N=$(find $RB/src/out/OM2G -path "*_${c}/*" -name "checkpoint_*.pt" -printf "%f\n" 2>/dev/null | sed 's/checkpoint_//;s/.pt//' | sort -n | tail -1)
  CK=$(find $RB/src/out/OM2G -path "*_${c}/*" -name "checkpoint_${N}.pt" | head -1)
  [ -z "$CK" ] && { echo "$(date +%H:%M) SKIP $c (no checkpoint)" >>$LOG; continue; }
  # A cell leaves the queue when it FINISHES or when it is killed (e.g. OOM).
  # Only a checkpoint covering most of the run is scored.
  MINFR=${OM2G_EVAL_MIN_FRAMES:-250000}
  if [ "${N:-0}" -lt "$MINFR" ]; then
    echo "$(date +%H:%M) SKIP $c (only ${N} frames < ${MINFR} -- died, not finished)" >>$LOG
    continue
  fi
  REL=${CK#$RB/src/}
  [ -f "$RB/src/$REL" ] || { echo "$(date +%H:%M) SKIP $c (bad path $REL)" >>$LOG; continue; }
  SEP=$(grep -aoE '^export OM2G_MINSEP=[0-9]+' "$S"|tail -1|cut -d= -f2)
  MNN=$(grep -aoE '^export OM2G_MAXNN=[0-9]+' "$S"|tail -1|cut -d= -f2)
  LAT=$(grep -aoE '^export OM2G_LATENT_DIM=[0-9]+' "$S"|tail -1|cut -d= -f2)
  # The arm is part of the MODEL/CHANNEL condition, not a label (a nocontrol
  # cell trains with beaconInterval 3600 s). It is read from the directory
  # prefix; a cell whose arm cannot be determined is skipped.
  ARM=$(find $RB/src/out/OM2G -maxdepth 1 -type d -name "*_${c}" -printf '%f\n' 2>/dev/null | head -1 | sed -E 's/^OM2G_(compressed|uncompressed|nocontrol)_.*/\1/')
  case "$ARM" in compressed|uncompressed|nocontrol) ;; *) echo "$(date +%H:%M) SKIP $c (cannot determine arm from dir)" >>$LOG; continue ;; esac
  W="OM2G_DRONES=8,OMNET_NUM_DRONES=8,OM2G_USER_SPAWN=archipelago,OM2G_CLUSTERS=4,OM2G_ARENA=3800,OM2G_SPREAD=25,OM2G_PCROSS=0.9,OM2G_MINSEP=${SEP},OM2G_MAXNN=${MNN},OM2G_MARGIN=300,OM2G_SPAWN=cluster,OM2G_USER_PARTNER_TAG=1,OM2G_PARTNER_TAG_MODE=group,OM2G_CTRL_SPREADING=8,OM2G_PHYSICAL_METRIC=1,OM2G_PEER_MEMORY=1"
  SH="OM2G_SHAPE_FIEDLER=0,OM2G_SHAPE_VORONOI=0,OM2G_SHAPE_EDGE=0,OM2G_SHAPE_SPREAD=0,OM2G_SHAPE_KERNEL=exp,OM2G_SHAPE_VOR_KERNEL=exp"
  EX=""; [ -n "$LAT" ] && EX=",OM2G_LATENT_DIM=${LAT},OMNET_USER_ENC_OUT_DIM=${LAT}"
  # world_from_cell.py drops OM2G_MSG_ABLATE ("identifies an arm, not a world"),
  # but a `sil` cell trained with mask_out must be scored with it too. The
  # ablation is part of the ARM, so it is carried over here.
  ABL=$(grep -aoE '^export OM2G_MSG_ABLATE=[^ ]*' "$S" | tail -1 | cut -d= -f2)
  [ -n "$ABL" ] && EX="${EX},OM2G_MSG_ABLATE=${ABL}"
  for SS in 0 10 20 30 40 50 60 70 80 90; do
    sbatch --parsable --export=ALL,$W,$SH$EX,OM2G_RUN_TAG=K${c}${SS} \
      $RB/om2g/om2g_eval_arton.sh $ARM "$REL" K_${c}_${N}_${SS} 10 $SS >/dev/null 2>&1
  done
  echo "$(date +%H:%M) fired n=100 eval $c @${N} sep=${SEP} maxnn=${MNN} lat=${LAT:-8} arm=${ARM}" >>$LOG
done
echo "$(date +%H:%M) generic watcher done: $*" >>$LOG
