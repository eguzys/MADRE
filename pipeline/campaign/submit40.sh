#!/bin/bash
# Submit the 40-user eval families HELD (-H) for one trained base cell.
# Usage: submit40.sh <rung> <cell> <seed> <families...>
#   families: FS SEP NDUB USPAWN CKFIN   (SEP = 1300..1700, NDUB = chi 0.1/0.2/0.4/0.6/1.0)
# Beacon OFF (shard.sbatch) for FS/SEP/USPAWN/CKFIN; beacon ON (shard_ub.sbatch + UB export) for NDUB only,
# with the same settings as the existing FS / SEP / USPAWN / CKFIN / NDUB families.
# Checkpoint: pick_best_ckpt over the cell's OWN normalizedmappo run dir (scoped, never bare "best"),
# passed explicitly so every shard loads the same file. CKFIN: checkpoint_300000.pt of that run dir.
# Idempotent: skips a family that already has a log, a CSV, or a queued shard. Writes NEW names only.
set -u
export PATH=/usr/sepp/bin:/usr/bin:/bin:$PATH
export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
unset OM2G_USER_BEACON OM2G_USER_BEACON_INTERVAL OM2G_USER_BEACON_BYTES OMNET_HOST_CODE OMNET_SMOKE_DIR DIAG_SPAWN DIAG_MINSEP DIAG_SENDERPROB DIAG_CKPT
UB=OM2G_USER_BEACON=1,OM2G_USER_BEACON_INTERVAL=1.0,OM2G_USER_BEACON_BYTES=1,OMNET_HOST_CODE=/usr/itetnas04/data-scratch-01/eguzys/data/ub_smoke,OMNET_SMOKE_DIR=/usr/itetnas04/data-scratch-01/eguzys/data/ub_smoke/omnet_smoke
RB=/usr/itetnas04/data-scratch-01/eguzys/data/resilient-backbone
S=/usr/itetnas04/data-scratch-01/eguzys/data/u100diag
PY=/usr/itetnas04/data-scratch-01/eguzys/data/padre_env/bin/python
LEDGER=$S/c4s/ledger40.tsv; mkdir -p $S/c4s
RUNG=$1; CELL=$2; SD=$3; shift 3
outer=$(ls -1d $RB/src/out/OM2G/OM2G_compressed_st_s${SD}_${CELL}/ 2>/dev/null | head -1)
[ -n "$outer" ] || { echo "FATAL no outer dir for $CELL"; exit 1; }
nrun=$(ls -1d ${outer}normalizedmappo_* 2>/dev/null | wc -l)
[ "$nrun" = 1 ] || { echo "FATAL $CELL has $nrun normalizedmappo run dirs (want exactly 1)"; exit 1; }
run=$(ls -1d ${outer}normalizedmappo_*); rel=${run#$RB/src/}
CK=$(cd $RB/src && $PY ../om2g/pick_best_ckpt.py "$rel" 2>/dev/null | tail -1)
[ -n "$CK" ] && [ -s "$RB/src/$CK" ] || { echo "FATAL $CELL pick_best gave '$CK'"; exit 1; }
case "$CK" in "$rel"/checkpoints/*) ;; *) echo "FATAL $CELL best $CK outside $rel"; exit 1;; esac
MEM=""; [ "$RUNG" = L512 ] && MEM="--mem=32G"
cd $RB
sub() {  # sub <NAME> <sbatch> <ckpt> <export>
  local NAME=$1 SB=$2 K=$3 EXP=$4 ss j
  if ls $S/logs/${NAME}_s0.log >/dev/null 2>&1 || ls $S/out/ 2>/dev/null | grep -q "_${NAME}_n10_seed"; then echo "SKIP $NAME (exists)"; return; fi
  if squeue -u eguzys -h -o "%j" | grep -q "^${NAME}_s"; then echo "SKIP $NAME (queued)"; return; fi
  for ss in 0 10 20 30 40 50 60 70 80 90; do
    j=$(sbatch -H --parsable --nice=3000 $MEM --export=$EXP --job-name=${NAME}_s$ss $S/$SB $CELL 40 $ss 10 "$K" $NAME 2>&1 | tail -1)
    case "$j" in ''|*[!0-9]*) echo "  FAIL $NAME s$ss -> $j"; continue;; esac
    scontrol update JobId=$j Partition=disco.med,cpu.normal,gpu.normal >/dev/null 2>&1
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -u +%FT%TZ)" $NAME $ss $j $CELL "$K" "$EXP" >> $LEDGER
  done
  echo "submitted HELD $NAME ($CELL, $SB, ckpt=$(basename $K), export=$EXP $MEM)"
}
for fam in "$@"; do
  case $fam in
    FS)     sub FS_${RUNG}s${SD} shard.sbatch "$CK" ALL;;
    SEP)    for d in 1300 1400 1500 1600 1700; do sub SEP${d}_${RUNG}s${SD} shard.sbatch "$CK" ALL,DIAG_MINSEP=$d; done;;
    USPAWN) sub USPAWN_${RUNG}s${SD} shard.sbatch "$CK" ALL,DIAG_SPAWN=uniform;;
    CKFIN)  F=$rel/checkpoints/checkpoint_300000.pt
            [ -s $RB/src/$F ] || { echo "FATAL no $F"; continue; }
            sub CKFIN_${RUNG}s${SD} shard.sbatch "$F" ALL;;
    NDUB)   for pair in 0.1:10 0.2:20 0.4:40 0.6:60 1.0:100; do
              p=${pair%%:*}; l=${pair##*:}
              sub NDUB${l}_${RUNG}s${SD} shard_ub.sbatch "$CK" ALL,DIAG_SENDERPROB=$p,$UB
            done;;
    *) echo "unknown family $fam";;
  esac
done
