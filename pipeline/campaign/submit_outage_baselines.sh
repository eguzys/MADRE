#!/bin/bash
# The three other non-learned baselines in the fig:outages world (GTMG, AGG, R1H).
# Same shard scripts and user-beacon export as the 100-user baselines (GTMG with SUITE_GTM_OBJ=global and
#   16 h wall, R1H via relay2_shard_ub (RELAY_MODE defaults to blind), AGG via suite_shard2_ub) but USERS=40
#   like NDUB*_ENG (baseline_shard_cl_ub, DIAG_CLUSTERS=4, br12rtL128s1 world = 1200 m, cluster spawn, arena
#   3800), chi 0.1..1.0, 10 shards x 10 episodes, seeds 0..99. Submitted HELD. Never resubmits an existing name.
set -u
export PATH=/usr/sepp/bin:/usr/bin:/bin:$PATH
export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
P=/usr/itetnas04/data-scratch-01/eguzys/data/u100diag/suite
S=/usr/itetnas04/data-scratch-01/eguzys/data/u100diag
UB=OM2G_USER_BEACON=1,OM2G_USER_BEACON_INTERVAL=1.0,OM2G_USER_BEACON_BYTES=1,OMNET_HOST_CODE=/usr/itetnas04/data-scratch-01/eguzys/data/ub_smoke,OMNET_SMOKE_DIR=/usr/itetnas04/data-scratch-01/eguzys/data/ub_smoke/omnet_smoke
LEDGER=$S/c4s/ledger_outage_baselines.tsv
for pair in 0.1:10 0.2:20 0.4:40 0.6:60 1.0:100; do
  SPR=${pair%%:*}; LBL=${pair##*:}
  for cp in R1H:relay_1hop GTMG:gtm AGG:aggregate; do
    CODE=${cp%%:*}; POL=${cp#*:}; NAME=NDUB${LBL}_${CODE}
    case $CODE in R1H) SH=$P/relay2_shard_ub.sbatch; X=""; T="";; GTMG) SH=$P/suite_shard2_ub.sbatch; X=",SUITE_GTM_OBJ=global"; T="--time=16:00:00";;
                  AGG) SH=$P/suite_shard2_ub.sbatch; X=""; T="";; esac
    if ls $P/logs/${NAME}_s0.log >/dev/null 2>&1 || squeue -u eguzys -h -o "%j" | grep -q "^${NAME}_s"; then echo "SKIP $NAME exists"; continue; fi
    for SS in 0 10 20 30 40 50 60 70 80 90; do
      j=$(sbatch -H --parsable $T --job-name=${NAME}_s${SS} --export=ALL,DIAG_SENDERPROB=${SPR},${UB}${X} "$SH" "$POL" 40 "$SS" 10 "$NAME" 2>&1 | tail -1)
      case "$j" in ''|*[!0-9]*) echo "FAIL $NAME s$SS -> $j"; continue;; esac
      scontrol update JobId=$j Partition=disco.med,cpu.normal,gpu.normal >/dev/null 2>&1
      printf "%s\t%s\t%s\t%s\n" "$(date -u +%FT%TZ)" $NAME $SS $j >> $LEDGER
    done
    echo "submitted HELD $NAME ($POL, chi=$SPR)"
  done
done
