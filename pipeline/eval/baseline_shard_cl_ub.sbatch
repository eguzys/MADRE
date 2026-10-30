#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH -o /usr/itetnas04/data-scratch-01/eguzys/data/u100diag/slurm/%x_%j.out
# Args: <policy> <users> <seed_start> <episodes> <name> ; env DIAG_SENDERPROB optional
export PATH=/usr/sepp/bin:/usr/bin:/bin:$PATH; export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
S=/usr/itetnas04/data-scratch-01/eguzys/data/u100diag; D=/usr/itetnas04/data-scratch-01/eguzys/data/resilient-backbone
POLICY=$1; USERS=$2; SS=$3; EP=$4; NAME=$5
while IFS= read -r line; do case "$line" in *OM2G_RUN_TAG=*|*OM2G_TMPDIR=*|*OM2G_EXTRA_OVERRIDES=*|*OM2G_SHAPE_*) continue;; esac; eval "$line"; done < <(grep -E "^\s*export (OM2G|OMNET)_" $D/arch_out/cells/br12rtL128s1.sh)
export OM2G_CLUSTERS=${DIAG_CLUSTERS:?DIAG_CLUSTERS required}
# $DIAG_MINSEP: override the cluster separation band of the sourced 1200 m cell;
# MAXNN = MINSEP + 300, as in ew.sh.
[ -n "${DIAG_MINSEP:-}" ] && { export OM2G_MINSEP=$DIAG_MINSEP; export OM2G_MAXNN=$((DIAG_MINSEP+300)); echo "[world override] MINSEP=$OM2G_MINSEP MAXNN=$OM2G_MAXNN"; }
export OM2G_ARENA=${DIAG_ARENA:-3800}
export OM2G_SPAWN=${DIAG_SPAWN:-cluster} OM2G_CODEC_MODE=e2e OM2G_LATENT_DIM=128 OM2G_RUN_TAG=$NAME OMP_NUM_THREADS=1
[ "$USERS" != "40" ] && export OM2G_USERS=$USERS
export OM2G_DUMP_TRACE=$S/out/trace_${NAME}_s${SS}.json
echo "=== $(hostname) job=$SLURM_JOB_ID $POLICY users=$USERS ss=$SS ep=$EP name=$NAME SENDERPROB=${DIAG_SENDERPROB:-} ==="
exec bash $S/baselines_diag_ub.sh $POLICY $EP $SS > $S/logs/${NAME}_s${SS}.log 2>&1
