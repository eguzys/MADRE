#!/bin/bash
#SBATCH --job-name=U100diag
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --time=08:00:00
#SBATCH -o /usr/itetnas04/data-scratch-01/eguzys/data/u100diag/slurm/%x_%j.out
# Args passed through: <cell-tag> <users> <seed_start> <episodes> <ckpt|best> <out-name>
export PATH=/usr/sepp/bin:/usr/bin:/bin:$PATH
export SLURM_CONF=/home/sladmitet/slurm/slurm.conf
S=/usr/itetnas04/data-scratch-01/eguzys/data/u100diag
echo "=== $(hostname) job=$SLURM_JOB_ID args: $* SENDERPROB=${DIAG_SENDERPROB:-} ==="
exec bash $S/launch2_ub.sh "$@"
