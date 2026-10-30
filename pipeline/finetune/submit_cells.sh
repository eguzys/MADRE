#!/bin/bash
# Submit fine-tune cells HELD (-H; slurm 23.11 ignores SBATCH_HOLD); release them with scontrol release.
# Usage: submit_cells.sh <cell...>. Skips a cell already queued, already in the submission ledger, or with an existing .out.
set -u
export PATH=/usr/sepp/bin:/usr/bin:/bin:$PATH SLURM_CONF=/home/sladmitet/slurm/slurm.conf
RB=/usr/itetnas04/data-scratch-01/eguzys/data/resilient-backbone; L=/usr/itetnas04/data-scratch-01/eguzys/data/u100diag/c4s/cells_jobs.tsv
cd $RB
for c in "$@"; do
  [ -f arch_out/cells/$c.sh ] || { echo "FATAL no cell $c"; continue; }
  squeue -u eguzys -h -o "%j" | grep -qx "$c" && { echo "SKIP $c queued"; continue; }
  awk -F'\t' -v c=$c '$2==c' $L 2>/dev/null | grep -q . && { echo "SKIP $c already submitted (ledger)"; continue; }
  [ -e om2g_logs/cells/$c.out ] && { echo "SKIP $c has .out (inspect by hand)"; continue; }
  j=$(sbatch -H --parsable arch_out/cells/$c.sh 2>&1 | tail -1)
  case "$j" in ''|*[!0-9]*) echo "FAIL $c -> $j"; continue;; esac
  scontrol update JobId=$j Partition=cpu.normal,gpu.normal >/dev/null 2>&1
  printf "%s\t%s\t%s\tinitial\n" "$(date -u +%FT%TZ)" "$c" "$j" | sed 's/^\([^\t]*\)\t/\1\t/' >> $L
  echo "HELD $c -> $j"
done
