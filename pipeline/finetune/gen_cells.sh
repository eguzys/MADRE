#!/bin/bash
# Generate beacon-ON 100-user fine-tune cells for one (rung, seed) at the given chis.
# Usage: gen_cells.sh <rung> <seed> <sp...>      rung in L8 L16 L32 L64 L128 L256 L512, sp in sp05 sp1 sp2 sp4 sp6 sp10
# Template = the seed-1 UB cell of the same rung/chi (L512: the seed-2 cell). Changed EXACTLY: tag (job-name, -o, -e,
# RUN_TAG, TMPDIR), OM2G_RESUME_FROM -> u100diag_ckpts/<pfx>_u100<sp>UB_s<seed>/checkpoints/checkpoint_<best>.pt,
# OMNET_RESUME_MAX_FRAMES = best + 48000, the "best on base dir = N" comment, launch line seed. Everything else
# (SENDER_PROB, STALL 21600, six UB lines, --exclude, --mem incl. 128G for L512 sp10, --time) is kept and asserted.
# best = the single checkpoint inside the per-chi UB stage. Writes NEW files only; appends to $S/u100diag/c4s/cells.tsv.
set -euo pipefail
R=$1; SD=$2; shift 2
D=/usr/itetnas04/data-scratch-01/eguzys/data; C=$D/resilient-backbone/arch_out/cells; K=$D/u100diag_ckpts
TSV=$D/u100diag/c4s/cells.tsv
[ -f $TSV ] || printf "cell\trung\tseed\tsp\ttemplate\tub_stage\tckpt\tresume_max_frames\tbase_outdir\n" > $TSV
case $R in L256|L512) CP=br12e2e$R; PF=e$R;; *) CP=br12rt$R; PF=$R;; esac
TS=1; [ $R = L512 ] && TS=2
for sp in "$@"; do
  NEW=${CP}s${SD}u100${sp}UB; OLD=${CP}s${TS}u100${sp}UB; T=$C/$OLD.sh; OUT=$C/$NEW.sh
  [ -f $T ] || { echo "FATAL no template $T"; exit 1; }
  [ -e $OUT ] && { echo "SKIP $NEW exists"; continue; }
  UBS=${PF}_u100${sp}UB_s${SD}
  n=$(ls $K/$UBS/checkpoints/checkpoint_*.pt 2>/dev/null | wc -l); [ "$n" = 1 ] || { echo "FATAL $UBS has $n ckpts"; exit 1; }
  ck=$(basename $(ls $K/$UBS/checkpoints/checkpoint_*.pt)); best=${ck#checkpoint_}; best=${best%.pt}; mf=$((best+48000))
  grep -qx "bash om2g/ub/om2g_launch_arton.sh compressed st ${TS} 300000" $T || { echo "FATAL launch line in $T"; exit 1; }
  tmp=$(mktemp)
  perl -pe "s/\\b\\Q$OLD\\E\\b/$NEW/g;
            s#u100diag_ckpts/[^/]+/checkpoints/checkpoint_[0-9]+\\.pt#u100diag_ckpts/$UBS/checkpoints/$ck#;
            s/^export OMNET_RESUME_MAX_FRAMES=[0-9]+\$/export OMNET_RESUME_MAX_FRAMES=$mf/;
            s/(best on base dir = )[0-9]+/\${1}$best/;
            s#^bash om2g/ub/om2g_launch_arton.sh compressed st $TS 300000\$#bash om2g/ub/om2g_launch_arton.sh compressed st $SD 300000#" $T > $tmp
  # assertions
  grep -qx "export OM2G_RUN_TAG=$NEW" $tmp
  grep -qx "#SBATCH --job-name=$NEW" $tmp
  grep -qx "export OM2G_RESUME_FROM=$K/$UBS/checkpoints/$ck" $tmp
  grep -qx "export OMNET_RESUME_MAX_FRAMES=$mf" $tmp
  grep -qx "bash om2g/ub/om2g_launch_arton.sh compressed st $SD 300000" $tmp
  ! grep -q "$OLD" $tmp
  for k in OM2G_SENDER_PROB OM2G_STALL_SECONDS OM2G_USER_BEACON OM2G_USER_BEACON_INTERVAL OM2G_USER_BEACON_BYTES OMNET_HOST_CODE OMNET_SMOKE_DIR OM2G_USERS OM2G_LATENT_DIM OM2G_CODEC_MODE; do
    [ "$(grep -E "^export $k=" $T)" = "$(grep -E "^export $k=" $tmp)" ] || { echo "FATAL $k changed in $NEW"; rm -f $tmp; exit 1; }
  done
  for k in "--exclude" "--mem" "--time" "--cpus-per-task"; do
    [ "$(grep -E "^#SBATCH $k" $T)" = "$(grep -E "^#SBATCH $k" $tmp)" ] || { echo "FATAL $k changed in $NEW"; rm -f $tmp; exit 1; }
  done
  nd=$(diff $T $tmp | grep -c '^>' || true)
  [ "$nd" -le 9 ] || { echo "FATAL $NEW differs from template in $nd lines"; diff $T $tmp; rm -f $tmp; exit 1; }
  mv $tmp $OUT; chmod 644 $OUT
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $NEW $R $SD $sp $OLD.sh $UBS $ck $mf OM2G_compressed_st_s${SD}_${CP}s${SD} >> $TSV
  echo "GEN $NEW  (template $OLD, stage $UBS, $ck, max $mf, $nd lines changed)"
done
