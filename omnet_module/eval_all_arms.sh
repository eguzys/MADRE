#!/bin/bash
# eval_all_arms.sh - submit the n=100 evaluation for every finished arm of the
# frozen/learned factorial, with the right flags per condition.
#
#   ./omnet_module/eval_all_arms.sh            # submit what is ready
#   DRY=1 ./omnet_module/eval_all_arms.sh      # show what WOULD be submitted
#
# Run ON arton (it submits sbatch jobs and reads $SCRATCH directly).
#
# Why a driver: the two conditions need DIFFERENT eval invocations and getting
# it wrong is silent.
#   frozen  -> ENC=<encoder.pt>  (eval_arton.sbatch then sets encoder_mode=
#              frozen + the two vars omnet_env.py demands). The encoder MUST be
#              the one the arm trained with -- a mismatch does NOT raise,
#              because any encoder of this shape yields a valid 8-dim latent.
#   learned -> no ENC (each arch pools with its own trained encoder)
set -uo pipefail
S="${OM2G_SCRATCH:-/usr/itetnas04/data-scratch-01/eguzys/data}"
R=$S/resilient-backbone
ENC_FROZEN="$S/frozen_enc/models/enc_mixed_sorted_drop_idtag_s0.pt"
EXPECT_SHA=26e76779f0515fc5
MIN_FRAMES=${MIN_FRAMES:-1000000}
cd "$R" || exit 1

# The frozen encoder must be the exact file the arms trained against.
got=$(sha256sum "$ENC_FROZEN" 2>/dev/null | cut -c1-16)
if [ "$got" != "$EXPECT_SHA" ]; then
  echo "FATAL: frozen encoder sha is '$got', expected '$EXPECT_SHA'."
  echo "       Scoring the arms against a different encoder silently changes"
  echo "       their observation and would NOT raise. Refusing."; exit 1
fi
echo "## frozen encoder sha OK: $got"

submitted=0; skipped=0
for mode in frozen learned; do
  for arch in st deepsets ffn; do
    for ts in 69 70 71; do
      tag="${mode}_${arch}_ts${ts}"
      # seed 69 of the learned column is the pre-existing canonical run, whose
      # tag does not follow this scheme -- skip it here and score it from its
      # own dir (selftok_ds / selftok_ffn / the bare _staging ST reference).
      if [ "$mode" = learned ] && [ "$ts" = 69 ]; then
        echo "-- skip $tag (canonical Aug-13 run; different tag scheme)"; skipped=$((skipped+1)); continue
      fi
      # frames check: refuse partial nets rather than scoring them
      fr=0
      for d in "$R"/src/out/benchmarl_omnet/*"${tag}"*/*/; do
        f=$(ls "$d"/checkpoints/ 2>/dev/null | grep -oE '^checkpoint_[0-9]+\.pt$' \
            | sed -E 's/checkpoint_([0-9]+)\.pt/\1/' | sort -n | tail -1)
        [ -n "$f" ] && [ "$f" -gt "$fr" ] && fr=$f
      done
      if [ "$fr" -lt "$MIN_FRAMES" ]; then
        echo "-- skip $tag (frame $fr < $MIN_FRAMES, still training)"; skipped=$((skipped+1)); continue
      fi
      if [ "$mode" = frozen ]; then
        cmd="sbatch --export=ALL,TAG=$tag,ENC=$ENC_FROZEN,OUTD=factorial omnet_module/eval_arton.sbatch"
      else
        cmd="sbatch --export=ALL,TAG=$tag,OUTD=factorial omnet_module/eval_arton.sbatch"
      fi
      if [ -n "${DRY:-}" ]; then echo "DRY: $cmd"; else eval "$cmd" >/dev/null 2>&1 && echo "++ submitted $tag (frame $fr)"; fi
      submitted=$((submitted+1))
    done
  done
done
echo "## submitted=$submitted skipped=$skipped"
echo "## results -> $S/eval_out/factorial/<tag>_n100.csv"
echo "## then:  python omnet_module/analyze_seeds.py \\"
echo "##          --glob 'out/factorial/frozen_*_n100.csv' \\"
echo "##          --pattern 'frozen_(?P<arch>st|deepsets|ffn)_ts(?P<seed>[0-9]+)' \\"
echo "##          --baseline out/n100/hover_n100.csv --ceiling out/n100/greedy_n100.csv"
