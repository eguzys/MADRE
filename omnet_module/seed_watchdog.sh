#!/bin/bash
# Seed-population watchdog. Backgrounded by train_omnet_arton.sh.
#
# WHY THIS EXISTS: if worker identity were INFERRED from a class counter that
# resets in every forked worker, all collector envs would simulate the SAME
# world. Unit tests do not see that, and MAPPO would train to completion on
# correlated copies of one trajectory with plausible-looking curves.
#
# The per-env tripwire in omnet_env.py catches collisions between DIFFERENT
# worker_index values. It CANNOT catch a uniform-index collapse -- every env
# claiming the same index agrees with itself. This checks the POPULATION:
# N envs must produce N distinct episodeSeeds. That is the assertion the
# subsystem is actually for.
#
# Usage: seed_watchdog.sh <slurm_job_id> <expected_n> [delay_seconds]
set -uo pipefail
JOBID="${1:?need slurm job id}"
EXPECTED="${2:?need expected env count}"
DELAY="${3:-240}"

# Bounded poll, not a fixed sleep. A fixed delay cannot tell a slow start from
# a dead run -- both look like "no children". The launcher's run.py either
# exists or it does not, and that is the discriminator:
#   run.py alive, no children yet  -> slow start, keep waiting
#   run.py gone                    -> the run DIED; FAIL loudly, do not SKIP
#   N distinct seeds >= expected   -> PASS immediately
#   window expires, run.py alive   -> INCONCLUSIVE, report, do not kill
LAUNCHER=$PPID
DEADLINE=$(( $(date +%s) + DELAY ))

descendants() {
  local parent=$1 child
  for child in $(ps -o pid= --ppid "$parent" 2>/dev/null); do
    echo "$child"
    descendants "$child"
  done
}

while :; do
  TREE=$(descendants "$LAUNCHER")
  PIDLIST=$(echo "$TREE" | tr '\n' ',' | sed 's/,$//')

  if [ -n "$PIDLIST" ]; then
    RUNPY=$(ps -o cmd= -p "$PIDLIST" 2>/dev/null | grep -c 'run\.py' || true)
    # Count CHILDREN and DISTINCT SEEDS separately. If env construction stalls
    # (e.g. at 8 of 11), only ONE child may exist, and "1 distinct seed" must
    # not be read as "ten envs sharing one world". Those are opposite problems:
    #   children ~= N, distinct < N -> real collapse, kill
    #   children  <  N              -> construction never finished, REPORT only
    KIDS=$(ps -o cmd= -p "$PIDLIST" 2>/dev/null | grep -c 'omnet_smoke .*episodeSeed' || true)
    SEEDS=$(ps -o cmd= -p "$PIDLIST" 2>/dev/null \
            | grep -o 'episodeSeed=[0-9]*' | sort -u)
  else
    RUNPY=0
    SEEDS=""
  fi
  N=$(printf '%s\n' "$SEEDS" | grep -c 'episodeSeed=' || true)

  if [ "$N" -ge "$EXPECTED" ]; then
    echo "SEED_CHECK: PASS -- $N distinct episodeSeeds (expected >= $EXPECTED)"
    printf '%s\n' "$SEEDS" | sed 's/^/SEED_CHECK:   /'
    exit 0
  fi

  if [ "$RUNPY" -eq 0 ]; then
    echo "SEED_CHECK: FAIL -- run.py is GONE. The training process died before"
    echo "SEED_CHECK: the seed population could be verified. This is a dead run,"
    echo "SEED_CHECK: not a slow one. Read the traceback above."
    exit 1
  fi

  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    if [ "$N" -eq 0 ]; then
      echo "SEED_CHECK: INCONCLUSIVE -- run.py alive after ${DELAY}s but no OMNeT"
      echo "SEED_CHECK: children ever appeared. Not a pass. Check by hand."
      exit 0
    fi
    if [ "$KIDS" -lt "$EXPECTED" ]; then
      echo "SEED_CHECK: STALL -- only $KIDS OMNeT children exist after ${DELAY}s"
      echo "SEED_CHECK: (expected >= $EXPECTED). Env CONSTRUCTION never finished."
      echo "SEED_CHECK: This is NOT a shared-world collapse -- the envs that would share a world"
      echo "SEED_CHECK: were never built. Not killing; the seed question is"
      echo "SEED_CHECK: unanswerable until construction completes. Check by hand:"
      echo "SEED_CHECK:   grep -c 'PettingZoo in TorchRL is tested' <joblog>"
      printf '%s\n' "$SEEDS" | sed 's/^/SEED_CHECK:   /'
      exit 0
    fi
    echo "SEED_CHECK: FAIL -- $KIDS children but only $N distinct episodeSeeds"
    echo "SEED_CHECK: (expected >= $EXPECTED)."
    printf '%s\n' "$SEEDS" | sed 's/^/SEED_CHECK:   /'
    echo "SEED_CHECK: this is a shared-world collapse: multiple envs simulating ONE world."
    echo "SEED_CHECK: training would finish and the curves would look fine."
    echo "SEED_CHECK: killing job $JOBID rather than burning the allocation."
    scancel "$JOBID"
    exit 1
  fi

  sleep 30
done
