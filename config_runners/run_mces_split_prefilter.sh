#!/bin/bash
#PBS -N mces_prefilter
#PBS -l select=1:ncpus=4:mem=16gb:scratch_local=10gb
#PBS -l walltime=4:00:00
# MCES-distance split, stage 1: rigorous two-bound pre-filter -> candidate-pair
# HDF5 batches. Pass via `qsub -v`: REPO_DIR, ENV_PREFIX, WORKDIR [, CEILING, BATCH_SIZE].
set -euo pipefail
test "$(whoami)" = "jozefov_147" || { echo "Refuse: wrong account"; exit 1; }
: "${REPO_DIR:?}"; : "${ENV_PREFIX:?}"; : "${WORKDIR:?}"
CEILING="${CEILING:-10}"; BATCH_SIZE="${BATCH_SIZE:-10000000}"

mkdir -p "$WORKDIR" "$SCRATCHDIR/tmp"; export TMPDIR="$SCRATCHDIR/tmp"
module add mambaforge; mamba activate "$ENV_PREFIX"
cd "$REPO_DIR"

# Batches are written straight to persistent WORKDIR (one sequential write).
python -m mces_split.prefilter \
  --data-dir "$REPO_DIR/data/nist" --out-dir "$WORKDIR" \
  --ceiling "$CEILING" --batch-size "$BATCH_SIZE"
clean_scratch
