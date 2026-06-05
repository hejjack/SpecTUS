#!/bin/bash
#PBS -N mces_report
#PBS -l select=1:ncpus=2:mem=32gb:scratch_local=10gb
#PBS -l walltime=2:00:00
# MCES-distance split, stage 3: build near_pairs.parquet and report the far/close
# split at the requested thresholds (default 6 7 8 9 10), excluding molecules with
# any incomputable pair so the far set is certain at every threshold. Pass via
# `qsub -v`: REPO_DIR, ENV_PREFIX, WORKDIR [, RESULTS, THRESHOLDS, EXCLUDE_INCOMPUTABLE].
set -euo pipefail
test "$(whoami)" = "jozefov_147" || { echo "Refuse: wrong account"; exit 1; }
: "${REPO_DIR:?}"; : "${ENV_PREFIX:?}"; : "${WORKDIR:?}"
RESULTS="${RESULTS:-$WORKDIR/results}"
THRESHOLDS="${THRESHOLDS:-6 7 8 9 10}"
EXCLUDE="${EXCLUDE_INCOMPUTABLE:-1}"

module add mambaforge; mamba activate "$ENV_PREFIX"
cd "$REPO_DIR"
FLAGS=""; [ "$EXCLUDE" = "1" ] && FLAGS="--exclude-incomputable"
python -m mces_split.split --out-dir "$WORKDIR" --results-dir "$RESULTS" --thresholds $THRESHOLDS $FLAGS
