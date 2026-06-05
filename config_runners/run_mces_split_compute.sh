#!/bin/bash
#PBS -N mces_compute
#PBS -l select=1:ncpus=32:mem=16gb:scratch_local=20gb
#PBS -l walltime=6:00:00
# MCES-distance split, stage 2 (ARRAY): one HDF5 batch per array index, computed
# with myopic-MCES using Fleming's recommended settings (dynamic bound, CPLEX/CBC
# single-threaded, silent). Submit with: qsub -J 0-<N-1> -v REPO_DIR,ENV_PREFIX,WORKDIR[,CEILING]
# NOTE: an explicit --solver is required for --solver_no_msg to take effect
# (the 'default' solver path ignores solver options and floods stdout).
set -euo pipefail
test "$(whoami)" = "jozefov_147" || { echo "Refuse: wrong account"; exit 1; }
: "${REPO_DIR:?}"; : "${ENV_PREFIX:?}"; : "${WORKDIR:?}"
CEILING="${CEILING:-10}"
SOLVER="${SOLVER:-PULP_CBC_CMD}"   # set to CPLEX_CMD if a CPLEX module is loaded
TIME_LIMIT="${TIME_LIMIT:-30}"    # per-pair ILP-solver cap (s)
WALL_CAP="${WALL_CAP:-45}"        # hard per-pair wall cap (s); bounds the matching-bound tail
NJOBS="${PBS_NCPUS:-$(nproc)}"

BATCH="$WORKDIR/batches/batch${PBS_ARRAY_INDEX}.hdf5"
# Idempotent: skip batches already computed, so the array can be re-run safely
# (durable per-batch checkpointing — a completed batch carries its 'mces' dataset).
if "$ENV_PREFIX/bin/python" -c "import h5py,sys; sys.exit(0 if 'mces' in h5py.File('$BATCH','r') else 1)" 2>/dev/null; then
  echo "batch ${PBS_ARRAY_INDEX} already computed — skipping"; exit 0
fi
mkdir -p "$SCRATCHDIR/tmp"; export TMPDIR="$SCRATCHDIR/tmp"
module add mambaforge; mamba activate "$ENV_PREFIX"
cd "$REPO_DIR"

cp "$BATCH" "$SCRATCHDIR/" || exit 2
LOCAL="$SCRATCHDIR/$(basename "$BATCH")"
# Computes via myopic-MCES (dynamic bound) and writes distances back into LOCAL.
python -m mces_split.compute \
  --batch "$LOCAL" --threshold "$CEILING" --solver "$SOLVER" \
  --solver-time-limit "$TIME_LIMIT" --wall-cap "$WALL_CAP" --num-jobs "$NJOBS" \
  > "$SCRATCHDIR/compute_${PBS_ARRAY_INDEX}.log" 2>&1 || { tail -50 "$SCRATCHDIR/compute_${PBS_ARRAY_INDEX}.log"; exit 3; }
cp "$LOCAL" "$BATCH" || exit 4   # batch now carries the computed distances
clean_scratch
