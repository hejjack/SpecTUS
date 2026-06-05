#!/bin/bash
# One-shot launcher for the MCES-distance split on MetaCentrum (plzen1).
# Stage 1 (prefilter, blocking) -> stage 2 (compute array). Stage 3 (report) is a
# quick final step printed to run once the array finishes. Override any default
# via environment, e.g. WORKDIR=... CEILING=10 bash config_runners/run_mces_split.sh
set -euo pipefail

PLZEN_HOME="${PLZEN_HOME:-/storage/plzen1/home/jozefov_147}"
export REPO_DIR="${REPO_DIR:-$PLZEN_HOME/projects/SpecTUS}"
export ENV_PREFIX="${ENV_PREFIX:-$PLZEN_HOME/.conda/envs/spectus_mces}"
export WORKDIR="${WORKDIR:-$PLZEN_HOME/projects/spectus_mces_runs}"   # holds batches (tens of GB) + results
export CEILING="${CEILING:-10}"
export BATCH_SIZE="${BATCH_SIZE:-10000000}"
RUN="$REPO_DIR/config_runners"
mkdir -p "$WORKDIR"

echo "[1/3] prefilter (blocks until the compute-node job finishes)..."
qsub -W block=true -v REPO_DIR,ENV_PREFIX,WORKDIR,CEILING,BATCH_SIZE "$RUN/run_mces_split_prefilter.sh"
N=$(python3 -c "import json; print(json.load(open('$WORKDIR/manifest.json'))['n_batches'])")
echo "  -> $N batches"

echo "[2/3] compute array 0-$((N - 1)) (idempotent: re-submit to recompute only missing batches)..."
AID=$(qsub -J "0-$((N - 1))" -v REPO_DIR,ENV_PREFIX,WORKDIR,CEILING "$RUN/run_mces_split_compute.sh")
echo "  -> array job $AID"

cat <<EOF
[3/3] Once the array finishes (watch: qstat -tu \$(whoami)), run the report:
  qsub -v REPO_DIR=$REPO_DIR,ENV_PREFIX=$ENV_PREFIX,WORKDIR=$WORKDIR $RUN/run_mces_split_report.sh
Results -> $WORKDIR/results/ (far_{valid,test}_t{6..10}_clean.smi, incomputable.smi, near_pairs.parquet, nn_distances.parquet)
EOF
