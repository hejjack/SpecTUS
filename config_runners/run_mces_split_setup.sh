#!/bin/bash
#PBS -N spectus_mces_env
#PBS -l select=1:ncpus=4:mem=16gb:scratch_local=20gb
#PBS -l walltime=2:00:00
# Create the dedicated `spectus_mces` env and install SpecTUS with the [mces]
# extra (no torch is pulled — setup.py has no core deps). Run installs on a
# compute node, not the frontend. Override paths via `qsub -v REPO_DIR,ENV_PREFIX`.
set -euo pipefail
test "$(whoami)" = "jozefov_147" || { echo "Refuse: wrong account"; exit 1; }
PLZEN_HOME=/storage/plzen1/home/jozefov_147
REPO_DIR="${REPO_DIR:-$PLZEN_HOME/projects/SpecTUS}"
ENV_PREFIX="${ENV_PREFIX:-$PLZEN_HOME/.conda/envs/spectus_mces}"

mkdir -p "$SCRATCHDIR/tmp"; export TMPDIR="$SCRATCHDIR/tmp"
module add mambaforge
if [ ! -d "$ENV_PREFIX" ]; then
  mamba create --prefix "$ENV_PREFIX" python=3.11 -y
fi
mamba run -p "$ENV_PREFIX" python -m pip install --upgrade pip setuptools wheel
mamba run -p "$ENV_PREFIX" python -m pip install -e "${REPO_DIR}[mces]"
mamba run -p "$ENV_PREFIX" python -c \
  "import mces_split, myopic_mces, rdkit, scipy, h5py, pandas, pyarrow; print('env OK')"
clean_scratch
