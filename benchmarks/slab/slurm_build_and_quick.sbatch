#!/bin/bash
#SBATCH --job-name=pace-slab-quick
#SBATCH --partition=jobmn01
#SBATCH --output=slurm-%j-quick.out
#SBATCH --error=slurm-%j-quick.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -euo pipefail

# Submit from the AMD-PACE repo root:
#   sbatch benchmarks/slab/slurm_build_and_quick.sbatch
#
# This cluster requires CPU-only work to use the CPU partition, not GPU.

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p benchmarks/slab/results

VENV_DIR="${PACE_VENV_DIR:-/data/scratch/${USER}/pace-venv}"
PY_BOOTSTRAP="${PY_BOOTSTRAP:-python3}"

echo "repo=${REPO_ROOT}"
echo "venv=${VENV_DIR}"
echo "bootstrap_python=$(${PY_BOOTSTRAP} --version 2>&1)"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  "${PY_BOOTSTRAP}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel
python -m pip install "setuptools>=77,<82"
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.12.0+cpu
python -m pip install -r requirements.txt
python -m pip install -r build_requirements.txt -v .

# The benchmark runs from the source checkout, so Python imports ./pace before
# the installed wheel. Copy the freshly built shared library into the source tree.
mkdir -p pace/lib
if [ ! -f pace/lib/libpace_cpp.so ]; then
  BUILT_LIB="$(find build -name libpace_cpp.so -print -quit)"
  if [ -z "${BUILT_LIB}" ]; then
    echo "Could not find libpace_cpp.so under build/ after installation."
    find build -name '*.so' -print || true
    exit 1
  fi
  cp "${BUILT_LIB}" pace/lib/libpace_cpp.so
fi
ls -lh pace/lib/libpace_cpp.so

python - <<'PY'
import os
import platform
import torch
import pace

print("platform:", platform.platform())
print("machine:", platform.machine())
print("torch:", torch.__version__)
print("threads:", torch.get_num_threads())
print("pace import: ok")
print("lib exists:", os.path.exists("pace/lib/libpace_cpp.so"))
PY

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export PACE_LOG_LEVEL=warning

python benchmarks/slab/bench_slab_layout_blocksize.py \
  --quick \
  --output benchmarks/slab/results/slab_quick_${SLURM_JOB_ID:-local}.csv
