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
