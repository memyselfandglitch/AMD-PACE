#!/bin/bash
#SBATCH --job-name=pace-slab-sweep
#SBATCH --partition=jobmn01
#SBATCH --output=slurm-%j-sweep.out
#SBATCH --error=slurm-%j-sweep.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=08:00:00

set -euo pipefail

# Submit from the AMD-PACE repo root after the environment is built:
#   sbatch benchmarks/slab/slurm_slab_sweep.sbatch
#
# This cluster requires CPU-only work to use the CPU partition, not GPU.

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p benchmarks/slab/results

VENV_DIR="${PACE_VENV_DIR:-/data/scratch/${USER}/pace-venv}"
PY_BOOTSTRAP="${PY_BOOTSTRAP:-python3}"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  echo "Missing virtualenv at ${VENV_DIR}."
  echo "Run: sbatch benchmarks/slab/slurm_setup_venv.sbatch"
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

if [ ! -f pace/lib/libpace_cpp.so ]; then
  mkdir -p pace/lib
  BUILT_LIB="$(find build -name libpace_cpp.so -print -quit)"
  if [ -z "${BUILT_LIB}" ]; then
    echo "Missing pace/lib/libpace_cpp.so."
    echo "Run first: sbatch benchmarks/slab/slurm_build_and_quick.sbatch"
    exit 1
  fi
  cp "${BUILT_LIB}" pace/lib/libpace_cpp.so
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export PACE_LOG_LEVEL=warning

python benchmarks/slab/bench_slab_layout_blocksize.py \
  --phases prefill,decode,mtd \
  --layouts head_major,block_major \
  --block-sizes 16,32,64,128,256 \
  --seq-lens 128,512,2048,8192 \
  --batch-sizes 1,4,16 \
  --shapes default \
  --exit-fractions 0.0,0.25,0.5,0.75 \
  --warmups 3 \
  --repeats 10 \
  --output benchmarks/slab/results/slab_sweep_${SLURM_JOB_ID:-local}.csv
