#!/bin/bash
#SBATCH --job-name=pace-slab-quick
#SBATCH --partition=jobmn01
#SBATCH --output=benchmarks/slab/results/slurm-%j-quick.out
#SBATCH --error=benchmarks/slab/results/slurm-%j-quick.err
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

mkdir -p benchmarks/slab/results

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate pace-env-py3.12
fi

python -m pip install --upgrade pip
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
