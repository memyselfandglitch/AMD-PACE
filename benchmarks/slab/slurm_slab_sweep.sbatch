#!/bin/bash
#SBATCH --job-name=pace-slab-sweep
#SBATCH --partition=jobmn01
#SBATCH --output=benchmarks/slab/results/slurm-%j-sweep.out
#SBATCH --error=benchmarks/slab/results/slurm-%j-sweep.err
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

mkdir -p benchmarks/slab/results

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate pace-env-py3.12
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
