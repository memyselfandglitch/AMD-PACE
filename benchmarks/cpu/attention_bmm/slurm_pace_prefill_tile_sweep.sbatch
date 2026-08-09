#!/bin/bash
#SBATCH --job-name=pace-tile-sweep
#SBATCH --partition=jobmn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PACE_REPO="${PACE_REPO:-${SLURM_SUBMIT_DIR}}"
PACE_VENV="${PACE_VENV:-/data/scratch/deveshisingh/pace-venv}"
PYTHON="${PACE_VENV}/bin/python"
RESULT_DIR="${PACE_RESULT_DIR:-${PACE_REPO}/benchmark_results/pace-prefill-tile/${SLURM_JOB_ID}}"
SHAPES="${PACE_TILE_SHAPES:-slm,llm}"
QUERY_LENS="${PACE_TILE_QUERY_LENS:-128,512}"
KV_LENS="${PACE_TILE_KV_LENS:-2048,8192,16384}"
BATCH_SIZES="${PACE_TILE_BATCH_SIZES:-1,4}"
TILES="${PACE_TILE_SIZES:-16,32,64,128}"
DATA_SEEDS="${PACE_TILE_DATA_SEEDS:-11,29,47}"

mkdir -p "${RESULT_DIR}" "${PACE_REPO}/pace/lib"
cd "${PACE_REPO}"
source "${PACE_VENV}/bin/activate"

# The native library must be rebuilt because query-tile selection is C++ code.
"${PYTHON}" setup.py build_clib
cp build/Release/pace/lib/libpace_cpp.so pace/lib/libpace_cpp.so

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-96}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DYNAMIC=false
export SLAB_LAYOUT=head_major
export SLAB_SCHEDULE=static
export PACE_LOG_LEVEL=none

{
    echo "job_id=${SLURM_JOB_ID}"
    echo "host=$(hostname)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "python=$(${PYTHON} --version)"
    echo "omp_num_threads=${OMP_NUM_THREADS}"
    echo "slab_layout=${SLAB_LAYOUT}"
    echo "slab_schedule=${SLAB_SCHEDULE}"
    echo "shapes=${SHAPES}"
    echo "query_lens=${QUERY_LENS}"
    echo "kv_lens=${KV_LENS}"
    echo "batch_sizes=${BATCH_SIZES}"
    echo "tiles=${TILES}"
    echo "data_seeds=${DATA_SEEDS}"
    echo "numa_node=${PACE_NUMA_NODE:-0}"
    lscpu
} > "${RESULT_DIR}/environment.txt"

RUN_PREFIX=()
if command -v numactl >/dev/null 2>&1; then
    RUN_PREFIX=(numactl --cpunodebind="${PACE_NUMA_NODE:-0}" --membind="${PACE_NUMA_NODE:-0}")
fi

"${RUN_PREFIX[@]}" "${PYTHON}" \
    benchmarks/cpu/attention_bmm/bench_pace_prefill_tiles.py \
    --out-dir "${RESULT_DIR}" \
    --shapes "${SHAPES}" \
    --query-lens "${QUERY_LENS}" \
    --kv-lens "${KV_LENS}" \
    --batch-sizes "${BATCH_SIZES}" \
    --tiles "${TILES}" \
    --data-seeds "${DATA_SEEDS}" \
    --warmups "${PACE_TILE_WARMUPS:-2}" \
    --repeats "${PACE_TILE_REPEATS:-20}" \
    --profile-one-call | tee "${RESULT_DIR}/pace_profile.log"

echo "Report: ${RESULT_DIR}/pace_prefill_tile_report.md"
