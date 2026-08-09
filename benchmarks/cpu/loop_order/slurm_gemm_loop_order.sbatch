#!/bin/bash
# Submit from the repository root:
#   sbatch benchmarks/cpu/loop_order/slurm_gemm_loop_order.sbatch
#
#SBATCH --job-name=pace-loop-order
#SBATCH --partition=jobmn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PACE_REPO="${PACE_REPO:-${SLURM_SUBMIT_DIR}}"
BENCH_DIR="${PACE_REPO}/benchmarks/cpu/loop_order"
RUN_NAME="${PACE_RUN_NAME:-${SLURM_JOB_ID}}"
RESULT_DIR="${PACE_RESULT_DIR:-${PACE_REPO}/benchmark_results/cpu-loop-order/${RUN_NAME}}"
CXX_BIN="${CXX:-g++}"
CPU_CORE="${PACE_CPU_CORE:-0}"
NUMA_NODE="${PACE_NUMA_NODE:-0}"
SIZES="${PACE_GEMM_SIZES:-32,48,64,80,96,112,128,160,192,224,256,320,384,448,512}"
WARMUPS="${PACE_GEMM_WARMUPS:-5}"
REPEATS="${PACE_GEMM_REPEATS:-50}"
SEED="${PACE_GEMM_SEED:-20260809}"
COMPILE_FLAGS=(-O3 -march=native -std=c++17 -DNDEBUG)
if "${CXX_BIN}" --version | head -n 1 | grep -qiE 'gcc|g\+\+'; then
    # Preserve the source loop permutation while retaining vectorization.
    COMPILE_FLAGS+=(-fno-loop-interchange)
fi

mkdir -p "${RESULT_DIR}"
cd "${PACE_REPO}"

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "started=$(date --iso-8601=seconds)"
echo "repository=${PACE_REPO}"
echo "results=${RESULT_DIR}"
echo "sizes=${SIZES} warmups=${WARMUPS} repeats=${REPEATS} seed=${SEED}"

# Record enough context to reproduce or reject a noisy run.
{
    echo "job_id=${SLURM_JOB_ID}"
    echo "host=$(hostname)"
    echo "date=$(date --iso-8601=seconds)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "compiler=$(${CXX_BIN} --version | head -n 1)"
    echo "compile_flags=${COMPILE_FLAGS[*]}"
    echo "sizes=${SIZES}"
    echo "warmups=${WARMUPS}"
    echo "repeats=${REPEATS}"
    echo "seed=${SEED}"
    echo "cpu_core=${CPU_CORE}"
    echo "numa_node=${NUMA_NODE}"
    echo "scaling_governor=$(cat /sys/devices/system/cpu/cpu${CPU_CORE}/cpufreq/scaling_governor 2>/dev/null || echo unavailable)"
    echo "boost=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo unavailable)"
    lscpu
} > "${RESULT_DIR}/environment.txt"

"${CXX_BIN}" "${COMPILE_FLAGS[@]}" \
    "${BENCH_DIR}/gemm_loop_order.cpp" \
    -o "${RESULT_DIR}/gemm_loop_order"

RUN_PREFIX=()
if command -v numactl >/dev/null 2>&1; then
    RUN_PREFIX=(numactl --physcpubind="${CPU_CORE}" --membind="${NUMA_NODE}")
else
    echo "WARNING: numactl unavailable; benchmark will not be explicitly pinned" >&2
fi

"${RUN_PREFIX[@]}" "${RESULT_DIR}/gemm_loop_order" \
    --sizes "${SIZES}" \
    --warmups "${WARMUPS}" \
    --repeats "${REPEATS}" \
    --seed "${SEED}" \
    --raw-csv "${RESULT_DIR}/gemm_loop_order_trials.csv" \
    --summary-csv "${RESULT_DIR}/gemm_loop_order_summary.csv"

python3 "${BENCH_DIR}/analyze_gemm_loop_threshold.py" \
    --trials "${RESULT_DIR}/gemm_loop_order_trials.csv" \
    --summary "${RESULT_DIR}/gemm_loop_order_summary.csv" \
    --out-csv "${RESULT_DIR}/gemm_loop_order_threshold.csv" \
    --out-md "${RESULT_DIR}/gemm_loop_order_threshold.md" \
    --seed "${SEED}"

echo "finished=$(date --iso-8601=seconds)"
echo "Summary: ${RESULT_DIR}/gemm_loop_order_summary.csv"
echo "Threshold report: ${RESULT_DIR}/gemm_loop_order_threshold.md"
