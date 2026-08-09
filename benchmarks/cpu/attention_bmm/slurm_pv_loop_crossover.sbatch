#!/bin/bash
# Submit from the repository root:
#   sbatch benchmarks/cpu/attention_bmm/slurm_pv_loop_crossover.sbatch
#
#SBATCH --job-name=pace-pv-cross
#SBATCH --partition=jobmn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PACE_REPO="${PACE_REPO:-${SLURM_SUBMIT_DIR}}"
BENCH_DIR="${PACE_REPO}/benchmarks/cpu/attention_bmm"
RUN_NAME="${PACE_RUN_NAME:-${SLURM_JOB_ID}}"
RESULT_DIR="${PACE_RESULT_DIR:-${PACE_REPO}/benchmark_results/cpu-pv-crossover/${RUN_NAME}}"
CXX_BIN="${CXX:-g++}"
CPU_CORE="${PACE_CPU_CORE:-0}"
NUMA_NODE="${PACE_NUMA_NODE:-0}"
SHAPES="${PACE_PV_SHAPES:-slm,llm}"
QUERY_LENS="${PACE_PV_QUERY_LENS:-1,2,4,8,16,32,64,128,256,512}"
KV_LENS="${PACE_PV_KV_LENS:-128,512,2048,8192,16384}"
DATA_SEEDS="${PACE_PV_DATA_SEEDS:-11,29,47}"
WARMUPS="${PACE_PV_WARMUPS:-2}"
REPEATS="${PACE_PV_REPEATS:-20}"
ORDER_SEED="${PACE_PV_ORDER_SEED:-20260809}"
BOOTSTRAP_SAMPLES="${PACE_PV_BOOTSTRAP_SAMPLES:-10000}"
COMPILE_FLAGS=(-O3 -march=native -std=c++17 -DNDEBUG)
if "${CXX_BIN}" --version | head -n 1 | grep -qiE 'gcc|g\+\+'; then
    COMPILE_FLAGS+=(-fno-loop-interchange)
fi

mkdir -p "${RESULT_DIR}"
cd "${PACE_REPO}"

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "started=$(date --iso-8601=seconds)"
echo "results=${RESULT_DIR}"
echo "shapes=${SHAPES} query_lens=${QUERY_LENS} kv_lens=${KV_LENS}"
echo "data_seeds=${DATA_SEEDS} warmups=${WARMUPS} repeats=${REPEATS}"

{
    echo "job_id=${SLURM_JOB_ID}"
    echo "host=$(hostname)"
    echo "date=$(date --iso-8601=seconds)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "compiler=$(${CXX_BIN} --version | head -n 1)"
    echo "compile_flags=${COMPILE_FLAGS[*]}"
    echo "shapes=${SHAPES}"
    echo "query_lens=${QUERY_LENS}"
    echo "kv_lens=${KV_LENS}"
    echo "data_seeds=${DATA_SEEDS}"
    echo "warmups=${WARMUPS}"
    echo "repeats=${REPEATS}"
    echo "order_seed=${ORDER_SEED}"
    echo "bootstrap_samples=${BOOTSTRAP_SAMPLES}"
    echo "cpu_core=${CPU_CORE}"
    echo "numa_node=${NUMA_NODE}"
    echo "scaling_governor=$(cat /sys/devices/system/cpu/cpu${CPU_CORE}/cpufreq/scaling_governor 2>/dev/null || echo unavailable)"
    echo "boost=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo unavailable)"
    lscpu
} > "${RESULT_DIR}/environment.txt"

"${CXX_BIN}" "${COMPILE_FLAGS[@]}" \
    "${BENCH_DIR}/pv_loop_crossover.cpp" \
    -o "${RESULT_DIR}/pv_loop_crossover"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1

RUN_PREFIX=()
if command -v numactl >/dev/null 2>&1; then
    RUN_PREFIX=(numactl --physcpubind="${CPU_CORE}" --membind="${NUMA_NODE}")
else
    echo "WARNING: numactl unavailable; benchmark will not be explicitly pinned" >&2
fi

"${RUN_PREFIX[@]}" "${RESULT_DIR}/pv_loop_crossover" \
    --shapes "${SHAPES}" \
    --query-lens "${QUERY_LENS}" \
    --kv-lens "${KV_LENS}" \
    --data-seeds "${DATA_SEEDS}" \
    --warmups "${WARMUPS}" \
    --repeats "${REPEATS}" \
    --order-seed "${ORDER_SEED}" \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
    --raw-csv "${RESULT_DIR}/pv_crossover_trials.csv" \
    --summary-csv "${RESULT_DIR}/pv_crossover_summary.csv" \
    --report-md "${RESULT_DIR}/pv_crossover_report.md"

echo "finished=$(date --iso-8601=seconds)"
echo "Report: ${RESULT_DIR}/pv_crossover_report.md"
