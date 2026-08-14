#!/bin/bash
# Submit from the repository root:
#   sbatch benchmarks/cpu/attention_bmm/slurm_pv_brgemm_dataflow.sbatch
#
#SBATCH --job-name=pace-pv-flow
#SBATCH --partition=jobmn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PACE_REPO="${PACE_REPO:-${SLURM_SUBMIT_DIR}}"
PACE_VENV="${PACE_VENV:-/data/scratch/deveshisingh/pace-venv}"
PYTHON="${PACE_VENV}/bin/python"
RESULT_DIR="${PACE_RESULT_DIR:-${PACE_REPO}/benchmark_results/pv-brgemm-dataflow/${SLURM_JOB_ID}}"
QUERY_LENS="${PACE_FLOW_QUERY_LENS:-64,128,256,512}"
KV_LENS="${PACE_FLOW_KV_LENS:-2048,8192,16384}"
HEAD_DIMS="${PACE_FLOW_HEAD_DIMS:-64,128}"
DATA_SEEDS="${PACE_FLOW_DATA_SEEDS:-11,29,47}"
WARMUPS="${PACE_FLOW_WARMUPS:-2}"
REPEATS="${PACE_FLOW_REPEATS:-20}"
ORDER_SEED="${PACE_FLOW_ORDER_SEED:-20260813}"
BOOTSTRAP_SAMPLES="${PACE_FLOW_BOOTSTRAP_SAMPLES:-10000}"

mkdir -p "${RESULT_DIR}"
cd "${PACE_REPO}"
source "${PACE_VENV}/bin/activate"

# Build PACE first so its pinned oneDNN dependency exists, then build only the
# standalone research target. No benchmark API is added to libpace_cpp.so.
"${PYTHON}" setup.py build_clib
TORCH_CMAKE_PREFIX="$(${PYTHON} -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DPACKAGE_NAME=pace_cpp \
    -DPACKAGE_VERSION=0.0.0 \
    -DCMAKE_PREFIX_PATH="${TORCH_CMAKE_PREFIX}" \
    -DCMAKE_INSTALL_PREFIX="${PACE_REPO}/build/Release/pace" \
    -DPACE_BUILD_BENCHMARKS=ON
cmake --build build --target pace_pv_brgemm_dataflow -j "${PACE_BUILD_JOBS:-16}"

BINARY="${PACE_REPO}/build/benchmarks/cpu/attention_bmm/pace_pv_brgemm_dataflow"
if [[ ! -x "${BINARY}" ]]; then
    echo "benchmark target was not produced: ${BINARY}" >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1

{
    echo "job_id=${SLURM_JOB_ID}"
    echo "host=$(hostname)"
    echo "date=$(date --iso-8601=seconds)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "query_lens=${QUERY_LENS}"
    echo "kv_lens=${KV_LENS}"
    echo "head_dims=${HEAD_DIMS}"
    echo "data_seeds=${DATA_SEEDS}"
    echo "warmups=${WARMUPS}"
    echo "repeats=${REPEATS}"
    echo "order_seed=${ORDER_SEED}"
    echo "cpu_core=${PACE_CPU_CORE:-0}"
    echo "numa_node=${PACE_NUMA_NODE:-0}"
    echo "scaling_governor=$(cat /sys/devices/system/cpu/cpu${PACE_CPU_CORE:-0}/cpufreq/scaling_governor 2>/dev/null || echo unavailable)"
    echo "boost=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo unavailable)"
    lscpu
} > "${RESULT_DIR}/environment.txt"

RUN_PREFIX=()
if command -v numactl >/dev/null 2>&1; then
    RUN_PREFIX=(
        numactl
        --physcpubind="${PACE_CPU_CORE:-0}"
        --membind="${PACE_NUMA_NODE:-0}"
    )
fi

"${RUN_PREFIX[@]}" "${BINARY}" \
    --query-lens "${QUERY_LENS}" \
    --kv-lens "${KV_LENS}" \
    --head-dims "${HEAD_DIMS}" \
    --data-seeds "${DATA_SEEDS}" \
    --warmups "${WARMUPS}" \
    --repeats "${REPEATS}" \
    --order-seed "${ORDER_SEED}" \
    --out "${RESULT_DIR}/pv_brgemm_dataflow_trials.csv"

"${PYTHON}" benchmarks/cpu/attention_bmm/analyze_pv_brgemm_dataflow.py \
    "${RESULT_DIR}/pv_brgemm_dataflow_trials.csv" \
    --summary "${RESULT_DIR}/pv_brgemm_dataflow_summary.csv" \
    --report "${RESULT_DIR}/pv_brgemm_dataflow_report.md" \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
    --bootstrap-seed "${ORDER_SEED}"

echo "Report: ${RESULT_DIR}/pv_brgemm_dataflow_report.md"
