#!/bin/bash
# Submit from the repository root:
#   sbatch benchmarks/cpu/attention_bmm/slurm_decode_layout_traversal.sbatch
#
#SBATCH --job-name=pace-dec-layout
#SBATCH --partition=jobmn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PACE_REPO="${PACE_REPO:-${SLURM_SUBMIT_DIR}}"
PACE_VENV="${PACE_VENV:-/data/scratch/deveshisingh/pace-venv}"
PYTHON="${PACE_VENV}/bin/python"
RESULT_DIR="${PACE_RESULT_DIR:-${PACE_REPO}/benchmark_results/decode-layout-traversal/${SLURM_JOB_ID}}"
PROFILE="${PACE_DECODE_PROFILE:-default}"
SHAPES="${PACE_DECODE_SHAPES:-slm_gqa,llama_gqa,mha}"
SHAPE_SPECS="${PACE_DECODE_SHAPE_SPECS:-}"
SEQ_LENS="${PACE_DECODE_SEQ_LENS:-512,2048,8192,16384}"
BATCH_SIZES="${PACE_DECODE_BATCH_SIZES:-1,4}"
DATA_SEEDS="${PACE_DECODE_DATA_SEEDS:-11,29,47}"
BLOCK_SIZES="${PACE_DECODE_BLOCK_SIZES:-${PACE_DECODE_BLOCK_SIZE:-64}}"
WARMUPS="${PACE_DECODE_WARMUPS:-2}"
REPEATS="${PACE_DECODE_REPEATS:-20}"
ORDER_SEED="${PACE_DECODE_ORDER_SEED:-20260816}"
BOOTSTRAP_SAMPLES="${PACE_DECODE_BOOTSTRAP_SAMPLES:-10000}"

if [[ "${PROFILE}" == "crossover" ]]; then
    SHAPE_SPECS="${PACE_DECODE_SHAPE_SPECS:-kv_heads/mha2:2:2:64,kv_heads/mha4:4:4:64,kv_heads/mha8:8:8:64,kv_heads/mha16:16:16:64,gqa_ratio/gqa1:8:8:64,gqa_ratio/gqa2:16:8:64,gqa_ratio/gqa4:32:8:64,gqa_ratio/gqa8:64:8:64,head_dim/d64:8:8:64,head_dim/d128:8:8:128,head_dim/d256:8:8:256}"
    SEQ_LENS="${PACE_DECODE_SEQ_LENS:-2048,4096,8192,12288,16384}"
    BLOCK_SIZES="${PACE_DECODE_BLOCK_SIZES:-64}"
fi

mkdir -p "${RESULT_DIR}"
cd "${PACE_REPO}"
source "${PACE_VENV}/bin/activate"

"${PYTHON}" setup.py build_clib
TORCH_CMAKE_PREFIX="$(${PYTHON} -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DPACKAGE_NAME=pace_cpp \
    -DPACKAGE_VERSION=0.0.0 \
    -DCMAKE_PREFIX_PATH="${TORCH_CMAKE_PREFIX}" \
    -DCMAKE_INSTALL_PREFIX="${PACE_REPO}/build/Release/pace" \
    -DPACE_BUILD_BENCHMARKS=ON
cmake --build build --target pace_decode_layout_traversal -j "${PACE_BUILD_JOBS:-16}"

BINARY="${PACE_REPO}/build/benchmarks/cpu/attention_bmm/pace_decode_layout_traversal"
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
    echo "profile=${PROFILE}"
    echo "shapes=${SHAPES}"
    echo "shape_specs=${SHAPE_SPECS}"
    echo "seq_lens=${SEQ_LENS}"
    echo "batch_sizes=${BATCH_SIZES}"
    echo "data_seeds=${DATA_SEEDS}"
    echo "block_sizes=${BLOCK_SIZES}"
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

BENCHMARK_ARGS=(
    --shapes "${SHAPES}"
    --seq-lens "${SEQ_LENS}"
    --batch-sizes "${BATCH_SIZES}"
    --data-seeds "${DATA_SEEDS}"
    --block-sizes "${BLOCK_SIZES}"
    --warmups "${WARMUPS}"
    --repeats "${REPEATS}"
    --order-seed "${ORDER_SEED}"
    --out "${RESULT_DIR}/decode_layout_traversal_trials.csv"
)
if [[ -n "${SHAPE_SPECS}" ]]; then
    BENCHMARK_ARGS+=(--shape-specs "${SHAPE_SPECS}")
fi

"${RUN_PREFIX[@]}" "${BINARY}" "${BENCHMARK_ARGS[@]}"

"${PYTHON}" benchmarks/cpu/attention_bmm/analyze_decode_layout_traversal.py \
    "${RESULT_DIR}/decode_layout_traversal_trials.csv" \
    --summary "${RESULT_DIR}/decode_layout_traversal_summary.csv" \
    --report "${RESULT_DIR}/decode_layout_traversal_report.md" \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
    --bootstrap-seed "${ORDER_SEED}"

echo "Report: ${RESULT_DIR}/decode_layout_traversal_report.md"
