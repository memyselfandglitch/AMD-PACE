#!/bin/bash
# Grouped-query GQA validation across a broad logical K+V payload range.
# Submit from the repository root:
#   sbatch benchmarks/cpu/attention_bmm/slurm_decode_gqa_grouped.sbatch
#
# Quick smoke test:
#   PACE_GQA_SEQ_LENS=512 PACE_GQA_KV_HEADS=32,8 \
#   PACE_GQA_BATCH_SIZES=1 PACE_DECODE_REPEATS=4 \
#   sbatch benchmarks/cpu/attention_bmm/slurm_decode_gqa_grouped.sbatch
#
#SBATCH --job-name=pace-gqa-group
#SBATCH --partition=jobmn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PACE_REPO="${PACE_REPO:-${SLURM_SUBMIT_DIR}}"
PACE_VENV="${PACE_VENV:-/data/scratch/deveshisingh/pace-venv}"
PYTHON="${PACE_VENV}/bin/python"
RESULT_DIR="${PACE_RESULT_DIR:-${PACE_REPO}/benchmark_results/decode-gqa-grouped/${SLURM_JOB_ID}}"
NUM_Q_HEADS="${PACE_GQA_NUM_Q_HEADS:-32}"
KV_HEADS="${PACE_GQA_KV_HEADS:-32,16,8,4,2,1}"
BATCH_SIZES="${PACE_GQA_BATCH_SIZES:-1,2,4}"
SEQ_LENS="${PACE_GQA_SEQ_LENS:-512,2048,8192,32768,65536}"
HEAD_DIM="${PACE_GQA_HEAD_DIM:-64}"
BLOCK_SIZE="${PACE_GQA_BLOCK_SIZE:-64}"
WARMUPS="${PACE_DECODE_WARMUPS:-5}"
REPEATS="${PACE_DECODE_REPEATS:-40}"
ORDER_SEED="${PACE_DECODE_ORDER_SEED:-20260820}"
LAUNCH_DATA_SEEDS="${PACE_DECODE_LAUNCH_DATA_SEEDS:-11,29,47}"
BOOTSTRAP_SAMPLES="${PACE_DECODE_BOOTSTRAP_SAMPLES:-10000}"

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

CASE_MANIFEST="${RESULT_DIR}/decode_gqa_grouped_cases.csv"
"${PYTHON}" benchmarks/cpu/attention_bmm/generate_decode_gqa_grouped_cases.py \
    --out "${CASE_MANIFEST}" \
    --num-q-heads "${NUM_Q_HEADS}" \
    --kv-heads "${KV_HEADS}" \
    --batch-sizes "${BATCH_SIZES}" \
    --seq-lens "${SEQ_LENS}" \
    --head-dim "${HEAD_DIM}" \
    --block-size "${BLOCK_SIZE}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1

{
    echo "job_id=${SLURM_JOB_ID}"
    echo "host=$(hostname)"
    echo "date=$(date --iso-8601=seconds)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "candidate_set=gqa_grouped"
    echo "case_manifest=${CASE_MANIFEST}"
    echo "num_q_heads=${NUM_Q_HEADS}"
    echo "kv_heads=${KV_HEADS}"
    echo "gqa_ratios=$(echo "${KV_HEADS}" | awk -v q="${NUM_Q_HEADS}" -F, '{for(i=1;i<=NF;i++) printf "%s%s", (i>1?",":""), q/$i}')"
    echo "batch_sizes=${BATCH_SIZES}"
    echo "seq_lens=${SEQ_LENS}"
    echo "head_dim=${HEAD_DIM}"
    echo "block_size=${BLOCK_SIZE}"
    echo "launch_data_seeds=${LAUNCH_DATA_SEEDS}"
    echo "warmups=${WARMUPS}"
    echo "repeats=${REPEATS}"
    echo "order_seed=${ORDER_SEED}"
    echo "batch_semantics=sequential_outer_loop"
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

IFS=',' read -r -a DATA_SEEDS <<< "${LAUNCH_DATA_SEEDS}"
if (( ${#DATA_SEEDS[@]} < 3 )); then
    echo "PACE_DECODE_LAUNCH_DATA_SEEDS must contain at least three seeds" >&2
    exit 1
fi

TRIAL_FILES=()
for launch in "${!DATA_SEEDS[@]}"; do
    trial_file="${RESULT_DIR}/decode_gqa_grouped_trials_launch${launch}.csv"
    launch_order_seed=$((ORDER_SEED + launch * 1000003))
    echo "launch=${launch} data_seed=${DATA_SEEDS[launch]} order_seed=${launch_order_seed}"
    "${RUN_PREFIX[@]}" "${BINARY}" \
        --candidate-set gqa_grouped \
        --cases "${CASE_MANIFEST}" \
        --data-seeds "${DATA_SEEDS[launch]}" \
        --warmups "${WARMUPS}" \
        --repeats "${REPEATS}" \
        --order-seed "${launch_order_seed}" \
        --process-launch "${launch}" \
        --out "${trial_file}"
    TRIAL_FILES+=("${trial_file}")
done

"${PYTHON}" benchmarks/cpu/attention_bmm/analyze_decode_gqa_grouped.py \
    "${TRIAL_FILES[@]}" \
    --summary "${RESULT_DIR}/decode_gqa_grouped_summary.csv" \
    --report "${RESULT_DIR}/decode_gqa_grouped_report.md" \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
    --bootstrap-seed "${ORDER_SEED}" \
    --minimum-launch-wins 2

echo "Report: ${RESULT_DIR}/decode_gqa_grouped_report.md"
