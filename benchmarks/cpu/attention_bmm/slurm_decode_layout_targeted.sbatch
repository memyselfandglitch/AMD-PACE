#!/bin/bash
# Three independent-process validation of byte-matched and MHA-2 transition cases.
# Submit from the repository root:
#   sbatch benchmarks/cpu/attention_bmm/slurm_decode_layout_targeted.sbatch
#
#SBATCH --job-name=pace-dec-target
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
RESULT_DIR="${PACE_RESULT_DIR:-${PACE_REPO}/benchmark_results/decode-layout-targeted/${SLURM_JOB_ID}}"
WARMUPS="${PACE_DECODE_WARMUPS:-5}"
REPEATS="${PACE_DECODE_REPEATS:-40}"
ORDER_SEED="${PACE_DECODE_ORDER_SEED:-20260816}"
LAUNCH_DATA_SEEDS="${PACE_DECODE_LAUNCH_DATA_SEEDS:-11,29,47}"
TARGET_MIB="${PACE_DECODE_TARGET_MIB:-8,16,24,32,48,64,96,128}"
KV_HEADS="${PACE_DECODE_KV_HEADS:-2,4,8,16}"
BATCH_SIZES="${PACE_DECODE_BATCH_SIZES:-1,4}"
TRANSITION_SEQ_LENS="${PACE_DECODE_TRANSITION_SEQ_LENS:-6144,8192,10240,12288,14336,16384}"
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

CASE_MANIFEST="${RESULT_DIR}/decode_layout_target_cases.csv"
"${PYTHON}" benchmarks/cpu/attention_bmm/generate_decode_layout_target_cases.py \
    --out "${CASE_MANIFEST}" \
    --target-mib "${TARGET_MIB}" \
    --kv-heads "${KV_HEADS}" \
    --batch-sizes "${BATCH_SIZES}" \
    --transition-seq-lens "${TRANSITION_SEQ_LENS}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1

{
    echo "job_id=${SLURM_JOB_ID}"
    echo "host=$(hostname)"
    echo "date=$(date --iso-8601=seconds)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "case_manifest=${CASE_MANIFEST}"
    echo "target_mib=${TARGET_MIB}"
    echo "kv_heads=${KV_HEADS}"
    echo "batch_sizes=${BATCH_SIZES}"
    echo "transition_seq_lens=${TRANSITION_SEQ_LENS}"
    echo "launch_data_seeds=${LAUNCH_DATA_SEEDS}"
    echo "warmups=${WARMUPS}"
    echo "repeats=${REPEATS}"
    echo "order_seed=${ORDER_SEED}"
    echo "batch_semantics=sequential_outer_loop"
    echo "alignment_bytes=64"
    echo "order_policy=latin_square_cycle4"
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
    trial_file="${RESULT_DIR}/decode_layout_target_trials_launch${launch}.csv"
    launch_order_seed=$((ORDER_SEED + launch * 1000003))
    echo "launch=${launch} data_seed=${DATA_SEEDS[launch]} order_seed=${launch_order_seed}"
    "${RUN_PREFIX[@]}" "${BINARY}" \
        --cases "${CASE_MANIFEST}" \
        --data-seeds "${DATA_SEEDS[launch]}" \
        --warmups "${WARMUPS}" \
        --repeats "${REPEATS}" \
        --order-seed "${launch_order_seed}" \
        --process-launch "${launch}" \
        --out "${trial_file}"
    TRIAL_FILES+=("${trial_file}")
done

"${PYTHON}" benchmarks/cpu/attention_bmm/analyze_decode_layout_traversal.py \
    "${TRIAL_FILES[@]}" \
    --summary "${RESULT_DIR}/decode_layout_target_summary.csv" \
    --report "${RESULT_DIR}/decode_layout_target_report.md" \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
    --bootstrap-seed "${ORDER_SEED}" \
    --minimum-launch-wins 2

echo "Report: ${RESULT_DIR}/decode_layout_target_report.md"
