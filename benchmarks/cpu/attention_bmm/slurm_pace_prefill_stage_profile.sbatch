#!/bin/bash
# Submit from the repository root:
#   sbatch benchmarks/cpu/attention_bmm/slurm_pace_prefill_stage_profile.sbatch
#
# Smoke test:
#   PACE_STAGE_SHAPES=slm PACE_STAGE_QUERY_LENS=128 \
#   PACE_STAGE_KV_LENS=2048 PACE_STAGE_BATCH_SIZES=1 \
#   PACE_STAGE_DATA_SEEDS=11 PACE_STAGE_REPEATS=3 \
#     sbatch --export=ALL \
#     benchmarks/cpu/attention_bmm/slurm_pace_prefill_stage_profile.sbatch
#
#SBATCH --job-name=pace-stage-prof
#SBATCH --partition=jobmn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PACE_REPO="${PACE_REPO:-${SLURM_SUBMIT_DIR}}"
PACE_VENV="${PACE_VENV:-/data/scratch/deveshisingh/pace-venv}"
PYTHON="${PACE_VENV}/bin/python"
RESULT_DIR="${PACE_RESULT_DIR:-${PACE_REPO}/benchmark_results/pace-prefill-stage/${SLURM_JOB_ID}}"
SHAPES="${PACE_STAGE_SHAPES:-slm,llm}"
QUERY_LENS="${PACE_STAGE_QUERY_LENS:-128,256,512,1024}"
KV_LENS="${PACE_STAGE_KV_LENS:-2048,8192}"
BATCH_SIZES="${PACE_STAGE_BATCH_SIZES:-1,4}"
DATA_SEEDS="${PACE_STAGE_DATA_SEEDS:-11,29,47}"
WARMUPS="${PACE_STAGE_WARMUPS:-2}"
REPEATS="${PACE_STAGE_REPEATS:-20}"
ORDER_SEED="${PACE_STAGE_ORDER_SEED:-20260809}"
BOOTSTRAP_SAMPLES="${PACE_STAGE_BOOTSTRAP_SAMPLES:-10000}"

mkdir -p "${RESULT_DIR}" "${PACE_REPO}/pace/lib"
cd "${PACE_REPO}"
source "${PACE_VENV}/bin/activate"
export PYTHONPATH="${PACE_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1

# Stage hooks are native C++ code, so every profiling run rebuilds the library.
"${PYTHON}" setup.py build_clib
BUILT_LIB="build/csrc/libpace_cpp.so"
TARGET_LIB="pace/lib/libpace_cpp.so"
if [[ ! -f "${BUILT_LIB}" ]]; then
    echo "PACE build did not produce ${BUILT_LIB}" >&2
    exit 1
fi
if [[ ! -f "${TARGET_LIB}" ]] || [[ ! "${BUILT_LIB}" -ef "${TARGET_LIB}" ]]; then
    cp "${BUILT_LIB}" "${TARGET_LIB}"
fi

# Fail before the expensive sweep if Python resolves an old PACE installation.
"${PYTHON}" - <<'PY'
from pathlib import Path

import pace
import torch

repo = Path.cwd().resolve()
module_path = Path(pace.__file__).resolve()
library_path = Path(pace._LIB_PATH).resolve()
if repo not in module_path.parents:
    raise RuntimeError(f"loaded PACE outside repository: {module_path}")
if repo not in library_path.parents:
    raise RuntimeError(f"loaded PACE library outside repository: {library_path}")

pool = torch.classes.pace.SlabPool(1, 1, 64, 64)
pool.set_stage_profile(False)
if list(pool.get_stage_profile()):
    raise RuntimeError("new SlabPool unexpectedly returned a stale profile")
print(f"PACE module preflight OK: {module_path}")
print(f"PACE library preflight OK: {library_path}")
PY

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
    echo "date=$(date --iso-8601=seconds)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "python=$(${PYTHON} --version)"
    echo "torch=$(${PYTHON} -c 'import torch; print(torch.__version__)')"
    echo "pace_module=$(${PYTHON} -c 'import pace; print(pace.__file__)')"
    echo "pace_library=$(${PYTHON} -c 'import pace; print(pace._LIB_PATH)')"
    echo "pace_library_sha256=$(sha256sum "${TARGET_LIB}" | cut -d' ' -f1)"
    echo "omp_num_threads=${OMP_NUM_THREADS}"
    echo "omp_places=${OMP_PLACES}"
    echo "omp_proc_bind=${OMP_PROC_BIND}"
    echo "omp_dynamic=${OMP_DYNAMIC}"
    echo "slab_layout=${SLAB_LAYOUT}"
    echo "slab_schedule=${SLAB_SCHEDULE}"
    echo "shapes=${SHAPES}"
    echo "query_lens=${QUERY_LENS}"
    echo "kv_lens=${KV_LENS}"
    echo "batch_sizes=${BATCH_SIZES}"
    echo "data_seeds=${DATA_SEEDS}"
    echo "warmups=${WARMUPS}"
    echo "repeats=${REPEATS}"
    echo "order_seed=${ORDER_SEED}"
    echo "bootstrap_samples=${BOOTSTRAP_SAMPLES}"
    echo "numa_node=${PACE_NUMA_NODE:-0}"
    echo "scaling_governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unavailable)"
    echo "boost=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo unavailable)"
    lscpu
} > "${RESULT_DIR}/environment.txt"

RUN_PREFIX=()
if command -v numactl >/dev/null 2>&1; then
    RUN_PREFIX=(
        numactl
        --cpunodebind="${PACE_NUMA_NODE:-0}"
        --membind="${PACE_NUMA_NODE:-0}"
    )
else
    echo "WARNING: numactl unavailable; job will not be NUMA-bound" >&2
fi

"${RUN_PREFIX[@]}" "${PYTHON}" \
    benchmarks/cpu/attention_bmm/bench_pace_prefill_stages.py \
    --out-dir "${RESULT_DIR}" \
    --shapes "${SHAPES}" \
    --query-lens "${QUERY_LENS}" \
    --kv-lens "${KV_LENS}" \
    --batch-sizes "${BATCH_SIZES}" \
    --data-seeds "${DATA_SEEDS}" \
    --warmups "${WARMUPS}" \
    --repeats "${REPEATS}" \
    --order-seed "${ORDER_SEED}" \
    --bootstrap-samples "${BOOTSTRAP_SAMPLES}"

echo "Report: ${RESULT_DIR}/pace_prefill_stage_report.md"
echo "Summary: ${RESULT_DIR}/pace_prefill_stage_summary.csv"
