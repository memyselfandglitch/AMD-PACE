#!/bin/bash
#SBATCH --job-name=pace-slab-perf
#SBATCH --partition=jobmn01
#SBATCH --output=benchmarks/slab/results/slurm-%j-perf.out
#SBATCH --error=benchmarks/slab/results/slurm-%j-perf.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=01:00:00

set -euo pipefail

# Submit from the AMD-PACE repo root:
#   sbatch benchmarks/slab/slurm_slab_perf.sbatch
#
# Override the selected case with environment variables, for example:
#   PHASE=mtd LAYOUT=head_major BLOCK_SIZE=16 BATCH_SIZE=16 SEQ_LEN=8192 \
#   SHAPE=slm_gqa:8:4:64 EXIT_FRACTION=0.5 \
#   sbatch benchmarks/slab/slurm_slab_perf.sbatch

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p benchmarks/slab/results/perf

VENV_DIR="${PACE_VENV_DIR:-/data/scratch/${USER}/pace-venv}"
if [ ! -x "${VENV_DIR}/bin/python" ]; then
  echo "Missing virtualenv at ${VENV_DIR}."
  echo "Run first: sbatch benchmarks/slab/slurm_setup_venv.sbatch"
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

if [ ! -f pace/lib/libpace_cpp.so ]; then
  mkdir -p pace/lib
  BUILT_LIB="$(find build -name libpace_cpp.so -print -quit)"
  if [ -z "${BUILT_LIB}" ]; then
    echo "Missing pace/lib/libpace_cpp.so."
    echo "Run first: sbatch benchmarks/slab/slurm_build_and_quick.sbatch"
    exit 1
  fi
  cp "${BUILT_LIB}" pace/lib/libpace_cpp.so
fi

if ! command -v perf >/dev/null 2>&1; then
  echo "perf was not found on this node. Ask the cluster admin whether Linux perf is available."
  exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-64}}"
export PACE_LOG_LEVEL="${PACE_LOG_LEVEL:-warning}"

PHASE="${PHASE:-mtd}"
LAYOUT="${LAYOUT:-head_major}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEQ_LEN="${SEQ_LEN:-8192}"
SHAPE="${SHAPE:-slm_gqa:8:4:64}"
EXIT_FRACTION="${EXIT_FRACTION:-0.5}"
WARMUPS="${WARMUPS:-5}"
REPEATS="${REPEATS:-20}"
SLAB_SCHEDULE_VALUE="${SLAB_SCHEDULE:-auto}"

if [ "${SLAB_SCHEDULE_VALUE}" = "auto" ]; then
  unset SLAB_SCHEDULE || true
else
  export SLAB_SCHEDULE="${SLAB_SCHEDULE_VALUE}"
fi

CASE_TAG="${PHASE}_${LAYOUT}_bs${BLOCK_SIZE}_b${BATCH_SIZE}_s${SEQ_LEN}_$(echo "${SHAPE}" | tr ':' '_')_exit${EXIT_FRACTION}_sched${SLAB_SCHEDULE_VALUE}"
BENCH_CSV="benchmarks/slab/results/perf/${CASE_TAG}_${SLURM_JOB_ID:-local}_bench.csv"
PERF_RAW="benchmarks/slab/results/perf/${CASE_TAG}_${SLURM_JOB_ID:-local}_perf_raw.csv"
PERF_PARSED="benchmarks/slab/results/perf/${CASE_TAG}_${SLURM_JOB_ID:-local}_perf.csv"

DEFAULT_EVENTS="task-clock,cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses"
GENERIC_EVENTS="task-clock,cycles,instructions,cache-references,cache-misses"
PERF_EVENTS="${PERF_EVENTS:-${DEFAULT_EVENTS}}"

echo "repo=${REPO_ROOT}"
echo "case=${CASE_TAG}"
echo "events=${PERF_EVENTS}"
echo "omp_threads=${OMP_NUM_THREADS}"

BENCH_CMD=(
  python benchmarks/slab/bench_slab_layout_blocksize.py
  --phases "${PHASE}"
  --layouts "${LAYOUT}"
  --block-sizes "${BLOCK_SIZE}"
  --batch-sizes "${BATCH_SIZE}"
  --seq-lens "${SEQ_LEN}"
  --shapes "${SHAPE}"
  --exit-fractions "${EXIT_FRACTION}"
  --warmups "${WARMUPS}"
  --repeats "${REPEATS}"
  --output "${BENCH_CSV}"
)

set +e
perf stat -x, -e "${PERF_EVENTS}" -o "${PERF_RAW}" -- "${BENCH_CMD[@]}"
PERF_STATUS=$?
set -e

if [ "${PERF_STATUS}" -ne 0 ] && [ "${PERF_EVENTS}" != "${GENERIC_EVENTS}" ]; then
  echo "perf failed with requested events. Retrying with generic cache counters only."
  PERF_EVENTS="${GENERIC_EVENTS}"
  set +e
  perf stat -x, -e "${PERF_EVENTS}" -o "${PERF_RAW}" -- "${BENCH_CMD[@]}"
  PERF_STATUS=$?
  set -e
fi

if [ "${PERF_STATUS}" -ne 0 ]; then
  echo "perf failed. This may be a perf_event permission issue on the server."
  echo "Raw perf output, if any:"
  cat "${PERF_RAW}" 2>/dev/null || true
  exit "${PERF_STATUS}"
fi

python benchmarks/slab/parse_perf_stat.py \
  --perf-raw "${PERF_RAW}" \
  --bench-csv "${BENCH_CSV}" \
  --output "${PERF_PARSED}" \
  --metadata "job_id=${SLURM_JOB_ID:-local}" \
  --metadata "omp_threads=${OMP_NUM_THREADS}" \
  --metadata "slab_schedule=${SLAB_SCHEDULE_VALUE}" \
  --metadata "perf_events=${PERF_EVENTS}"

echo "bench_csv=${BENCH_CSV}"
echo "perf_raw=${PERF_RAW}"
echo "perf_csv=${PERF_PARSED}"
cat "${PERF_PARSED}"
