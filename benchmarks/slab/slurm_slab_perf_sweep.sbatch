#!/bin/bash
#SBATCH --job-name=pace-slab-perf-sweep
#SBATCH --partition=jobmn01
#SBATCH --output=benchmarks/slab/results/slurm-%j-perf-sweep.out
#SBATCH --error=benchmarks/slab/results/slurm-%j-perf-sweep.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=08:00:00

set -euo pipefail

# Submit from the AMD-PACE repo root:
#   sbatch benchmarks/slab/slurm_slab_perf_sweep.sbatch
#
# Default matrix is focused on autotuning evidence:
# - phases: prefill, decode, mtd
# - short and long contexts: 128, 8192
# - both layouts
# - representative block sizes: 16, 64, 128, 256
# - three attention shapes
# - no-exit and 50% premature-exit for decode/MTD
#
# Override with env vars, e.g.:
#   PHASES=decode,mtd SEQ_LENS=2048,8192 REPEATS=20 \
#   sbatch benchmarks/slab/slurm_slab_perf_sweep.sbatch

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
mkdir -p benchmarks/slab/results/perf_sweeps

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

PHASES="${PHASES:-prefill,decode,mtd}"
LAYOUTS="${LAYOUTS:-head_major,block_major}"
BLOCK_SIZES="${BLOCK_SIZES:-16,64,128,256}"
BATCH_SIZES="${BATCH_SIZES:-16}"
SEQ_LENS="${SEQ_LENS:-128,8192}"
SHAPES="${SHAPES:-slm_gqa:8:4:64,llama_gqa:32:8:128,mha:8:8:64}"
EXIT_FRACTIONS="${EXIT_FRACTIONS:-0.0,0.5}"
WARMUPS="${WARMUPS:-3}"
REPEATS="${REPEATS:-10}"
SLAB_SCHEDULE_VALUE="${SLAB_SCHEDULE:-auto}"

DEFAULT_EVENTS="task-clock,cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses"
GENERIC_EVENTS="task-clock,cycles,instructions,cache-references,cache-misses"
PERF_EVENTS="${PERF_EVENTS:-${DEFAULT_EVENTS}}"

JOB_ID="${SLURM_JOB_ID:-local}"
OUT_DIR="benchmarks/slab/results/perf_sweeps"
COMBINED_CSV="${OUT_DIR}/perf_sweep_${JOB_ID}.csv"
ANALYSIS_DIR="${OUT_DIR}/analysis_${JOB_ID}"
WORK_DIR="${PERF_WORK_DIR:-${SLURM_TMPDIR:-/tmp}/pace_slab_perf_${JOB_ID}}"
mkdir -p "${WORK_DIR}"

echo "repo=${REPO_ROOT}"
echo "combined_csv=${COMBINED_CSV}"
echo "analysis_dir=${ANALYSIS_DIR}"
echo "work_dir=${WORK_DIR}"
echo "events=${PERF_EVENTS}"
echo "omp_threads=${OMP_NUM_THREADS}"

split_csv() {
  local raw="$1"
  raw="${raw//,/ }"
  echo "${raw}"
}

append_csv() {
  local src="$1"
  local dst="$2"
  if [ ! -f "${dst}" ]; then
    cat "${src}" > "${dst}"
  else
    tail -n +2 "${src}" >> "${dst}"
  fi
}

run_one_case() {
  local phase="$1"
  local layout="$2"
  local block_size="$3"
  local batch_size="$4"
  local seq_len="$5"
  local shape="$6"
  local exit_fraction="$7"

  if [ "${SLAB_SCHEDULE_VALUE}" = "auto" ]; then
    unset SLAB_SCHEDULE || true
  else
    export SLAB_SCHEDULE="${SLAB_SCHEDULE_VALUE}"
  fi

  local shape_tag
  shape_tag="$(echo "${shape}" | tr ':' '_')"
  local case_tag="${phase}_${layout}_bs${block_size}_b${batch_size}_s${seq_len}_${shape_tag}_exit${exit_fraction}_sched${SLAB_SCHEDULE_VALUE}"
  local bench_csv="${WORK_DIR}/${case_tag}_bench.csv"
  local perf_raw="${WORK_DIR}/${case_tag}_perf_raw.csv"
  local perf_csv="${WORK_DIR}/${case_tag}_perf.csv"
  local effective_events="${PERF_EVENTS}"

  local -a bench_cmd=(
    python benchmarks/slab/bench_slab_layout_blocksize.py
    --phases "${phase}"
    --layouts "${layout}"
    --block-sizes "${block_size}"
    --batch-sizes "${batch_size}"
    --seq-lens "${seq_len}"
    --shapes "${shape}"
    --exit-fractions "${exit_fraction}"
    --warmups "${WARMUPS}"
    --repeats "${REPEATS}"
    --output "${bench_csv}"
  )

  echo "case=${case_tag}"
  set +e
  perf stat -x, -e "${effective_events}" -o "${perf_raw}" -- "${bench_cmd[@]}"
  local perf_status=$?
  set -e

  if [ "${perf_status}" -ne 0 ] && [ "${effective_events}" != "${GENERIC_EVENTS}" ]; then
    echo "perf failed with requested events; retrying generic cache counters for ${case_tag}."
    effective_events="${GENERIC_EVENTS}"
    set +e
    perf stat -x, -e "${effective_events}" -o "${perf_raw}" -- "${bench_cmd[@]}"
    perf_status=$?
    set -e
  fi

  if [ "${perf_status}" -ne 0 ]; then
    echo "perf failed for ${case_tag}. Raw perf output, if any:"
    cat "${perf_raw}" 2>/dev/null || true
    exit "${perf_status}"
  fi

  python benchmarks/slab/parse_perf_stat.py \
    --perf-raw "${perf_raw}" \
    --bench-csv "${bench_csv}" \
    --output "${perf_csv}" \
    --metadata "job_id=${JOB_ID}" \
    --metadata "case_tag=${case_tag}" \
    --metadata "perf_events=${effective_events}"

  append_csv "${perf_csv}" "${COMBINED_CSV}"
}

case_count=0
for phase in $(split_csv "${PHASES}"); do
  for layout in $(split_csv "${LAYOUTS}"); do
    for block_size in $(split_csv "${BLOCK_SIZES}"); do
      for batch_size in $(split_csv "${BATCH_SIZES}"); do
        for seq_len in $(split_csv "${SEQ_LENS}"); do
          for shape in $(split_csv "${SHAPES}"); do
            if [ "${phase}" = "prefill" ]; then
              run_one_case "${phase}" "${layout}" "${block_size}" "${batch_size}" "${seq_len}" "${shape}" "0.0"
              case_count=$((case_count + 1))
            else
              for exit_fraction in $(split_csv "${EXIT_FRACTIONS}"); do
                run_one_case "${phase}" "${layout}" "${block_size}" "${batch_size}" "${seq_len}" "${shape}" "${exit_fraction}"
                case_count=$((case_count + 1))
              done
            fi
          done
        done
      done
    done
  done
done

python benchmarks/slab/summarize_perf_autotune.py "${COMBINED_CSV}" \
  --out-dir "${ANALYSIS_DIR}"

echo "Wrote ${case_count} rows to ${COMBINED_CSV}"
echo "Summary: ${ANALYSIS_DIR}/perf_autotune_summary.md"
echo "Decisions: ${ANALYSIS_DIR}/perf_autotune_decisions.csv"
head -n 5 "${COMBINED_CSV}"
