#!/bin/bash
#SBATCH --job-name=pace-bs-selperf
#SBATCH --partition=jobmn01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=benchmarks/slab/results/blocksize_perf_selected/slurm-%j-selected-perf.out
#SBATCH --error=benchmarks/slab/results/blocksize_perf_selected/slurm-%j-selected-perf.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
VENV_DIR="${VENV_DIR:-/data/scratch/$USER/pace-venv}"
PYTHON="${PYTHON:-python3}"

cd "$REPO_DIR"

if [ -f "$VENV_DIR/bin/activate" ]; then
  source "$VENV_DIR/bin/activate"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-64}}"
export SLAB_LAYOUT="${SLAB_LAYOUT:-head_major}"

JOB_ID="${SLURM_JOB_ID:-manual}"
OUT_DIR="${OUT_DIR:-benchmarks/slab/results/blocksize_perf_selected}"
ANALYSIS_DIR="${ANALYSIS_DIR:-benchmarks/slab/results/blocksize/analysis_6979}"
DECISIONS_CSV="${DECISIONS_CSV:-$ANALYSIS_DIR/decisions.csv}"
CASES_CSV="${CASES_CSV:-$ANALYSIS_DIR/selected_perf_cases_6979.csv}"
PERF_CSV="${PERF_CSV:-$OUT_DIR/selected_perf_${JOB_ID}.csv}"
RAW_DIR="${RAW_DIR:-$OUT_DIR/raw_${JOB_ID}}"

mkdir -p "$OUT_DIR" "$RAW_DIR"

if [ ! -f "$CASES_CSV" ]; then
  "$PYTHON" benchmarks/slab/select_blocksize_perf_cases.py "$DECISIONS_CSV" \
    --out "$CASES_CSV" \
    --max-cases "${MAX_CASES:-10}"
fi

"$PYTHON" benchmarks/slab/run_blocksize_perf_cases.py \
  --cases "$CASES_CSV" \
  --out "$PERF_CSV" \
  --raw-dir "$RAW_DIR" \
  --layout "$SLAB_LAYOUT" \
  --warmups "${WARMUPS:-3}" \
  --repeats "${REPEATS:-10}" \
  --python "$PYTHON" \
  --perf-events "${PERF_EVENTS:-task-clock,cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses}"

echo "Selected cases: $CASES_CSV"
echo "Perf CSV: $PERF_CSV"
