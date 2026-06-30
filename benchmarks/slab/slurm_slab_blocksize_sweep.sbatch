#!/bin/bash
#SBATCH --job-name=pace-bs-sweep
#SBATCH --partition=jobmn01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=benchmarks/slab/results/blocksize/slurm-%j-blocksize.out
#SBATCH --error=benchmarks/slab/results/blocksize/slurm-%j-blocksize.err

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

OUT_DIR="${OUT_DIR:-benchmarks/slab/results/blocksize}"
mkdir -p "$OUT_DIR"

JOB_ID="${SLURM_JOB_ID:-manual}"
CSV="${CSV:-$OUT_DIR/blocksize_sweep_${JOB_ID}.csv}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$OUT_DIR/analysis_${JOB_ID}}"

"$PYTHON" benchmarks/slab/bench_slab_blocksize.py \
  --out "$CSV" \
  --layouts "$SLAB_LAYOUT" \
  --block-sizes "${BLOCK_SIZES:-16,32,64,128,256}" \
  --phases "${PHASES:-prefill,decode,mtd}" \
  --seq-lens "${SEQ_LENS:-128,512,2048,8192}" \
  --batch-sizes "${BATCH_SIZES:-1,4,16}" \
  --shapes "${SHAPES:-default}" \
  --exit-fractions "${EXIT_FRACTIONS:-0.0,0.25,0.5,0.75}" \
  --warmups "${WARMUPS:-3}" \
  --repeats "${REPEATS:-10}" \
  --verbose

"$PYTHON" benchmarks/slab/analyze_blocksize_autotune.py "$CSV" \
  --layout "$SLAB_LAYOUT" \
  --out-dir "$ANALYSIS_DIR"

echo "CSV: $CSV"
echo "Analysis: $ANALYSIS_DIR"
