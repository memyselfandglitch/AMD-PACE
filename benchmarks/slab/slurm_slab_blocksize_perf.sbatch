#!/bin/bash
#SBATCH --job-name=pace-bs-perf
#SBATCH --partition=jobmn01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=benchmarks/slab/results/blocksize_perf/slurm-%j-perf.out
#SBATCH --error=benchmarks/slab/results/blocksize_perf/slurm-%j-perf.err

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

OUT_DIR="${OUT_DIR:-benchmarks/slab/results/blocksize_perf}"
JOB_ID="${SLURM_JOB_ID:-manual}"
RAW_DIR="$OUT_DIR/raw_${JOB_ID}"
PERF_CSV="${PERF_CSV:-$OUT_DIR/blocksize_perf_${JOB_ID}.csv}"

mkdir -p "$RAW_DIR"

PERF_EVENTS="${PERF_EVENTS:-task-clock,cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses}"

IFS=',' read -r -a PHASE_ARR <<< "${PHASES:-prefill,decode,mtd}"
IFS=',' read -r -a SHAPE_ARR <<< "${SHAPES:-slm_gqa,llama_gqa,mha}"
IFS=',' read -r -a SEQ_ARR <<< "${SEQ_LENS:-128,8192}"
IFS=',' read -r -a EXIT_ARR <<< "${EXIT_FRACTIONS:-0.0,0.5}"
IFS=',' read -r -a BLOCK_ARR <<< "${BLOCK_SIZES:-16,32,64,128,256}"

rm -f "$PERF_CSV"

for phase in "${PHASE_ARR[@]}"; do
  for shape in "${SHAPE_ARR[@]}"; do
    for seq_len in "${SEQ_ARR[@]}"; do
      for exit_fraction in "${EXIT_ARR[@]}"; do
        if [ "$phase" = "prefill" ] && [ "$exit_fraction" != "0.0" ]; then
          continue
        fi
        for block_size in "${BLOCK_ARR[@]}"; do
          tag="${phase}_${shape}_seq${seq_len}_exit${exit_fraction}_bs${block_size}"
          latency_csv="$RAW_DIR/${tag}.latency.csv"
          perf_stat="$RAW_DIR/${tag}.perf.csv"

          echo "perf case: $tag"
          perf stat -x, -e "$PERF_EVENTS" -o "$perf_stat" -- \
            "$PYTHON" benchmarks/slab/bench_slab_blocksize.py \
              --out "$latency_csv" \
              --layouts "$SLAB_LAYOUT" \
              --block-sizes "$block_size" \
              --phases "$phase" \
              --seq-lens "$seq_len" \
              --batch-sizes "${BATCH_SIZE:-16}" \
              --shapes "$shape" \
              --exit-fractions "$exit_fraction" \
              --warmups "${WARMUPS:-2}" \
              --repeats "${REPEATS:-5}"

          "$PYTHON" benchmarks/slab/parse_perf_stat.py \
            --latency-csv "$latency_csv" \
            --perf-stat "$perf_stat" \
            --out "$PERF_CSV"
        done
      done
    done
  done
done

"$PYTHON" benchmarks/slab/analyze_blocksize_autotune.py "$PERF_CSV" \
  --layout "$SLAB_LAYOUT" \
  --out-dir "$OUT_DIR/analysis_${JOB_ID}"

echo "Perf CSV: $PERF_CSV"
echo "Analysis: $OUT_DIR/analysis_${JOB_ID}"
