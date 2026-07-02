#!/bin/bash
#SBATCH --job-name=pace-bs-uprof
#SBATCH --partition=jobmn01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=benchmarks/slab/results/blocksize_uprof_pcm_selected/slurm-%j-uprof.out
#SBATCH --error=benchmarks/slab/results/blocksize_uprof_pcm_selected/slurm-%j-uprof.err

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

find_uprof_pcm() {
  if [ -n "${AMDUPROFPCM:-}" ] && [ -x "$AMDUPROFPCM" ]; then
    printf '%s\n' "$AMDUPROFPCM"
    return 0
  fi

  if command -v AMDuProfPcm >/dev/null 2>&1; then
    command -v AMDuProfPcm
    return 0
  fi

  for candidate in \
    /opt/AMDuProf*/bin/AMDuProfPcm \
    /opt/AMD/AMDuProf*/bin/AMDuProfPcm \
    "$HOME"/AMDuProf*/bin/AMDuProfPcm
  do
    if [ -x "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

AMDUPROFPCM_BIN="$(find_uprof_pcm || true)"
if [ -z "$AMDUPROFPCM_BIN" ]; then
  cat >&2 <<'EOF'
Could not find AMDuProfPcm.

Set AMDUPROFPCM to the full path, for example:
  AMDUPROFPCM=/path/to/AMDuProf/bin/AMDuProfPcm sbatch benchmarks/slab/slurm_slab_blocksize_uprof_pcm_selected.sbatch

If AMD uProf is not installed on this server, ask the admin/AMD contact whether
AMDuProfPcm is available on jobmn01 and whether user-level profiling is allowed.
EOF
  exit 1
fi

JOB_ID="${SLURM_JOB_ID:-manual}"
OUT_DIR="${OUT_DIR:-benchmarks/slab/results/blocksize_uprof_pcm_selected}"
ANALYSIS_DIR="${ANALYSIS_DIR:-benchmarks/slab/results/blocksize/analysis_6979}"
DECISIONS_CSV="${DECISIONS_CSV:-$ANALYSIS_DIR/decisions.csv}"
CASES_CSV="${CASES_CSV:-$ANALYSIS_DIR/selected_perf_cases_6979.csv}"
UPROF_MANIFEST="${UPROF_MANIFEST:-$OUT_DIR/selected_uprof_pcm_${JOB_ID}.csv}"
RAW_DIR="${RAW_DIR:-$OUT_DIR/raw_${JOB_ID}}"

mkdir -p "$OUT_DIR" "$RAW_DIR"

if [ ! -f "$CASES_CSV" ]; then
  "$PYTHON" benchmarks/slab/select_blocksize_perf_cases.py "$DECISIONS_CSV" \
    --out "$CASES_CSV" \
    --max-cases "${MAX_CASES:-10}"
fi

"$PYTHON" benchmarks/slab/run_blocksize_uprof_pcm_cases.py \
  --cases "$CASES_CSV" \
  --out "$UPROF_MANIFEST" \
  --raw-dir "$RAW_DIR" \
  --amd-uprof-pcm "$AMDUPROFPCM_BIN" \
  --layout "$SLAB_LAYOUT" \
  --warmups "${WARMUPS:-10}" \
  --repeats "${REPEATS:-5000}" \
  --python "$PYTHON" \
  --metrics "${UPROF_METRICS:-ipc,l2,l3}" \
  --scope-args "${UPROF_SCOPE_ARGS:--a}"

echo "AMDuProfPcm: $AMDUPROFPCM_BIN"
echo "Selected cases: $CASES_CSV"
echo "uProf PCM manifest: $UPROF_MANIFEST"
echo "Raw uProf CSVs: $RAW_DIR"
