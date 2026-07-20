#!/bin/bash
# Submit from the repository root with:
#   sbatch --partition=CPU_PARTITION benchmarks/llm/performance/slurm_block_size_sweep.sh
# CPU_PARTITION must be a CPU-only partition shown by `sinfo`; do not use GPU.
#
#SBATCH --job-name=pace-decode-blocks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# The submission directory is assumed to be the PACE repository. Override when
# submitting from elsewhere: sbatch --export=ALL,PACE_REPO=/path/to/AMD-PACE ...
PACE_REPO="${PACE_REPO:-${SLURM_SUBMIT_DIR}}"
BENCH_DIR="${PACE_REPO}/benchmarks/llm/performance"
SPEC="${PACE_SWEEP_SPEC:-${BENCH_DIR}/block_size_sweep_decode.json}"
SCRATCH_ROOT="${PACE_SCRATCH_ROOT:-/data/scratch/${USER}/pace-block-size}"
RUN_NAME="${PACE_RUN_NAME:-${SLURM_JOB_ID}}"
RESULT_DIR="${SCRATCH_ROOT}/results/${RUN_NAME}"

# Keep large Hugging Face/model caches and temporary files out of the 20 GB
# home allocation. Scratch is cleaned periodically, so copy valuable results out.
export HF_HOME="${SCRATCH_ROOT}/huggingface"
export TORCH_HOME="${SCRATCH_ROOT}/torch"
export TMPDIR="${SCRATCH_ROOT}/tmp/${SLURM_JOB_ID}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}" "${RESULT_DIR}"

# One Slurm CPU is one physical EPYC core for this request. These settings keep
# OpenMP workers pinned and prevent thread migration during measurements.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DYNAMIC=false

cd "${BENCH_DIR}"

# Activate the environment prepared on the cluster. Either export
# PACE_ACTIVATE=/path/to/venv/bin/activate, or ensure the current environment
# already contains PACE and its benchmark dependencies.
if [[ -n "${PACE_ACTIVATE:-}" ]]; then
    # shellcheck disable=SC1090
    source "${PACE_ACTIVATE}"
fi

# Prefer an explicitly supplied interpreter, then the environment's `python`,
# and finally `python3` (many cluster images do not provide a `python` alias).
if [[ -n "${PACE_PYTHON:-}" ]]; then
    PYTHON_BIN="${PACE_PYTHON}"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "ERROR: Python was not found. Submit with PACE_PYTHON=/absolute/path/to/python." >&2
    exit 127
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: PACE_PYTHON is not executable: ${PYTHON_BIN}" >&2
    exit 126
fi

echo "job_id=${SLURM_JOB_ID}"
echo "host=$(hostname)"
echo "started=$(date --iso-8601=seconds)"
echo "repository=${PACE_REPO}"
echo "spec=${SPEC}"
echo "results=${RESULT_DIR}"

# --exclusive asks Slurm not to colocate any other job on this node. Verify that
# the scheduler actually honored it before collecting performance numbers.
NODE_NAME="${SLURMD_NODENAME:-$(hostname -s)}"
OTHER_JOB_IDS="$(
    squeue --noheader --nodelist="${NODE_NAME}" --states=RUNNING --format='%A' \
        | awk -v own_job="${SLURM_JOB_ID}" '$1 != own_job {print $1}'
)"
echo "running allocations on ${NODE_NAME}:"
squeue --nodelist="${NODE_NAME}" --states=RUNNING \
    --format='%.18i %.12u %.12P %.10T %.6C %R'
if [[ -n "${OTHER_JOB_IDS}" ]]; then
    echo "ERROR: another Slurm allocation is running on exclusive node ${NODE_NAME}: ${OTHER_JOB_IDS}" >&2
    exit 2
fi

# Slurm exclusivity cannot remove OS daemons or detect work launched outside
# Slurm. Record the busiest processes so anomalous background load is visible.
echo "top CPU-consuming processes before benchmark:"
ps -eo user,pid,psr,pcpu,pmem,comm --sort=-pcpu | head -n 20 || true

lscpu
numactl --hardware 2>/dev/null || true
echo "python=${PYTHON_BIN}"
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" -c 'import torch; import pace; print("PACE Python imports: OK")'

# This cluster's Slurm build does not support --cpu-bind. Pin the complete
# process tree to one EPYC socket instead; child benchmark processes inherit
# both the CPU affinity and local-memory policy. Override with PACE_NUMA_NODE.
NUMA_PREFIX=()
if command -v numactl >/dev/null 2>&1; then
    PACE_NUMA_NODE="${PACE_NUMA_NODE:-0}"
    NUMA_PREFIX=(numactl --cpunodebind="${PACE_NUMA_NODE}" --membind="${PACE_NUMA_NODE}")
    echo "NUMA binding: node ${PACE_NUMA_NODE}"
else
    echo "WARNING: numactl is unavailable; proceeding without explicit NUMA binding" >&2
fi

"${NUMA_PREFIX[@]}" "${PYTHON_BIN}" -u block_size_sweep.py run \
    --spec "${SPEC}" \
    --output-dir "${RESULT_DIR}"

echo "finished=$(date --iso-8601=seconds)"
echo "Copy ${RESULT_DIR} to persistent storage before the weekly scratch cleanup."
