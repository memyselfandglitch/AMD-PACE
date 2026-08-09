#!/bin/bash
#SBATCH --job-name=pace-pv-uprof
#SBATCH --partition=jobmn01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PACE_REPO="${PACE_REPO:-${SLURM_SUBMIT_DIR}}"
BENCH_DIR="${PACE_REPO}/benchmarks/cpu/attention_bmm"
RESULT_DIR="${PACE_RESULT_DIR:-${PACE_REPO}/benchmark_results/cpu-pv-crossover-uprof/${SLURM_JOB_ID}}"
AMDUPROFPCM="${AMDUPROFPCM:-/data/scratch/deveshisingh/tools/AMDuProf_Linux_x64_5.3.521/bin/AMDuProfPcm}"
CXX_BIN="${CXX:-g++}"
CPU_CORE="${PACE_CPU_CORE:-0}"
NUMA_NODE="${PACE_NUMA_NODE:-0}"

mkdir -p "${RESULT_DIR}"
cd "${PACE_REPO}"

if [[ ! -x "${AMDUPROFPCM}" ]]; then
    echo "AMDuProfPcm is not executable: ${AMDUPROFPCM}" >&2
    exit 1
fi

"${CXX_BIN}" -O3 -march=native -std=c++17 -DNDEBUG -fno-loop-interchange \
    "${BENCH_DIR}/pv_profile_case.cpp" \
    -o "${RESULT_DIR}/pv_profile_case"

RUN_PREFIX=()
if command -v numactl >/dev/null 2>&1; then
    RUN_PREFIX=(numactl --physcpubind="${CPU_CORE}" --membind="${NUMA_NODE}")
fi

# uProf launches this wrapper so both the benchmark and its memory stay on one NUMA node.
WRAPPER="${RESULT_DIR}/run_bound.sh"
cat > "${WRAPPER}" <<EOF
#!/bin/bash
exec ${RUN_PREFIX[*]} "${RESULT_DIR}/pv_profile_case" "\$@"
EOF
chmod +x "${WRAPPER}"

python3 "${BENCH_DIR}/run_pv_uprof_cases.py" \
    --binary "${WRAPPER}" \
    --amd-uprof-pcm "${AMDUPROFPCM}" \
    --out-dir "${RESULT_DIR}" \
    --minimum-seconds "${PACE_PROFILE_SECONDS:-5}" \
    --scope-args=-a

echo "Results: ${RESULT_DIR}"
