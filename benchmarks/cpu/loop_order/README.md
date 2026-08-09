# CPU GEMM Loop-Order Benchmark

This benchmark isolates the locality effect of the six legal loop orders for
row-major square matrix multiplication:

```text
C[i, j] += A[i, k] * B[k, j]
```

It is intentionally single-threaded. This removes OpenMP scheduling and load
balancing from the first experiment, leaving loop order and memory traversal as
the variables under study.

## Experiment

For each matrix size, every round runs all six orders exactly once:

```text
ijk, ikj, jik, jki, kij, kji
```

Their order is shuffled independently in every warmup and measured round. Input
matrices are generated once per size from a deterministic seed. Clearing `C` and
checking correctness are outside the timed region.

Default threshold-sweep settings:

- Sizes: `32,48,64,80,96,112,128,160,192,224,256,320,384,448,512`
- Warmup rounds: `5`
- Measured paired rounds: `50`
- Threads: `1`
- Compiler flags: `-O3 -march=native -std=c++17`; GCC additionally receives
  `-fno-loop-interchange` so it cannot rewrite the source permutation being
  measured. Normal SIMD vectorization remains enabled.

The benchmark writes:

- `gemm_loop_order_trials.csv`: one row per loop order per measured round.
- `gemm_loop_order_summary.csv`: mean, median, p95, range, GFLOP/s, correctness,
  median rank/best marker, and speedup relative to `ijk` for each matrix size
  and loop order.
- `gemm_loop_order_threshold.csv`: paired `ikj` versus `kij` statistics and a
  strict decision at each sampled size, including the combined `A+B+C`
  working-set size.
- `gemm_loop_order_threshold.md`: an automatic report that only identifies a
  crossover when the strict winners form one consistent size transition.
- `environment.txt`: CPU, compiler, git commit, affinity, governor, and boost
  state for reproducibility.

## Run On mn01

Submit from the PACE repository root:

```bash
sbatch benchmarks/cpu/loop_order/slurm_gemm_loop_order.sbatch
```

Results stay inside the checkout at:

```text
benchmark_results/cpu-loop-order/<job_id>/
```

Optional overrides can be supplied with Slurm's environment export:

```bash
PACE_GEMM_SIZES=48,64,80,96,112,128 PACE_GEMM_REPEATS=50 PACE_GEMM_SEED=42 \
  sbatch --export=ALL benchmarks/cpu/loop_order/slurm_gemm_loop_order.sbatch
```

## Interpretation

In row-major storage, increasing the final index accesses adjacent memory.
Orders with `j` innermost (`ikj`, `kij`) therefore stream through rows of `B`
and `C`. Orders with `i` innermost (`jki`, `kji`) stride through rows of `A`
and `C`. The measurements establish how strongly that distinction matters on
the EPYC CPU before adding blocking, batching, or attention-specific shapes.

This benchmark is the first step only. The next experiment extends the same
method to rectangular attention matrices and batched matrix multiplication.
