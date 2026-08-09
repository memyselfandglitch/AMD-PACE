# CPU Attention BMM Loop-Order And Layout Benchmark

This benchmark extends the square GEMM experiment to the two matrix
multiplications used by attention:

```text
QK^T: scores[M,N] = query[M,D] * key[N,D]^T
P*V:  output[M,D] = probability[M,N] * value[N,D]
```

Here `M` is query length, `N` is KV-history length, and `D` is head dimension.
The benchmark processes all query heads and maps groups of query heads to a
shared KV head, matching grouped-query attention.

## Physical K/V Layouts

Both layouts contain identical K/V values:

```text
head_major:  [batch, kv_head, token, dim]
token_major: [batch, token, kv_head, dim]
```

The query, probability, and output matrices remain head-major. Layout is
therefore varied independently from mathematical loop order.

## Loop Names

For `QK^T`, `i,j,k` mean:

```text
i = query row, j = KV token, k = head dimension
```

For `P*V`, they mean standard GEMM dimensions:

```text
i = query row, j = output/head dimension, k = KV token
```

All six permutations (`ijk`, `ikj`, `jik`, `jki`, `kij`, `kji`) are compiled
separately. GCC loop interchange is disabled so the compiler cannot rewrite the
source permutation, while SIMD vectorization remains enabled.

## Default Experiment

- SLM proxy: Qwen2.5-0.5B-style `14` query heads, `2` KV heads, `D=64`
- LLM proxy: Qwen2.5-7B-style `28` query heads, `4` KV heads, `D=128`
- Query lengths: `1,16`
- KV lengths: `128,512`
- Batch size: `1`
- K/V layouts: `head_major,token_major`
- Operations: `QK^T,P*V`
- Warmups: `2`
- Measured randomized rounds: `10`
- CPU threads: `1`

Every round runs all 24 operation/layout/order candidates exactly once in a new
random order. Input data is deterministic, output clearing and correctness
checking happen outside the timed region, and all candidates see semantically
identical K/V values.

## Outputs

Results stay inside the repository at
`benchmark_results/cpu-attention-bmm/<job_id>/`:

- `attention_bmm_trials.csv`: one row per measured candidate invocation.
- `attention_bmm_summary.csv`: median, p95, GFLOP/s, overall rank, rank within
  each layout, and speedup over `head_major:ijk`.
- `attention_bmm_decisions.csv`: the best layout/order pair and runner-up for
  each operation and workload.
- `environment.txt`: compiler, git commit, CPU binding, governor, boost, and
  complete machine information.

## Run On mn01

From the repository root:

```bash
sbatch benchmarks/cpu/attention_bmm/slurm_attention_bmm.sbatch
```

After validating the default run, longer or batched workloads can be submitted
without changing code:

```bash
PACE_ATTN_QUERY_LENS=1,16,64 \
PACE_ATTN_KV_LENS=512,2048 \
PACE_ATTN_BATCH_SIZES=1,4 \
PACE_ATTN_REPEATS=10 \
  sbatch --export=ALL benchmarks/cpu/attention_bmm/slurm_attention_bmm.sbatch
```

## Scope

This is an FP32, single-threaded locality microbenchmark. It excludes softmax,
masking, block tables, OpenMP scheduling, and PACE's BF16 kernels. Its purpose
is to determine which loop/layout interactions deserve implementation and
hardware-counter profiling in PACE, not to predict end-to-end inference speed.
