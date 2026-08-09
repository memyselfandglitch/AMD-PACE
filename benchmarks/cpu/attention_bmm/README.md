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

## Focused Long-Context P*V Claim

`pv_loop_crossover.cpp` tests the strongest pattern from the initial BMM run
without carrying all 24 candidates into a larger sweep. It fixes V to one
head-major attention head and compares only:

```text
ikj: complete one query row while streaming through V
kij: reuse each V[token, :] row across all query rows
```

The pre-registered claim is:

- At `M=1`, the loops are equivalent and must be reported as a tie.
- At long context (`M=16..128`, `N>=2048`), `kij` should win by reusing each
  `V[token, :]` row across query rows.
- Other cells are exploratory and locate where the locality advantage appears
  or reverses. A universal order is rejected only if each order earns at least
  two strict workload wins.

Defaults cover `M=1..512`, `N=128..16384`, both head dimensions, and three
independent random dense inputs. Each workload has `60` measured pairs: three
data seeds times 20 repeats. Candidate order is randomized within every pair.

A winner requires all of the following:

- At least 5% paired median improvement.
- At least 80% paired wins.
- A 95% bootstrap confidence interval excluding 1.

Run from the repository root:

```bash
sbatch benchmarks/cpu/attention_bmm/slurm_pv_loop_crossover.sbatch
```

Results are written to `benchmark_results/cpu-pv-crossover/<job_id>/`. The
generated `pv_crossover_report.md` states how many pre-registered workloads
matched the claim and lists every counterexample automatically.

## Hardware-Counter Validation

The selected uProf run isolates each loop order in its own process for four
cases: a decode control, the `kij` side of the SLM `M=512` crossover, and the
two long-context `ikj` cases. Each process spends at least five seconds in one
kernel so setup is a small fraction of process-level PCM totals.

```bash
sbatch benchmarks/cpu/attention_bmm/slurm_pv_crossover_uprof.sbatch
```

Results are written to `benchmark_results/cpu-pv-crossover-uprof/<job_id>/`.
`pv_uprof_metrics.csv` contains all parsed IPC, L2, L3, and memory metrics in
one long-form table; `raw/` preserves the complete AMD uProf reports.

## Real SlabPool Prefill Tile Sweep

PACE prefill already computes `QK^T` and softmax-times-V with tiled BF16
oneDNN BRGeMM rather than scalar loops. `PACE_SLAB_PREFILL_Q_TILE` therefore
exposes the corresponding production tuning knob while retaining `64` as the
default. Valid values are `16,32,64,128`; decode and MTD dispatch are unchanged.

The real-kernel sweep covers SLM/LLM shapes, query lengths `128,512`, KV
lengths `2048,8192,16384`, batch sizes `1,4`, four tiles, three random inputs,
two warmups, and 20 measurements. It checks every candidate output against the
default tile and automatically recommends a non-default tile only when it is
at least 5% faster, wins at least 80% of paired rounds, has a bootstrap CI above
one, and does not regress p95.

```bash
sbatch benchmarks/cpu/attention_bmm/slurm_pace_prefill_tile_sweep.sbatch
```

Run this one-workload smoke test first:

```bash
PACE_TILE_SHAPES=slm \
PACE_TILE_QUERY_LENS=128 \
PACE_TILE_KV_LENS=2048 \
PACE_TILE_BATCH_SIZES=1 \
PACE_TILE_DATA_SEEDS=11 \
PACE_TILE_WARMUPS=1 \
PACE_TILE_REPEATS=3 \
  sbatch --export=ALL \
  benchmarks/cpu/attention_bmm/slurm_pace_prefill_tile_sweep.sbatch
```

This job rebuilds the native PACE library because the selectable tile is C++
code. Results are written to `benchmark_results/pace-prefill-tile/<job_id>/`,
including raw trials, per-tile summaries, workload decisions, environment
metadata, and a ready-to-read Markdown report.
