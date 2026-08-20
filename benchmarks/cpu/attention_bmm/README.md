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

## Real Prefill Stage Profile And Packing-Reuse Gate

The stage profiler measures where time is spent inside the production BF16
SlabPool prefill kernel before attempting a K/V-reuse rewrite. It records:

```text
BRGeMM cache/context setup, Q preparation, K packing, QK^T,
mask/online softmax, V packing, P*V, and output normalization
```

Profiling is disabled by default. When enabled through the diagnostic SlabPool
API, timing accumulates in thread-local structs and is reduced after the OpenMP
region; the timed kernel contains no file I/O, mutex, or shared atomic update.
The benchmark randomizes one profiled and one unprofiled call in every paired
round. It separately reports:

- the runtime cost of each timer pair;
- the empty interval observed between two adjacent timer reads;
- measured profiled/unprofiled wall-latency overhead;
- bias-corrected K+V packing fraction with a bootstrap 95% interval;
- the ideal Amdahl speedup ceiling if packing were reused across query tiles.

Packing reuse is grouped by query-tile count (`query_len / 64`), because that
is the number of times the current query-tile-outer kernel revisits each KV
block. The report recommends a synthetic reuse prototype only when profiler
overhead's upper 95% bound is at most 2%, estimated timer bias's upper 95%
bound is at most 1%, and the lower 95% bound of the ideal reuse ceiling is at
least `1.05x`. This is a go/no-go gate, not a claim that the ideal speedup is
achievable.

Run the complete first-stage experiment from the repository root:

```bash
sbatch benchmarks/cpu/attention_bmm/slurm_pace_prefill_stage_profile.sbatch
```

Run a small build and schema smoke test first:

```bash
PACE_STAGE_SHAPES=slm \
PACE_STAGE_QUERY_LENS=128 \
PACE_STAGE_KV_LENS=2048 \
PACE_STAGE_BATCH_SIZES=1 \
PACE_STAGE_DATA_SEEDS=11 \
PACE_STAGE_REPEATS=3 \
  sbatch --export=ALL \
  benchmarks/cpu/attention_bmm/slurm_pace_prefill_stage_profile.sbatch
```

Results are written to `benchmark_results/pace-prefill-stage/<job_id>/`:

- `pace_prefill_stage_latency_trials.csv`: randomized paired wall times;
- `pace_prefill_stage_profiles.csv`: raw and corrected stage counters;
- `pace_prefill_stage_summary.csv`: per-workload confidence intervals and gate;
- `pace_prefill_stage_report.md`: ready-to-read decision table;
- `environment.txt`: exact hardware, binding, software, and sweep settings.

This first gate intentionally profiles the attention call only. Slab allocation
and `cache_update` happen once during workload setup and remain outside the
paired timing region. Allocation/copy timing is a separate follow-up because it
answers a different question from repeated K/V packing inside prefill.

## Tiled BF16 IKJ Versus KIJ

`pv_brgemm_dataflow.cpp` translates the two strongest scalar P*V loop orders
into macro-tiled dataflows while retaining PACE's oneDNN BF16 BRGeMM ukernel:

```text
IKJ: query tile -> KV block -> output-dimension tile
KIJ: KV block -> query tile -> output-dimension tile
```

Both candidates consume identical pre-tiled probability and V buffers. Operand
conversion and V packing happen before timing, so the comparison isolates the
cache effect of traversal order rather than measuring different preprocessing.
The initial gate is single-head and single-threaded by design; physical
head-major/block-major layout, GQA batching, online softmax, and OpenMP are held
out until KIJ demonstrates at least two strict wins.

The default matrix covers query lengths `64,128,256,512`, KV lengths
`2048,8192,16384`, head dimensions `64,128`, and three random inputs. Here the
P*V dimensions are `M=query length`, `K=KV length`, and `N=head dimension`;
head dimension alone is not treated as a complete SLM/LLM model shape. Each
workload receives two warmups and 20 paired measurements in randomized order.
A strict winner needs a 5% paired median effect, 80% pair wins, a bootstrap 95%
confidence interval excluding one, and non-regressing p95.

Run a small build/correctness smoke test:

```bash
PACE_FLOW_QUERY_LENS=64,128 \
PACE_FLOW_KV_LENS=2048 \
PACE_FLOW_HEAD_DIMS=64 \
PACE_FLOW_DATA_SEEDS=11 \
PACE_FLOW_WARMUPS=1 \
PACE_FLOW_REPEATS=3 \
  sbatch --export=ALL \
  benchmarks/cpu/attention_bmm/slurm_pv_brgemm_dataflow.sbatch
```

Then submit the default experiment:

```bash
sbatch benchmarks/cpu/attention_bmm/slurm_pv_brgemm_dataflow.sbatch
```

Results are written under `benchmark_results/pv-brgemm-dataflow/<job_id>/` as
the raw paired trials, workload summary, automatic Markdown report, and exact
environment metadata.

## QK Transpose Full-K Versus IKJ/KIJ

`qk_brgemm_dataflow.cpp` applies the same controlled method to QK transpose.
Unlike P*V, a faithful tiled translation of scalar `IKJ` and `KIJ` must expose
the reduction dimension as multiple chunks:

```text
PACE baseline: query tile -> KV tile, one full-head-dimension BRGeMM
IKJ:           query tile -> K chunk -> KV tile
KIJ:           K chunk -> query tile -> KV tile
```

The default K chunk is `32`; query and KV output tiles remain `64x64`. The PACE
baseline is essential because split-K execution introduces more BRGeMM calls.
An ordering is useful only if its locality benefit exceeds that overhead and
improves the implementation PACE uses today. Q and K generation plus K packing
are identical and outside the timed region.

Run the build and correctness smoke test:

```bash
PACE_QK_QUERY_LENS=64,128 \
PACE_QK_KV_LENS=2048 \
PACE_QK_HEAD_DIMS=64 \
PACE_QK_DATA_SEEDS=11 \
PACE_QK_WARMUPS=1 \
PACE_QK_REPEATS=3 \
  sbatch --export=ALL \
  benchmarks/cpu/attention_bmm/slurm_qk_brgemm_dataflow.sbatch
```

Then run the default 24-workload experiment:

```bash
sbatch benchmarks/cpu/attention_bmm/slurm_qk_brgemm_dataflow.sbatch
```

Results are written under `benchmark_results/qk-brgemm-dataflow/<job_id>/`.
The generated report retains the current PACE baseline unless IKJ or KIJ earns
a strict improvement in at least two workloads.
# Decode KV layout versus traversal

`pace_decode_layout_traversal` isolates the interaction between physical KV
storage and fused decode traversal. It compares:

- head-major storage with head-first traversal (current design),
- block-major storage with head-first traversal (layout-only control),
- head-major storage with block-first traversal (traversal-only control), and
- block-major storage with block-first traversal (co-designed candidate).

The benchmark uses BF16 K/V and queries, AVX-512 BF16 dot products, PACE's
blockwise online-softmax structure, GQA sharing, randomized paired repeats, and
correctness checks. It is deliberately single-threaded; a production SlabPool
prototype and OpenMP study come only after a repeatable co-design signal.

Run on the CPU node from the repository root:

```bash
sbatch benchmarks/cpu/attention_bmm/slurm_decode_layout_traversal.sbatch
```

Run the controlled crossover study, which independently varies KV-head count,
GQA ratio, and head dimension:

```bash
PACE_DECODE_PROFILE=crossover \
  sbatch --export=ALL \
  benchmarks/cpu/attention_bmm/slurm_decode_layout_traversal.sbatch
```

Custom shapes use
`family/name:num_q_heads:num_kv_heads:head_dim`. Multiple block sizes may be
specified with `PACE_DECODE_BLOCK_SIZES=16,32,64,128,256`; the crossover
profile intentionally fixes block size at 64 until a shape/length boundary is
identified.

Run the preregistered targeted validation after the broad crossover sweep:

```bash
sbatch benchmarks/cpu/attention_bmm/slurm_decode_layout_targeted.sbatch
```

This job generates and runs two lanes without manual post-processing:

- a 64-case matrix matching total logical K+V bytes per call across MHA KV-head
  counts `2,4,8,16` and sequential batch sizes `1,4`; and
- a six-case MHA-2, batch-4 transition lane spanning sequence lengths
  `6144` through `16384`, retaining all four layout/traversal configurations.

The harness records that batch is processed by a sequential outer loop, reports
both per-sequence and per-call KV bytes and block counts, uses 64-byte-aligned
storage, and balances candidate position with randomized four-round Latin-square
cycles. The Slurm job performs three independent process launches and only calls
a result repeatable when it passes the strict paired criterion in at least two
launches. Outputs are written under
`benchmark_results/decode-layout-targeted/<job_id>/`.

Inspect a targeted summary directly from the terminal:

```bash
python3 benchmarks/cpu/attention_bmm/inspect_decode_layout_results.py \
  benchmark_results/decode-layout-targeted/7130/decode_layout_target_summary.csv \
  --family kv_byte_target
```

Drill down to individual workloads or the three independent process launches:

```bash
python3 benchmarks/cpu/attention_bmm/inspect_decode_layout_results.py \
  benchmark_results/decode-layout-targeted/7130/decode_layout_target_summary.csv \
  --family kv_byte_target --payload 56 --batch 4 --view workloads

python3 benchmarks/cpu/attention_bmm/inspect_decode_layout_results.py \
  benchmark_results/decode-layout-targeted/7130/decode_layout_target_summary.csv \
  --family kv_byte_target --payload 56 --batch 4 --view launches
```

Filters include `--payload`, `--batch`, `--kv-heads`, and
`--decision win|tie|baseline`. Add `--csv` to emit the filtered view as CSV.

### Grouped-Query GQA Decode

The grouped-query experiment asks whether one K/V block can be consumed more
efficiently by jointly processing all query heads mapped to its KV head. It
compares the existing HM/HF and BM/BF paths with grouped versions of each, while
holding the attention result and randomized paired protocol constant.

Run a small server smoke test first:

```bash
PACE_GQA_SEQ_LENS=512 \
PACE_GQA_KV_HEADS=32,8 \
PACE_GQA_BATCH_SIZES=1 \
PACE_DECODE_WARMUPS=2 \
PACE_DECODE_REPEATS=4 \
sbatch benchmarks/cpu/attention_bmm/slurm_decode_gqa_grouped.sbatch
```

After the smoke job builds successfully and all candidates pass correctness,
run the full three-launch sweep:

```bash
sbatch benchmarks/cpu/attention_bmm/slurm_decode_gqa_grouped.sbatch
```

The default matrix holds query heads at `32`, varies KV heads
`32,16,8,4,2,1` (GQA ratios `1,2,4,8,16,32`), uses batch sizes `1,2,4`, and
sequence lengths `512,2048,8192,32768,65536`. Holding sequence geometry fixed
while changing KV-head count isolates query sharing; the resulting logical K+V
payload spans `0.125 MiB` through `2048 MiB`, rather than assuming the earlier
40-72 MiB region. Results and the generated report are written under
`benchmark_results/decode-gqa-grouped/<job_id>/`.
