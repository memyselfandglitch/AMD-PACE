# SlabPool Layout and Block-Size Benchmark

This directory contains the first CPU-focused experiment for characterizing
SlabPool KV-cache layout and block-size choices.

The benchmark calls `torch.classes.pace.SlabPool` directly so the first study
isolates cache-update and attention behavior from tokenizer, model, scheduler,
and sampling overhead.

## What to Read First

Read these in order while working through the benchmark:

1. `docs/SlabAttention.md`
   Start with "Pool Tensor", "Pool Navigation", "Block Size Auto-Tuning", and
   "Attention Dispatch". This explains the two layouts and why block size is
   tied to cache residency.

2. `csrc/ops/attention/slab/slab_pool.h`
   This gives the C++ object model: pool tensor, sequence state, block metadata,
   layout strides, and the public methods used by the benchmark.

3. `csrc/ops/attention/slab/slab_pool_avx512.cpp`
   Focus on the constructor, `SLAB_LAYOUT`, `autotune_block_size`, sequence
   lifecycle, and `cache_update`.

4. `csrc/ops/attention/slab/slab_attention_avx512.cpp`
   Read the dispatch logic after the above pieces are clear. This is where
   decode, multi-token decode, and prefill split into different kernel paths.

5. `tests/attention/test_slab_attention.py`
   Use this as the correctness map. It shows how the raw SlabPool API is called
   for prefill, decode, GQA, MHA, and longer sequence cases.

## Quick Run

PACE must be built or installed so `pace/lib/libpace_cpp.so` exists.

```bash
python benchmarks/slab/bench_slab_layout_blocksize.py --quick
```

The default output is:

```text
benchmarks/slab/results/slab_layout_blocksize.csv
```

## Focused Examples

Single decode comparison:

```bash
python benchmarks/slab/bench_slab_layout_blocksize.py \
  --phases decode \
  --layouts head_major,block_major \
  --block-sizes 16,32,64,128 \
  --seq-lens 2048 \
  --batch-sizes 16 \
  --shapes llama_gqa:32:8:128
```

Prefill-only sweep:

```bash
python benchmarks/slab/bench_slab_layout_blocksize.py \
  --phases prefill \
  --seq-lens 128,512,2048 \
  --batch-sizes 1,4 \
  --block-sizes 32,64,128
```

Premature-exit decode workload:

```bash
python benchmarks/slab/bench_slab_layout_blocksize.py \
  --phases decode \
  --batch-sizes 16 \
  --seq-lens 2048 \
  --exit-fractions 0.0,0.25,0.5,0.75
```

## Hardware Counter Pass

After the latency sweep works, collect hardware counters for a few selected
cases with `perf` on Linux. This is the next step after identifying interesting
layout/block-size wins and losses from the sweep.

```bash
PHASE=mtd LAYOUT=head_major BLOCK_SIZE=16 BATCH_SIZE=16 SEQ_LEN=8192 \
SHAPE=slm_gqa:8:4:64 EXIT_FRACTION=0.5 \
sbatch benchmarks/slab/slurm_slab_perf.sbatch
```

The Slurm script writes:

- one benchmark CSV with latency and tokens/sec
- one raw `perf stat -x,` CSV
- one parsed CSV with counters and derived metrics

The parsed CSV includes `cycles`, `instructions`, `cache-references`,
`cache-misses`, `LLC-loads`, `LLC-load-misses`, `ipc`, cache miss rates,
cycles/token, and instructions/token. If the server does not expose LLC events,
the script retries with generic cache counters and records unavailable events.

For the report, compare latency trends against cache miss trends. The strongest
result would be a layout/block-size choice that improves latency and also
reduces cache misses, cycles/token, or instructions/token for a specific phase
and model shape.

Good first cases:

1. Strong head-major win: `PHASE=mtd LAYOUT=head_major BLOCK_SIZE=16 BATCH_SIZE=16 SEQ_LEN=8192 SHAPE=slm_gqa:8:4:64 EXIT_FRACTION=0.5`
2. Matching block-major case: same parameters with `LAYOUT=block_major`
3. Larger-block block-major case: same parameters with `LAYOUT=block_major BLOCK_SIZE=256`
4. Prefill case: `PHASE=prefill LAYOUT=head_major BLOCK_SIZE=64 BATCH_SIZE=1 SEQ_LEN=8192 SHAPE=slm_gqa:8:4:64`

## Analyze Results

After a sweep finishes, summarize the raw CSV into winner tables:

```bash
python benchmarks/slab/analyze_slab_results.py \
  benchmarks/slab/results/slab_sweep_<jobid>.csv \
  --out-dir benchmarks/slab/results/analysis_<jobid>
```

Start with `median_ms` rather than `mean_ms`; it is less sensitive to occasional
system noise. The generated `summary.md` counts which layout/block-size choice
won for each fixed workload and compares empirical best block size against the
PACE L2-based heuristic.
