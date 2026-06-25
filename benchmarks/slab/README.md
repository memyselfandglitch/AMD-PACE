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

## Cache Counter Pass

After the latency sweep works, collect hardware counters with `perf` on Linux:

```bash
perf stat -e LLC-loads,LLC-load-misses,cache-references,cache-misses \
  python benchmarks/slab/bench_slab_layout_blocksize.py --quick
```

For the report, compare latency trends against LLC miss trends. The strongest
result would be a layout/block-size choice that improves both latency and cache
miss rate for a specific phase and model shape.

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
