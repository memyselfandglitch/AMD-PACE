# SlabPool Block-Size Study

This directory contains a focused CPU-first benchmark for studying SlabPool
block size in AMD PACE.

The first version intentionally holds layout fixed, usually `head_major`, so
the main variable is block size. This makes the experiment easier to defend:
we compare the current PACE-style L2 heuristic against an empirical best block
size for each workload.

## Latency Sweep

Submit the default CPU sweep on the `jobmn01` partition:

```bash
sbatch benchmarks/slab/slurm_slab_blocksize_sweep.sbatch
```

Useful overrides:

```bash
BLOCK_SIZES=16,32,64,128,256 \
SEQ_LENS=128,512,2048,8192 \
BATCH_SIZES=1,4,16 \
SHAPES=default \
SLAB_LAYOUT=head_major \
sbatch benchmarks/slab/slurm_slab_blocksize_sweep.sbatch
```

Outputs:

- `benchmarks/slab/results/blocksize/blocksize_sweep_<job>.csv`
- `benchmarks/slab/results/blocksize/analysis_<job>/summary.md`
- `benchmarks/slab/results/blocksize/analysis_<job>/decisions.csv`
- `benchmarks/slab/results/blocksize/analysis_<job>/blocksize_rule.csv`
- `benchmarks/slab/results/blocksize/analysis_<job>/validation.csv`

## Hardware-Counter Sweep

Run a focused perf sweep over selected sequence lengths and block sizes:

```bash
sbatch benchmarks/slab/slurm_slab_blocksize_perf.sbatch
```

By default this runs:

- phases: `prefill,decode,mtd`
- shapes: `slm_gqa,llama_gqa,mha`
- sequence lengths: `128,8192`
- batch size: `16`
- exit fractions: `0.0,0.5` for decode/MTD
- block sizes: `16,32,64,128,256`

The perf script writes one combined CSV:

```text
benchmarks/slab/results/blocksize_perf/blocksize_perf_<job>.csv
```

It requests generic cache counters and LLC counters. If the machine does not
expose `LLC-loads` or `LLC-load-misses`, the parser records those events in the
`unavailable_events` column instead of silently treating them as zero.

## Selected Hardware-Counter Cases

After a latency sweep has produced `analysis_<job>/decisions.csv`, select a
small number of high-value cases and run only baseline-vs-best comparisons:

```bash
python3 benchmarks/slab/select_blocksize_perf_cases.py \
  benchmarks/slab/results/blocksize/analysis_6979/decisions.csv \
  --out benchmarks/slab/results/blocksize/analysis_6979/selected_perf_cases_6979.csv

sbatch benchmarks/slab/slurm_slab_blocksize_perf_selected.sbatch
```

The selected perf script runs each case twice: once with the current heuristic
block size and once with the empirical best block size. This is the right next
step when the question is "why did this block size win?" rather than "which
block size won?"

## What The Analyzer Tests

The analyzer compares:

- current PACE-style L2 heuristic block size
- empirical best block size for each workload
- a learned lookup rule using `(phase, shape, sequence bucket)`
- leave-one-sequence-length-out validation

Sequence buckets are:

- `short`: `seq_len <= 512`
- `medium`: `512 < seq_len <= 2048`
- `long`: `seq_len > 2048`

The key claim we want to test is not "larger blocks are always better." The
claim is:

```text
The best SlabPool block size is workload-dependent, and a lightweight
workload-aware rule can beat the current static cache heuristic.
```
