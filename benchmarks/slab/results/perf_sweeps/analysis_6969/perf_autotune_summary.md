# SlabPool Perf Autotuning Summary

- CSV: `benchmarks/slab/results/perf_sweeps/perf_sweep_6969.csv`
- Rows: `240`
- Workloads compared: `30`
- Baseline: `head_major` + PACE L2 heuristic block size when present
- Non-default best choices: `23/30`
- Best choice beat baseline by >= 1.05x: `16/30`

## Best Choices

| layout | block_size | wins |
| --- | --- | --- |
| head_major | 256 | 8 |
| head_major | 128 | 7 |
| head_major | 64 | 4 |
| block_major | 256 | 4 |
| block_major | 128 | 2 |
| block_major | 16 | 2 |
| head_major | 16 | 2 |
| block_major | 64 | 1 |

## Best Choices By Phase

| phase | layout | block_size | wins |
| --- | --- | --- | --- |
| decode | head_major | 256 | 4 |
| mtd | head_major | 128 | 4 |
| decode | head_major | 64 | 3 |
| mtd | head_major | 256 | 3 |
| mtd | block_major | 256 | 3 |
| decode | block_major | 16 | 2 |
| decode | head_major | 128 | 2 |
| mtd | head_major | 16 | 2 |
| prefill | block_major | 128 | 1 |
| prefill | block_major | 64 | 1 |
| prefill | head_major | 64 | 1 |
| prefill | head_major | 256 | 1 |
| prefill | head_major | 128 | 1 |
| prefill | block_major | 256 | 1 |
| decode | block_major | 128 | 1 |

## Best Choices By Shape

| shape | layout | block_size | wins |
| --- | --- | --- | --- |
| slm_gqa | head_major | 256 | 4 |
| mha | block_major | 256 | 3 |
| llama_gqa | head_major | 256 | 3 |
| mha | head_major | 128 | 3 |
| mha | head_major | 64 | 2 |
| llama_gqa | head_major | 128 | 2 |
| slm_gqa | head_major | 64 | 2 |
| slm_gqa | head_major | 128 | 2 |
| llama_gqa | head_major | 16 | 2 |
| slm_gqa | block_major | 128 | 1 |
| llama_gqa | block_major | 64 | 1 |
| slm_gqa | block_major | 16 | 1 |
| mha | block_major | 16 | 1 |
| llama_gqa | block_major | 128 | 1 |
| mha | head_major | 256 | 1 |
| llama_gqa | block_major | 256 | 1 |

## Largest Baseline Improvements

| phase | shape | batch | seq | exit | best | baseline | speedup | miss_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prefill | mha | 16 | 128 | 0.0 | head_major:64 | head_major:128 | 2.38x | 0.9% |
| prefill | llama_gqa | 16 | 128 | 0.0 | block_major:64 | head_major:64 | 1.46x | 0.9% |
| prefill | slm_gqa | 16 | 128 | 0.0 | block_major:128 | head_major:256 | 1.44x | 0.2% |
| mtd | mha | 16 | 128 | 0.0 | block_major:256 | head_major:128 | 1.28x | -0.7% |
| decode | llama_gqa | 16 | 8192 | 0.5 | block_major:128 | head_major:64 | 1.23x | -10.5% |
| decode | llama_gqa | 16 | 128 | 0.5 | head_major:256 | head_major:64 | 1.22x | 0.3% |
| mtd | mha | 16 | 8192 | 0.0 | block_major:256 | head_major:128 | 1.14x | -11.9% |
| mtd | slm_gqa | 16 | 128 | 0.0 | head_major:128 | head_major:256 | 1.12x | 0.4% |
| decode | slm_gqa | 16 | 8192 | 0.0 | head_major:64 | head_major:256 | 1.11x | -1.3% |
| mtd | llama_gqa | 16 | 8192 | 0.0 | head_major:16 | head_major:64 | 1.11x | -10.8% |
| mtd | slm_gqa | 16 | 8192 | 0.5 | head_major:128 | head_major:256 | 1.11x | 0.6% |
| decode | llama_gqa | 16 | 128 | 0.0 | head_major:128 | head_major:64 | 1.11x | 0.0% |
| mtd | llama_gqa | 16 | 128 | 0.5 | head_major:256 | head_major:64 | 1.09x | -1.0% |
| decode | llama_gqa | 16 | 8192 | 0.0 | head_major:256 | head_major:64 | 1.05x | -9.0% |
| decode | slm_gqa | 16 | 128 | 0.5 | head_major:64 | head_major:256 | 1.05x | -0.3% |
| mtd | llama_gqa | 16 | 128 | 0.0 | block_major:256 | head_major:64 | 1.05x | -2.2% |
| decode | mha | 16 | 128 | 0.5 | block_major:16 | head_major:128 | 1.05x | 0.7% |
| decode | mha | 16 | 128 | 0.0 | head_major:64 | head_major:128 | 1.04x | -0.0% |
| decode | slm_gqa | 16 | 128 | 0.0 | block_major:16 | head_major:256 | 1.04x | 0.7% |
| mtd | llama_gqa | 16 | 8192 | 0.5 | head_major:16 | head_major:64 | 1.03x | -4.9% |

