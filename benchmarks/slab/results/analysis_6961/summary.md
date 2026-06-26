# SlabPool Sweep Summary

- CSV: `benchmarks/slab/results/slab_sweep_6961.csv`
- Rows: `3240`
- Workloads compared: `324`
- Metric: `median_ms`
- Assumed L2/core: `1024 KiB`
- L2 heuristic matched empirical best block size: `97/324`
- Best-vs-worst speedup across layout/block choices: avg `1.25x`, max `2.25x`

## Wins By Layout

| layout | wins |
| --- | --- |
| head_major | 221 |
| block_major | 103 |

## Wins By Block Size

| block_size | wins |
| --- | --- |
| 256 | 115 |
| 128 | 73 |
| 64 | 70 |
| 32 | 36 |
| 16 | 30 |

## Wins By Layout And Block Size

| layout | block_size | wins |
| --- | --- | --- |
| head_major | 256 | 85 |
| head_major | 64 | 55 |
| head_major | 128 | 43 |
| block_major | 128 | 30 |
| block_major | 256 | 30 |
| head_major | 32 | 21 |
| head_major | 16 | 17 |
| block_major | 64 | 15 |
| block_major | 32 | 15 |
| block_major | 16 | 13 |

## Wins By Phase

| phase | layout | block_size | wins |
| --- | --- | --- | --- |
| decode | head_major | 256 | 50 |
| mtd | head_major | 256 | 34 |
| mtd | head_major | 128 | 33 |
| mtd | head_major | 64 | 23 |
| prefill | head_major | 64 | 20 |
| decode | block_major | 256 | 19 |
| decode | block_major | 128 | 19 |
| mtd | head_major | 32 | 16 |
| decode | block_major | 16 | 13 |
| decode | head_major | 64 | 12 |
| decode | block_major | 32 | 11 |
| mtd | head_major | 16 | 11 |
| mtd | block_major | 128 | 9 |
| decode | block_major | 64 | 8 |
| mtd | block_major | 256 | 8 |
| decode | head_major | 128 | 6 |
| mtd | block_major | 64 | 6 |
| decode | head_major | 16 | 5 |
| prefill | head_major | 32 | 4 |
| prefill | head_major | 128 | 4 |
| mtd | block_major | 32 | 4 |
| prefill | block_major | 256 | 3 |
| prefill | block_major | 128 | 2 |
| prefill | block_major | 64 | 1 |
| prefill | head_major | 16 | 1 |
| prefill | head_major | 256 | 1 |
| decode | head_major | 32 | 1 |

## Wins By Shape

| shape | layout | block_size | wins |
| --- | --- | --- | --- |
| llama_gqa | head_major | 256 | 33 |
| mha | head_major | 256 | 27 |
| llama_gqa | head_major | 64 | 25 |
| slm_gqa | head_major | 256 | 25 |
| slm_gqa | head_major | 128 | 17 |
| mha | head_major | 64 | 16 |
| slm_gqa | head_major | 64 | 14 |
| mha | head_major | 128 | 14 |
| mha | block_major | 128 | 14 |
| slm_gqa | block_major | 256 | 13 |
| llama_gqa | head_major | 128 | 12 |
| slm_gqa | block_major | 128 | 12 |
| mha | block_major | 256 | 10 |
| slm_gqa | head_major | 32 | 7 |
| llama_gqa | head_major | 32 | 7 |
| mha | head_major | 32 | 7 |
| llama_gqa | block_major | 32 | 7 |
| llama_gqa | block_major | 256 | 7 |
| mha | block_major | 16 | 7 |
| slm_gqa | head_major | 16 | 6 |
| llama_gqa | block_major | 64 | 6 |
| llama_gqa | head_major | 16 | 6 |
| slm_gqa | block_major | 32 | 5 |
| mha | block_major | 64 | 5 |
| slm_gqa | block_major | 16 | 5 |
| mha | head_major | 16 | 5 |
| slm_gqa | block_major | 64 | 4 |
| llama_gqa | block_major | 128 | 4 |
| mha | block_major | 32 | 3 |
| llama_gqa | block_major | 16 | 1 |

## Top 15 Largest Differences

| phase | shape | batch | seq_len | exit | best | best_median_ms | worst | worst_median_ms | speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mtd | slm_gqa | 1 | 8192 | 0.0 | head_major:128 | 0.160745 | head_major:16 | 0.361799 | 2.25x |
| mtd | llama_gqa | 4 | 128 | 0.0 | block_major:64 | 0.279236 | head_major:32 | 0.584641 | 2.09x |
| prefill | slm_gqa | 16 | 128 | 0.0 | head_major:32 | 0.257043 | head_major:256 | 0.500308 | 1.95x |
| mtd | slm_gqa | 16 | 8192 | 0.5 | head_major:16 | 1.32237 | block_major:16 | 2.39657 | 1.81x |
| mtd | mha | 4 | 8192 | 0.5 | head_major:32 | 0.371854 | block_major:16 | 0.667826 | 1.80x |
| mtd | slm_gqa | 4 | 8192 | 0.5 | head_major:128 | 0.328114 | block_major:16 | 0.586071 | 1.79x |
| prefill | llama_gqa | 16 | 128 | 0.0 | head_major:64 | 1.35627 | head_major:256 | 2.41789 | 1.78x |
| prefill | llama_gqa | 1 | 128 | 0.0 | head_major:64 | 0.366835 | head_major:256 | 0.653029 | 1.78x |
| mtd | mha | 16 | 8192 | 0.75 | head_major:128 | 0.723183 | block_major:16 | 1.27833 | 1.77x |
| prefill | slm_gqa | 1 | 128 | 0.0 | head_major:32 | 0.148877 | head_major:16 | 0.26208 | 1.76x |
| mtd | mha | 4 | 8192 | 0.75 | head_major:256 | 0.174171 | block_major:16 | 0.303921 | 1.74x |
| mtd | mha | 4 | 8192 | 0.0 | head_major:128 | 0.721527 | block_major:16 | 1.23898 | 1.72x |
| prefill | mha | 1 | 128 | 0.0 | head_major:64 | 0.12756 | head_major:256 | 0.218268 | 1.71x |
| mtd | mha | 4 | 8192 | 0.25 | head_major:256 | 0.514424 | block_major:16 | 0.876966 | 1.70x |
| mtd | slm_gqa | 4 | 8192 | 0.0 | head_major:256 | 0.570948 | block_major:16 | 0.956442 | 1.68x |

