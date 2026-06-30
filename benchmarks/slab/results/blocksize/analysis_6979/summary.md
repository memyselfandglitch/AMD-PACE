# SlabPool Block-Size Autotuning Summary

- CSV rows: `1440`
- Workloads analyzed: `288`
- Layout held fixed: `head_major`
- Winner metric: `median_ms`
- Assumed L2 bytes when missing from CSV: `1048576`
- Baseline: current PACE-style L2 heuristic block size for each shape

## Main Result

- Best empirical block size improves over the heuristic by >=5% in `83/288` workloads (28.8%).
- Best empirical block size improves over the heuristic by >=10% in `41/288` workloads (14.2%).
- >=5% wins with p95 not worse: `80/288` workloads (27.8%).
- Median speedup over heuristic: `1.02x`.
- Max speedup over heuristic: `2.10x`.
- Median p95 speedup: `1.02x`.
- Larger block size is monotonically better in only `38/288` workloads (13.2%).

## Best Block-Size Distribution

| block_size | workloads |
| --- | --- |
| 256 | 84 |
| 64 | 84 |
| 128 | 57 |
| 32 | 43 |
| 16 | 20 |

## Median Speedup By Phase

| phase | median_speedup |
| --- | --- |
| decode | 1.01x |
| mtd | 1.03x |

## Median Speedup By Sequence Length

| seq_len | median_speedup |
| --- | --- |
| 128 | 1.02x |
| 512 | 1.02x |
| 2048 | 1.02x |
| 8192 | 1.01x |

## Median Speedup By Shape

| shape | median_speedup |
| --- | --- |
| llama_gqa | 1.02x |
| mha | 1.02x |
| slm_gqa | 1.01x |

## Learned Rule

Rule features: `(phase, shape, sequence bucket)`, where buckets are `short <= 512`, `medium <= 2048`, and `long > 2048`.

| phase | shape | seq_bucket | chosen_block_size | training_workloads | median_regret_vs_oracle |
| --- | --- | --- | --- | --- | --- |
| decode | llama_gqa | long | 256 | 12 | 1.006x |
| decode | llama_gqa | medium | 128 | 12 | 1.003x |
| decode | llama_gqa | short | 128 | 24 | 1.001x |
| decode | mha | long | 256 | 12 | 1.000x |
| decode | mha | medium | 256 | 12 | 1.000x |
| decode | mha | short | 128 | 24 | 1.002x |
| decode | slm_gqa | long | 256 | 12 | 1.000x |
| decode | slm_gqa | medium | 256 | 12 | 1.000x |
| decode | slm_gqa | short | 128 | 24 | 1.008x |
| mtd | llama_gqa | long | 64 | 12 | 1.007x |
| mtd | llama_gqa | medium | 256 | 12 | 1.000x |
| mtd | llama_gqa | short | 32 | 24 | 1.005x |
| mtd | mha | long | 64 | 12 | 1.001x |
| mtd | mha | medium | 64 | 12 | 1.000x |
| mtd | mha | short | 32 | 24 | 1.001x |
| mtd | slm_gqa | long | 256 | 12 | 1.005x |
| mtd | slm_gqa | medium | 64 | 12 | 1.000x |
| mtd | slm_gqa | short | 64 | 24 | 1.000x |

## Leave-One-Sequence-Length-Out Validation

- Validation rows: `288`
- Median rule speedup over heuristic: `1.00x`
- Rule within 5% of oracle best block: `240/288` workloads (83.3%)
- Median regret vs oracle: `1.008x`

## Files

- `decisions.csv`: per-workload baseline block, best block, speedup, and p95 comparison.
- `blocksize_rule.csv`: learned lookup rule from `(phase, shape, seq_bucket)` to block size.
- `validation.csv`: leave-one-sequence-length-out rule validation.
