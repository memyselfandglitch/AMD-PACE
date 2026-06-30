# SlabPool Block-Size Autotuning Summary

- CSV rows: `180`
- Workloads analyzed: `36`
- Layout held fixed: `head_major`
- Winner metric: `median_ms`
- Assumed L2 bytes when missing from CSV: `1048576`
- Baseline: current PACE-style L2 heuristic block size for each shape

## Main Result

- Best empirical block size improves over the heuristic by >=5% in `10/36` workloads (27.8%).
- Best empirical block size improves over the heuristic by >=10% in `5/36` workloads (13.9%).
- >=5% wins with p95 not worse: `8/36` workloads (22.2%).
- Median speedup over heuristic: `1.02x`.
- Max speedup over heuristic: `2.05x`.
- Median p95 speedup: `1.01x`.
- Larger block size is monotonically better in only `5/36` workloads (13.9%).

## Best Block-Size Distribution

| block_size | workloads |
| --- | --- |
| 64 | 15 |
| 128 | 14 |
| 256 | 5 |
| 16 | 1 |
| 32 | 1 |

## Median Speedup By Phase

| phase | median_speedup |
| --- | --- |
| prefill | 1.02x |

## Median Speedup By Sequence Length

| seq_len | median_speedup |
| --- | --- |
| 128 | 1.06x |
| 512 | 1.03x |
| 2048 | 1.01x |
| 8192 | 1.01x |

## Median Speedup By Shape

| shape | median_speedup |
| --- | --- |
| llama_gqa | 1.02x |
| mha | 1.01x |
| slm_gqa | 1.04x |

## Learned Rule

Rule features: `(phase, shape, sequence bucket)`, where buckets are `short <= 512`, `medium <= 2048`, and `long > 2048`.

| phase | shape | seq_bucket | chosen_block_size | training_workloads | median_regret_vs_oracle |
| --- | --- | --- | --- | --- | --- |
| prefill | llama_gqa | long | 128 | 3 | 1.000x |
| prefill | llama_gqa | medium | 128 | 3 | 1.000x |
| prefill | llama_gqa | short | 128 | 6 | 1.005x |
| prefill | mha | long | 256 | 3 | 1.000x |
| prefill | mha | medium | 128 | 3 | 1.000x |
| prefill | mha | short | 64 | 6 | 1.000x |
| prefill | slm_gqa | long | 256 | 3 | 1.000x |
| prefill | slm_gqa | medium | 128 | 3 | 1.000x |
| prefill | slm_gqa | short | 64 | 6 | 1.000x |

## Leave-One-Sequence-Length-Out Validation

- Validation rows: `36`
- Median rule speedup over heuristic: `1.01x`
- Rule within 5% of oracle best block: `32/36` workloads (88.9%)
- Median regret vs oracle: `1.001x`

## Files

- `decisions.csv`: per-workload baseline block, best block, speedup, and p95 comparison.
- `blocksize_rule.csv`: learned lookup rule from `(phase, shape, seq_bucket)` to block size.
- `validation.csv`: leave-one-sequence-length-out rule validation.
