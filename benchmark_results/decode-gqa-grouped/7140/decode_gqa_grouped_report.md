# Grouped-Query GQA Decode Summary

## Experiment

- Baseline: head-major/head-first, one query head at a time.
- Grouped mode reuses each K vector and V vector across query heads sharing one KV head.
- Four candidates isolate BM/BF alone, grouping alone, and their combination.
- Strict win: >=5% paired median effect, >=80% pair wins, 95% CI excluding one, and non-regressing p95.
- Repeatability requires >=2 independent launch wins.

## Main Result

- Workloads: `2`
- Existing BM/BF versus baseline: median `0.998x`, repeatable wins `0/2`
- Grouping under HM/HF versus baseline: median `0.444x`, repeatable wins `0/2`
- Grouping added under BM/BF: median `0.443x`, repeatable wins `0/2`
- Grouped BM/BF versus baseline: median `0.441x`, repeatable wins `0/2`

### Recommendations

- `head_major_head_first`: `2` workloads
- `block_major_block_first`: `0` workloads
- `head_major_head_first_grouped`: `0` workloads
- `block_major_block_first_grouped`: `0` workloads

## By GQA Ratio

| ratio | workloads | existing BM/BF | grouped HM/HF | grouped BM/BF | grouped BM/BF wins |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0.999x | 0.299x | 0.299x | 0/1 |
| 4 | 1 | 0.998x | 0.589x | 0.584x | 0/1 |

## By Sequence Length

| sequence | workloads | existing BM/BF | grouped HM/HF | grouped BM/BF | grouped BM/BF wins |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 2 | 0.998x | 0.444x | 0.441x | 0/2 |

## By Logical K+V Payload

| payload MiB | workloads | existing BM/BF | grouped HM/HF | grouped BM/BF | grouped BM/BF wins |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0.998x | 0.589x | 0.584x | 0/1 |
| 4 | 1 | 0.999x | 0.299x | 0.299x | 0/1 |

## Guardrail

This is a single-threaded synthetic fused-decode mechanism benchmark. It does not yet modify production SlabPool, use fragmented block tables, or test OpenMP scaling. Logical K+V payload is not measured DRAM traffic.
