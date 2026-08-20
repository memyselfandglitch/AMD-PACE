# Grouped-Query GQA Decode Summary

## Experiment

- Baseline: head-major/head-first, one query head at a time.
- Grouped mode reuses each K vector and V vector across query heads sharing one KV head.
- Four candidates isolate BM/BF alone, grouping alone, and their combination.
- Strict win: >=5% paired median effect, >=80% pair wins, 95% CI excluding one, and non-regressing p95.
- Repeatability requires >=2 independent launch wins.

## Main Result

- Workloads: `90`
- Existing BM/BF versus baseline: median `0.998x`, repeatable wins `1/90`
- Grouping under HM/HF versus baseline: median `0.659x`, repeatable wins `0/90`
- Grouping added under BM/BF: median `0.652x`, repeatable wins `0/90`
- Grouped BM/BF versus baseline: median `0.648x`, repeatable wins `0/90`

### Recommendations

- `head_major_head_first`: `89` workloads
- `block_major_block_first`: `1` workloads
- `head_major_head_first_grouped`: `0` workloads
- `block_major_block_first_grouped`: `0` workloads

## By GQA Ratio

| ratio | workloads | existing BM/BF | grouped HM/HF | grouped BM/BF | grouped BM/BF wins |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 15 | 0.994x | 0.397x | 0.397x | 0/15 |
| 2 | 15 | 0.964x | 0.453x | 0.454x | 0/15 |
| 4 | 15 | 0.976x | 0.657x | 0.655x | 0/15 |
| 8 | 15 | 0.999x | 0.668x | 0.660x | 0/15 |
| 16 | 15 | 1.000x | 0.693x | 0.691x | 0/15 |
| 32 | 15 | 0.999x | 0.698x | 0.684x | 0/15 |

## By Sequence Length

| sequence | workloads | existing BM/BF | grouped HM/HF | grouped BM/BF | grouped BM/BF wins |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 18 | 0.999x | 0.613x | 0.613x | 0/18 |
| 2048 | 18 | 0.999x | 0.634x | 0.633x | 0/18 |
| 8192 | 18 | 0.993x | 0.660x | 0.646x | 0/18 |
| 32768 | 18 | 0.997x | 0.686x | 0.676x | 0/18 |
| 65536 | 18 | 0.994x | 0.680x | 0.677x | 0/18 |

## By Logical K+V Payload

| payload MiB | workloads | existing BM/BF | grouped HM/HF | grouped BM/BF | grouped BM/BF wins |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.125 | 1 | 0.996x | 0.697x | 0.686x | 0/1 |
| 0.25 | 2 | 0.996x | 0.699x | 0.684x | 0/2 |
| 0.5 | 4 | 0.999x | 0.690x | 0.680x | 0/4 |
| 1 | 5 | 0.999x | 0.689x | 0.680x | 0/5 |
| 2 | 7 | 0.998x | 0.638x | 0.641x | 0/7 |
| 4 | 8 | 1.001x | 0.614x | 0.605x | 0/8 |
| 8 | 9 | 1.001x | 0.636x | 0.636x | 0/9 |
| 16 | 10 | 0.998x | 0.651x | 0.646x | 0/10 |
| 32 | 10 | 0.999x | 0.694x | 0.678x | 0/10 |
| 64 | 10 | 0.997x | 0.681x | 0.672x | 0/10 |
| 128 | 8 | 0.979x | 0.668x | 0.660x | 0/8 |
| 256 | 7 | 0.970x | 0.469x | 0.467x | 0/7 |
| 512 | 5 | 0.976x | 0.469x | 0.468x | 0/5 |
| 1024 | 3 | 0.993x | 0.425x | 0.425x | 0/3 |
| 2048 | 1 | 0.994x | 0.424x | 0.426x | 0/1 |

## Guardrail

This is a single-threaded synthetic fused-decode mechanism benchmark. It does not yet modify production SlabPool, use fragmented block tables, or test OpenMP scaling. Logical K+V payload is not measured DRAM traffic.
