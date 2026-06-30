# Decision-Level Report for Job 6979

This report mines `decisions.csv`: one row per workload, comparing the current PACE-style heuristic block size against the empirically best block size under fixed `head_major` layout.

## Overall

- Workloads: `288`
- >=5% median wins over heuristic: `83/288` (28.8%)
- >=10% median wins over heuristic: `41/288` (14.2%)
- >=5% wins with p95 not worse: `80/288` (27.8%)
- Median speedup: `1.017x`
- Max speedup: `2.103x`

## By Phase

| group | n | >=5% wins | >=10% wins | median speedup | top best blocks |
| --- | --- | --- | --- | --- | --- |
| decode | 144 | 21.5% | 13.2% | 1.009x | [(256, 60), (128, 45), (64, 21)] |
| mtd | 144 | 36.1% | 15.3% | 1.030x | [(64, 63), (32, 39), (256, 24)] |

## By Phase And Shape

| group | n | >=5% wins | >=10% wins | median speedup | top best blocks |
| --- | --- | --- | --- | --- | --- |
| decode / llama_gqa | 48 | 41.7% | 25.0% | 1.039x | [(128, 21), (256, 12), (16, 8)] |
| decode / mha | 48 | 18.8% | 14.6% | 1.007x | [(256, 21), (128, 16), (64, 6)] |
| decode / slm_gqa | 48 | 4.2% | 0.0% | 1.000x | [(256, 27), (64, 9), (128, 8)] |
| mtd / llama_gqa | 48 | 12.5% | 0.0% | 1.010x | [(64, 15), (256, 14), (32, 10)] |
| mtd / mha | 48 | 56.2% | 27.1% | 1.056x | [(64, 21), (32, 17), (128, 4)] |
| mtd / slm_gqa | 48 | 39.6% | 18.8% | 1.040x | [(64, 27), (32, 12), (256, 7)] |

## By Phase And Sequence Bucket

| group | n | >=5% wins | >=10% wins | median speedup | top best blocks |
| --- | --- | --- | --- | --- | --- |
| decode / long | 36 | 33.3% | 16.7% | 1.010x | [(256, 23), (128, 7), (64, 3)] |
| decode / medium | 36 | 22.2% | 22.2% | 1.005x | [(256, 18), (128, 9), (16, 4)] |
| decode / short | 72 | 15.3% | 6.9% | 1.010x | [(128, 29), (256, 19), (64, 14)] |
| mtd / long | 36 | 22.2% | 8.3% | 1.006x | [(64, 13), (256, 11), (128, 7)] |
| mtd / medium | 36 | 44.4% | 13.9% | 1.045x | [(64, 17), (256, 9), (32, 6)] |
| mtd / short | 72 | 38.9% | 19.4% | 1.036x | [(64, 33), (32, 29), (16, 4)] |

## By Shape

| group | n | >=5% wins | >=10% wins | median speedup | top best blocks |
| --- | --- | --- | --- | --- | --- |
| llama_gqa | 96 | 27.1% | 12.5% | 1.020x | [(128, 27), (256, 26), (64, 21)] |
| mha | 96 | 37.5% | 20.8% | 1.020x | [(64, 27), (256, 24), (128, 20)] |
| slm_gqa | 96 | 21.9% | 9.4% | 1.012x | [(64, 36), (256, 34), (32, 14)] |

## Top Stable Wins

| phase | shape | batch | seq | exit | heur | best | speed | p95 | heur_ms | best_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode | mha | 16 | 2048 | 0.25 | 128 | 16 | 2.10x | 2.29x | 0.473733 | 0.22522 |
| decode | llama_gqa | 4 | 2048 | 0.25 | 64 | 16 | 1.99x | 2.46x | 0.388813 | 0.195246 |
| decode | mha | 16 | 512 | 0.0 | 128 | 16 | 1.84x | 1.80x | 0.221245 | 0.120332 |
| decode | mha | 16 | 512 | 0.25 | 128 | 16 | 1.83x | 1.70x | 0.170688 | 0.0934465 |
| decode | llama_gqa | 16 | 512 | 0.0 | 64 | 16 | 1.75x | 1.57x | 0.404758 | 0.231796 |
| decode | mha | 16 | 2048 | 0.0 | 128 | 32 | 1.61x | 1.66x | 0.734557 | 0.457092 |
| mtd | mha | 16 | 2048 | 0.75 | 128 | 32 | 1.54x | 2.00x | 0.271691 | 0.17599 |
| mtd | mha | 4 | 2048 | 0.0 | 128 | 32 | 1.54x | 1.90x | 0.258846 | 0.168265 |
| decode | llama_gqa | 16 | 512 | 0.25 | 64 | 16 | 1.47x | 1.53x | 0.293529 | 0.199848 |
| decode | llama_gqa | 1 | 8192 | 0.25 | 64 | 256 | 1.45x | 1.45x | 0.295252 | 0.203358 |
| decode | llama_gqa | 16 | 8192 | 0.25 | 64 | 16 | 1.43x | 1.97x | 2.22627 | 1.55824 |
| decode | llama_gqa | 4 | 8192 | 0.5 | 64 | 128 | 1.42x | 1.44x | 0.308457 | 0.217129 |
| mtd | slm_gqa | 16 | 8192 | 0.5 | 256 | 64 | 1.36x | 1.54x | 2.35091 | 1.7251 |
| mtd | mha | 16 | 2048 | 0.5 | 128 | 16 | 1.34x | 1.62x | 0.690924 | 0.513912 |
| decode | llama_gqa | 16 | 2048 | 0.25 | 64 | 16 | 1.34x | 1.38x | 1.05842 | 0.792325 |
| mtd | slm_gqa | 1 | 128 | 0.25 | 256 | 32 | 1.31x | 1.28x | 0.041783 | 0.031808 |
| mtd | mha | 4 | 2048 | 0.25 | 128 | 32 | 1.25x | 1.87x | 0.179717 | 0.143898 |
| decode | llama_gqa | 1 | 2048 | 0.75 | 64 | 128 | 1.25x | 1.14x | 0.102445 | 0.082209 |
| decode | llama_gqa | 1 | 2048 | 0.5 | 64 | 256 | 1.24x | 1.11x | 0.101183 | 0.0817385 |
| decode | mha | 16 | 8192 | 0.0 | 128 | 16 | 1.22x | 1.03x | 2.31294 | 1.89081 |

## Heuristic Already Optimal

- Exact heuristic-best workloads: `75/288` (26.0%).
- Within 5% of oracle: `205/288` (71.2%).

## Selected Perf Cases

Wrote `benchmarks/slab/results/blocksize/analysis_6979/selected_perf_cases_6979.csv` with `12` cases. For each case, run perf on both the heuristic block and the best block.

| label | phase | shape | batch | seq | exit | heur | best | speed | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_strong_short_llama | decode | llama_gqa | 16 | 512 | 0.0 | 64 | 16 | 1.75x | 1.57x |
| decode_strong_medium_mha | decode | mha | 16 | 2048 | 0.25 | 128 | 16 | 2.10x | 2.29x |
| decode_strong_long_llama | decode | llama_gqa | 1 | 8192 | 0.25 | 64 | 256 | 1.45x | 1.45x |
| mtd_strong_short_llama | mtd | llama_gqa | 1 | 128 | 0.5 | 64 | 16 | 1.07x | 1.10x |
| mtd_strong_medium_mha | mtd | mha | 16 | 2048 | 0.75 | 128 | 32 | 1.54x | 2.00x |
| mtd_strong_long_slm | mtd | slm_gqa | 16 | 8192 | 0.5 | 256 | 64 | 1.36x | 1.54x |
| top_stable_win | decode | llama_gqa | 4 | 2048 | 0.25 | 64 | 16 | 1.99x | 2.46x |
| top_stable_win | decode | mha | 16 | 512 | 0.0 | 128 | 16 | 1.84x | 1.80x |
| top_stable_win | decode | mha | 16 | 512 | 0.25 | 128 | 16 | 1.83x | 1.70x |
| top_stable_win | decode | mha | 16 | 2048 | 0.0 | 128 | 32 | 1.61x | 1.66x |
| decode_heuristic_control | decode | llama_gqa | 16 | 8192 | 0.0 | 64 | 32 | 1.01x | 0.99x |
| mtd_heuristic_control | mtd | llama_gqa | 16 | 8192 | 0.25 | 64 | 64 | 1.00x | 1.00x |
