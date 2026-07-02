# Selected Perf Comparison 6981

- Paired heuristic-vs-best cases: `11`
- Single-run controls: `1`
- LLC events unavailable in all rows: `True`

## Important Caveat

`perf stat` wrapped the full Python benchmark process, including import, tensor allocation, SlabPool construction, warmups, and benchmark repeats. Therefore task-clock/cycles/instructions/cache counters are useful only as coarse process-level signals. They are not clean kernel-only counters yet.

## Pairwise Results

| case | phase | shape | batch | seq | exit | blocks | measured speedup | p95 speedup | cycles reduction | instruction reduction | cache-miss reduction | miss rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| decode_strong_medium_mha | decode | mha | 16 | 2048 | 0.25 | 128->16 | 2.46x | 3.32x | 0.6% | 0.1% | 0.2% | 0.134->0.134 |
| top_stable_win | decode | llama_gqa | 4 | 2048 | 0.25 | 64->16 | 2.31x | 2.69x | 0.9% | 0.2% | 0.2% | 0.131->0.129 |
| top_stable_win | decode | mha | 16 | 2048 | 0.0 | 128->32 | 2.13x | 1.89x | 3.0% | 0.3% | 0.5% | 0.132->0.133 |
| mtd_strong_medium_mha | mtd | mha | 16 | 2048 | 0.75 | 128->32 | 2.00x | 2.92x | 0.8% | -0.3% | -1.4% | 0.128->0.128 |
| top_stable_win | decode | mha | 16 | 512 | 0.0 | 128->16 | 1.92x | 1.26x | 1.9% | 0.2% | 0.2% | 0.134->0.135 |
| decode_strong_short_llama | decode | llama_gqa | 16 | 512 | 0.0 | 64->16 | 1.82x | 1.71x | -1.1% | -0.5% | 0.3% | 0.133->0.133 |
| top_stable_win | decode | mha | 16 | 512 | 0.25 | 128->16 | 1.80x | 1.90x | -2.4% | -0.6% | -2.0% | 0.134->0.136 |
| decode_strong_long_llama | decode | llama_gqa | 1 | 8192 | 0.25 | 64->256 | 1.04x | 1.14x | -3.4% | -1.1% | -1.4% | 0.129->0.128 |
| mtd_strong_long_slm | mtd | slm_gqa | 16 | 8192 | 0.5 | 256->64 | 0.94x | 0.93x | 1.5% | 0.4% | 0.6% | 0.112->0.114 |
| decode_heuristic_control | decode | llama_gqa | 16 | 8192 | 0.0 | 64->32 | 0.85x | 0.82x | 21.0% | 2.0% | 5.2% | 0.101->0.112 |
| mtd_strong_short_llama | mtd | llama_gqa | 1 | 128 | 0.5 | 64->16 | 0.79x | 0.81x | -2.1% | -0.5% | -2.3% | 0.134->0.134 |

## Aggregate

- Median measured speedup: `1.82x`
- Median p95 speedup: `1.71x`
- Best block had fewer process-level cycles in `7/11` cases.
- Best block had fewer process-level instructions in `6/11` cases.
- Best block had fewer process-level generic cache misses in `7/11` cases.
- Best block had lower generic cache miss rate in `3/11` cases.
