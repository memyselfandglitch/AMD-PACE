# Decode KV Layout and Traversal Summary

## Experiment

- Current baseline: head-major storage plus head-first traversal.
- Layout-only control: block-major storage plus head-first traversal.
- Traversal-only control: head-major storage plus block-first traversal.
- Co-designed candidate: block-major storage plus block-first traversal.
- Each candidate performs the same fused BF16 decode: QK dot products, blockwise online softmax, and weighted V accumulation.
- Candidate order uses randomized four-round Latin-square cycles, so every candidate occupies every timing position equally often. All outputs must pass correctness before timing.
- Recorded batch semantics: `sequential_outer_loop`. In `sequential_outer_loop`, batch elements contribute to total call traffic but are not interleaved as one simultaneous cache working set.
- Strict winner: >=5% paired median effect, >=80% pair wins, 95% CI excluding one, and non-regressing p95. A repeatable winner must also be strict in >=2 independent process launches when that many launches are available.

## Main Result

- Workloads: `70`
- Strict block-major/block-first wins over current baseline: `23/70`
- Strict current-baseline wins over block-major/block-first: `0/70`
- Repeatable block-major/block-first wins: `22/70`
- Repeatable current-baseline wins: `0/70`
- Recommended `head_major_head_first`: `48` workloads
- Recommended `block_major_head_first`: `0` workloads
- Recommended `head_major_block_first`: `0` workloads
- Recommended `block_major_block_first`: `22` workloads

## KV-Byte Target Matrix

`call MiB` is total logical K+V traffic across the sequential batch loop. `sequence MiB` is the cache-relevant footprint of one sequence. Block count is reported separately because bytes and traversal iterations co-vary at fixed block size.

| call MiB | geometries | sequence MiB range | blocks/call range | median co-design speedup | repeatable wins |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 8 | 2-8 | 32-256 | 1.000x | 0/8 |
| 16 | 8 | 4-16 | 64-512 | 1.013x | 0/8 |
| 24 | 8 | 6-24 | 96-768 | 1.032x | 0/8 |
| 32 | 8 | 8-32 | 128-1024 | 1.090x | 5/8 |
| 48 | 8 | 12-48 | 192-1536 | 1.195x | 6/8 |
| 64 | 8 | 16-64 | 256-2048 | 1.176x | 6/8 |
| 96 | 8 | 24-96 | 384-3072 | 1.025x | 2/8 |
| 128 | 8 | 32-128 | 512-4096 | 1.024x | 1/8 |

## MHA-2 Transition Control

This lane keeps all four configurations and directly retests the isolated head-major/block-first recommendation around sequence 12288.

| seq | call MiB | blocks/sequence | HM/BF speedup | HM/BF launches | BM/BF speedup | BM/BF launches | recommendation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 6144 | 12 | 96 | 1.008x | 0/3 | 1.005x | 0/3 | head_major_head_first |
| 8192 | 16 | 128 | 1.017x | 0/3 | 1.010x | 0/3 | head_major_head_first |
| 10240 | 20 | 160 | 1.039x | 0/3 | 1.023x | 0/3 | head_major_head_first |
| 12288 | 24 | 192 | 1.047x | 1/3 | 1.025x | 0/3 | head_major_head_first |
| 14336 | 28 | 224 | 1.047x | 1/3 | 1.065x | 3/3 | block_major_block_first |
| 16384 | 32 | 256 | 1.031x | 0/3 | 1.100x | 3/3 | block_major_block_first |

## Controlled Shape Families

| family | shape | Q/KV | D | GQA ratio | workloads | median co-design speedup | strict wins |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| kv_byte_target | mha16 | 16/16 | 64 | 1 | 16 | 1.037x | 2/16 |
| kv_byte_target | mha2 | 2/2 | 64 | 1 | 16 | 1.072x | 8/16 |
| kv_byte_target | mha4 | 4/4 | 64 | 1 | 16 | 1.059x | 7/16 |
| kv_byte_target | mha8 | 8/8 | 64 | 1 | 16 | 1.040x | 4/16 |
| mha2_transition | mha2 | 2/2 | 64 | 1 | 6 | 1.024x | 2/6 |

## Sequence And Batch Regions

| family | sequence | batch | shapes | median co-design speedup | strict wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| kv_byte_target | 512 | 4 | 1 | 1.013x | 0/1 |
| kv_byte_target | 1024 | 4 | 2 | 1.026x | 0/2 |
| kv_byte_target | 1536 | 4 | 1 | 1.049x | 0/1 |
| kv_byte_target | 2048 | 1 | 1 | 0.999x | 0/1 |
| kv_byte_target | 2048 | 4 | 3 | 1.015x | 0/3 |
| kv_byte_target | 3072 | 4 | 2 | 1.090x | 1/2 |
| kv_byte_target | 4096 | 1 | 2 | 1.006x | 0/2 |
| kv_byte_target | 4096 | 4 | 4 | 1.073x | 2/4 |
| kv_byte_target | 6144 | 1 | 1 | 1.032x | 0/1 |
| kv_byte_target | 6144 | 4 | 3 | 1.039x | 1/3 |
| kv_byte_target | 8192 | 1 | 3 | 1.011x | 0/3 |
| kv_byte_target | 8192 | 4 | 4 | 1.060x | 2/4 |
| kv_byte_target | 12288 | 1 | 2 | 1.076x | 0/2 |
| kv_byte_target | 12288 | 4 | 3 | 1.024x | 1/3 |
| kv_byte_target | 16384 | 1 | 4 | 1.041x | 1/4 |
| kv_byte_target | 16384 | 4 | 3 | 1.094x | 2/3 |
| kv_byte_target | 24576 | 1 | 3 | 1.029x | 0/3 |
| kv_byte_target | 24576 | 4 | 2 | 1.091x | 1/2 |
| kv_byte_target | 32768 | 1 | 4 | 1.058x | 1/4 |
| kv_byte_target | 32768 | 4 | 2 | 1.092x | 1/2 |
| kv_byte_target | 49152 | 1 | 3 | 1.048x | 1/3 |
| kv_byte_target | 49152 | 4 | 1 | 1.074x | 0/1 |
| kv_byte_target | 65536 | 1 | 3 | 1.078x | 1/3 |
| kv_byte_target | 65536 | 4 | 1 | 1.054x | 1/1 |
| kv_byte_target | 98304 | 1 | 2 | 1.162x | 2/2 |
| kv_byte_target | 131072 | 1 | 2 | 1.126x | 1/2 |
| kv_byte_target | 196608 | 1 | 1 | 1.120x | 1/1 |
| kv_byte_target | 262144 | 1 | 1 | 1.072x | 1/1 |
| mha2_transition | 6144 | 4 | 1 | 1.005x | 0/1 |
| mha2_transition | 8192 | 4 | 1 | 1.010x | 0/1 |
| mha2_transition | 10240 | 4 | 1 | 1.023x | 0/1 |
| mha2_transition | 12288 | 4 | 1 | 1.025x | 0/1 |
| mha2_transition | 14336 | 4 | 1 | 1.065x | 1/1 |
| mha2_transition | 16384 | 4 | 1 | 1.100x | 1/1 |

## Co-design Win/Loss Maps

Cells show current-baseline latency divided by block-major/block-first latency. `+` is a strict co-design win, `-` is a strict current-baseline win, and `~` is a tie under the preregistered criterion.

### kv_byte_target, block size 64

| shape | S512/B4 | S1024/B4 | S1536/B4 | S2048/B1 | S2048/B4 | S3072/B4 | S4096/B1 | S4096/B4 | S6144/B1 | S6144/B4 | S8192/B1 | S8192/B4 | S12288/B1 | S12288/B4 | S16384/B1 | S16384/B4 | S24576/B1 | S24576/B4 | S32768/B1 | S32768/B4 | S49152/B1 | S49152/B4 | S65536/B1 | S65536/B4 | S98304/B1 | S131072/B1 | S196608/B1 | S262144/B1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
