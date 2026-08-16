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
| mha16 | 1.013x ~ | 1.041x ~ | 1.049x ~ | 0.999x ~ | 1.047x ~ | 1.148x + | 1.013x ~ | 1.174x + | 1.032x ~ | 1.002x ~ | 1.086x ~ | 0.994x ~ | 1.131x ~ | - | 1.071x ~ | - | 0.995x ~ | - | 0.992x ~ | - | - | - | - | - | - | - | - | - |
| mha2 | - | - | - | - | - | - | - | 1.002x ~ | - | - | - | 1.009x ~ | - | 1.024x ~ | 0.998x ~ | 1.094x + | - | 1.194x + | 1.014x ~ | 1.193x + | 1.032x ~ | 1.074x ~ | 1.071x ~ | 1.054x + | 1.190x + | 1.174x + | 1.120x + | 1.072x + |
| mha4 | - | - | - | - | 1.002x ~ | - | - | 1.018x ~ | - | 1.039x ~ | 0.999x ~ | 1.111x + | - | 1.207x + | 1.012x ~ | 1.180x + | 1.029x ~ | 0.988x ~ | 1.102x + | 0.991x ~ | 1.196x + | - | 1.179x + | - | 1.134x + | 1.079x ~ | - | - |
| mha8 | - | 1.010x ~ | - | - | 1.015x ~ | 1.033x ~ | 0.999x ~ | 1.129x + | - | 1.213x + | 1.011x ~ | 1.183x + | 1.022x ~ | 0.993x ~ | 1.087x + | 0.991x ~ | 1.202x ~ | - | 1.169x ~ | - | 1.048x ~ | - | 1.078x ~ | - | - | - | - | - |

### mha2_transition, block size 64

| shape | S6144/B4 | S8192/B4 | S10240/B4 | S12288/B4 | S14336/B4 | S16384/B4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mha2 | 1.005x ~ | 1.010x ~ | 1.023x ~ | 1.025x ~ | 1.065x + | 1.100x + |


## Individual Workloads

| family | shape | batch | seq | block | current ms | layout-only | traversal-only | co-designed | recommendation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| kv_byte_target | mha16 | 1 | 32768 | 64 | 3.7412 | 0.576x [head_major_head_first] | 0.693x [head_major_head_first] | 0.992x [tie] | head_major_head_first |
| kv_byte_target | mha2 | 1 | 262144 | 64 | 3.7200 | 0.425x [head_major_head_first] | 0.965x [tie] | 1.072x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha4 | 1 | 131072 | 64 | 3.7258 | 0.433x [head_major_head_first] | 0.924x [head_major_head_first] | 1.079x [tie] | head_major_head_first |
| kv_byte_target | mha8 | 1 | 65536 | 64 | 3.6993 | 0.454x [head_major_head_first] | 0.761x [head_major_head_first] | 1.078x [tie] | head_major_head_first |
| kv_byte_target | mha16 | 4 | 8192 | 64 | 3.7489 | 0.670x [head_major_head_first] | 0.694x [head_major_head_first] | 0.994x [tie] | head_major_head_first |
| kv_byte_target | mha2 | 4 | 65536 | 64 | 3.7219 | 0.447x [head_major_head_first] | 0.969x [tie] | 1.054x [block_major_block_first] | head_major_head_first |
| kv_byte_target | mha4 | 4 | 32768 | 64 | 3.7453 | 0.561x [head_major_head_first] | 0.913x [head_major_head_first] | 0.991x [tie] | head_major_head_first |
| kv_byte_target | mha8 | 4 | 16384 | 64 | 3.7496 | 0.642x [head_major_head_first] | 0.759x [head_major_head_first] | 0.991x [tie] | head_major_head_first |
| kv_byte_target | mha16 | 1 | 4096 | 64 | 0.3448 | 0.679x [head_major_head_first] | 0.723x [head_major_head_first] | 1.013x [tie] | head_major_head_first |
| kv_byte_target | mha2 | 1 | 32768 | 64 | 0.3427 | 0.598x [head_major_head_first] | 1.016x [tie] | 1.014x [tie] | head_major_head_first |
| kv_byte_target | mha4 | 1 | 16384 | 64 | 0.3422 | 0.627x [head_major_head_first] | 0.972x [tie] | 1.012x [tie] | head_major_head_first |
| kv_byte_target | mha8 | 1 | 8192 | 64 | 0.3435 | 0.654x [head_major_head_first] | 0.770x [head_major_head_first] | 1.011x [tie] | head_major_head_first |
| kv_byte_target | mha16 | 4 | 1024 | 64 | 0.3606 | 0.738x [head_major_head_first] | 0.744x [head_major_head_first] | 1.041x [tie] | head_major_head_first |
| kv_byte_target | mha2 | 4 | 8192 | 64 | 0.3420 | 0.618x [head_major_head_first] | 1.017x [tie] | 1.009x [tie] | head_major_head_first |
| kv_byte_target | mha4 | 4 | 4096 | 64 | 0.3462 | 0.658x [head_major_head_first] | 0.960x [tie] | 1.018x [tie] | head_major_head_first |
| kv_byte_target | mha8 | 4 | 2048 | 64 | 0.3491 | 0.691x [head_major_head_first] | 0.781x [head_major_head_first] | 1.015x [tie] | head_major_head_first |
| kv_byte_target | mha16 | 1 | 6144 | 64 | 0.5480 | 0.558x [head_major_head_first] | 0.605x [head_major_head_first] | 1.032x [tie] | head_major_head_first |
| kv_byte_target | mha2 | 1 | 49152 | 64 | 0.5453 | 0.461x [head_major_head_first] | 1.045x [tie] | 1.032x [tie] | head_major_head_first |
| kv_byte_target | mha4 | 1 | 24576 | 64 | 0.5447 | 0.507x [head_major_head_first] | 0.997x [tie] | 1.029x [tie] | head_major_head_first |
| kv_byte_target | mha8 | 1 | 12288 | 64 | 0.5456 | 0.546x [head_major_head_first] | 0.654x [head_major_head_first] | 1.022x [tie] | head_major_head_first |
| kv_byte_target | mha16 | 4 | 1536 | 64 | 0.5739 | 0.598x [head_major_head_first] | 0.636x [head_major_head_first] | 1.049x [tie] | head_major_head_first |
| kv_byte_target | mha2 | 4 | 12288 | 64 | 0.5458 | 0.489x [head_major_head_first] | 1.050x [head_major_block_first] | 1.024x [tie] | head_major_head_first |
| kv_byte_target | mha4 | 4 | 6144 | 64 | 0.5528 | 0.548x [head_major_head_first] | 0.978x [tie] | 1.039x [tie] | head_major_head_first |
| kv_byte_target | mha8 | 4 | 3072 | 64 | 0.5540 | 0.590x [head_major_head_first] | 0.679x [head_major_head_first] | 1.033x [tie] | head_major_head_first |
| kv_byte_target | mha16 | 1 | 8192 | 64 | 0.7797 | 0.598x [head_major_head_first] | 0.616x [head_major_head_first] | 1.086x [tie] | head_major_head_first |
| kv_byte_target | mha2 | 1 | 65536 | 64 | 0.7662 | 0.450x [head_major_head_first] | 1.033x [tie] | 1.071x [tie] | head_major_head_first |
| kv_byte_target | mha4 | 1 | 32768 | 64 | 0.7821 | 0.509x [head_major_head_first] | 0.973x [tie] | 1.102x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha8 | 1 | 16384 | 64 | 0.7776 | 0.563x [head_major_head_first] | 0.678x [head_major_head_first] | 1.087x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha16 | 4 | 2048 | 64 | 0.7669 | 0.621x [head_major_head_first] | 0.625x [head_major_head_first] | 1.047x [tie] | head_major_head_first |
| kv_byte_target | mha2 | 4 | 16384 | 64 | 0.7873 | 0.506x [head_major_head_first] | 1.050x [tie] | 1.094x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha4 | 4 | 8192 | 64 | 0.7974 | 0.582x [head_major_head_first] | 0.999x [tie] | 1.111x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha8 | 4 | 4096 | 64 | 0.8050 | 0.658x [head_major_head_first] | 0.681x [head_major_head_first] | 1.129x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha16 | 1 | 12288 | 64 | 1.3192 | 0.604x [head_major_head_first] | 0.647x [head_major_head_first] | 1.131x [tie] | head_major_head_first |
| kv_byte_target | mha2 | 1 | 98304 | 64 | 1.3021 | 0.474x [head_major_head_first] | 1.023x [tie] | 1.190x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha4 | 1 | 49152 | 64 | 1.3089 | 0.507x [head_major_head_first] | 0.967x [tie] | 1.196x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha8 | 1 | 24576 | 64 | 1.3151 | 0.541x [head_major_head_first] | 0.729x [head_major_head_first] | 1.202x [tie] | head_major_head_first |
| kv_byte_target | mha16 | 4 | 3072 | 64 | 1.2837 | 0.661x [head_major_head_first] | 0.653x [head_major_head_first] | 1.148x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha2 | 4 | 24576 | 64 | 1.3070 | 0.515x [head_major_head_first] | 1.013x [tie] | 1.194x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha4 | 4 | 12288 | 64 | 1.3237 | 0.588x [head_major_head_first] | 0.968x [tie] | 1.207x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha8 | 4 | 6144 | 64 | 1.3434 | 0.655x [head_major_head_first] | 0.744x [head_major_head_first] | 1.213x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha16 | 1 | 16384 | 64 | 1.8354 | 0.584x [head_major_head_first] | 0.685x [head_major_head_first] | 1.071x [tie] | head_major_head_first |
| kv_byte_target | mha2 | 1 | 131072 | 64 | 1.7928 | 0.455x [head_major_head_first] | 0.980x [tie] | 1.174x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha4 | 1 | 65536 | 64 | 1.8010 | 0.479x [head_major_head_first] | 0.969x [tie] | 1.179x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha8 | 1 | 32768 | 64 | 1.8117 | 0.527x [head_major_head_first] | 0.741x [head_major_head_first] | 1.169x [tie] | head_major_head_first |
| kv_byte_target | mha16 | 4 | 4096 | 64 | 1.7982 | 0.668x [head_major_head_first] | 0.681x [head_major_head_first] | 1.174x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha2 | 4 | 32768 | 64 | 1.7991 | 0.494x [head_major_head_first] | 0.982x [tie] | 1.193x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha4 | 4 | 16384 | 64 | 1.8040 | 0.546x [head_major_head_first] | 0.965x [tie] | 1.180x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha8 | 4 | 8192 | 64 | 1.8216 | 0.608x [head_major_head_first] | 0.756x [head_major_head_first] | 1.183x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha16 | 1 | 2048 | 64 | 0.1607 | 0.921x [head_major_head_first] | 0.953x [tie] | 0.999x [tie] | head_major_head_first |
| kv_byte_target | mha2 | 1 | 16384 | 64 | 0.1598 | 0.878x [head_major_head_first] | 0.997x [tie] | 0.998x [tie] | head_major_head_first |
| kv_byte_target | mha4 | 1 | 8192 | 64 | 0.1600 | 0.887x [head_major_head_first] | 0.990x [tie] | 0.999x [tie] | head_major_head_first |
| kv_byte_target | mha8 | 1 | 4096 | 64 | 0.1601 | 0.881x [head_major_head_first] | 0.960x [tie] | 0.999x [tie] | head_major_head_first |
| kv_byte_target | mha16 | 4 | 512 | 64 | 0.1654 | 0.973x [tie] | 0.954x [tie] | 1.013x [tie] | head_major_head_first |
| kv_byte_target | mha2 | 4 | 4096 | 64 | 0.1603 | 0.894x [head_major_head_first] | 1.000x [tie] | 1.002x [tie] | head_major_head_first |
| kv_byte_target | mha4 | 4 | 2048 | 64 | 0.1615 | 0.919x [head_major_head_first] | 0.995x [tie] | 1.002x [tie] | head_major_head_first |
| kv_byte_target | mha8 | 4 | 1024 | 64 | 0.1636 | 0.950x [head_major_head_first] | 0.971x [tie] | 1.010x [tie] | head_major_head_first |
| kv_byte_target | mha16 | 1 | 24576 | 64 | 2.7966 | 0.618x [head_major_head_first] | 0.712x [head_major_head_first] | 0.995x [tie] | head_major_head_first |
| kv_byte_target | mha2 | 1 | 196608 | 64 | 2.7419 | 0.434x [head_major_head_first] | 1.018x [tie] | 1.120x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha4 | 1 | 98304 | 64 | 2.7866 | 0.448x [head_major_head_first] | 0.988x [tie] | 1.134x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha8 | 1 | 49152 | 64 | 2.7886 | 0.482x [head_major_head_first] | 0.760x [head_major_head_first] | 1.048x [tie] | head_major_head_first |
| kv_byte_target | mha16 | 4 | 6144 | 64 | 2.8114 | 0.639x [head_major_head_first] | 0.710x [head_major_head_first] | 1.002x [tie] | head_major_head_first |
| kv_byte_target | mha2 | 4 | 49152 | 64 | 2.7841 | 0.465x [head_major_head_first] | 1.017x [tie] | 1.074x [tie] | head_major_head_first |
| kv_byte_target | mha4 | 4 | 24576 | 64 | 2.7934 | 0.534x [head_major_head_first] | 0.976x [tie] | 0.988x [tie] | head_major_head_first |
| kv_byte_target | mha8 | 4 | 12288 | 64 | 2.8014 | 0.591x [head_major_head_first] | 0.761x [head_major_head_first] | 0.993x [tie] | head_major_head_first |
| mha2_transition | mha2 | 4 | 10240 | 64 | 0.4450 | 0.522x [head_major_head_first] | 1.039x [tie] | 1.023x [tie] | head_major_head_first |
| mha2_transition | mha2 | 4 | 12288 | 64 | 0.5470 | 0.489x [head_major_head_first] | 1.047x [tie] | 1.025x [tie] | head_major_head_first |
| mha2_transition | mha2 | 4 | 14336 | 64 | 0.6607 | 0.501x [head_major_head_first] | 1.047x [tie] | 1.065x [block_major_block_first] | block_major_block_first |
| mha2_transition | mha2 | 4 | 16384 | 64 | 0.7884 | 0.511x [head_major_head_first] | 1.031x [tie] | 1.100x [block_major_block_first] | block_major_block_first |
| mha2_transition | mha2 | 4 | 6144 | 64 | 0.2447 | 0.777x [head_major_head_first] | 1.008x [tie] | 1.005x [tie] | head_major_head_first |
| mha2_transition | mha2 | 4 | 8192 | 64 | 0.3424 | 0.606x [head_major_head_first] | 1.017x [tie] | 1.010x [tie] | head_major_head_first |

## Scope Guardrail

This is a single-threaded decode-mechanism benchmark. It mirrors PACE's GQA blockwise online-softmax dataflow and AVX-512 BF16 arithmetic, but it does not yet modify SlabPool, exercise its allocator's non-contiguous physical block mapping, use Split-K, or include OpenMP scheduling. A production prototype is justified only if a repeatable co-design signal survives this controlled test.
