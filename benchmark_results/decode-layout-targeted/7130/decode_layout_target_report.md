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
- Strict block-major/block-first wins over current baseline: `48/70`
- Strict current-baseline wins over block-major/block-first: `0/70`
- Repeatable block-major/block-first wins: `46/70`
- Repeatable current-baseline wins: `0/70`
- Recommended `head_major_head_first`: `24` workloads
- Recommended `block_major_head_first`: `0` workloads
- Recommended `head_major_block_first`: `1` workloads
- Recommended `block_major_block_first`: `45` workloads

## KV-Byte Target Matrix

`call MiB` is total logical K+V traffic across the sequential batch loop. `sequence MiB` is the cache-relevant footprint of one sequence. Block count is reported separately because bytes and traversal iterations co-vary at fixed block size.

| call MiB | geometries | sequence MiB range | blocks/call range | median co-design speedup | repeatable wins |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 16 | 5-40 | 160-1280 | 1.161x | 13/16 |
| 56 | 16 | 7-56 | 224-1792 | 1.193x | 14/16 |
| 72 | 16 | 9-72 | 288-2304 | 1.157x | 12/16 |
| 80 | 16 | 10-80 | 320-2560 | 1.088x | 5/16 |

## MHA-2 Transition Control

This lane keeps all four configurations and directly retests the isolated head-major/block-first recommendation around sequence 12288.

| seq | call MiB | blocks/sequence | HM/BF speedup | HM/BF launches | BM/BF speedup | BM/BF launches | recommendation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 6144 | 12 | 96 | 1.006x | 0/3 | 1.008x | 0/3 | head_major_head_first |
| 8192 | 16 | 128 | 1.017x | 0/3 | 1.010x | 0/3 | head_major_head_first |
| 10240 | 20 | 160 | 1.039x | 0/3 | 1.025x | 0/3 | head_major_head_first |
| 12288 | 24 | 192 | 1.048x | 1/3 | 1.030x | 0/3 | head_major_head_first |
| 14336 | 28 | 224 | 1.063x | 3/3 | 1.059x | 3/3 | head_major_block_first |
| 16384 | 32 | 256 | 1.038x | 0/3 | 1.096x | 3/3 | block_major_block_first |

## Controlled Shape Families

| family | shape | Q/KV | D | GQA ratio | workloads | median co-design speedup | strict wins |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| kv_byte_target | mha16 | 16/16 | 64 | 1 | 16 | 1.109x | 9/16 |
| kv_byte_target | mha2 | 2/2 | 64 | 1 | 16 | 1.168x | 15/16 |
| kv_byte_target | mha4 | 4/4 | 64 | 1 | 16 | 1.171x | 13/16 |
| kv_byte_target | mha8 | 8/8 | 64 | 1 | 16 | 1.150x | 9/16 |
| mha2_transition | mha2 | 2/2 | 64 | 1 | 6 | 1.027x | 2/6 |

## Sequence And Batch Regions

| family | sequence | batch | shapes | median co-design speedup | strict wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| kv_byte_target | 1280 | 8 | 1 | 1.044x | 0/1 |
| kv_byte_target | 1792 | 8 | 1 | 1.080x | 1/1 |
| kv_byte_target | 2304 | 8 | 1 | 1.119x | 1/1 |
| kv_byte_target | 2560 | 4 | 1 | 1.100x | 1/1 |
| kv_byte_target | 2560 | 8 | 2 | 1.117x | 2/2 |
| kv_byte_target | 3584 | 4 | 1 | 1.173x | 1/1 |
| kv_byte_target | 3584 | 8 | 1 | 1.208x | 1/1 |
| kv_byte_target | 4608 | 4 | 1 | 1.145x | 1/1 |
| kv_byte_target | 4608 | 8 | 1 | 1.141x | 1/1 |
| kv_byte_target | 5120 | 2 | 1 | 1.167x | 1/1 |
| kv_byte_target | 5120 | 4 | 2 | 1.118x | 1/2 |
| kv_byte_target | 5120 | 8 | 2 | 1.114x | 1/2 |
| kv_byte_target | 7168 | 2 | 1 | 1.183x | 1/1 |
| kv_byte_target | 7168 | 4 | 1 | 1.215x | 1/1 |
| kv_byte_target | 7168 | 8 | 1 | 1.205x | 1/1 |
| kv_byte_target | 9216 | 2 | 1 | 1.130x | 1/1 |
| kv_byte_target | 9216 | 4 | 1 | 1.134x | 0/1 |
| kv_byte_target | 9216 | 8 | 1 | 1.161x | 1/1 |
| kv_byte_target | 10240 | 1 | 1 | 1.142x | 0/1 |
| kv_byte_target | 10240 | 2 | 2 | 1.111x | 1/2 |
| kv_byte_target | 10240 | 4 | 2 | 1.121x | 2/2 |
| kv_byte_target | 10240 | 8 | 2 | 1.126x | 1/2 |
| kv_byte_target | 14336 | 1 | 1 | 1.036x | 0/1 |
| kv_byte_target | 14336 | 2 | 1 | 1.201x | 1/1 |
| kv_byte_target | 14336 | 4 | 1 | 1.203x | 1/1 |
| kv_byte_target | 14336 | 8 | 1 | 1.209x | 1/1 |
| kv_byte_target | 18432 | 1 | 1 | 0.997x | 0/1 |
| kv_byte_target | 18432 | 2 | 1 | 1.159x | 1/1 |
| kv_byte_target | 18432 | 4 | 1 | 1.150x | 1/1 |
| kv_byte_target | 18432 | 8 | 1 | 1.176x | 1/1 |
| kv_byte_target | 20480 | 1 | 2 | 1.069x | 0/2 |
| kv_byte_target | 20480 | 2 | 2 | 1.133x | 1/2 |
| kv_byte_target | 20480 | 4 | 2 | 1.132x | 1/2 |
| kv_byte_target | 20480 | 8 | 1 | 1.086x | 0/1 |
| kv_byte_target | 28672 | 1 | 1 | 1.191x | 0/1 |
| kv_byte_target | 28672 | 2 | 1 | 1.196x | 1/1 |
| kv_byte_target | 28672 | 4 | 1 | 1.191x | 1/1 |
| kv_byte_target | 36864 | 1 | 1 | 1.165x | 0/1 |
| kv_byte_target | 36864 | 2 | 1 | 1.173x | 1/1 |
| kv_byte_target | 36864 | 4 | 1 | 1.156x | 1/1 |
| kv_byte_target | 40960 | 1 | 2 | 1.115x | 1/2 |
| kv_byte_target | 40960 | 2 | 2 | 1.140x | 1/2 |
| kv_byte_target | 40960 | 4 | 1 | 1.133x | 1/1 |
| kv_byte_target | 57344 | 1 | 1 | 1.200x | 1/1 |
| kv_byte_target | 57344 | 2 | 1 | 1.190x | 1/1 |
| kv_byte_target | 73728 | 1 | 1 | 1.182x | 1/1 |
| kv_byte_target | 73728 | 2 | 1 | 1.163x | 1/1 |
| kv_byte_target | 81920 | 1 | 2 | 1.163x | 2/2 |
| kv_byte_target | 81920 | 2 | 1 | 1.149x | 1/1 |
| kv_byte_target | 114688 | 1 | 1 | 1.189x | 1/1 |
| kv_byte_target | 147456 | 1 | 1 | 1.175x | 1/1 |
| kv_byte_target | 163840 | 1 | 1 | 1.154x | 1/1 |
| mha2_transition | 6144 | 4 | 1 | 1.008x | 0/1 |
| mha2_transition | 8192 | 4 | 1 | 1.010x | 0/1 |
| mha2_transition | 10240 | 4 | 1 | 1.025x | 0/1 |
| mha2_transition | 12288 | 4 | 1 | 1.030x | 0/1 |
| mha2_transition | 14336 | 4 | 1 | 1.059x | 1/1 |
| mha2_transition | 16384 | 4 | 1 | 1.096x | 1/1 |

## Co-design Win/Loss Maps

Cells show current-baseline latency divided by block-major/block-first latency. `+` is a strict co-design win, `-` is a strict current-baseline win, and `~` is a tie under the preregistered criterion.

### kv_byte_target, block size 64

| shape | S1280/B8 | S1792/B8 | S2304/B8 | S2560/B4 | S2560/B8 | S3584/B4 | S3584/B8 | S4608/B4 | S4608/B8 | S5120/B2 | S5120/B4 | S5120/B8 | S7168/B2 | S7168/B4 | S7168/B8 | S9216/B2 | S9216/B4 | S9216/B8 | S10240/B1 | S10240/B2 | S10240/B4 | S10240/B8 | S14336/B1 | S14336/B2 | S14336/B4 | S14336/B8 | S18432/B1 | S18432/B2 | S18432/B4 | S18432/B8 | S20480/B1 | S20480/B2 | S20480/B4 | S20480/B8 | S28672/B1 | S28672/B2 | S28672/B4 | S36864/B1 | S36864/B2 | S36864/B4 | S40960/B1 | S40960/B2 | S40960/B4 | S57344/B1 | S57344/B2 | S73728/B1 | S73728/B2 | S81920/B1 | S81920/B2 | S114688/B1 | S147456/B1 | S163840/B1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mha16 | 1.044x ~ | 1.080x + | 1.119x + | 1.100x + | 1.131x + | 1.173x + | - | 1.145x + | - | 1.167x + | 1.058x ~ | - | 1.183x + | - | - | 1.130x + | - | - | 1.142x ~ | 1.047x ~ | - | - | 1.036x ~ | - | - | - | 0.997x ~ | - | - | - | 0.998x ~ | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| mha2 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 1.183x + | - | - | - | 1.209x + | - | - | - | 1.176x + | - | - | 1.173x + | 1.086x ~ | - | - | 1.191x + | - | - | 1.156x + | - | 1.155x + | 1.133x + | - | 1.190x + | - | 1.163x + | 1.156x + | 1.149x + | 1.189x + | 1.175x + | 1.154x + |
| mha4 | - | - | - | - | - | - | - | - | - | - | - | 1.181x + | - | - | 1.205x + | - | - | 1.161x + | - | - | 1.165x + | 1.070x ~ | - | - | 1.203x + | - | - | - | 1.150x + | - | - | 1.173x + | 1.090x ~ | - | - | 1.196x + | - | - | 1.173x + | - | 1.157x + | 1.125x ~ | - | 1.200x + | - | 1.182x + | - | 1.170x + | - | - | - | - |
| mha8 | - | - | - | - | 1.102x + | - | 1.208x + | - | 1.141x + | - | 1.178x + | 1.048x ~ | - | 1.215x + | - | - | 1.134x ~ | - | - | 1.175x + | 1.078x + | - | - | 1.201x + | - | - | - | 1.159x + | - | - | 1.140x ~ | 1.092x ~ | - | - | 1.191x ~ | - | - | 1.165x ~ | - | - | 1.074x ~ | - | - | - | - | - | - | - | - | - | - | - |

### mha2_transition, block size 64

| shape | S6144/B4 | S8192/B4 | S10240/B4 | S12288/B4 | S14336/B4 | S16384/B4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mha2 | 1.008x ~ | 1.010x ~ | 1.025x ~ | 1.030x ~ | 1.059x + | 1.096x + |


## Individual Workloads

| family | shape | batch | seq | block | current ms | layout-only | traversal-only | co-designed | recommendation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| kv_byte_target | mha16 | 1 | 10240 | 64 | 1.0594 | 0.619x [head_major_head_first] | 0.662x [head_major_head_first] | 1.142x [tie] | head_major_head_first |
| kv_byte_target | mha2 | 1 | 81920 | 64 | 1.0396 | 0.475x [head_major_head_first] | 1.019x [tie] | 1.156x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha4 | 1 | 40960 | 64 | 1.0497 | 0.516x [head_major_head_first] | 0.980x [tie] | 1.157x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha8 | 1 | 20480 | 64 | 1.0348 | 0.545x [head_major_head_first] | 0.709x [head_major_head_first] | 1.140x [tie] | head_major_head_first |
| kv_byte_target | mha16 | 2 | 5120 | 64 | 1.0568 | 0.654x [head_major_head_first] | 0.664x [head_major_head_first] | 1.167x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha2 | 2 | 40960 | 64 | 1.0392 | 0.486x [head_major_head_first] | 1.002x [tie] | 1.155x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha4 | 2 | 20480 | 64 | 1.0564 | 0.554x [head_major_head_first] | 0.968x [tie] | 1.173x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha8 | 2 | 10240 | 64 | 1.0600 | 0.603x [head_major_head_first] | 0.730x [head_major_head_first] | 1.175x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha16 | 4 | 2560 | 64 | 1.0178 | 0.644x [head_major_head_first] | 0.649x [head_major_head_first] | 1.100x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha2 | 4 | 20480 | 64 | 1.0536 | 0.520x [head_major_head_first] | 1.032x [tie] | 1.173x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha4 | 4 | 10240 | 64 | 1.0584 | 0.580x [head_major_head_first] | 0.947x [tie] | 1.165x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha8 | 4 | 5120 | 64 | 1.0632 | 0.655x [head_major_head_first] | 0.732x [head_major_head_first] | 1.178x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha16 | 8 | 1280 | 64 | 0.9866 | 0.634x [head_major_head_first] | 0.631x [head_major_head_first] | 1.044x [tie] | head_major_head_first |
| kv_byte_target | mha2 | 8 | 10240 | 64 | 1.0585 | 0.537x [head_major_head_first] | 1.041x [tie] | 1.183x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha4 | 8 | 5120 | 64 | 1.0673 | 0.625x [head_major_head_first] | 0.996x [tie] | 1.181x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha8 | 8 | 2560 | 64 | 1.0180 | 0.640x [head_major_head_first] | 0.711x [head_major_head_first] | 1.102x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha16 | 1 | 14336 | 64 | 1.5854 | 0.599x [head_major_head_first] | 0.693x [head_major_head_first] | 1.036x [tie] | head_major_head_first |
| kv_byte_target | mha2 | 1 | 114688 | 64 | 1.5410 | 0.464x [head_major_head_first] | 1.023x [tie] | 1.189x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha4 | 1 | 57344 | 64 | 1.5558 | 0.496x [head_major_head_first] | 0.980x [tie] | 1.200x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha8 | 1 | 28672 | 64 | 1.5745 | 0.537x [head_major_head_first] | 0.747x [head_major_head_first] | 1.191x [tie] | head_major_head_first |
| kv_byte_target | mha16 | 2 | 7168 | 64 | 1.5471 | 0.628x [head_major_head_first] | 0.684x [head_major_head_first] | 1.183x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha2 | 2 | 57344 | 64 | 1.5531 | 0.483x [head_major_head_first] | 1.033x [tie] | 1.190x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha4 | 2 | 28672 | 64 | 1.5605 | 0.525x [head_major_head_first] | 0.972x [tie] | 1.196x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha8 | 2 | 14336 | 64 | 1.5723 | 0.581x [head_major_head_first] | 0.749x [head_major_head_first] | 1.201x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha16 | 4 | 3584 | 64 | 1.5449 | 0.685x [head_major_head_first] | 0.684x [head_major_head_first] | 1.173x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha2 | 4 | 28672 | 64 | 1.5408 | 0.503x [head_major_head_first] | 1.005x [tie] | 1.191x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha4 | 4 | 14336 | 64 | 1.5702 | 0.563x [head_major_head_first] | 0.994x [tie] | 1.203x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha8 | 4 | 7168 | 64 | 1.5808 | 0.634x [head_major_head_first] | 0.751x [head_major_head_first] | 1.215x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha16 | 8 | 1792 | 64 | 1.4753 | 0.649x [head_major_head_first] | 0.673x [head_major_head_first] | 1.080x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha2 | 8 | 14336 | 64 | 1.5655 | 0.525x [head_major_head_first] | 1.031x [tie] | 1.209x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha4 | 8 | 7168 | 64 | 1.5822 | 0.604x [head_major_head_first] | 0.995x [tie] | 1.205x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha8 | 8 | 3584 | 64 | 1.5887 | 0.689x [head_major_head_first] | 0.757x [head_major_head_first] | 1.208x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha16 | 1 | 18432 | 64 | 2.0624 | 0.613x [head_major_head_first] | 0.695x [head_major_head_first] | 0.997x [tie] | head_major_head_first |
| kv_byte_target | mha2 | 1 | 147456 | 64 | 2.0371 | 0.452x [head_major_head_first] | 1.025x [tie] | 1.175x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha4 | 1 | 73728 | 64 | 2.0549 | 0.467x [head_major_head_first] | 0.989x [tie] | 1.182x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha8 | 1 | 36864 | 64 | 2.0607 | 0.512x [head_major_head_first] | 0.752x [head_major_head_first] | 1.165x [tie] | head_major_head_first |
| kv_byte_target | mha16 | 2 | 9216 | 64 | 2.0894 | 0.606x [head_major_head_first] | 0.705x [head_major_head_first] | 1.130x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha2 | 2 | 73728 | 64 | 2.0329 | 0.464x [head_major_head_first] | 1.008x [tie] | 1.163x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha4 | 2 | 36864 | 64 | 2.0557 | 0.500x [head_major_head_first] | 0.980x [tie] | 1.173x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha8 | 2 | 18432 | 64 | 2.0909 | 0.548x [head_major_head_first] | 0.765x [head_major_head_first] | 1.159x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha16 | 4 | 4608 | 64 | 2.0853 | 0.663x [head_major_head_first] | 0.705x [head_major_head_first] | 1.145x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha2 | 4 | 36864 | 64 | 2.0459 | 0.483x [head_major_head_first] | 1.015x [tie] | 1.156x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha4 | 4 | 18432 | 64 | 2.0564 | 0.539x [head_major_head_first] | 0.982x [tie] | 1.150x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha8 | 4 | 9216 | 64 | 2.0717 | 0.599x [head_major_head_first] | 0.752x [head_major_head_first] | 1.134x [tie] | head_major_head_first |
| kv_byte_target | mha16 | 8 | 2304 | 64 | 2.0030 | 0.668x [head_major_head_first] | 0.695x [head_major_head_first] | 1.119x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha2 | 8 | 18432 | 64 | 2.0630 | 0.506x [head_major_head_first] | 1.019x [tie] | 1.176x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha4 | 8 | 9216 | 64 | 2.0514 | 0.573x [head_major_head_first] | 0.979x [tie] | 1.161x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha8 | 8 | 4608 | 64 | 2.0945 | 0.660x [head_major_head_first] | 0.761x [head_major_head_first] | 1.141x [block_major_block_first] | head_major_head_first |
| kv_byte_target | mha16 | 1 | 20480 | 64 | 2.3262 | 0.626x [head_major_head_first] | 0.710x [head_major_head_first] | 0.998x [tie] | head_major_head_first |
| kv_byte_target | mha2 | 1 | 163840 | 64 | 2.2772 | 0.443x [head_major_head_first] | 1.023x [tie] | 1.154x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha4 | 1 | 81920 | 64 | 2.3122 | 0.464x [head_major_head_first] | 0.995x [tie] | 1.170x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha8 | 1 | 40960 | 64 | 2.3198 | 0.505x [head_major_head_first] | 0.758x [head_major_head_first] | 1.074x [tie] | head_major_head_first |
| kv_byte_target | mha16 | 2 | 10240 | 64 | 2.3218 | 0.598x [head_major_head_first] | 0.704x [head_major_head_first] | 1.047x [tie] | head_major_head_first |
| kv_byte_target | mha2 | 2 | 81920 | 64 | 2.2767 | 0.453x [head_major_head_first] | 1.014x [tie] | 1.149x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha4 | 2 | 40960 | 64 | 2.2898 | 0.490x [head_major_head_first] | 0.975x [tie] | 1.125x [tie] | head_major_head_first |
| kv_byte_target | mha8 | 2 | 20480 | 64 | 2.3253 | 0.541x [head_major_head_first] | 0.755x [head_major_head_first] | 1.092x [tie] | head_major_head_first |
| kv_byte_target | mha16 | 4 | 5120 | 64 | 2.3483 | 0.652x [head_major_head_first] | 0.707x [head_major_head_first] | 1.058x [tie] | head_major_head_first |
| kv_byte_target | mha2 | 4 | 40960 | 64 | 2.2860 | 0.476x [head_major_head_first] | 1.012x [tie] | 1.133x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha4 | 4 | 20480 | 64 | 2.3239 | 0.536x [head_major_head_first] | 0.980x [tie] | 1.090x [tie] | head_major_head_first |
| kv_byte_target | mha8 | 4 | 10240 | 64 | 2.3310 | 0.587x [head_major_head_first] | 0.750x [head_major_head_first] | 1.078x [block_major_block_first] | head_major_head_first |
| kv_byte_target | mha16 | 8 | 2560 | 64 | 2.2614 | 0.669x [head_major_head_first] | 0.700x [head_major_head_first] | 1.131x [block_major_block_first] | block_major_block_first |
| kv_byte_target | mha2 | 8 | 20480 | 64 | 2.3169 | 0.497x [head_major_head_first] | 1.020x [tie] | 1.086x [tie] | head_major_head_first |
| kv_byte_target | mha4 | 8 | 10240 | 64 | 2.3160 | 0.568x [head_major_head_first] | 0.975x [tie] | 1.070x [tie] | head_major_head_first |
| kv_byte_target | mha8 | 8 | 5120 | 64 | 2.3381 | 0.646x [head_major_head_first] | 0.761x [head_major_head_first] | 1.048x [tie] | head_major_head_first |
| mha2_transition | mha2 | 4 | 10240 | 64 | 0.4450 | 0.524x [head_major_head_first] | 1.039x [tie] | 1.025x [tie] | head_major_head_first |
| mha2_transition | mha2 | 4 | 12288 | 64 | 0.5461 | 0.496x [head_major_head_first] | 1.048x [tie] | 1.030x [tie] | head_major_head_first |
| mha2_transition | mha2 | 4 | 14336 | 64 | 0.6575 | 0.500x [head_major_head_first] | 1.063x [head_major_block_first] | 1.059x [block_major_block_first] | head_major_block_first |
| mha2_transition | mha2 | 4 | 16384 | 64 | 0.7859 | 0.510x [head_major_head_first] | 1.038x [tie] | 1.096x [block_major_block_first] | block_major_block_first |
| mha2_transition | mha2 | 4 | 6144 | 64 | 0.2447 | 0.782x [head_major_head_first] | 1.006x [tie] | 1.008x [tie] | head_major_head_first |
| mha2_transition | mha2 | 4 | 8192 | 64 | 0.3416 | 0.611x [head_major_head_first] | 1.017x [tie] | 1.010x [tie] | head_major_head_first |

## Scope Guardrail

This is a single-threaded decode-mechanism benchmark. It mirrors PACE's GQA blockwise online-softmax dataflow and AVX-512 BF16 arithmetic, but it does not yet modify SlabPool, exercise its allocator's non-contiguous physical block mapping, use Split-K, or include OpenMP scheduling. A production prototype is justified only if a repeatable co-design signal survives this controlled test.
