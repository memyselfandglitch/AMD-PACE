# Decode KV Layout and Traversal Summary

## Experiment

- Current baseline: head-major storage plus head-first traversal.
- Layout-only control: block-major storage plus head-first traversal.
- Traversal-only control: head-major storage plus block-first traversal.
- Co-designed candidate: block-major storage plus block-first traversal.
- Each candidate performs the same fused BF16 decode: QK dot products, blockwise online softmax, and weighted V accumulation.
- Candidate order is randomized within paired repeats and all outputs must pass correctness before timing.
- Strict winner: >=5% paired median effect, >=80% pair wins, 95% CI excluding one, and non-regressing p95.

## Main Result

- Workloads: `110`
- Strict block-major/block-first wins over current baseline: `16/110`
- Strict current-baseline wins over block-major/block-first: `8/110`
- Recommended `head_major_head_first`: `93` workloads
- Recommended `block_major_head_first`: `0` workloads
- Recommended `head_major_block_first`: `1` workloads
- Recommended `block_major_block_first`: `16` workloads

## Controlled Shape Families

| family | shape | Q/KV | D | GQA ratio | workloads | median co-design speedup | strict wins |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gqa_ratio | gqa1 | 8/8 | 64 | 1 | 10 | 1.003x | 3/10 |
| gqa_ratio | gqa2 | 16/8 | 64 | 2 | 10 | 1.002x | 0/10 |
| gqa_ratio | gqa4 | 32/8 | 64 | 4 | 10 | 0.991x | 0/10 |
| gqa_ratio | gqa8 | 64/8 | 64 | 8 | 10 | 1.000x | 0/10 |
| head_dim | d128 | 8/8 | 128 | 1 | 10 | 0.975x | 1/10 |
| head_dim | d256 | 8/8 | 256 | 1 | 10 | 0.982x | 0/10 |
| head_dim | d64 | 8/8 | 64 | 1 | 10 | 1.028x | 3/10 |
| kv_heads | mha16 | 16/16 | 64 | 1 | 10 | 1.037x | 1/10 |
| kv_heads | mha2 | 2/2 | 64 | 1 | 10 | 0.989x | 1/10 |
| kv_heads | mha4 | 4/4 | 64 | 1 | 10 | 1.026x | 2/10 |
| kv_heads | mha8 | 8/8 | 64 | 1 | 10 | 1.056x | 5/10 |

## Sequence And Batch Regions

| family | sequence | batch | shapes | median co-design speedup | strict wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| gqa_ratio | 2048 | 1 | 4 | 0.966x | 0/4 |
| gqa_ratio | 2048 | 4 | 4 | 0.997x | 0/4 |
| gqa_ratio | 4096 | 1 | 4 | 0.952x | 0/4 |
| gqa_ratio | 4096 | 4 | 4 | 1.021x | 1/4 |
| gqa_ratio | 8192 | 1 | 4 | 0.998x | 0/4 |
| gqa_ratio | 8192 | 4 | 4 | 1.006x | 1/4 |
| gqa_ratio | 12288 | 1 | 4 | 0.995x | 0/4 |
| gqa_ratio | 12288 | 4 | 4 | 0.997x | 0/4 |
| gqa_ratio | 16384 | 1 | 4 | 1.013x | 1/4 |
| gqa_ratio | 16384 | 4 | 4 | 0.994x | 0/4 |
| head_dim | 2048 | 1 | 3 | 0.969x | 0/3 |
| head_dim | 2048 | 4 | 3 | 1.010x | 0/3 |
| head_dim | 4096 | 1 | 3 | 0.990x | 0/3 |
| head_dim | 4096 | 4 | 3 | 1.080x | 2/3 |
| head_dim | 8192 | 1 | 3 | 0.986x | 0/3 |
| head_dim | 8192 | 4 | 3 | 0.983x | 1/3 |
| head_dim | 12288 | 1 | 3 | 1.010x | 0/3 |
| head_dim | 12288 | 4 | 3 | 0.982x | 0/3 |
| head_dim | 16384 | 1 | 3 | 1.033x | 1/3 |
| head_dim | 16384 | 4 | 3 | 0.981x | 0/3 |
| kv_heads | 2048 | 1 | 4 | 1.034x | 1/4 |
| kv_heads | 2048 | 4 | 4 | 1.014x | 0/4 |
| kv_heads | 4096 | 1 | 4 | 1.025x | 1/4 |
| kv_heads | 4096 | 4 | 4 | 1.025x | 2/4 |
| kv_heads | 8192 | 1 | 4 | 1.031x | 1/4 |
| kv_heads | 8192 | 4 | 4 | 1.026x | 1/4 |
| kv_heads | 12288 | 1 | 4 | 1.010x | 0/4 |
| kv_heads | 12288 | 4 | 4 | 1.005x | 1/4 |
| kv_heads | 16384 | 1 | 4 | 1.049x | 0/4 |
| kv_heads | 16384 | 4 | 4 | 1.027x | 2/4 |

## Co-design Win/Loss Maps

Cells show current-baseline latency divided by block-major/block-first latency. `+` is a strict co-design win, `-` is a strict current-baseline win, and `~` is a tie under the preregistered criterion.

### gqa_ratio, block size 64

| shape | S2048/B1 | S2048/B4 | S4096/B1 | S4096/B4 | S8192/B1 | S8192/B4 | S12288/B1 | S12288/B4 | S16384/B1 | S16384/B4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gqa1 | 0.988x ~ | 0.994x ~ | 0.954x ~ | 1.060x + | 0.993x ~ | 1.131x + | 1.074x ~ | 1.005x ~ | 1.051x + | 1.000x ~ |
| gqa2 | 0.980x ~ | 1.024x ~ | 0.984x ~ | 1.020x ~ | 1.135x ~ | 1.002x ~ | 1.011x ~ | 0.978x ~ | 1.002x ~ | 0.982x ~ |
| gqa4 | 0.946x - | 0.990x ~ | 0.949x - | 1.022x ~ | 0.956x ~ | 1.007x ~ | 0.978x ~ | 0.994x ~ | 1.015x ~ | 0.991x ~ |
| gqa8 | 0.951x - | 1.000x ~ | 0.950x - | 1.012x ~ | 1.003x ~ | 1.005x ~ | 0.947x - | 0.999x ~ | 1.011x ~ | 0.997x ~ |

### head_dim, block size 64

| shape | S2048/B1 | S2048/B4 | S4096/B1 | S4096/B4 | S8192/B1 | S8192/B4 | S12288/B1 | S12288/B4 | S16384/B1 | S16384/B4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d128 | 0.916x - | 1.012x ~ | 0.924x - | 1.092x + | 0.982x ~ | 0.962x ~ | 1.010x ~ | 0.968x ~ | 1.033x ~ | 0.965x ~ |
| d256 | 0.969x ~ | 1.007x ~ | 0.990x ~ | 0.983x ~ | 1.008x ~ | 0.983x ~ | 0.980x ~ | 0.982x ~ | 0.982x ~ | 0.981x ~ |
| d64 | 1.027x ~ | 1.010x ~ | 1.028x ~ | 1.080x + | 0.986x ~ | 1.147x + | 1.045x ~ | 1.003x ~ | 1.069x + | 0.997x ~ |

### kv_heads, block size 64

| shape | S2048/B1 | S2048/B4 | S4096/B1 | S4096/B4 | S8192/B1 | S8192/B4 | S12288/B1 | S12288/B4 | S16384/B1 | S16384/B4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mha16 | 1.045x ~ | 1.033x ~ | 1.025x ~ | 1.120x + | 1.041x ~ | 0.998x ~ | 1.135x ~ | 1.001x ~ | 1.081x ~ | 1.000x ~ |
| mha2 | 0.983x ~ | 1.031x ~ | 0.982x ~ | 0.994x ~ | 0.979x ~ | 1.004x ~ | 0.979x ~ | 1.006x ~ | 0.977x ~ | 1.054x + |
| mha4 | 1.024x ~ | 0.946x - | 1.025x ~ | 0.978x ~ | 1.022x ~ | 1.047x ~ | 1.027x ~ | 1.151x + | 1.040x ~ | 1.138x + |
| mha8 | 1.056x + | 0.997x ~ | 1.056x + | 1.056x + | 1.055x + | 1.140x + | 0.992x ~ | 1.004x ~ | 1.057x ~ | 1.000x ~ |


## Individual Workloads

| family | shape | batch | seq | block | current ms | layout-only | traversal-only | co-designed | recommendation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gqa_ratio | gqa1 | 1 | 2048 | 64 | 0.0838 | 0.929x [head_major_head_first] | 0.968x [tie] | 0.988x [tie] | head_major_head_first |
| gqa_ratio | gqa1 | 1 | 4096 | 64 | 0.1597 | 0.846x [head_major_head_first] | 0.954x [tie] | 0.954x [tie] | head_major_head_first |
| gqa_ratio | gqa1 | 1 | 8192 | 64 | 0.3476 | 0.643x [head_major_head_first] | 0.779x [head_major_head_first] | 0.993x [tie] | head_major_head_first |
| gqa_ratio | gqa1 | 1 | 12288 | 64 | 0.5737 | 0.558x [head_major_head_first] | 0.673x [head_major_head_first] | 1.074x [tie] | head_major_head_first |
| gqa_ratio | gqa1 | 1 | 16384 | 64 | 0.8026 | 0.566x [head_major_head_first] | 0.691x [head_major_head_first] | 1.051x [block_major_block_first] | block_major_block_first |
| gqa_ratio | gqa1 | 4 | 2048 | 64 | 0.3676 | 0.703x [head_major_head_first] | 0.785x [head_major_head_first] | 0.994x [tie] | head_major_head_first |
| gqa_ratio | gqa1 | 4 | 4096 | 64 | 0.8259 | 0.640x [head_major_head_first] | 0.684x [head_major_head_first] | 1.060x [block_major_block_first] | block_major_block_first |
| gqa_ratio | gqa1 | 4 | 8192 | 64 | 1.8130 | 0.590x [head_major_head_first] | 0.707x [head_major_head_first] | 1.131x [block_major_block_first] | block_major_block_first |
| gqa_ratio | gqa1 | 4 | 12288 | 64 | 2.8224 | 0.550x [head_major_head_first] | 0.722x [head_major_head_first] | 1.005x [tie] | head_major_head_first |
| gqa_ratio | gqa1 | 4 | 16384 | 64 | 3.7854 | 0.580x [head_major_head_first] | 0.717x [head_major_head_first] | 1.000x [tie] | head_major_head_first |
| gqa_ratio | gqa2 | 1 | 2048 | 64 | 0.1677 | 0.966x [tie] | 0.989x [tie] | 0.980x [tie] | head_major_head_first |
| gqa_ratio | gqa2 | 1 | 4096 | 64 | 0.3356 | 0.939x [head_major_head_first] | 0.984x [tie] | 0.984x [tie] | head_major_head_first |
| gqa_ratio | gqa2 | 1 | 8192 | 64 | 0.7791 | 0.919x [head_major_head_first] | 0.924x [head_major_head_first] | 1.135x [tie] | head_major_head_first |
| gqa_ratio | gqa2 | 1 | 12288 | 64 | 1.1012 | 0.725x [head_major_head_first] | 0.801x [head_major_head_first] | 1.011x [tie] | head_major_head_first |
| gqa_ratio | gqa2 | 1 | 16384 | 64 | 1.7172 | 0.803x [head_major_head_first] | 0.870x [head_major_head_first] | 1.002x [tie] | head_major_head_first |
| gqa_ratio | gqa2 | 4 | 2048 | 64 | 0.7550 | 0.907x [head_major_head_first] | 0.925x [head_major_head_first] | 1.024x [tie] | head_major_head_first |
| gqa_ratio | gqa2 | 4 | 4096 | 64 | 1.7643 | 0.873x [head_major_head_first] | 0.887x [head_major_head_first] | 1.020x [tie] | head_major_head_first |
| gqa_ratio | gqa2 | 4 | 8192 | 64 | 3.6128 | 0.832x [head_major_head_first] | 0.896x [head_major_head_first] | 1.002x [tie] | head_major_head_first |
| gqa_ratio | gqa2 | 4 | 12288 | 64 | 5.4196 | 0.804x [head_major_head_first] | 0.898x [head_major_head_first] | 0.978x [tie] | head_major_head_first |
| gqa_ratio | gqa2 | 4 | 16384 | 64 | 7.2311 | 0.802x [head_major_head_first] | 0.896x [head_major_head_first] | 0.982x [tie] | head_major_head_first |
| gqa_ratio | gqa4 | 1 | 2048 | 64 | 0.3194 | 0.937x [head_major_head_first] | 0.992x [tie] | 0.946x [head_major_head_first] | head_major_head_first |
| gqa_ratio | gqa4 | 1 | 4096 | 64 | 0.6417 | 0.929x [head_major_head_first] | 0.993x [tie] | 0.949x [head_major_head_first] | head_major_head_first |
| gqa_ratio | gqa4 | 1 | 8192 | 64 | 1.4631 | 0.924x [head_major_head_first] | 0.968x [tie] | 0.956x [tie] | head_major_head_first |
| gqa_ratio | gqa4 | 1 | 12288 | 64 | 2.3839 | 0.932x [head_major_head_first] | 0.991x [tie] | 0.978x [tie] | head_major_head_first |
| gqa_ratio | gqa4 | 1 | 16384 | 64 | 3.2466 | 0.910x [head_major_head_first] | 0.980x [tie] | 1.015x [tie] | head_major_head_first |
| gqa_ratio | gqa4 | 4 | 2048 | 64 | 1.4090 | 0.965x [tie] | 0.975x [tie] | 0.990x [tie] | head_major_head_first |
| gqa_ratio | gqa4 | 4 | 4096 | 64 | 3.2316 | 0.971x [tie] | 0.984x [tie] | 1.022x [tie] | head_major_head_first |
| gqa_ratio | gqa4 | 4 | 8192 | 64 | 6.5430 | 0.915x [head_major_head_first] | 0.975x [tie] | 1.007x [tie] | head_major_head_first |
| gqa_ratio | gqa4 | 4 | 12288 | 64 | 9.8314 | 0.907x [head_major_head_first] | 0.980x [tie] | 0.994x [tie] | head_major_head_first |
| gqa_ratio | gqa4 | 4 | 16384 | 64 | 13.1480 | 0.908x [head_major_head_first] | 0.977x [tie] | 0.991x [tie] | head_major_head_first |
| gqa_ratio | gqa8 | 1 | 2048 | 64 | 0.6355 | 0.944x [head_major_head_first] | 0.999x [tie] | 0.951x [head_major_head_first] | head_major_head_first |
| gqa_ratio | gqa8 | 1 | 4096 | 64 | 1.2742 | 0.940x [head_major_head_first] | 0.994x [tie] | 0.950x [head_major_head_first] | head_major_head_first |
| gqa_ratio | gqa8 | 1 | 8192 | 64 | 2.6829 | 0.980x [tie] | 0.987x [tie] | 1.003x [tie] | head_major_head_first |
| gqa_ratio | gqa8 | 1 | 12288 | 64 | 4.1888 | 0.923x [head_major_head_first] | 0.994x [tie] | 0.947x [head_major_head_first] | head_major_head_first |
| gqa_ratio | gqa8 | 1 | 16384 | 64 | 5.8766 | 0.959x [tie] | 0.994x [tie] | 1.011x [tie] | head_major_head_first |
| gqa_ratio | gqa8 | 4 | 2048 | 64 | 2.7062 | 0.989x [tie] | 0.990x [tie] | 1.000x [tie] | head_major_head_first |
| gqa_ratio | gqa8 | 4 | 4096 | 64 | 5.9024 | 0.989x [tie] | 0.996x [tie] | 1.012x [tie] | head_major_head_first |
| gqa_ratio | gqa8 | 4 | 8192 | 64 | 11.8856 | 0.963x [tie] | 0.992x [tie] | 1.005x [tie] | head_major_head_first |
| gqa_ratio | gqa8 | 4 | 12288 | 64 | 17.8789 | 0.959x [tie] | 0.995x [tie] | 0.999x [tie] | head_major_head_first |
| gqa_ratio | gqa8 | 4 | 16384 | 64 | 23.8276 | 0.957x [tie] | 0.992x [tie] | 0.997x [tie] | head_major_head_first |
| head_dim | d128 | 1 | 2048 | 64 | 0.1403 | 0.848x [head_major_head_first] | 0.944x [head_major_head_first] | 0.916x [head_major_head_first] | head_major_head_first |
| head_dim | d128 | 1 | 4096 | 64 | 0.3119 | 0.747x [head_major_head_first] | 0.815x [head_major_head_first] | 0.924x [head_major_head_first] | head_major_head_first |
| head_dim | d128 | 1 | 8192 | 64 | 0.7281 | 0.686x [head_major_head_first] | 0.760x [head_major_head_first] | 0.982x [tie] | head_major_head_first |
| head_dim | d128 | 1 | 12288 | 64 | 1.2058 | 0.697x [head_major_head_first] | 0.788x [head_major_head_first] | 1.010x [tie] | head_major_head_first |
| head_dim | d128 | 1 | 16384 | 64 | 1.7489 | 0.733x [head_major_head_first] | 0.814x [head_major_head_first] | 1.033x [tie] | head_major_head_first |
| head_dim | d128 | 4 | 2048 | 64 | 0.7342 | 0.738x [head_major_head_first] | 0.765x [head_major_head_first] | 1.012x [tie] | head_major_head_first |
| head_dim | d128 | 4 | 4096 | 64 | 1.7314 | 0.806x [head_major_head_first] | 0.828x [head_major_head_first] | 1.092x [block_major_block_first] | block_major_block_first |
| head_dim | d128 | 4 | 8192 | 64 | 3.7810 | 0.813x [head_major_head_first] | 0.838x [head_major_head_first] | 0.962x [tie] | head_major_head_first |
| head_dim | d128 | 4 | 12288 | 64 | 5.6741 | 0.804x [head_major_head_first] | 0.838x [head_major_head_first] | 0.968x [tie] | head_major_head_first |
| head_dim | d128 | 4 | 16384 | 64 | 7.5731 | 0.794x [head_major_head_first] | 0.806x [head_major_head_first] | 0.965x [tie] | head_major_head_first |
| head_dim | d256 | 1 | 2048 | 64 | 0.3831 | 0.902x [head_major_head_first] | 0.981x [tie] | 0.969x [tie] | head_major_head_first |
| head_dim | d256 | 1 | 4096 | 64 | 0.9469 | 0.934x [head_major_head_first] | 0.985x [tie] | 0.990x [tie] | head_major_head_first |
| head_dim | d256 | 1 | 8192 | 64 | 2.0845 | 0.938x [head_major_head_first] | 0.985x [tie] | 1.008x [tie] | head_major_head_first |
| head_dim | d256 | 1 | 12288 | 64 | 3.3127 | 0.936x [head_major_head_first] | 0.988x [tie] | 0.980x [tie] | head_major_head_first |
| head_dim | d256 | 1 | 16384 | 64 | 4.4698 | 0.929x [head_major_head_first] | 0.985x [tie] | 0.982x [tie] | head_major_head_first |
| head_dim | d256 | 4 | 2048 | 64 | 2.0591 | 0.976x [tie] | 0.995x [tie] | 1.007x [tie] | head_major_head_first |
| head_dim | d256 | 4 | 4096 | 64 | 4.4798 | 0.970x [tie] | 0.985x [tie] | 0.983x [tie] | head_major_head_first |
| head_dim | d256 | 4 | 8192 | 64 | 8.9667 | 0.969x [tie] | 0.985x [tie] | 0.983x [tie] | head_major_head_first |
| head_dim | d256 | 4 | 12288 | 64 | 13.4356 | 0.954x [tie] | 0.983x [tie] | 0.982x [tie] | head_major_head_first |
| head_dim | d256 | 4 | 16384 | 64 | 17.9107 | 0.930x [head_major_head_first] | 0.979x [tie] | 0.981x [tie] | head_major_head_first |
| head_dim | d64 | 1 | 2048 | 64 | 0.0829 | 0.968x [tie] | 0.959x [tie] | 1.027x [tie] | head_major_head_first |
| head_dim | d64 | 1 | 4096 | 64 | 0.1652 | 0.918x [head_major_head_first] | 0.948x [head_major_head_first] | 1.028x [tie] | head_major_head_first |
| head_dim | d64 | 1 | 8192 | 64 | 0.3555 | 0.646x [head_major_head_first] | 0.742x [head_major_head_first] | 0.986x [tie] | head_major_head_first |
| head_dim | d64 | 1 | 12288 | 64 | 0.5768 | 0.554x [head_major_head_first] | 0.651x [head_major_head_first] | 1.045x [tie] | head_major_head_first |
| head_dim | d64 | 1 | 16384 | 64 | 0.7965 | 0.550x [head_major_head_first] | 0.679x [head_major_head_first] | 1.069x [block_major_block_first] | block_major_block_first |
| head_dim | d64 | 4 | 2048 | 64 | 0.3537 | 0.692x [head_major_head_first] | 0.774x [head_major_head_first] | 1.010x [tie] | head_major_head_first |
| head_dim | d64 | 4 | 4096 | 64 | 0.8241 | 0.639x [head_major_head_first] | 0.698x [head_major_head_first] | 1.080x [block_major_block_first] | block_major_block_first |
| head_dim | d64 | 4 | 8192 | 64 | 1.8202 | 0.590x [head_major_head_first] | 0.709x [head_major_head_first] | 1.147x [block_major_block_first] | block_major_block_first |
| head_dim | d64 | 4 | 12288 | 64 | 2.8302 | 0.558x [head_major_head_first] | 0.726x [head_major_head_first] | 1.003x [tie] | head_major_head_first |
| head_dim | d64 | 4 | 16384 | 64 | 3.7844 | 0.620x [head_major_head_first] | 0.723x [head_major_head_first] | 0.997x [tie] | head_major_head_first |
| kv_heads | mha16 | 1 | 2048 | 64 | 0.1712 | 0.955x [tie] | 0.941x [head_major_head_first] | 1.045x [tie] | head_major_head_first |
| kv_heads | mha16 | 1 | 4096 | 64 | 0.3657 | 0.687x [head_major_head_first] | 0.703x [head_major_head_first] | 1.025x [tie] | head_major_head_first |
| kv_heads | mha16 | 1 | 8192 | 64 | 0.7972 | 0.575x [head_major_head_first] | 0.596x [head_major_head_first] | 1.041x [tie] | head_major_head_first |
| kv_heads | mha16 | 1 | 12288 | 64 | 1.3490 | 0.602x [head_major_head_first] | 0.670x [head_major_head_first] | 1.135x [tie] | head_major_head_first |
| kv_heads | mha16 | 1 | 16384 | 64 | 1.8647 | 0.578x [head_major_head_first] | 0.665x [head_major_head_first] | 1.081x [tie] | head_major_head_first |
| kv_heads | mha16 | 4 | 2048 | 64 | 0.8132 | 0.625x [head_major_head_first] | 0.633x [head_major_head_first] | 1.033x [tie] | head_major_head_first |
| kv_heads | mha16 | 4 | 4096 | 64 | 1.7894 | 0.643x [head_major_head_first] | 0.641x [head_major_head_first] | 1.120x [block_major_block_first] | block_major_block_first |
| kv_heads | mha16 | 4 | 8192 | 64 | 3.7886 | 0.614x [head_major_head_first] | 0.671x [head_major_head_first] | 0.998x [tie] | head_major_head_first |
| kv_heads | mha16 | 4 | 12288 | 64 | 5.7096 | 0.655x [head_major_head_first] | 0.673x [head_major_head_first] | 1.001x [tie] | head_major_head_first |
| kv_heads | mha16 | 4 | 16384 | 64 | 7.5867 | 0.644x [head_major_head_first] | 0.663x [head_major_head_first] | 1.000x [tie] | head_major_head_first |
| kv_heads | mha2 | 1 | 2048 | 64 | 0.0576 | 0.952x [head_major_head_first] | 1.001x [tie] | 0.983x [tie] | head_major_head_first |
| kv_heads | mha2 | 1 | 4096 | 64 | 0.0422 | 0.902x [head_major_head_first] | 1.000x [tie] | 0.982x [tie] | head_major_head_first |
| kv_heads | mha2 | 1 | 8192 | 64 | 0.0839 | 0.887x [head_major_head_first] | 1.000x [tie] | 0.979x [tie] | head_major_head_first |
| kv_heads | mha2 | 1 | 12288 | 64 | 0.1257 | 0.883x [head_major_head_first] | 1.000x [tie] | 0.979x [tie] | head_major_head_first |
| kv_heads | mha2 | 1 | 16384 | 64 | 0.1673 | 0.880x [head_major_head_first] | 0.999x [tie] | 0.977x [tie] | head_major_head_first |
| kv_heads | mha2 | 4 | 2048 | 64 | 0.0839 | 0.958x [tie] | 1.000x [tie] | 1.031x [tie] | head_major_head_first |
| kv_heads | mha2 | 4 | 4096 | 64 | 0.1671 | 0.909x [head_major_head_first] | 0.989x [tie] | 0.994x [tie] | head_major_head_first |
| kv_heads | mha2 | 4 | 8192 | 64 | 0.3571 | 0.628x [head_major_head_first] | 1.030x [tie] | 1.004x [tie] | head_major_head_first |
| kv_heads | mha2 | 4 | 12288 | 64 | 0.5624 | 0.491x [head_major_head_first] | 1.053x [head_major_block_first] | 1.006x [tie] | head_major_block_first |
| kv_heads | mha2 | 4 | 16384 | 64 | 0.7954 | 0.492x [head_major_head_first] | 1.047x [tie] | 1.054x [block_major_block_first] | block_major_block_first |
| kv_heads | mha4 | 1 | 2048 | 64 | 0.0421 | 0.983x [tie] | 0.994x [tie] | 1.024x [tie] | head_major_head_first |
| kv_heads | mha4 | 1 | 4096 | 64 | 0.0837 | 0.959x [tie] | 0.996x [tie] | 1.025x [tie] | head_major_head_first |
| kv_heads | mha4 | 1 | 8192 | 64 | 0.1668 | 0.931x [head_major_head_first] | 0.995x [tie] | 1.022x [tie] | head_major_head_first |
| kv_heads | mha4 | 1 | 12288 | 64 | 0.2547 | 0.822x [head_major_head_first] | 0.991x [tie] | 1.027x [tie] | head_major_head_first |
| kv_heads | mha4 | 1 | 16384 | 64 | 0.3591 | 0.668x [head_major_head_first] | 1.004x [tie] | 1.040x [tie] | head_major_head_first |
| kv_heads | mha4 | 4 | 2048 | 64 | 0.1631 | 0.861x [head_major_head_first] | 0.988x [tie] | 0.946x [head_major_head_first] | head_major_head_first |
| kv_heads | mha4 | 4 | 4096 | 64 | 0.3521 | 0.666x [head_major_head_first] | 0.990x [tie] | 0.978x [tie] | head_major_head_first |
| kv_heads | mha4 | 4 | 8192 | 64 | 0.8080 | 0.572x [head_major_head_first] | 0.992x [tie] | 1.047x [tie] | head_major_head_first |
| kv_heads | mha4 | 4 | 12288 | 64 | 1.3496 | 0.569x [head_major_head_first] | 0.984x [tie] | 1.151x [block_major_block_first] | block_major_block_first |
| kv_heads | mha4 | 4 | 16384 | 64 | 1.8292 | 0.539x [head_major_head_first] | 0.964x [tie] | 1.138x [block_major_block_first] | block_major_block_first |
| kv_heads | mha8 | 1 | 2048 | 64 | 0.0849 | 0.999x [tie] | 0.972x [tie] | 1.056x [block_major_block_first] | block_major_block_first |
| kv_heads | mha8 | 1 | 4096 | 64 | 0.1694 | 0.943x [head_major_head_first] | 0.963x [tie] | 1.056x [block_major_block_first] | block_major_block_first |
| kv_heads | mha8 | 1 | 8192 | 64 | 0.3596 | 0.698x [head_major_head_first] | 0.777x [head_major_head_first] | 1.055x [block_major_block_first] | block_major_block_first |
| kv_heads | mha8 | 1 | 12288 | 64 | 0.5514 | 0.528x [head_major_head_first] | 0.686x [head_major_head_first] | 0.992x [tie] | head_major_head_first |
| kv_heads | mha8 | 1 | 16384 | 64 | 0.7892 | 0.541x [head_major_head_first] | 0.663x [head_major_head_first] | 1.057x [tie] | head_major_head_first |
| kv_heads | mha8 | 4 | 2048 | 64 | 0.3539 | 0.685x [head_major_head_first] | 0.790x [head_major_head_first] | 0.997x [tie] | head_major_head_first |
| kv_heads | mha8 | 4 | 4096 | 64 | 0.8128 | 0.627x [head_major_head_first] | 0.664x [head_major_head_first] | 1.056x [block_major_block_first] | block_major_block_first |
| kv_heads | mha8 | 4 | 8192 | 64 | 1.8207 | 0.598x [head_major_head_first] | 0.715x [head_major_head_first] | 1.140x [block_major_block_first] | block_major_block_first |
| kv_heads | mha8 | 4 | 12288 | 64 | 2.8341 | 0.561x [head_major_head_first] | 0.726x [head_major_head_first] | 1.004x [tie] | head_major_head_first |
| kv_heads | mha8 | 4 | 16384 | 64 | 3.7828 | 0.609x [head_major_head_first] | 0.725x [head_major_head_first] | 1.000x [tie] | head_major_head_first |

## Scope Guardrail

This is a single-threaded decode-mechanism benchmark. It mirrors PACE's GQA blockwise online-softmax dataflow and AVX-512 BF16 arithmetic, but it does not yet modify SlabPool, exercise its allocator's non-contiguous physical block mapping, use Split-K, or include OpenMP scheduling. A production prototype is justified only if a repeatable co-design signal survives this controlled test.
