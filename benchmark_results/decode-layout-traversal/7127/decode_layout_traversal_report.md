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

- Workloads: `3`
- Strict block-major/block-first wins over current baseline: `0/3`
- Strict current-baseline wins over block-major/block-first: `0/3`
- Recommended `head_major_head_first`: `3` workloads
- Recommended `block_major_head_first`: `0` workloads
- Recommended `head_major_block_first`: `0` workloads
- Recommended `block_major_block_first`: `0` workloads

## Controlled Shape Families

| family | shape | Q/KV | D | GQA ratio | workloads | median co-design speedup | strict wins |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gqa_ratio | gqa4 | 32/8 | 64 | 4 | 1 | 0.996x | 0/1 |
| head_dim | d256 | 8/8 | 256 | 1 | 1 | 0.959x | 0/1 |
| kv_heads | mha2 | 2/2 | 64 | 1 | 1 | 1.004x | 0/1 |

## Sequence And Batch Regions

| family | sequence | batch | shapes | median co-design speedup | strict wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| gqa_ratio | 2048 | 1 | 1 | 0.996x | 0/1 |
| head_dim | 2048 | 1 | 1 | 0.959x | 0/1 |
| kv_heads | 2048 | 1 | 1 | 1.004x | 0/1 |

## Co-design Win/Loss Maps

Cells show current-baseline latency divided by block-major/block-first latency. `+` is a strict co-design win, `-` is a strict current-baseline win, and `~` is a tie under the preregistered criterion.

### gqa_ratio, block size 64

| shape | S2048/B1 |
| --- | ---: |
| gqa4 | 0.996x ~ |

### head_dim, block size 64

| shape | S2048/B1 |
| --- | ---: |
| d256 | 0.959x ~ |

### kv_heads, block size 64

| shape | S2048/B1 |
| --- | ---: |
| mha2 | 1.004x ~ |


## Individual Workloads

| family | shape | batch | seq | block | current ms | layout-only | traversal-only | co-designed | recommendation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| gqa_ratio | gqa4 | 1 | 2048 | 64 | 0.3326 | 0.990x [tie] | 0.992x [tie] | 0.996x [tie] | head_major_head_first |
| head_dim | d256 | 1 | 2048 | 64 | 0.3906 | 0.894x [head_major_head_first] | 0.963x [tie] | 0.959x [tie] | head_major_head_first |
| kv_heads | mha2 | 1 | 2048 | 64 | 0.0581 | 0.970x [tie] | 1.002x [tie] | 1.004x [tie] | head_major_head_first |

## Scope Guardrail

This is a single-threaded decode-mechanism benchmark. It mirrors PACE's GQA blockwise online-softmax dataflow and AVX-512 BF16 arithmetic, but it does not yet modify SlabPool, exercise its allocator's non-contiguous physical block mapping, use Split-K, or include OpenMP scheduling. A production prototype is justified only if a repeatable co-design signal survives this controlled test.
