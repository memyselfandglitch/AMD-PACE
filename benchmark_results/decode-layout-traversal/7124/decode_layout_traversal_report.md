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

- Workloads: `1`
- Strict block-major/block-first wins over current baseline: `0/1`
- Strict current-baseline wins over block-major/block-first: `0/1`
- Recommended `head_major_head_first`: `1` workloads
- Recommended `block_major_head_first`: `0` workloads
- Recommended `head_major_block_first`: `0` workloads
- Recommended `block_major_block_first`: `0` workloads

## Workloads

| shape | batch | seq | current ms | layout-only | traversal-only | co-designed | recommendation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| slm_gqa | 1 | 512 | 0.0532 | 0.998x [tie] | 0.982x [tie] | 0.983x [tie] | head_major_head_first |

## Scope Guardrail

This is a single-threaded decode-mechanism benchmark. It mirrors PACE's GQA blockwise online-softmax dataflow and AVX-512 BF16 arithmetic, but it does not yet modify SlabPool, exercise its allocator's non-contiguous physical block mapping, use Split-K, or include OpenMP scheduling. A production prototype is justified only if a repeatable co-design signal survives this controlled test.
