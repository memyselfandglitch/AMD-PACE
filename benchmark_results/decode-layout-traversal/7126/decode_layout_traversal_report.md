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

- Workloads: `24`
- Strict block-major/block-first wins over current baseline: `4/24`
- Strict current-baseline wins over block-major/block-first: `0/24`
- Recommended `head_major_head_first`: `20` workloads
- Recommended `block_major_head_first`: `0` workloads
- Recommended `head_major_block_first`: `0` workloads
- Recommended `block_major_block_first`: `4` workloads

## Workloads

| shape | batch | seq | current ms | layout-only | traversal-only | co-designed | recommendation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| llama_gqa | 1 | 16384 | 5.6334 | 0.943x [head_major_head_first] | 0.986x [tie] | 1.001x [tie] | head_major_head_first |
| llama_gqa | 1 | 2048 | 0.5937 | 0.996x [tie] | 0.997x [tie] | 1.012x [tie] | head_major_head_first |
| llama_gqa | 1 | 512 | 0.1476 | 0.972x [tie] | 0.996x [tie] | 0.975x [tie] | head_major_head_first |
| llama_gqa | 1 | 8192 | 2.7002 | 0.939x [head_major_head_first] | 0.983x [tie] | 0.992x [tie] | head_major_head_first |
| llama_gqa | 4 | 16384 | 22.9977 | 0.966x [tie] | 0.975x [tie] | 0.995x [tie] | head_major_head_first |
| llama_gqa | 4 | 2048 | 2.7162 | 0.971x [tie] | 0.985x [tie] | 0.996x [tie] | head_major_head_first |
| llama_gqa | 4 | 512 | 0.5986 | 1.004x [tie] | 0.994x [tie] | 1.009x [tie] | head_major_head_first |
| llama_gqa | 4 | 8192 | 11.5042 | 0.973x [tie] | 0.993x [tie] | 0.994x [tie] | head_major_head_first |
| mha | 1 | 16384 | 0.7989 | 0.546x [head_major_head_first] | 0.676x [head_major_head_first] | 1.050x [tie] | head_major_head_first |
| mha | 1 | 2048 | 0.0852 | 0.975x [tie] | 0.968x [tie] | 1.044x [tie] | head_major_head_first |
| mha | 1 | 512 | 0.0221 | 1.024x [tie] | 0.975x [tie] | 1.050x [block_major_block_first] | block_major_block_first |
| mha | 1 | 8192 | 0.3675 | 0.680x [head_major_head_first] | 0.760x [head_major_head_first] | 1.058x [block_major_block_first] | block_major_block_first |
| mha | 4 | 16384 | 3.7888 | 0.603x [head_major_head_first] | 0.706x [head_major_head_first] | 0.998x [tie] | head_major_head_first |
| mha | 4 | 2048 | 0.3690 | 0.663x [head_major_head_first] | 0.784x [head_major_head_first] | 0.999x [tie] | head_major_head_first |
| mha | 4 | 512 | 0.0869 | 0.956x [tie] | 0.978x [tie] | 0.986x [tie] | head_major_head_first |
| mha | 4 | 8192 | 1.8423 | 0.591x [head_major_head_first] | 0.716x [head_major_head_first] | 1.136x [block_major_block_first] | block_major_block_first |
| slm_gqa | 1 | 16384 | 0.7736 | 0.873x [head_major_head_first] | 0.919x [head_major_head_first] | 1.057x [block_major_block_first] | block_major_block_first |
| slm_gqa | 1 | 2048 | 0.0814 | 0.939x [tie] | 0.994x [tie] | 0.960x [tie] | head_major_head_first |
| slm_gqa | 1 | 512 | 0.0517 | 0.965x [tie] | 0.993x [tie] | 0.969x [tie] | head_major_head_first |
| slm_gqa | 1 | 8192 | 0.3351 | 0.952x [head_major_head_first] | 0.980x [tie] | 0.993x [tie] | head_major_head_first |
| slm_gqa | 4 | 16384 | 3.8601 | 0.800x [head_major_head_first] | 0.957x [tie] | 1.000x [tie] | head_major_head_first |
| slm_gqa | 4 | 2048 | 0.3402 | 0.970x [tie] | 0.990x [tie] | 1.001x [tie] | head_major_head_first |
| slm_gqa | 4 | 512 | 0.0856 | 0.974x [tie] | 0.995x [tie] | 0.985x [tie] | head_major_head_first |
| slm_gqa | 4 | 8192 | 1.9034 | 0.876x [head_major_head_first] | 0.958x [tie] | 1.003x [tie] | head_major_head_first |

## Scope Guardrail

This is a single-threaded decode-mechanism benchmark. It mirrors PACE's GQA blockwise online-softmax dataflow and AVX-512 BF16 arithmetic, but it does not yet modify SlabPool, exercise its allocator's non-contiguous physical block mapping, use Split-K, or include OpenMP scheduling. A production prototype is justified only if a repeatable co-design signal survives this controlled test.
