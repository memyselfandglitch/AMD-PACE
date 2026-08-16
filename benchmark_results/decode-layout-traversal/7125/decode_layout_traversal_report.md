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
- Strict block-major/block-first wins over current baseline: `2/24`
- Strict current-baseline wins over block-major/block-first: `0/24`
- Recommended `head_major_head_first`: `22` workloads
- Recommended `block_major_head_first`: `0` workloads
- Recommended `head_major_block_first`: `0` workloads
- Recommended `block_major_block_first`: `2` workloads

## Workloads

| shape | batch | seq | current ms | layout-only | traversal-only | co-designed | recommendation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| llama_gqa | 1 | 16384 | 5.6412 | 0.938x [head_major_head_first] | 0.984x [tie] | 1.005x [tie] | head_major_head_first |
| llama_gqa | 1 | 2048 | 0.5895 | 0.986x [tie] | 0.993x [tie] | 1.005x [tie] | head_major_head_first |
| llama_gqa | 1 | 512 | 0.1478 | 0.976x [tie] | 0.995x [tie] | 0.979x [tie] | head_major_head_first |
| llama_gqa | 1 | 8192 | 2.7113 | 0.942x [head_major_head_first] | 0.980x [tie] | 0.998x [tie] | head_major_head_first |
| llama_gqa | 4 | 16384 | 23.0582 | 0.965x [tie] | 0.973x [tie] | 0.995x [tie] | head_major_head_first |
| llama_gqa | 4 | 2048 | 2.7300 | 0.970x [tie] | 0.983x [tie] | 0.999x [tie] | head_major_head_first |
| llama_gqa | 4 | 512 | 0.5971 | 1.001x [tie] | 0.995x [tie] | 1.007x [tie] | head_major_head_first |
| llama_gqa | 4 | 8192 | 11.5334 | 0.967x [tie] | 0.991x [tie] | 0.994x [tie] | head_major_head_first |
| mha | 1 | 16384 | 0.8012 | 0.550x [head_major_head_first] | 0.678x [head_major_head_first] | 1.049x [tie] | head_major_head_first |
| mha | 1 | 2048 | 0.0863 | 0.980x [tie] | 0.970x [tie] | 1.047x [tie] | head_major_head_first |
| mha | 1 | 512 | 0.0220 | 1.014x [tie] | 0.976x [tie] | 1.042x [tie] | head_major_head_first |
| mha | 1 | 8192 | 0.3682 | 0.668x [head_major_head_first] | 0.753x [head_major_head_first] | 1.048x [tie] | head_major_head_first |
| mha | 4 | 16384 | 3.7835 | 0.594x [head_major_head_first] | 0.706x [head_major_head_first] | 0.997x [tie] | head_major_head_first |
| mha | 4 | 2048 | 0.3709 | 0.672x [head_major_head_first] | 0.780x [head_major_head_first] | 1.002x [tie] | head_major_head_first |
| mha | 4 | 512 | 0.0878 | 0.968x [tie] | 0.980x [tie] | 0.999x [tie] | head_major_head_first |
| mha | 4 | 8192 | 1.8400 | 0.597x [head_major_head_first] | 0.711x [head_major_head_first] | 1.133x [block_major_block_first] | block_major_block_first |
| slm_gqa | 1 | 16384 | 0.7929 | 0.899x [head_major_head_first] | 0.940x [head_major_head_first] | 1.060x [block_major_block_first] | block_major_block_first |
| slm_gqa | 1 | 2048 | 0.0816 | 0.935x [head_major_head_first] | 0.990x [tie] | 0.954x [tie] | head_major_head_first |
| slm_gqa | 1 | 512 | 0.0519 | 0.958x [tie] | 0.994x [tie] | 0.957x [tie] | head_major_head_first |
| slm_gqa | 1 | 8192 | 0.3384 | 0.955x [tie] | 0.987x [tie] | 0.997x [tie] | head_major_head_first |
| slm_gqa | 4 | 16384 | 3.8831 | 0.814x [head_major_head_first] | 0.964x [tie] | 1.002x [tie] | head_major_head_first |
| slm_gqa | 4 | 2048 | 0.3401 | 0.972x [tie] | 0.988x [tie] | 1.003x [tie] | head_major_head_first |
| slm_gqa | 4 | 512 | 0.0857 | 0.977x [tie] | 0.995x [tie] | 0.988x [tie] | head_major_head_first |
| slm_gqa | 4 | 8192 | 1.9262 | 0.885x [head_major_head_first] | 0.967x [tie] | 1.011x [tie] | head_major_head_first |

## Scope Guardrail

This is a single-threaded decode-mechanism benchmark. It mirrors PACE's GQA blockwise online-softmax dataflow and AVX-512 BF16 arithmetic, but it does not yet modify SlabPool, exercise its allocator's non-contiguous physical block mapping, use Split-K, or include OpenMP scheduling. A production prototype is justified only if a repeatable co-design signal survives this controlled test.
