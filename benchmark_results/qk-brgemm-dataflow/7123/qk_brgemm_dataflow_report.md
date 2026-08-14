# QK^T BF16 BRGeMM Dataflow Comparison

## Experiment

- `pace_fullk`: query tile -> KV tile, reducing the complete head dimension in one BRGeMM, as PACE does today.
- `ikj`: query tile -> K chunk -> KV tile.
- `kij`: K chunk -> query tile -> KV tile.
- All candidates use identical BF16 Q and prepacked K inputs, 64x64 output tiles, and the same oneDNN ukernel API.
- Packing is outside timing; execution is single-threaded and candidate order is randomized.
- Strict winner: >=5% paired median effect, >=80% pair wins, 95% CI excluding one, and non-regressing p95.

## Result

- Workloads: `24`
- Strict IKJ improvements over PACE: `0`
- Strict KIJ improvements over PACE: `0`
- Proceed to a PACE prototype only if IKJ or KIJ strictly beats the full-K baseline in at least two workloads.

| N | M | K | K chunk | PACE ms | IKJ ms | KIJ ms | PACE/IKJ | PACE/KIJ | IKJ/KIJ | recommendation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2048 | 64 | 64 | 32 | 0.0767 | 0.0800 | 0.0799 | 0.959x [tie] | 0.961x [tie] | 1.000x [tie] | pace_fullk |
| 8192 | 64 | 64 | 32 | 0.3083 | 0.3204 | 0.3200 | 0.963x [tie] | 0.962x [tie] | 1.000x [tie] | pace_fullk |
| 16384 | 64 | 64 | 32 | 0.6200 | 0.6445 | 0.6440 | 0.964x [tie] | 0.963x [tie] | 1.000x [tie] | pace_fullk |
| 2048 | 128 | 64 | 32 | 0.1532 | 0.1597 | 0.1599 | 0.960x [tie] | 0.958x [tie] | 0.999x [tie] | pace_fullk |
| 8192 | 128 | 64 | 32 | 0.6155 | 0.6396 | 0.6398 | 0.962x [tie] | 0.961x [tie] | 0.999x [tie] | pace_fullk |
| 16384 | 128 | 64 | 32 | 1.2436 | 1.2936 | 1.2949 | 0.961x [tie] | 0.959x [tie] | 1.000x [tie] | pace_fullk |
| 2048 | 256 | 64 | 32 | 0.3061 | 0.3187 | 0.3194 | 0.961x [tie] | 0.959x [tie] | 0.998x [tie] | pace_fullk |
| 8192 | 256 | 64 | 32 | 1.2322 | 1.2818 | 1.2829 | 0.962x [tie] | 0.960x [tie] | 0.999x [tie] | pace_fullk |
| 16384 | 256 | 64 | 32 | 2.4911 | 2.6144 | 2.6192 | 0.953x [tie] | 0.949x [pace_fullk] | 0.997x [tie] | pace_fullk |
| 2048 | 512 | 64 | 32 | 0.6126 | 0.6372 | 0.6388 | 0.961x [tie] | 0.959x [tie] | 0.998x [tie] | pace_fullk |
| 8192 | 512 | 64 | 32 | 2.4705 | 2.5858 | 2.5936 | 0.955x [tie] | 0.951x [pace_fullk] | 0.997x [tie] | pace_fullk |
| 16384 | 512 | 64 | 32 | 5.2870 | 6.2390 | 6.4664 | 0.849x [pace_fullk] | 0.816x [pace_fullk] | 0.960x [tie] | pace_fullk |
| 2048 | 64 | 128 | 32 | 0.1477 | 0.1593 | 0.1594 | 0.928x [pace_fullk] | 0.927x [pace_fullk] | 0.999x [tie] | pace_fullk |
| 8192 | 64 | 128 | 32 | 0.5943 | 0.6413 | 0.6407 | 0.927x [pace_fullk] | 0.927x [pace_fullk] | 0.999x [tie] | pace_fullk |
| 16384 | 64 | 128 | 32 | 1.2014 | 1.2908 | 1.2916 | 0.932x [pace_fullk] | 0.930x [pace_fullk] | 0.998x [tie] | pace_fullk |
| 2048 | 128 | 128 | 32 | 0.2954 | 0.3195 | 0.3196 | 0.925x [pace_fullk] | 0.924x [pace_fullk] | 0.999x [tie] | pace_fullk |
| 8192 | 128 | 128 | 32 | 1.1895 | 1.2836 | 1.2836 | 0.928x [pace_fullk] | 0.927x [pace_fullk] | 1.000x [tie] | pace_fullk |
| 16384 | 128 | 128 | 32 | 2.4087 | 2.5835 | 2.5819 | 0.931x [pace_fullk] | 0.931x [pace_fullk] | 0.999x [tie] | pace_fullk |
| 2048 | 256 | 128 | 32 | 0.5911 | 0.6370 | 0.6381 | 0.928x [pace_fullk] | 0.926x [pace_fullk] | 0.998x [tie] | pace_fullk |
| 8192 | 256 | 128 | 32 | 2.3843 | 2.5701 | 2.5672 | 0.928x [pace_fullk] | 0.929x [pace_fullk] | 1.002x [tie] | pace_fullk |
| 16384 | 256 | 128 | 32 | 4.8195 | 5.2180 | 5.1938 | 0.924x [pace_fullk] | 0.928x [pace_fullk] | 1.001x [tie] | pace_fullk |
| 2048 | 512 | 128 | 32 | 1.1823 | 1.2769 | 1.2798 | 0.926x [pace_fullk] | 0.924x [pace_fullk] | 0.997x [tie] | pace_fullk |
| 8192 | 512 | 128 | 32 | 4.7767 | 5.1556 | 5.1669 | 0.927x [pace_fullk] | 0.924x [pace_fullk] | 0.998x [tie] | pace_fullk |
| 16384 | 512 | 128 | 32 | 10.2143 | 11.2957 | 11.9040 | 0.904x [pace_fullk] | 0.855x [pace_fullk] | 0.939x [ikj] | pace_fullk |

## Scope Guardrail

The split-K candidates are faithful tiled translations of scalar IKJ/KIJ, but splitting K adds BRGeMM invocations relative to PACE. That overhead is intentionally included because a useful ordering must improve the existing full-K implementation, not only beat another split-K candidate. This benchmark excludes softmax, P*V, physical SlabPool layout, GQA batching, and OpenMP scheduling.
