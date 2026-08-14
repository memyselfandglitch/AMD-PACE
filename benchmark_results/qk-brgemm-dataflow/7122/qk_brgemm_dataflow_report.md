# QK^T BF16 BRGeMM Dataflow Comparison

## Experiment

- `pace_fullk`: query tile -> KV tile, reducing the complete head dimension in one BRGeMM, as PACE does today.
- `ikj`: query tile -> K chunk -> KV tile.
- `kij`: K chunk -> query tile -> KV tile.
- All candidates use identical BF16 Q and prepacked K inputs, 64x64 output tiles, and the same oneDNN ukernel API.
- Packing is outside timing; execution is single-threaded and candidate order is randomized.
- Strict winner: >=5% paired median effect, >=80% pair wins, 95% CI excluding one, and non-regressing p95.

## Result

- Workloads: `2`
- Strict IKJ improvements over PACE: `0`
- Strict KIJ improvements over PACE: `0`
- Proceed to a PACE prototype only if IKJ or KIJ strictly beats the full-K baseline in at least two workloads.

| N | M | K | K chunk | PACE ms | IKJ ms | KIJ ms | PACE/IKJ | PACE/KIJ | IKJ/KIJ | recommendation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2048 | 64 | 64 | 32 | 0.1947 | 0.2016 | 0.2018 | 0.966x [tie] | 0.965x [tie] | 0.999x [tie] | pace_fullk |
| 2048 | 128 | 64 | 32 | 0.1565 | 0.1618 | 0.1624 | 0.967x [tie] | 0.965x [tie] | 0.998x [tie] | pace_fullk |

## Scope Guardrail

The split-K candidates are faithful tiled translations of scalar IKJ/KIJ, but splitting K adds BRGeMM invocations relative to PACE. That overhead is intentionally included because a useful ordering must improve the existing full-K implementation, not only beat another split-K candidate. This benchmark excludes softmax, P*V, physical SlabPool layout, GQA batching, and OpenMP scheduling.
