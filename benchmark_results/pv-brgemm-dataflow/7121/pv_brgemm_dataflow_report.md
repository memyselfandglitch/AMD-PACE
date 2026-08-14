# Tiled BF16 BRGeMM Dataflow Comparison

## Experiment

- Candidate `IKJ`: query tile -> KV block -> dimension tile.
- Candidate `KIJ`: KV block -> query tile -> dimension tile.
- Query tile and KV block are fixed at `64`.
- Both candidates use the same oneDNN BF16 BRGeMM ukernel.
- Operands are pre-tiled identically outside the timed region.
- Execution is single-threaded and candidate order is randomized.
- Strict winner: >=5% paired median effect, >=80% pair wins, 95% CI excluding one, and non-regressing p95.

## Result

- Workloads: `24`
- Decisions: `IKJ=0`, `KIJ=0`, `tie=24`
- Proceed to batched/OpenMP validation only if KIJ earns at least two strict wins.

| head dim (N) | query len (M) | query tiles | KV len (K) | KV blocks | IKJ ms | KIJ ms | KIJ speedup (95% CI) | win rate | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 64 | 64 | 1 | 2048 | 32 | 0.1880 | 0.1881 | 1.000x (0.999-1.000) | 48.3% | tie |
| 64 | 64 | 1 | 8192 | 128 | 0.3009 | 0.3010 | 1.000x (1.000-1.001) | 48.3% | tie |
| 64 | 64 | 1 | 16384 | 256 | 0.6017 | 0.6021 | 1.000x (0.999-1.000) | 40.0% | tie |
| 64 | 128 | 2 | 2048 | 32 | 0.1508 | 0.1510 | 0.999x (0.998-0.999) | 25.0% | tie |
| 64 | 128 | 2 | 8192 | 128 | 0.6039 | 0.6054 | 0.998x (0.997-0.998) | 11.7% | tie |
| 64 | 128 | 2 | 16384 | 256 | 1.2054 | 1.2082 | 1.000x (0.998-1.000) | 40.0% | tie |
| 64 | 256 | 4 | 2048 | 32 | 0.3007 | 0.3006 | 1.000x (1.000-1.001) | 65.0% | tie |
| 64 | 256 | 4 | 8192 | 128 | 1.2087 | 1.2099 | 0.999x (0.999-1.000) | 36.7% | tie |
| 64 | 256 | 4 | 16384 | 256 | 2.4201 | 2.4190 | 1.001x (0.999-1.002) | 51.7% | tie |
| 64 | 512 | 8 | 2048 | 32 | 0.6030 | 0.6042 | 0.998x (0.998-0.999) | 25.0% | tie |
| 64 | 512 | 8 | 8192 | 128 | 2.4143 | 2.4230 | 0.996x (0.994-0.998) | 8.3% | tie |
| 64 | 512 | 8 | 16384 | 256 | 4.8635 | 4.8903 | 0.995x (0.994-0.995) | 1.7% | tie |
| 128 | 64 | 1 | 2048 | 32 | 0.1500 | 0.1502 | 0.999x (0.999-1.000) | 38.3% | tie |
| 128 | 64 | 1 | 8192 | 128 | 0.6040 | 0.6039 | 1.000x (1.000-1.001) | 60.0% | tie |
| 128 | 64 | 1 | 16384 | 256 | 1.2084 | 1.2087 | 1.000x (0.999-1.000) | 40.0% | tie |
| 128 | 128 | 2 | 2048 | 32 | 0.3018 | 0.3020 | 1.000x (0.999-1.000) | 33.3% | tie |
| 128 | 128 | 2 | 8192 | 128 | 1.2027 | 1.2029 | 0.999x (0.997-1.000) | 38.3% | tie |
| 128 | 128 | 2 | 16384 | 256 | 2.4115 | 2.4136 | 0.999x (0.998-1.001) | 43.3% | tie |
| 128 | 256 | 4 | 2048 | 32 | 0.6003 | 0.6007 | 0.999x (0.998-1.000) | 33.3% | tie |
| 128 | 256 | 4 | 8192 | 128 | 2.4095 | 2.4137 | 0.998x (0.997-1.000) | 35.0% | tie |
| 128 | 256 | 4 | 16384 | 256 | 4.8343 | 4.8395 | 0.999x (0.998-0.999) | 28.3% | tie |
| 128 | 512 | 8 | 2048 | 32 | 1.2010 | 1.2030 | 0.998x (0.997-0.999) | 31.7% | tie |
| 128 | 512 | 8 | 8192 | 128 | 4.8304 | 4.8355 | 0.999x (0.999-1.000) | 28.3% | tie |
| 128 | 512 | 8 | 16384 | 256 | 9.7108 | 9.7485 | 0.996x (0.995-0.997) | 3.3% | tie |

## Scope Guardrail

This isolates the P*V traversal of pre-tiled operands. It does not include QK^T, online softmax, K/V packing, physical SlabPool layout, GQA batching, or OpenMP scheduling. A KIJ win is evidence to build the next prototype, not yet evidence of faster complete PACE attention.
Head dimensions 64 and 128 are per-head attention dimensions, not complete SLM and LLM model shapes by themselves.
