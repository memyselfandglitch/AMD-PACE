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

- Workloads: `2`
- Decisions: `IKJ=0`, `KIJ=0`, `tie=2`
- Proceed to batched/OpenMP validation only if KIJ earns at least two strict wins.

| head dim (N) | query len (M) | query tiles | KV len (K) | KV blocks | IKJ ms | KIJ ms | KIJ speedup (95% CI) | win rate | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 64 | 64 | 1 | 2048 | 32 | 0.1884 | 0.1887 | 0.998x (0.998-0.999) | 0.0% | tie |
| 64 | 128 | 2 | 2048 | 32 | 0.3753 | 0.3752 | 1.000x (1.000-1.001) | 100.0% | tie |

## Scope Guardrail

This isolates the P*V traversal of pre-tiled operands. It does not include QK^T, online softmax, K/V packing, physical SlabPool layout, GQA batching, or OpenMP scheduling. A KIJ win is evidence to build the next prototype, not yet evidence of faster complete PACE attention.
Head dimensions 64 and 128 are per-head attention dimensions, not complete SLM and LLM model shapes by themselves.
