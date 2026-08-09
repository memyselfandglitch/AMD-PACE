# P*V Loop-Crossover Hypothesis

## Pre-Registered Claim

- `M=1`: `ikj` and `kij` are equivalent, so the expected decision is `tie`.
- Long-context `M=16..128, N>=2048`: `kij` is expected to win by reusing each V row across query rows.
- Other cells are exploratory and locate where the locality advantage appears or reverses.
- Evidence against a universal loop order requires at least two strict workload wins for each order.

A strict winner requires at least 5% paired median improvement, 80% pair wins, a 95% bootstrap CI excluding 1, and no post-hoc threshold changes.

## Result

- Claim-scored workloads: `34`
- Matches: `23`
- Counterexamples: `11`
- Strict decisions across the full matrix: `kij=57`, `ikj=12`, `tie=31`
- Universal-order test: `rejected; the preferred order is workload-dependent`

## Trend By Query Length

| query_len | workloads | median kij speedup | kij wins | ties | ikj wins |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | 0.976x | 0 | 10 | 0 |
| 2 | 10 | 1.311x | 6 | 4 | 0 |
| 4 | 10 | 1.200x | 8 | 2 | 0 |
| 8 | 10 | 1.251x | 8 | 2 | 0 |
| 16 | 10 | 1.124x | 6 | 4 | 0 |
| 32 | 10 | 1.091x | 5 | 1 | 4 |
| 64 | 10 | 1.041x | 4 | 5 | 1 |
| 128 | 10 | 1.181x | 9 | 0 | 1 |
| 256 | 10 | 1.162x | 6 | 2 | 2 |
| 512 | 10 | 1.104x | 5 | 1 | 4 |

## Counterexamples

| shape | M | N | expected | observed | speedup | 95% CI | win rate |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| llm | 16 | 8192 | kij | tie | 1.038x | 1.037-1.041 | 1.000 |
| llm | 16 | 16384 | kij | tie | 1.040x | 1.037-1.041 | 1.000 |
| llm | 32 | 2048 | kij | ikj | 0.852x | 0.851-0.853 | 0.000 |
| llm | 32 | 8192 | kij | ikj | 0.838x | 0.837-0.840 | 0.000 |
| llm | 32 | 16384 | kij | ikj | 0.840x | 0.839-0.841 | 0.000 |
| llm | 64 | 8192 | kij | tie | 0.986x | 0.985-0.986 | 0.000 |
| llm | 64 | 16384 | kij | ikj | 0.929x | 0.928-0.930 | 0.000 |
| llm | 128 | 8192 | kij | ikj | 0.883x | 0.883-0.884 | 0.000 |
| slm | 64 | 2048 | kij | tie | 1.031x | 1.030-1.033 | 1.000 |
| slm | 64 | 8192 | kij | tie | 0.981x | 0.981-0.982 | 0.000 |
| slm | 64 | 16384 | kij | tie | 1.044x | 1.040-1.045 | 0.667 |

## Scope Guardrail

This is a single-head FP32 microbenchmark with random dense values. It tests P*V loop locality, not complete attention or end-to-end PACE latency.
