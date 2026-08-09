# Square GEMM Loop-Order Threshold

- Sizes tested: `15`
- Paired rounds per size: `50`
- Strict effect threshold: `5.0%`
- Minimum pair-win rate: `80.0%`
- Decisions: `ikj=3`, `kij=8`, `tie=4`

## Threshold Result

Among strict decisions, a candidate crossover lies between N=160 and N=384: kij wins below the interval and ikj wins above it. This statement applies only to the sampled square FP32 GEMM sizes.

## Paired Results

| N | A+B+C KiB | all-six fastest | ikj ms | kij ms | ikj speedup | 95% CI | ikj win rate | decision |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 12.0 | kij | 0.009525 | 0.005603 | 0.588x | 0.578-0.591 | 0.0% | kij |
| 48 | 27.0 | kij | 0.025433 | 0.017436 | 0.688x | 0.686-0.691 | 2.0% | kij |
| 64 | 48.0 | kij | 0.019449 | 0.015784 | 0.812x | 0.811-0.813 | 0.0% | kij |
| 80 | 75.0 | kij | 0.059710 | 0.032754 | 0.548x | 0.547-0.550 | 0.0% | kij |
| 96 | 108.0 | kij | 0.091809 | 0.061007 | 0.665x | 0.664-0.666 | 0.0% | kij |
| 112 | 147.0 | kij | 0.130593 | 0.096501 | 0.740x | 0.738-0.741 | 0.0% | kij |
| 128 | 192.0 | kij | 0.177548 | 0.140432 | 0.792x | 0.790-0.794 | 0.0% | kij |
| 160 | 300.0 | kij | 0.304681 | 0.269563 | 0.885x | 0.883-0.885 | 0.0% | kij |
| 192 | 432.0 | kij | 0.480226 | 0.461397 | 0.961x | 0.960-0.962 | 0.0% | tie |
| 224 | 588.0 | kij | 0.854316 | 0.832763 | 0.975x | 0.974-0.976 | 2.0% | tie |
| 256 | 768.0 | ikj | 1.204505 | 1.223323 | 1.015x | 1.014-1.016 | 100.0% | tie |
| 320 | 1200.0 | ikj | 2.327060 | 2.383686 | 1.025x | 1.023-1.026 | 100.0% | tie |
| 384 | 1728.0 | ikj | 3.810833 | 4.129554 | 1.083x | 1.083-1.084 | 100.0% | ikj |
| 448 | 2352.0 | ikj | 5.892238 | 6.770134 | 1.148x | 1.147-1.150 | 100.0% | ikj |
| 512 | 3072.0 | ikj | 8.962322 | 10.044116 | 1.120x | 1.118-1.123 | 100.0% | ikj |

## Guardrail

An `ikj` speedup is paired `kij latency / ikj latency`; values above 1 favor ikj.
Strict winners require the minimum median effect, pair-win rate, a bootstrap CI excluding 1, and non-regressing p95.
This is unblocked, single-threaded, square FP32 GEMM; it does not establish an attention-kernel threshold.
