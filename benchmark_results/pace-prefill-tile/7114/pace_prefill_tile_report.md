# Real PACE SlabPool Prefill Tile Summary

- Workloads: `24`
- Current/default tile: `64`
- Strictly improved workloads: `0/24`
- Strict win: >=5% paired median improvement, >=80% pair wins, 95% bootstrap CI above 1, and p95 not worse.

| shape | batch | query | KV | best | recommended | speedup | p95 baseline->best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| llm | 1 | 128 | 2048 | 128 | 64 | 1.017x | 2.654->2.600 ms |
| llm | 1 | 128 | 8192 | 32 | 64 | 1.001x | 10.096->10.060 ms |
| llm | 1 | 128 | 16384 | 32 | 64 | 1.000x | 20.171->20.168 ms |
| llm | 1 | 512 | 2048 | 128 | 64 | 1.003x | 4.006->2.757 ms |
| llm | 1 | 512 | 8192 | 16 | 64 | 1.001x | 10.540->10.527 ms |
| llm | 1 | 512 | 16384 | 64 | 64 | 1.000x | 21.034->21.034 ms |
| llm | 4 | 128 | 2048 | 64 | 64 | 1.000x | 2.799->2.799 ms |
| llm | 4 | 128 | 8192 | 32 | 64 | 1.002x | 10.744->11.643 ms |
| llm | 4 | 128 | 16384 | 32 | 64 | 1.001x | 21.332->21.165 ms |
| llm | 4 | 512 | 2048 | 16 | 64 | 1.005x | 6.352->6.946 ms |
| llm | 4 | 512 | 8192 | 128 | 64 | 1.002x | 39.424->23.853 ms |
| llm | 4 | 512 | 16384 | 64 | 64 | 1.000x | 47.557->47.557 ms |
| slm | 1 | 128 | 2048 | 128 | 64 | 1.003x | 1.534->1.496 ms |
| slm | 1 | 128 | 8192 | 32 | 64 | 1.001x | 5.879->5.879 ms |
| slm | 1 | 128 | 16384 | 16 | 64 | 1.002x | 11.455->11.433 ms |
| slm | 1 | 512 | 2048 | 64 | 64 | 1.000x | 1.545->1.545 ms |
| slm | 1 | 512 | 8192 | 32 | 64 | 1.003x | 5.899->5.876 ms |
| slm | 1 | 512 | 16384 | 64 | 64 | 1.000x | 11.414->11.414 ms |
| slm | 4 | 128 | 2048 | 32 | 64 | 1.003x | 1.551->1.544 ms |
| slm | 4 | 128 | 8192 | 128 | 64 | 1.019x | 5.895->5.788 ms |
| slm | 4 | 128 | 16384 | 128 | 64 | 1.002x | 11.449->11.417 ms |
| slm | 4 | 512 | 2048 | 128 | 64 | 1.010x | 1.656->1.652 ms |
| slm | 4 | 512 | 8192 | 32 | 64 | 1.002x | 6.623->6.786 ms |
| slm | 4 | 512 | 16384 | 32 | 64 | 0.999x | 22.224->13.400 ms |

## Guardrail

This measures the real BF16 SlabPool prefill path, including packing, online softmax, oneDNN BRGeMM, OpenMP dispatch, and output normalization. It is not an end-to-end model-generation benchmark.
