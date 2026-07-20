# Job 7055 latency summary by block size

Source: `results.csv`, using the measured `average_gen_time` in seconds. Each
block-size value is itself the aggregate of five timed generations after two
warm-ups. Lowest and highest identify both latency and requested block size.

Mean and p90 are calculated across the five block-size aggregates (`auto`, 32,
64, 128, 256). P90 is therefore not a per-request tail-latency percentile; the
underlying individual timings were not retained by job 7055.

Mean improvement versus auto is
`(PACE auto latency - mean latency) / PACE auto latency * 100`. Positive means
the mean across tested settings was faster than auto; negative means auto was
faster than that mean.

| Model | Input | Output | Batch | Auto (s) | BS32 (s) | BS64 (s) | BS128 (s) | BS256 (s) | Lowest (s, block) | Highest (s, block) | Mean (s) | P90 (s) | Mean improvement vs auto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct (LLM) | 128 | 128 | 1 | 7.5945 | 7.5478 | 7.6078 | 7.5450 | 7.6356 | 7.5450 (128) | 7.6356 (256) | 7.5861 | 7.6244 | +0.11% |
| Qwen2.5-7B-Instruct (LLM) | 128 | 128 | 4 | 8.3927 | 8.2614 | 8.3621 | 8.3993 | 8.4081 | 8.2614 (32) | 8.4081 (256) | 8.3647 | 8.4046 | +0.33% |
| Qwen2.5-7B-Instruct (LLM) | 512 | 128 | 1 | 8.0828 | 8.0064 | 8.0135 | 8.0371 | 8.0678 | 8.0064 (32) | 8.0828 (auto) | 8.0415 | 8.0768 | +0.51% |
| Qwen2.5-7B-Instruct (LLM) | 512 | 128 | 4 | 10.2788 | 10.1931 | 10.2410 | 10.2584 | 10.2635 | 10.1931 (32) | 10.2788 (auto) | 10.2469 | 10.2727 | +0.31% |
| Qwen2.5-7B-Instruct (LLM) | 2048 | 128 | 1 | 10.5962 | 10.1264 | 10.2661 | 10.2821 | 10.3623 | 10.1264 (32) | 10.5962 (auto) | 10.3266 | 10.5027 | +2.54% |
| Qwen2.5-7B-Instruct (LLM) | 2048 | 128 | 4 | 23.9589 | 23.9934 | 23.9756 | 23.8802 | 23.8661 | 23.8661 (256) | 23.9934 (32) | 23.9349 | 23.9863 | +0.10% |
| Qwen2.5-0.5B (SLM) | 128 | 128 | 1 | 1.4185 | 1.3906 | 1.3991 | 1.4002 | 1.3919 | 1.3906 (32) | 1.4185 (auto) | 1.4001 | 1.4112 | +1.30% |
| Qwen2.5-0.5B (SLM) | 128 | 128 | 4 | 1.6906 | 1.6833 | 1.6928 | 1.6775 | 1.6769 | 1.6769 (256) | 1.6928 (64) | 1.6842 | 1.6919 | +0.38% |
| Qwen2.5-0.5B (SLM) | 512 | 128 | 1 | 1.4845 | 1.5013 | 1.5034 | 1.4988 | 1.4943 | 1.4845 (auto) | 1.5034 (64) | 1.4965 | 1.5026 | -0.81% |
| Qwen2.5-0.5B (SLM) | 512 | 128 | 4 | 1.9415 | 1.9704 | 1.9741 | 1.9572 | 1.9677 | 1.9415 (auto) | 1.9741 (64) | 1.9622 | 1.9726 | -1.07% |
| Qwen2.5-0.5B (SLM) | 2048 | 128 | 1 | 1.9062 | 1.9528 | 1.9454 | 1.9042 | 1.9217 | 1.9042 (128) | 1.9528 (32) | 1.9261 | 1.9498 | -1.04% |
| Qwen2.5-0.5B (SLM) | 2048 | 128 | 4 | 3.3768 | 3.3747 | 3.3821 | 3.3696 | 3.3940 | 3.3696 (128) | 3.3940 (256) | 3.3794 | 3.3893 | -0.08% |
