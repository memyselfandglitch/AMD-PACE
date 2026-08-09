#!/bin/bash
exec numactl --physcpubind=0 --membind=0 "/data/scratch/deveshisingh/AMD-PACE/benchmark_results/cpu-pv-crossover-uprof/7113/pv_profile_case" "$@"
