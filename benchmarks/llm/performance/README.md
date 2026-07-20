# Benchmarking Large Language Models

## Contents
1. [Scenarios](#scenarios)
    1. [Offline](#offline)
2. [Benchmarking Infrastructure](#benchmarking-infrastructure)
    1. [Data Generator](#data-generator)
    2. [System Metrics](#system-metrics)
    3. [Token Metrics](#token-metrics)

## Scenarios

### Offline

This document provides an overview of the benchmarking process for language models using the `benchmark_llm_offline.py` script. The script benchmarks the offline performance (throughput and latency) of language models using different frameworks, with different configurations and data types.

#### Configuration

The benchmarking process is configured using a JSON file. Below is an example configuration file:

```json
{
    "frameworks": [
        "pace",
        "hf"
    ],
    "model_args": {
        "model_name": "meta-llama/Llama-3.1-8B",
        "dtype": "bf16",
        "llm_operators": {
            "Norm": "NATIVE",
            "QKVProjection": "TPP",
            "Attention" : "JIT",
            "OutProjection": "TPP",
            "MLP": "TPP",
            "LMHead": "TPP"
        },
        "spec_config": {
            "model_name": "amd/PARD-Llama-3.2-1B",
            "num_speculated_tokens": 12
        }
    },
    "use_real_data": true,
    "generation_args": {
        "input_tokens": 128,
        "output_tokens": 128,
        "batch_size": 1,
        "kv_cache_type": "BMC",
        "do_sample": false,
        "manual_seed": 0
    },
    "warmup_runs": 2,
    "num_runs": 5,
    "verbose": true,
    "output_dir": "./benchmark_results",
    "token_metrics": {
        "time_to_first_token": false,
        "time_per_tokens": false
    },
    "system_metrics": false
}
```

#### Configuration Parameters

- `frameworks`: List of frameworks to benchmark (e.g., `"pace"`, `"hf"`, `"vllm"`, `"vllm_zentorch"`, `"zentorch"`).
- `model_args`: Dictionary of model arguments, including:
    - `model_name`: Name of the model to benchmark (e.g., `"facebook/opt-125m"`).
    - `dtype`: Data type to use (e.g., `"bf16"`).
    - `llm_operators`: Dictionary specifying operator implementations (e.g., `"Norm": "NATIVE"`).
- `use_real_data`: Boolean indicating whether to use real data from a dataset.
- `generation_args`: Dictionary of generation parameters, including:
    - `input_tokens`: Number of input tokens.
    - `output_tokens`: Number of output tokens.
    - `batch_size`: Batch size for benchmarking.
    - `kv_cache_type`: Type of key-value cache (e.g., `"BMC"`).
    - `do_sample`: Boolean indicating whether to use sampling.
    - `manual_seed`: Random seed for reproducibility.
- `warmup_runs`: Number of warm-up runs before benchmarking.
- `num_runs`: Number of benchmark runs.
- `verbose`: Boolean indicating whether to print detailed logs.
- `output_dir`: Directory to save benchmark results. Output files are named using the pattern `{frameworks}_{model}_{dtype}_bs{batch}_it{input}_nt{output}_{timestamp}_results.json`, where `{frameworks}` is the list of frameworks joined by `-` and `{timestamp}` is `YYYYMMDD_HHMMSS` to ensure uniqueness across runs.
- `token_metrics`: Dictionary specifying token-level metrics to collect (e.g., `time_to_first_token`, `time_per_tokens`).
- `system_metrics`: Boolean indicating whether to collect system-level metrics.

NOTE:
1. Either `verbose` or `output_dir` should be set, otherwise the script will not run.

#### Usage
Install required dependencies:

```bash
pip install -r requirements.txt
```

> Note: While benchmarking for ZenTorch, vLLM, or vLLM+ZenTorch, it's the user's responsibility to ensure that the respective frameworks are installed and properly configured with all their dependencies. For `vllm_zentorch`, both `vllm` and `zentorch` must be installed; vLLM will automatically detect and use the zentorch platform plugin.

To run the benchmarking script, use the following command:

```bash
python benchmark_llm_offline.py --config /path/to/config.json
```

### Decode block-size sweep

`block_size_sweep.py` runs isolated SlabPool experiments across model class,
prompt length, decode length, batch size, and KV block size. `null` in the
`block_sizes` axis means the existing L2-based auto-tuner; explicit values set
`SLAB_BLOCK_SIZE` for PACE (`PACE_VLLM_SLAB_BLOCK_SIZE` for
`vllm_zentorch`). Every point uses `SLAB_POOL`, greedy decoding, and a fresh
process so an earlier model initialization cannot leak into another point.

Start by inspecting the 60-point SLM/LLM matrix:

```bash
python block_size_sweep.py run \
  --spec block_size_sweep_quick.json \
  --output-dir benchmark_results/block_size_quick \
  --dry-run
```

Run it on an otherwise idle, performance-configured AMD host:

```bash
python block_size_sweep.py run \
  --spec block_size_sweep_quick.json \
  --output-dir benchmark_results/block_size_quick \
  --keep-going
```

Completed points are reused when the same command is restarted. Pass `--rerun`
to replace them. Raw JSON and logs are retained per point, while `results.csv`
is updated after every success. To select per-workload winners:

```bash
python block_size_sweep.py summarize \
  --results benchmark_results/block_size_quick/results.csv \
  --min-margin-pct 3
```

The summary marks a winner stable only if its output-token throughput is at
least 3% above the runner-up. Treat other cells as ties and prefer `auto` (or
the larger block if reducing allocator metadata matters). Before encoding an
auto-tuning rule, repeat stable candidates in randomized order and add prompt
lengths near observed crossover points. Useful second-stage axes are input
lengths `[64, 128, 256, 512, 1024, 2048, 4096, 8192]`, batch sizes
`[1, 2, 4, 8]`, and output lengths `[32, 128, 512]`. Keep CPU affinity, NUMA
placement, thread counts, memory policy, model revision, and machine firmware
fixed and record them alongside the output directory.

#### AMD cluster / Slurm submission

The supplied cluster policy requires every workload to run through Slurm and
forbids CPU-only jobs in GPU queues. The cluster document names GPU partitions
but does not identify the CPU partition, so first inspect the live scheduler:

```bash
sinfo -o '%P %a %l %D %C %G'
```

Choose a partition with CPU capacity and no GPU allocation, then submit the
provided single-node EPYC job from the repository root. Replace
`CPU_PARTITION` with that partition's exact name; do **not** use `GPU` or
`jobgn01` for this benchmark.

```bash
sbatch --partition=CPU_PARTITION \
  benchmarks/llm/performance/slurm_block_size_sweep.sh
```

If the PACE Python environment is not already active, pass its activation
script. A venv example is:

```bash
sbatch --partition=CPU_PARTITION \
  --export=ALL,PACE_ACTIVATE=/path/to/venv/bin/activate \
  benchmarks/llm/performance/slurm_block_size_sweep.sh
```

Monitor or cancel the job with `squeue -j JOB_ID` and `scancel JOB_ID`. The job
requests one exclusive node, 96 physical cores, 512 GB RAM, and 24 hours,
writes Slurm logs in the submission directory, pins OpenMP workers to cores,
and stores models plus measurements under `/scratch/$USER/pace-block-size`.
`--exclusive` prevents Slurm from placing another job on the node even though
the benchmark requests fewer logical CPUs than the machine exposes. At startup
the script also queries `squeue` for the assigned node and aborts if it sees a
different running allocation. To check manually after the job starts:

```bash
NODE=$(squeue -h -j JOB_ID -o '%N')
squeue -w "$NODE" -t RUNNING -o '%.18i %.12u %.12P %.10T %.6C %R'
scontrol show job JOB_ID | grep -E 'NodeList=|Shared='
```

Only the benchmark job should appear in the node-filtered `squeue` output.
Slurm exclusivity does not stop kernel threads, system services, or a process
improperly launched outside Slurm, so the script records the 20 busiest
processes in its log before benchmarking. If unexplained CPU use remains,
retain the log and ask the administrator rather than killing another user's
processes. Partition `OverSubscribe` policy can override job sharing options;
inspect it with `scontrol show partition CPU_PARTITION` and confirm the
exclusive request with the cluster administrator if another allocation ever
appears.

Scratch is cleaned weekly on this cluster, so copy the result directory to
persistent storage after the run. If the CPU partition exposes fewer than 96
cores or has different memory/time limits, adjust the corresponding `#SBATCH`
lines before submitting.

## Benchmarking Infrastructure

### Data Generator

The `data.py` file provides utilities for generating input data for benchmarking. It includes a class `BenchMarkDataGenerator` that can generate random input data or use real data from a dataset.

The `BenchMarkDataGenerator` class is used to generate input data for benchmarking the language model. It can generate random input data or use real data from a specified dataset.

If using real data, it loads and tokenizes the dataset using the provided tokenizer, and is saved in memory according to the batch size and input tokens. If using random data, it picks random tokens based on the input tokens and batch size from the tokenizer and the same data will be reused across runs. Once the data is generated, it can be accessed using the `next` method, and it will infinitely loop through the data, giving a random batch of input data each time.

NOTE: The data generator cannot be used for evaluation purposes, as it does not guarantee the same data across runs and is not representative of the dataset.

#### Example Usage

```python
from transformers import AutoTokenizer
from data import BenchMarkDataGenerator

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

# Create data generator
data_gen = BenchMarkDataGenerator(
    tokenizer=tokenizer,
    input_tokens=128,
    batch_size=8,
    use_real_data=True
)

# Get data for benchmarking
for _ in range(N):
    data = next(data_gen)
    ...
```

### System Metrics

The SystemMonitor class is designed to capture real-time CPU and RAM usage in a background thread during benchmarking. This can be useful for understanding the system resource consumption of your benchmarks. It includes methods to start and stop monitoring and provides a usage history over time. The data is collected using psutil library, and ran in the background thread to avoid blocking the main thread.

#### Example Usage

```python
from metrics import SystemMonitor

monitor = SystemMonitor(interval=1)
monitor.start()

# ... run the benchmark ...

monitor.stop()
metrics = monitor.get_history()
print("CPU usage:", metrics.cpu_usage)
print("RAM usage (MB):", metrics.ram_usage)
print("Peak RAM usage (MB):", metrics.peak_ram_usage)
```

NOTE:
1. The system monitor should be off when benchmarking for absolute performance, as it can affect the performance from the overhead of collecting metrics.
2. RAM usage is calculated by the difference between the current memory usage and the memory usage at the start of monitoring. If background processes are running, the RAM usage may not be accurate.


### Token Metrics

1. FirstTokenTimerStreamer: The FirstTokenTimerStreamer class is designed to calculate the time taken for the first time token to be generated during the token generation process.

    #### Example Usage

    ```python
    from metrics import FirstTokenTimerStreamer

    first_token_streamer = FirstTokenTimerStreamer()

    start_time = time.time()
    out = model.generate(inputs, streamer=first_token_streamer, **gen_kwargs)

    ttft = streamer.end_time_first_token - start_time
    ```

2. TokenLatencyStreamer: The TokenLatencyStreamer class is designed to calculate the time taken for each token to be generated during the token generation process.

    ##### Example Usage

    ```python
    from metrics import TokenLatencyStreamer

    token_latency_streamer = TokenLatencyStreamer()

    start_time = time.time()
    out = model.generate(inputs, streamer=token_latency_streamer, **gen_kwargs)

    time_per_tokens = streamer.time_per_tokens()
    ```
