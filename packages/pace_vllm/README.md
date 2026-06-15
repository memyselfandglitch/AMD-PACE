# pace-vllm

vLLM platform plugin for AMD **PACE** (Platform Aware Compute Engine) on CPU.

Drop-in: install it next to vLLM, and `vllm serve` automatically routes through PACE — same vLLM CLI and Python API, PACE's kernels and KV cache underneath. No changes to your existing scripts.

## What you get

- **Drop-in** — `vllm.platform_plugins` entry point is auto-discovered at startup; nothing to wire up.
- **PACE kernels for free** — vLLM's CPU attention backend, KV cache, and Linear / RMSNorm layers are transparently replaced with their PACE equivalents (`pace::rmsnorm`, `pace::libxsmmlinear_plain`, `SlabPool`-backed KV).
- **Fused MLP under compile mode** — gated SwiGLU / GeGLU and ungated `fc1->act->fc2` MLPs are rewritten into a single `pace::libxsmm_fused_mlp` call.
- **Self-contained** — bundles its own `libpace_cpp.so`, so you don't need `amd-pace` installed alongside.

## Contents

* [Requirements](#requirements)
* [Install](#install)
* [Usage](#usage)
* [What works](#what-works)
* [What doesn't (yet)](#what-doesnt-yet)
* [Tests](#tests)
* [Internals (deep dive)](docs/Internals.md)
* [License](#license)

## Requirements

- Linux x86_64 with AVX512F + AVX512_BF16 (AMD Zen4 / EPYC 5th Gen or newer)
- Python 3.10 – 3.13
- vLLM 0.21.x (CPU build) — see [docs/Internals.md](docs/Internals.md#vllm-version-compatibility) for the exact range and how it's enforced.

## Install

### From PyPI (recommended)

1. **Create a Python 3.12 env.** Either `uv` (vLLM's recommendation) or `conda` (PACE's recommendation) works — pick whichever you prefer.

   With **uv**:
   ```bash
   uv venv pace-vllm-env --python 3.12
   source pace-vllm-env/bin/activate
   ```

   With **conda** (use [miniforge](https://conda-forge.org/miniforge/)):
   ```bash
   conda create -n pace-vllm-env python=3.12 -y
   conda activate pace-vllm-env
   ```

   > pace-vllm supports Python 3.10 – 3.13; 3.12 is the most thoroughly exercised version.

2. **Install vLLM (CPU build).** The `+cpu` wheel isn't on PyPI, so pull it from vLLM's GitHub release directly:

   ```bash
   pip install --extra-index-url https://download.pytorch.org/whl/cpu \
     https://github.com/vllm-project/vllm/releases/download/v0.21.0/vllm-0.21.0+cpu-cp38-abi3-manylinux_2_34_x86_64.whl
   ```

3. **Install pace-vllm:**

   ```bash
   pip install pace-vllm
   ```

### From source (developers)

1. Create a Python 3.12 env with miniforge. Install miniforge from [here](https://conda-forge.org/miniforge/), then:

    ```bash
    conda create -n pace-vllm-env-py3.12 python=3.12 -y
    conda activate pace-vllm-env-py3.12
    ```

2. Build and install pace-vllm:

    ```bash
    cd amd-pace
    pip install -r packages/pace_vllm/build_requirements.txt
    pip install --no-build-isolation -v packages/pace_vllm
    ```

> **Do not use editable install** (`pip install -e .`). It skips the CMake step that builds and copies `libpace_cpp.so` into the package — the resulting install will fail to load the native library at runtime. Always use the `pip install --no-build-isolation` flow above.

## Usage

vLLM auto-discovers pace-vllm at startup; just run vLLM normally.

CLI:

```bash
LD_PRELOAD="${CONDA_PREFIX}/lib/libtcmalloc.so" \
  numactl --physcpubind=0-95 --membind=0 \
  VLLM_CPU_KVCACHE_SPACE=350 \
  vllm serve meta-llama/Llama-3.1-8B
```

Python:

```python
from vllm import LLM, SamplingParams


def main() -> None:
    llm = LLM(model="meta-llama/Llama-3.1-8B", dtype="bfloat16")
    out = llm.generate(
        ["The capital of France is"], SamplingParams(max_tokens=8)
    )
    print(out[0].outputs[0].text)


if __name__ == "__main__":
    main()
```

> The `if __name__ == "__main__":` guard is **required**: vLLM v1's engine spawns a subprocess for the worker, and without the guard the subprocess re-imports the script and recursively spawns until the OS refuses.

You'll see `Loading plugin pace ... pace-vllm: plugin active, PacePlatform registered.` in the startup logs — that's confirmation it's live. The full annotated log breakdown lives in [docs/Internals.md](docs/Internals.md#startup-log-lines).

To pin discovery to pace-vllm only:

```bash
VLLM_PLUGINS=pace vllm serve <model>
```

To disable pace-vllm and run pure stock vLLM:

```bash
VLLM_PLUGINS="" vllm serve <model>
```

## What works

- **Models** — Llama 3.x family (primary target; e.g. `meta-llama/Llama-3.1-8B`, `meta-llama/Llama-3.2-1B`), Gemma 3 (sliding window + full attention mix), GPT-OSS (attention sinks) etc with [Slab Attention](../../docs/SlabAttention.md) backend.
- **MLP fusion** (compile mode only) — gated SwiGLU / GeGLU and ungated `fc1->act->fc2`, across silu / gelu-tanh / gelu-exact / relu.
- **KV sharing across layers**.
- **vLLM preemption** — slab K/V resets when vLLM frees blocks; the sequence re-prefills from token 0 on resume.
- **Decoder-only attention**.

## What doesn't (yet)

- Encoder / cross / encoder-only attention.
- vLLM prefix caching (force-disabled at startup).
- MLA, FP8 KV, BNBA quantization.
- Multi-process tensor / pipeline parallel (one worker per instance).
- AMX instruction path — PACE's libXSMM kernels fail on AMX. On AMX-enabled CPUs, set `LIBXSMM_TARGET=cpx` to force the AVX-512 (CPX) path and disable AMX.

## Tests

To run all the unit tests:

```
cd packages/pace_vllm/tests
python -m unittest [-v]
```

To run a specific test:

```
cd packages/pace_vllm/tests
python -m unittest kv_cache.test_kv_cache.TestPaceKVCacheSpec.test_uniform_factory_replicates_num_blocks
```

## License

MIT. Copyright (c) 2026 Advanced Micro Devices, Inc. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for third-party notices.
