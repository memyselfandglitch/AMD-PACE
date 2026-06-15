# AMD PACE

High-performance LLM inference on AMD EPYC CPUs. PACE is a PyTorch C++ extension with custom AVX512 kernels, slab/paged KV cache, fused operators, and a production-ready serving stack. Check out the [GitHub repository](https://github.com/amd/amd-pace) for more information.

PACE achieves **1.6x higher autoregressive** and **3.2x higher speculative-decoding** throughput compared to vLLM on 5th Gen AMD EPYC processors. [More details and technical results here.](https://www.amd.com/en/developer/resources/technical-articles/2026/amd-pace---high-performance-platform-aware-compute-engine.html)

## Highlights

- **SlabPool attention** - CPU-native KV cache and attention backend with O(1) slab allocation, L2-aware block sizing, and a unified dispatcher that picks the optimal kernel path per sequence (GQA decode, multi-token decode, tiled prefill) within one OMP dispatch. Continuous batching, sliding-window, and sink attention go through a single entry point.
- **Inference server** - `pace-server` provides a router/engine serving stack with continuous batching, multi-instance NUMA-aware execution, and built-in metrics. The launcher partitions CPU cores across engine instances and binds memory to the local NUMA node.
- **Paged attention** - vLLM-style paged KV cache on CPU, fully integrated with PACE's serving stack and all supported models.
- **Fused AVX512 kernels** - fused Add+RMSNorm, Add+LayerNorm, RoPE, QKV projections, and a fused MLP kernel (via TPP/libXSMM). Default for all supported models.
- **Broad model support** - Llama (up to 3.3), Qwen2/2.5, Phi3/4, Gemma 3, GPT-J, OPT, and GPT-OSS, all running in BF16 under one operator and backend framework. Adding a new architecture is a single-file effort.
- **Speculative decoding (PARD)** - built-in parallel-draft speculation, up to **5x throughput** over standard autoregressive decoding.

## Requirements

- Linux x86_64 with AVX512F + AVX512_BF16 (AMD Zen4 or newer)
- Python 3.10 – 3.13

## Install

```bash
# 1. CPU PyTorch (the +cpu build is not on PyPI; needs PyTorch's index).
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.12.0+cpu

# 2. amd-pace
pip install amd-pace
```

## Quick example

Inference server (router + engine, OpenAI-compatible endpoint):

```bash
pace-server --server_model meta-llama/Llama-3.1-8B --kv_cache_type SLAB_POOL --serve_type continuous_prefill_first
```

For offline programmatic generation (the `pace.llm.LLMModel` API needs a
tokenizer and an `OperatorConfig` that picks a backend per op), see the
runnable scripts at
[`examples/`](https://github.com/amd/amd-pace/tree/main/examples) --
[`pace_llm_basic.py`](https://github.com/amd/amd-pace/blob/main/examples/pace_llm_basic.py)
is the smallest starting point.

## Support

We welcome feedback, suggestions, and bug reports. Should you have any of these, please kindly file an issue on the PACE GitHub page [here](https://github.com/amd/amd-pace/issues).

## License

AMD PACE is licensed under the MIT License. See the [LICENSE](https://github.com/amd/amd-pace/blob/main/LICENSE) file for details. Third-party notices are in [NOTICE](https://github.com/amd/amd-pace/blob/main/NOTICE).
