# pace-vllm

vLLM platform plugin for AMD PACE. Once installed, vLLM auto-discovers it and routes its CPU worker through PACE's kernels and KV cache, with no changes to your vLLM scripts. Check out the [GitHub repository](https://github.com/amd/amd-pace) for more information.

The PACE vLLM plugin brings PACE's CPU optimizations to vLLM with no application code changes, retaining **~95% of standalone PACE efficiency** and delivering **~1.3x the performance of native vLLM 0.21** on 5th Gen AMD EPYC processors. [More details and technical results here.](https://www.amd.com/en/developer/resources/technical-articles/2026/amd-pace-integrates-with-vllm.html)

## What it does

`pace-vllm` registers PACE as a vLLM CPU platform via the `vllm.platform_plugins` entry point. The plugin replaces vLLM's stock CPU worker, attention backend, KV cache, and Linear/RMSNorm layers with PACE equivalents; in compile mode it also installs a post-grad pattern matcher that fuses gated/ungated MLP blocks into a single libxsmm call.

## Highlights

- **Drop-in plugin** - no changes to your vLLM serve script; the `vllm.platform_plugins` entry point is discovered automatically.
- **SlabPool KV cache** - one slab per attention layer, owned by PACE, with sliding-window and sink-attention support.
- **Fused MLP pass** - gated SwiGLU/GeGLU and ungated fc1->act->fc2 MLPs (silu / gelu-tanh / gelu-exact / relu) are rewritten into a single `pace::libxsmm_fused_mlp` call under compile mode.

## Requirements

- Linux x86_64 with AVX512F + AVX512_BF16 (AMD Zen4 / EPYC 5th Gen or newer)
- Python 3.10 – 3.13
- vLLM 0.21.x (CPU build)

## Install

```bash
# 1. vLLM CPU build (pace-vllm is a plugin; it no-ops without vllm).
pip install https://github.com/vllm-project/vllm/releases/download/v0.21.0/vllm-0.21.0+cpu-cp38-abi3-manylinux_2_34_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cpu

# 2. pace-vllm
pip install pace-vllm
```

## Quick example

CLI (vLLM auto-discovers the plugin):

```bash
vllm serve meta-llama/Llama-3.1-8B
```

Python:

```python
from vllm import LLM, SamplingParams


def main() -> None:
    llm = LLM(model="meta-llama/Llama-3.1-8B", dtype="bfloat16")
    out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=8))
    print(out[0].outputs[0].text)


if __name__ == "__main__":
    main()
```

> The `if __name__ == "__main__":` guard is **required**: vLLM v1's engine spawns a subprocess for the worker, and without the guard the subprocess re-imports the script and recursively spawns until the OS refuses.

## Support

We welcome feedback, suggestions, and bug reports. Should you have any of these, please kindly file an issue on the PACE GitHub page [here](https://github.com/amd/amd-pace/issues).

## License

pace-vllm is licensed under the MIT License. See the [LICENSE](https://github.com/amd/amd-pace/blob/main/LICENSE) file for details. Third-party notices are in [NOTICE](https://github.com/amd/amd-pace/blob/main/NOTICE).
