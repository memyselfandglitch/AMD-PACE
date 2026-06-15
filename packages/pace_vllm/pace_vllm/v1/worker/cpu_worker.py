# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""PaceWorker: vLLM v1 CPU worker that builds PaceModelRunner.

Thin subclass of `CPUWorker`. `init_device` does three things:

1. `_register_pace_oots()` imports the modules that carry our
   `PluggableLayer.register_oot` / `CustomOp.register_oot` decorators.
   Must fire before model construction so the OOT substitution lands
   on every `QKVParallelLinear(...)` / `RMSNorm(...)` / etc. call.
2. `_install_fused_mlp_pass()` stashes the post-grad pattern matcher
   into `compilation_config.inductor_compile_config` in this
   subprocess -- `FusedMLPPass` captures `torch.ops.pace.*` OpOverload
   handles that are not pickleable across the worker spawn.
3. Delegate to `super().init_device()` for libtcmalloc / OMP / thread
   binding / distributed init / seed, then swap the throwaway
   `CPUModelRunner` super() built for `PaceModelRunner`.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.v1.worker.cpu_worker import CPUWorker

from pace_vllm.v1.worker.cpu_model_runner import PaceModelRunner

logger = init_logger("pace_vllm.v1.worker")

_PACE_OOTS_REGISTERED = False


def _register_pace_oots() -> None:
    """Install pace-vllm OOTs (TPP Linear + ParallelLMHead + RMSNorm).

    Idempotent. Must run before model construction so the
    `PluggableLayer.register_oot` / `CustomOp.register_oot` substitutions
    land on every QKVParallelLinear / RMSNorm / etc. in the model graph.
    """
    global _PACE_OOTS_REGISTERED
    if _PACE_OOTS_REGISTERED:
        return
    import pace_vllm.model_executor.layers.layernorm  # noqa: F401
    import pace_vllm.model_executor.layers.linear  # noqa: F401
    import pace_vllm.model_executor.layers.vocab_parallel_embedding  # noqa: F401

    _PACE_OOTS_REGISTERED = True
    logger.info(
        "pace-vllm: registered OOTs (TPP Linear + ParallelLMHead + JIT RMSNorm)."
    )


def _install_fused_mlp_pass(vllm_config) -> None:
    """Install the pace fused-MLP post-grad pass for this subprocess.

    `FusedMLPPass` captures `torch.ops.pace.*` OpOverload handles that
    are not pickleable; building it in the main process would break
    the worker-spawn pickle. Doing it here is safe because the worker
    has already unpickled `vllm_config`. No-op in eager mode.
    """
    from vllm.config.compilation import CompilationMode

    if vllm_config.compilation_config.mode == CompilationMode.NONE:
        return

    key = "post_grad_custom_post_pass"
    inductor_config = vllm_config.compilation_config.inductor_compile_config
    existing = inductor_config.get(key)
    if existing is not None:
        logger.warning(
            "pace-vllm: post_grad_custom_post_pass already set (%s); "
            "pace FusedMLPPass registration skipped.",
            type(existing).__name__,
        )
        return

    from pace_vllm.compilation.fused_mlp_pass import FusedMLPPass

    inductor_config[key] = FusedMLPPass(vllm_config)
    logger.info("pace-vllm: installed FusedMLPPass into post_grad_custom_post_pass.")


class PaceWorker(CPUWorker):
    """CPU worker that instantiates `PaceModelRunner` instead of `CPUModelRunner`."""

    def init_device(self) -> None:
        logger.info("pace-vllm: PaceWorker.init_device starting.")

        _register_pace_oots()
        _install_fused_mlp_pass(self.vllm_config)

        super().init_device()

        # Swap super()'s throwaway CPUModelRunner for our subclass; weights
        # aren't loaded until load_model, so the discarded instance only
        # costs its in-memory setup.
        self.model_runner: PaceModelRunner = PaceModelRunner(
            self.vllm_config, torch.device("cpu")
        )

        logger.info(
            "pace-vllm: PaceWorker active (worker=%s, model_runner=%s).",
            type(self).__name__,
            type(self.model_runner).__name__,
        )

    def determine_available_memory(self) -> int:
        # vLLM's CPUWorker computes the CPU KV-cache budget (explicit env /
        # CLI value, or auto via `numa_total * memory_fraction - process_RSS`)
        # and returns it, but the engine consumes the return value without
        # writing it back to cache_config. PaceKVCache reads
        # cache_config.kv_cache_memory_bytes, so mirror super's answer there.
        budget = int(super().determine_available_memory())
        if budget <= 0:
            # CPUWorker raises on `<= 0` in its auto path, but the explicit
            # path doesn't guard the same way (a `--kv-cache-memory-bytes=0`
            # would slip through). Catch here so PaceKVCache never has to
            # reason about a non-positive budget.
            raise RuntimeError(
                f"pace-vllm: vLLM returned a non-positive CPU KV cache "
                f"budget ({budget} bytes). Set --kv-cache-memory-bytes or "
                "VLLM_CPU_KVCACHE_SPACE to a positive value."
            )
        self.vllm_config.cache_config.kv_cache_memory_bytes = budget
        return budget
