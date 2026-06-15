# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""PacePlatform: vLLM CPU platform for PACE.

Subclass of `CpuPlatform` that swaps the vLLM worker class to `PaceWorker`,
routes attention to `PaceAttentionBackend`, and force-disables vLLM prefix
caching (`PaceAttentionImpl` does not honor `num_computed_tokens` cache
hits). All other CPU defaults are inherited.
"""

from __future__ import annotations

import os
import platform as py_platform
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.platforms.cpu import CpuPlatform

if TYPE_CHECKING:  # pragma: no cover
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import AttentionSelectorConfig

logger = init_logger("pace_vllm.platform")

_PACE_WORKER_CLS = "pace_vllm.v1.worker.cpu_worker.PaceWorker"
_PACE_BACKEND_CLS = "pace_vllm.v1.attention.backends.pace_attn.PaceAttentionBackend"

# Pace `CustomOp` OOTs that need explicit `compilation_config.custom_ops`
# opt-in (vLLM defaults custom_ops=['none'] under Inductor on CPU, which
# routes every CustomOp through forward_native and bypasses our forward_cpu
# overrides). Grouped by the friendly shorthand PACE_VLLM_CUSTOM_OPS accepts.
_PACE_CUSTOM_OP_GROUPS: dict[str, list[str]] = {
    "rms_norm": ["RMSNorm", "GemmaRMSNorm"],
}

_PACE_CUSTOM_OPS_ENV = "PACE_VLLM_CUSTOM_OPS"


def _cpu_supports_avx512_bf16() -> bool:
    """Return True on x86_64 Linux with AVX512F + AVX512_BF16 (Zen4+, SPR+)."""
    if py_platform.system() != "Linux":
        return False
    if py_platform.machine() != "x86_64":
        return False
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.startswith("flags"):
                    continue
                flags = line.split(":", 1)[1].split()
                return "avx512f" in flags and "avx512_bf16" in flags
    except OSError:
        return False
    return False


def _resolve_pace_custom_op_groups() -> list[str]:
    """Resolve PACE_VLLM_CUSTOM_OPS into a list of group keys.

    Accepts: unset / `all` / `true` / `1` (default = all groups), `none` /
    `""` / `false` / `0` (empty list), or a comma-separated subset of the
    keys in `_PACE_CUSTOM_OP_GROUPS` (currently `rms_norm`). Unknown
    groups log a warning and are ignored. Group keys are what users pass
    in PACE_VLLM_CUSTOM_OPS, so logging *these* (not the underlying class
    names) gives copy-pasteable diagnostics.
    """
    raw = os.environ.get(_PACE_CUSTOM_OPS_ENV)
    if raw is None:
        return list(_PACE_CUSTOM_OP_GROUPS.keys())

    normalized = raw.strip().lower()
    if normalized in ("", "none", "0", "false"):
        return []
    if normalized in ("all", "1", "true"):
        return list(_PACE_CUSTOM_OP_GROUPS.keys())

    requested = [tok.strip().lower() for tok in normalized.split(",") if tok.strip()]
    groups: list[str] = []
    for tok in requested:
        if tok in _PACE_CUSTOM_OP_GROUPS:
            groups.append(tok)
        else:
            logger.warning(
                "pace-vllm: ignoring unknown %s group %r (known: %s).",
                _PACE_CUSTOM_OPS_ENV,
                tok,
                sorted(_PACE_CUSTOM_OP_GROUPS.keys()),
            )
    return groups


def _resolve_pace_custom_op_names() -> list[str]:
    """Flatten the active groups into vLLM's class-name contract for
    `compilation_config.custom_ops`."""
    return [
        name
        for group in _resolve_pace_custom_op_groups()
        for name in _PACE_CUSTOM_OP_GROUPS[group]
    ]


def _enable_pace_custom_ops(vllm_config: "VllmConfig") -> list[str]:
    """Append pace OOT opt-ins to `compilation_config.custom_ops`.

    Each pace OOT selected by PACE_VLLM_CUSTOM_OPS is added as `+<Name>`,
    skipping any already `+`-listed by the user / another plugin or
    `-`-listed (explicitly disabled). Returns the list of *group keys*
    that were resolved -- these are the env-var-shape strings users can
    copy back into `PACE_VLLM_CUSTOM_OPS`.
    """
    cops = vllm_config.compilation_config.custom_ops
    appended_any = False
    for name in _resolve_pace_custom_op_names():
        plus = f"+{name}"
        minus = f"-{name}"
        if plus in cops or minus in cops:
            continue
        cops.append(plus)
        appended_any = True

    # CustomOp.default_on() asserts exactly one of "none" / "all" is in cops.
    # Only add a "none" baseline when we actually appended a `+<Name>` and
    # no baseline exists -- otherwise we'd silently flip vLLM-wide
    # CustomOp dispatch to forward_native for the user.
    if appended_any and cops.count("all") == 0 and cops.count("none") == 0:
        cops.append("none")

    return _resolve_pace_custom_op_groups()


class PacePlatform(CpuPlatform):
    """PACE CPU platform.

    `_enum = PlatformEnum.CPU` stays so `CustomOp` dispatch continues to route
    to `forward_cpu`.
    """

    @classmethod
    def is_available(cls) -> bool:
        return _cpu_supports_avx512_bf16()

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        """Return the qualname of `PaceAttentionBackend`. vLLM's selector
        imports it and uses `get_impl_cls()` / `get_builder_cls()` from there."""
        if selected_backend is not None:
            logger.debug(
                "pace-vllm: ignoring selected_backend=%s; forcing PACE_SLAB.",
                selected_backend,
            )
        return _PACE_BACKEND_CLS

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        if vllm_config.model_config is None:
            return
        # Pace's slab attention C++ kernel is bf16-only. Catch a non-bf16
        # model dtype here so the user gets an actionable startup error.
        model_dtype = vllm_config.model_config.dtype
        if model_dtype != torch.bfloat16:
            raise ValueError(
                f"pace-vllm: requires dtype=torch.bfloat16, got {model_dtype}. "
                "Pass `dtype='bfloat16'` to vLLM "
                "(LLM/AsyncLLMEngine `dtype=` argument or `--dtype bfloat16` "
                "on the CLI), or unset VLLM_PLUGINS to use stock vLLM CPU."
            )

        # vLLM 0.20+ CpuPlatform rewrites distributed_executor_backend
        # 'uni' -> 'mp' for its OMP shim. PACE sets OMP via the launch
        # script, so we restore 'uni' for TP=1 (saves ~20% IPC on BS=1).
        parallel_config = vllm_config.parallel_config
        prev_executor = parallel_config.distributed_executor_backend
        keep_uniproc = parallel_config.world_size == 1 and prev_executor in (
            None,
            "uni",
        )

        super().check_and_update_config(vllm_config)

        if keep_uniproc and parallel_config.distributed_executor_backend == "mp":
            parallel_config.distributed_executor_backend = "uni"
            logger.info(
                "pace-vllm: keeping distributed_executor_backend='uni' "
                "for TP=1 (CpuPlatform forced 'mp'; OMP env is set by "
                "the launch script)."
            )

        prev = parallel_config.worker_cls
        parallel_config.worker_cls = _PACE_WORKER_CLS

        # PaceAttentionImpl does not honor num_computed_tokens cache hits, so
        # a cache hit would attend over a partial K/V tail (wrong output).
        cache_config = vllm_config.cache_config
        if cache_config.enable_prefix_caching:
            logger.warning(
                "pace-vllm: disabling vLLM prefix caching (PaceAttentionImpl "
                "does not yet honor num_computed_tokens cache hits)."
            )
            cache_config.enable_prefix_caching = False

        auto_enabled = _enable_pace_custom_ops(vllm_config)

        logger.info(
            "pace-vllm: PacePlatform active (worker_cls=%s, was=%s, "
            "attn_backend=%s, prefix_caching=%s, auto_custom_ops=%s).",
            parallel_config.worker_cls,
            prev,
            _PACE_BACKEND_CLS,
            cache_config.enable_prefix_caching,
            auto_enabled,
        )
