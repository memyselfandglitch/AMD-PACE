# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Pace RMSNorm / GemmaRMSNorm OOTs for vLLM (JIT backend).

Routes CPU forwards through `torch.ops.pace.rmsnorm` (no residual) and
`torch.ops.pace.fused_add_rmsnorm` (residual path). PACE's
`fused_add_rmsnorm` returns `(normed, x + residual)` in the same order
vLLM's `(x_normed, residual)` contract expects, so the adapter is a
direct pass-through.

GemmaRMSNorm scales by `(1 + w)` rather than `w`; PACE's kernel
multiplies by the raw weight, so we pre-compute `w + 1.0` once on
first use and cache it on the layer.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm

logger = init_logger("pace_vllm.model_executor.layers.layernorm")


@RMSNorm.register_oot
class PaceRMSNorm(RMSNorm):
    def forward_cpu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Use getattr so the attribute is purely per-instance (no class-level
        # default that could be misread as shared across instances).
        ready = getattr(self, "_pace_ready", None)
        if ready is None:
            ready = (
                self.has_weight
                and self.weight.data.dtype == torch.bfloat16
                and self.variance_size_override is None
            )
            self._pace_ready = ready

        if not ready or x.dtype != torch.bfloat16:
            return self.forward_native(x, residual)

        w = self.weight.data
        if residual is None:
            return torch.ops.pace.rmsnorm(x, w, self.variance_epsilon)
        normed, x_plus_res = torch.ops.pace.fused_add_rmsnorm(
            x, residual, w, self.variance_epsilon
        )
        return normed, x_plus_res


@GemmaRMSNorm.register_oot
class PaceGemmaRMSNorm(GemmaRMSNorm):
    """GemmaRMSNorm scales by `(1 + w)`. PACE's kernel takes the
    pre-shifted weight directly and only accepts bf16 (asserted in
    `csrc/ops/norm.cpp:19-22`), so we must round-trip the (w + 1)
    sum back to bf16 before the call. Stock vLLM's `forward_native`
    keeps the (w + 1) intermediate in fp32, which produces ~1-2 bf16
    ULP less drift than this path. The trade-off is intentional --
    the bf16 weight is the kernel contract -- and pinned by
    `tests/layers/test_layernorm.py::TestPaceGemmaRMSNormDrift`.
    """

    def _pace_weight_plus_one(self) -> torch.Tensor:
        cached = getattr(self, "_pace_w_plus_one_cache", None)
        w = self.weight.data
        if cached is not None and cached.shape == w.shape and cached.dtype == w.dtype:
            return cached
        # fp32 sum then cast back: bf16-bf16 add would lose ~1 ULP per
        # element vs this version, and the kernel demands bf16 anyway.
        updated = (w.float() + 1.0).to(w.dtype).contiguous()
        self._pace_w_plus_one_cache = updated
        return updated

    def forward_cpu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        ready = getattr(self, "_pace_ready", None)
        if ready is None:
            ready = self.weight.data.dtype == torch.bfloat16
            self._pace_ready = ready

        if not ready or x.dtype != torch.bfloat16:
            return self.forward_native(x, residual)

        w_plus_one = self._pace_weight_plus_one()
        if residual is None:
            return torch.ops.pace.rmsnorm(x, w_plus_one, self.variance_epsilon)
        normed, x_plus_res = torch.ops.pace.fused_add_rmsnorm(
            x, residual, w_plus_one, self.variance_epsilon
        )
        return normed, x_plus_res


__all__ = ["PaceRMSNorm", "PaceGemmaRMSNorm"]
