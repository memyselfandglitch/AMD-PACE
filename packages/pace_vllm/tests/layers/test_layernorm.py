# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Drift bound test for `PaceGemmaRMSNorm`.

The PACE GemmaRMSNorm path round-trips `(w + 1)` through bf16 because
PACE's C kernel asserts a bf16 weight at `csrc/ops/norm.cpp:19-22`.
Stock vLLM's `forward_native` keeps that intermediate in fp32, so the
two paths differ by a fixed-bound bf16 round-off. This test pins the
bound so a future kernel change that widens the gap fails CI.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager

import torch
from vllm.config import VllmConfig, set_current_vllm_config

import pace_vllm

# Native lib must be loaded before constructing PaceGemmaRMSNorm (which
# calls into torch.ops.pace.rmsnorm).
pace_vllm._load_pace_native()


_BF16_EPS = float(torch.finfo(torch.bfloat16).eps)


@contextmanager
def _dummy_vllm_config():
    """vLLM CustomOp instantiation reads the current VllmConfig to decide
    whether to dispatch to forward_native or forward_cpu. Tests need a
    minimal config in scope before constructing the OOTs."""
    with set_current_vllm_config(VllmConfig()):
        yield


class TestPaceGemmaRMSNormDrift(unittest.TestCase):
    """Compare PaceGemmaRMSNorm.forward_cpu against stock GemmaRMSNorm's
    forward_native. We allow a small absolute drift -- the deliberate
    bf16 cast on the weight side leaves ~1-2 ULP per element -- but
    pin the bound so it can't silently widen."""

    def _build(self, hidden_size: int):
        from pace_vllm.model_executor.layers.layernorm import PaceGemmaRMSNorm
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm

        torch.manual_seed(0)
        weight = torch.randn(hidden_size, dtype=torch.bfloat16) * 0.1
        with _dummy_vllm_config():
            ref = GemmaRMSNorm(hidden_size, eps=1e-6)
            pace = PaceGemmaRMSNorm(hidden_size, eps=1e-6)
        with torch.no_grad():
            ref.weight.data.copy_(weight)
            pace.weight.data.copy_(weight)
        return ref, pace

    def test_no_residual_drift_within_bound(self) -> None:
        hidden = 256
        ref, pace = self._build(hidden)
        x = torch.randn(4, hidden, dtype=torch.bfloat16)

        ref_out = ref.forward_native(x)
        pace_out = pace.forward_cpu(x)

        # 4x bf16 eps in absolute -- comfortable headroom over the
        # observed ~1-2 ULP drift while still flagging a regression.
        max_abs = (ref_out.float() - pace_out.float()).abs().max().item()
        self.assertLess(
            max_abs,
            4 * _BF16_EPS,
            f"PaceGemmaRMSNorm drift {max_abs:.3e} exceeds 4*bf16_eps "
            f"({4 * _BF16_EPS:.3e}); the deliberate (w+1) bf16 cast "
            "should leave ~1-2 ULP only.",
        )

    def test_residual_path_drift_within_bound(self) -> None:
        # The residual path (fused_add_rmsnorm) uses the same (w + 1)
        # cast; bound applies to both outputs.
        hidden = 256
        ref, pace = self._build(hidden)
        x = torch.randn(4, hidden, dtype=torch.bfloat16)
        residual = torch.randn(4, hidden, dtype=torch.bfloat16)

        ref_normed, ref_resid = ref.forward_native(x.clone(), residual.clone())
        pace_normed, pace_resid = pace.forward_cpu(x.clone(), residual.clone())

        normed_drift = (ref_normed.float() - pace_normed.float()).abs().max().item()
        resid_drift = (ref_resid.float() - pace_resid.float()).abs().max().item()
        self.assertLess(normed_drift, 4 * _BF16_EPS)
        # Residual is just `x + residual` -- no kernel work, so drift
        # should be at most 1 ULP of bf16 add.
        self.assertLess(resid_drift, 2 * _BF16_EPS)


if __name__ == "__main__":
    unittest.main()
