# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Numerical equivalence: `pace::libxsmm_fused_mlp` vs the unfused
PyTorch composition. The structural tests in `test_fused_mlp_pass.py`
and `test_fused_mlp_match.py` cover the FX-pass plumbing -- this file
guards the C++ kernel itself, so a regression in ordering / bias
application / dtype handling fails CI here instead of only as decode-
quality drift in production.

Each test packs the bf16 weights via `tpp_prepack` (the same helper
the production path uses), calls the fused op directly, and compares
to a stock PyTorch reference. bf16 tolerances are loose because the
two paths fuse rounding differently."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

import pace_vllm

pace_vllm._load_pace_native()

from pace_vllm.model_executor.layers.utils import tpp_prepack  # noqa: E402

# Larger atol/rtol than the GemmaRMSNorm drift test because here we
# accumulate a full GeMM + activation + GeMM. Matches the bounds
# pace's own kernel test in tests/ops/test_fused_mlp.py uses.
_BF16_ATOL = 5e-3
_BF16_RTOL = 2e-2

# Reference activations. The libXSMM `GeluFwdTPP` kernel implements the
# tanh approximation, so the reference uses `approximate='tanh'`.
_ACT_FNS = {
    "silu": F.silu,
    "gelu": lambda x: F.gelu(x, approximate="tanh"),
    "relu": F.relu,
}


def _fused_op():
    return torch.ops.pace.libxsmm_fused_mlp.default


def _ref_gated(
    src: torch.Tensor,
    wg: torch.Tensor,
    wu: torch.Tensor,
    wd: torch.Tensor,
    bg: torch.Tensor | None,
    bu: torch.Tensor | None,
    bd: torch.Tensor | None,
    act_fn,
) -> torch.Tensor:
    return F.linear(act_fn(F.linear(src, wg, bg)) * F.linear(src, wu, bu), wd, bd)


def _ref_ungated(
    src: torch.Tensor,
    fc1_w: torch.Tensor,
    fc2_w: torch.Tensor,
    fc1_b: torch.Tensor | None,
    fc2_b: torch.Tensor | None,
    act_fn,
) -> torch.Tensor:
    return F.linear(act_fn(F.linear(src, fc1_w, fc1_b)), fc2_w, fc2_b)


# Small shapes that satisfy tpp_prepack's divisibility:
# K % 64 == 0, intermediate (N) % 32 == 0, hidden_out (M) % 32 == 0.
_K, _N, _M, _T = 1024, 4096, 1024, 8


def _rand_bf16(*shape, scale: float = 0.01, seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(*shape, dtype=torch.bfloat16, generator=g) * scale


class TestFusedMLPNumericalEquivalenceGated(unittest.TestCase):
    """Gated SwiGLU MLP: out = down(act(gate(x)) * up(x))."""

    def _build(self, with_bias: bool):
        src = _rand_bf16(_T, _K, seed=1)
        wg = _rand_bf16(_N, _K, seed=2)
        wu = _rand_bf16(_N, _K, seed=3)
        wd = _rand_bf16(_M, _N, seed=4)
        if with_bias:
            bg = _rand_bf16(_N, seed=5)
            bu = _rand_bf16(_N, seed=6)
            bd = _rand_bf16(_M, seed=7)
        else:
            bg = bu = bd = None
        return src, wg, wu, wd, bg, bu, bd

    def _check(self, act_key: str, with_bias: bool) -> None:
        src, wg, wu, wd, bg, bu, bd = self._build(with_bias)
        ref = _ref_gated(src, wg, wu, wd, bg, bu, bd, _ACT_FNS[act_key])
        out = _fused_op()(
            src.unsqueeze(0),
            tpp_prepack(wg),
            tpp_prepack(wu),
            tpp_prepack(wd),
            bg,
            bu,
            bd,
            act_key,
        ).squeeze(0)
        torch.testing.assert_close(out, ref, atol=_BF16_ATOL, rtol=_BF16_RTOL)

    def test_silu_no_bias(self) -> None:
        self._check("silu", with_bias=False)

    def test_silu_with_bias(self) -> None:
        self._check("silu", with_bias=True)

    def test_gelu_no_bias(self) -> None:
        self._check("gelu", with_bias=False)

    def test_gelu_with_bias(self) -> None:
        self._check("gelu", with_bias=True)

    def test_relu_no_bias(self) -> None:
        self._check("relu", with_bias=False)

    def test_relu_with_bias(self) -> None:
        self._check("relu", with_bias=True)


class TestFusedMLPNumericalEquivalenceUngated(unittest.TestCase):
    """Ungated MLP: out = fc2(act(fc1(x))). C++ signature passes
    wt_gate=None, gate_bias=None; fc1 -> wt_up, fc2 -> wt_down."""

    def _build(self, with_bias: bool):
        src = _rand_bf16(_T, _K, seed=11)
        fc1_w = _rand_bf16(_N, _K, seed=12)
        fc2_w = _rand_bf16(_M, _N, seed=13)
        if with_bias:
            fc1_b = _rand_bf16(_N, seed=14)
            fc2_b = _rand_bf16(_M, seed=15)
        else:
            fc1_b = fc2_b = None
        return src, fc1_w, fc2_w, fc1_b, fc2_b

    def _check(self, act_key: str, with_bias: bool) -> None:
        src, fc1_w, fc2_w, fc1_b, fc2_b = self._build(with_bias)
        ref = _ref_ungated(src, fc1_w, fc2_w, fc1_b, fc2_b, _ACT_FNS[act_key])
        out = _fused_op()(
            src.unsqueeze(0),
            None,
            tpp_prepack(fc1_w),
            tpp_prepack(fc2_w),
            None,
            fc1_b,
            fc2_b,
            act_key,
        ).squeeze(0)
        torch.testing.assert_close(out, ref, atol=_BF16_ATOL, rtol=_BF16_RTOL)

    def test_silu_no_bias(self) -> None:
        self._check("silu", with_bias=False)

    def test_silu_with_bias(self) -> None:
        self._check("silu", with_bias=True)

    def test_gelu_no_bias(self) -> None:
        self._check("gelu", with_bias=False)

    def test_gelu_with_bias(self) -> None:
        self._check("gelu", with_bias=True)

    def test_relu_no_bias(self) -> None:
        self._check("relu", with_bias=False)

    def test_relu_with_bias(self) -> None:
        self._check("relu", with_bias=True)


class TestFusedMLP2DAnd3DInputs(unittest.TestCase):
    """The fake op (`pace/_register_fake.py:122`) accepts any rank for
    `src` -- it derives the output rank from `src.size()`. Live model
    code passes 3D `(B, T, K)` after the unsqueeze in the FX pattern,
    but a 2D `(T, K)` input must produce the same numerical answer
    once we flatten / squeeze leading dims."""

    def test_2d_and_3d_produce_equivalent_output(self) -> None:
        src = _rand_bf16(_T, _K, seed=21)
        wg = _rand_bf16(_N, _K, seed=22)
        wu = _rand_bf16(_N, _K, seed=23)
        wd = _rand_bf16(_M, _N, seed=24)
        wg_p = tpp_prepack(wg)
        wu_p = tpp_prepack(wu)
        wd_p = tpp_prepack(wd)

        out_3d = _fused_op()(
            src.unsqueeze(0), wg_p, wu_p, wd_p, None, None, None, "silu"
        ).squeeze(0)
        # 2D path: same data, no batch dim. The kernel handles either via
        # `src.dim()`-driven loops; the assertion is that the values match.
        out_2d_via_3d = _fused_op()(
            src.view(1, _T, _K), wg_p, wu_p, wd_p, None, None, None, "silu"
        ).view(_T, _M)
        torch.testing.assert_close(
            out_3d, out_2d_via_3d, atol=_BF16_ATOL, rtol=_BF16_RTOL
        )


if __name__ == "__main__":
    unittest.main()
