# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""End-to-end pattern match for `FusedMLPPass`.

Builds a synthetic post-grad graph for one MLP block (gated SwiGLU and
ungated fc1->relu->fc2 with bias), runs `FusedMLPPass`, and asserts the
graph is rewritten to `pace::libxsmm_fused_mlp`. No model load; relies
on the registered fake impls in `pace_vllm._fakes_snapshot` so make_fx
can trace under `FakeTensorMode`.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

import pace_vllm

pace_vllm._load_pace_native()

from pace_vllm.compilation.fused_mlp_pass import FusedMLPPass  # noqa: E402

_INT64_MAX = 9223372036854775807


def _fake_vllm_config() -> SimpleNamespace:
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            splitting_ops=[],
            use_inductor_graph_partition=False,
            pass_config=SimpleNamespace(),
            inductor_compile_config={},
        ),
        model_config=None,
        device_config=None,
        compile_debug_dump_path=lambda: None,
    )


def _restore_reshape(gm: torch.fx.GraphModule) -> None:
    """make_fx decomposes `aten.reshape.default` into `aten.view.default`,
    but Inductor's post-grad graph keeps it as reshape -- which is what
    the pattern is hand-built to match. Swap targets in place; meta['val']
    survives the rewrite."""
    for node in list(gm.graph.nodes):
        if node.target is torch.ops.aten.view.default:
            node.target = torch.ops.aten.reshape.default
    gm.recompile()


def _build_gated_swiglu_module() -> torch.fx.GraphModule:
    """Llama-3.1-8B shaped gated SwiGLU MLP, traced through FakeTensorMode
    with the pace fake impls so meta['val'] is populated on every node."""
    T, HID, INT_, OUT = 16, 4096, 14336, 4096

    def mlp(x, merged_w, down_w):
        x_3d = torch.ops.aten.unsqueeze.default(x, 0)
        gate_up_3d = torch.ops.pace.libxsmmlinear_plain.default(x_3d, merged_w, None)
        gate_up_2d = torch.ops.aten.reshape.default(gate_up_3d, [T, 2 * INT_])
        gate = torch.ops.aten.slice.Tensor(gate_up_2d, 1, 0, INT_)
        up = torch.ops.aten.slice.Tensor(gate_up_2d, 1, INT_, _INT64_MAX)
        gate_f32 = torch.ops.prims.convert_element_type.default(gate, torch.float32)
        sig = torch.ops.aten.sigmoid.default(gate_f32)
        silu_f32 = torch.ops.aten.mul.Tensor(gate_f32, sig)
        silu_bf16 = torch.ops.prims.convert_element_type.default(
            silu_f32, torch.bfloat16
        )
        prod = torch.ops.aten.mul.Tensor(silu_bf16, up)
        prod_3d = torch.ops.aten.unsqueeze.default(prod, 0)
        down_3d = torch.ops.pace.libxsmmlinear_plain.default(prod_3d, down_w, None)
        return torch.ops.aten.reshape.default(down_3d, [T, OUT])

    with FakeTensorMode():
        x = torch.empty(T, HID, dtype=torch.bfloat16)
        merged_w = torch.empty(
            (2 * INT_) // 32, HID // 64, 32, 32, 2, dtype=torch.bfloat16
        )
        down_w = torch.empty(OUT // 32, INT_ // 64, 32, 32, 2, dtype=torch.bfloat16)
        gm = make_fx(mlp)(x, merged_w, down_w)

    _restore_reshape(gm)
    return gm


def _build_gated_swiglu_with_bias_module() -> torch.fx.GraphModule:
    """Same SwiGLU shape as `_build_gated_swiglu_module` but with bias on
    both the merged gate_up and down projections. Exercises the gated
    + bias variant: pattern captures the merged bias, replacement
    splits it on dim-0 the same way as the merged weight."""
    T, HID, INT_, OUT = 16, 4096, 14336, 4096

    def mlp(x, merged_w, merged_b, down_w, down_b):
        x_3d = torch.ops.aten.unsqueeze.default(x, 0)
        gate_up_3d = torch.ops.pace.libxsmmlinear_plain.default(
            x_3d, merged_w, merged_b
        )
        gate_up_2d = torch.ops.aten.reshape.default(gate_up_3d, [T, 2 * INT_])
        gate = torch.ops.aten.slice.Tensor(gate_up_2d, 1, 0, INT_)
        up = torch.ops.aten.slice.Tensor(gate_up_2d, 1, INT_, _INT64_MAX)
        gate_f32 = torch.ops.prims.convert_element_type.default(gate, torch.float32)
        sig = torch.ops.aten.sigmoid.default(gate_f32)
        silu_f32 = torch.ops.aten.mul.Tensor(gate_f32, sig)
        silu_bf16 = torch.ops.prims.convert_element_type.default(
            silu_f32, torch.bfloat16
        )
        prod = torch.ops.aten.mul.Tensor(silu_bf16, up)
        prod_3d = torch.ops.aten.unsqueeze.default(prod, 0)
        down_3d = torch.ops.pace.libxsmmlinear_plain.default(prod_3d, down_w, down_b)
        return torch.ops.aten.reshape.default(down_3d, [T, OUT])

    with FakeTensorMode():
        x = torch.empty(T, HID, dtype=torch.bfloat16)
        merged_w = torch.empty(
            (2 * INT_) // 32, HID // 64, 32, 32, 2, dtype=torch.bfloat16
        )
        merged_b = torch.empty(2 * INT_, dtype=torch.bfloat16)
        down_w = torch.empty(OUT // 32, INT_ // 64, 32, 32, 2, dtype=torch.bfloat16)
        down_b = torch.empty(OUT, dtype=torch.bfloat16)
        gm = make_fx(mlp)(x, merged_w, merged_b, down_w, down_b)

    _restore_reshape(gm)
    return gm


def _build_gated_swiglu_div_module() -> torch.fx.GraphModule:
    """Llama-3.2-3B shaped gated SwiGLU MLP that emits the silu in the
    torch >= 2.11 / vLLM >= 0.20 decomposed form `x / (1 + exp(-x))`
    instead of `x * sigmoid(x)`. Inductor's post-grad in the new
    torch decomposes `aten.sigmoid` into primitives, so the matcher
    has to track this shape too -- otherwise every Llama / Qwen / Phi
    MLP runs unfused and decode tps drops ~30%."""
    T, HID, INT_, OUT = 16, 3072, 8192, 3072

    def mlp(x, merged_w, down_w):
        x_3d = torch.ops.aten.unsqueeze.default(x, 0)
        gate_up_3d = torch.ops.pace.libxsmmlinear_plain.default(x_3d, merged_w, None)
        gate_up_2d = torch.ops.aten.reshape.default(gate_up_3d, [T, 2 * INT_])
        gate = torch.ops.aten.slice.Tensor(gate_up_2d, 1, 0, INT_)
        up = torch.ops.aten.slice.Tensor(gate_up_2d, 1, INT_, _INT64_MAX)
        gate_f32 = torch.ops.prims.convert_element_type.default(gate, torch.float32)
        # silu_div shape: gate / (1 + exp(-gate))
        neg = torch.ops.aten.neg.default(gate_f32)
        exp_neg = torch.ops.aten.exp.default(neg)
        denom = torch.ops.aten.add.Tensor(exp_neg, 1)
        silu_f32 = torch.ops.aten.div.Tensor(gate_f32, denom)
        silu_bf16 = torch.ops.prims.convert_element_type.default(
            silu_f32, torch.bfloat16
        )
        prod = torch.ops.aten.mul.Tensor(silu_bf16, up)
        prod_3d = torch.ops.aten.unsqueeze.default(prod, 0)
        down_3d = torch.ops.pace.libxsmmlinear_plain.default(prod_3d, down_w, None)
        return torch.ops.aten.reshape.default(down_3d, [T, OUT])

    with FakeTensorMode():
        x = torch.empty(T, HID, dtype=torch.bfloat16)
        merged_w = torch.empty(
            (2 * INT_) // 32, HID // 64, 32, 32, 2, dtype=torch.bfloat16
        )
        down_w = torch.empty(OUT // 32, INT_ // 64, 32, 32, 2, dtype=torch.bfloat16)
        gm = make_fx(mlp)(x, merged_w, down_w)

    _restore_reshape(gm)
    return gm


def _build_ungated_relu_with_bias_module() -> torch.fx.GraphModule:
    """OPT-shaped ungated fc1->relu->fc2 with bias. Confirms with-bias
    variants register and match independently of the bias=None path."""
    T, HID, INT_, OUT = 16, 4096, 16384, 4096

    def mlp(x, fc1_w, fc1_b, fc2_w, fc2_b):
        x_3d = torch.ops.aten.unsqueeze.default(x, 0)
        fc1_3d = torch.ops.pace.libxsmmlinear_plain.default(x_3d, fc1_w, fc1_b)
        fc1_2d = torch.ops.aten.reshape.default(fc1_3d, [T, INT_])
        act = torch.ops.aten.relu.default(fc1_2d)
        act_3d = torch.ops.aten.unsqueeze.default(act, 0)
        fc2_3d = torch.ops.pace.libxsmmlinear_plain.default(act_3d, fc2_w, fc2_b)
        return torch.ops.aten.reshape.default(fc2_3d, [T, OUT])

    with FakeTensorMode():
        x = torch.empty(T, HID, dtype=torch.bfloat16)
        fc1_w = torch.empty(INT_ // 32, HID // 64, 32, 32, 2, dtype=torch.bfloat16)
        fc1_b = torch.empty(INT_, dtype=torch.bfloat16)
        fc2_w = torch.empty(OUT // 32, INT_ // 64, 32, 32, 2, dtype=torch.bfloat16)
        fc2_b = torch.empty(OUT, dtype=torch.bfloat16)
        gm = make_fx(mlp)(x, fc1_w, fc1_b, fc2_w, fc2_b)

    _restore_reshape(gm)
    return gm


class TestSyntheticMatch(unittest.TestCase):
    """One match per topology; downstream variants all share the same
    plumbing, so a covering smoke per topology is enough to keep the
    pass honest."""

    def test_gated_swiglu_rewrites_to_fused_mlp(self) -> None:
        gm = _build_gated_swiglu_module()
        pass_obj = FusedMLPPass(_fake_vllm_config())

        pass_obj(gm.graph)

        self.assertEqual(pass_obj.matched_count, 1)
        fused = [
            n
            for n in gm.graph.nodes
            if n.target is torch.ops.pace.libxsmm_fused_mlp.default
        ]
        leftover_linears = [
            n
            for n in gm.graph.nodes
            if n.target is torch.ops.pace.libxsmmlinear_plain.default
        ]
        self.assertEqual(len(fused), 1, "expected exactly one libxsmm_fused_mlp node")
        self.assertEqual(
            len(leftover_linears),
            0,
            "all libxsmmlinear_plain calls should have been fused",
        )
        # The fused call carries the activation enum string in the last
        # positional arg (signature: x, gate_w, fc_w, down_w, *biases, op_str).
        self.assertEqual(fused[0].args[-1], "silu")

    def test_gated_swiglu_silu_div_rewrites_to_fused_mlp(self) -> None:
        # torch >= 2.11 / vLLM >= 0.20 emits the gate side as
        # `x / (1 + exp(-x))` instead of `x * sigmoid(x)`. Without
        # the silu_div pattern the matcher reports
        # `FusedMLPPass matched 0 MLP sites` and every MLP runs
        # unfused. This test guards the new pattern's plumbing.
        gm = _build_gated_swiglu_div_module()
        pass_obj = FusedMLPPass(_fake_vllm_config())

        pass_obj(gm.graph)

        self.assertEqual(pass_obj.matched_count, 1)
        fused = [
            n
            for n in gm.graph.nodes
            if n.target is torch.ops.pace.libxsmm_fused_mlp.default
        ]
        leftover_linears = [
            n
            for n in gm.graph.nodes
            if n.target is torch.ops.pace.libxsmmlinear_plain.default
        ]
        self.assertEqual(len(fused), 1)
        self.assertEqual(len(leftover_linears), 0)
        # silu_div maps to the same C++ activation enum as silu.
        self.assertEqual(fused[0].args[-1], "silu")

    def test_gated_swiglu_with_bias_rewrites_to_fused_mlp(self) -> None:
        gm = _build_gated_swiglu_with_bias_module()
        pass_obj = FusedMLPPass(_fake_vllm_config())

        pass_obj(gm.graph)

        self.assertEqual(pass_obj.matched_count, 1)
        fused = [
            n
            for n in gm.graph.nodes
            if n.target is torch.ops.pace.libxsmm_fused_mlp.default
        ]
        leftover_linears = [
            n
            for n in gm.graph.nodes
            if n.target is torch.ops.pace.libxsmmlinear_plain.default
        ]
        self.assertEqual(len(fused), 1)
        self.assertEqual(len(leftover_linears), 0)
        # Bias args land at positions 4 / 5 / 6 of the fused call
        # (signature: x, gate_w, up_w, down_w, gate_bias, up_bias,
        # down_bias, op_str). They must all be tensor nodes (not None)
        # so the kernel actually applies the biases.
        gate_bias, up_bias, down_bias = fused[0].args[4:7]
        for slot, name in (
            (gate_bias, "gate_bias"),
            (up_bias, "up_bias"),
            (down_bias, "down_bias"),
        ):
            with self.subTest(slot=name):
                self.assertIsNotNone(slot, f"{name} must not be None")
        self.assertEqual(fused[0].args[-1], "silu")

    def test_ungated_relu_with_bias_rewrites_to_fused_mlp(self) -> None:
        gm = _build_ungated_relu_with_bias_module()
        pass_obj = FusedMLPPass(_fake_vllm_config())

        pass_obj(gm.graph)

        self.assertEqual(pass_obj.matched_count, 1)
        fused = [
            n
            for n in gm.graph.nodes
            if n.target is torch.ops.pace.libxsmm_fused_mlp.default
        ]
        leftover_linears = [
            n
            for n in gm.graph.nodes
            if n.target is torch.ops.pace.libxsmmlinear_plain.default
        ]
        self.assertEqual(len(fused), 1)
        self.assertEqual(len(leftover_linears), 0)
        self.assertEqual(fused[0].args[-1], "relu")


def _build_small_gated_swiglu_module() -> torch.fx.GraphModule:
    """Tiny gated SwiGLU MLP for the derived-offset assertion: with
    intermediate_size=128 and block_size=32, the activation-axis slices
    are (0, 128) and (128, 256), and the replacement must rewrite them
    to dim-0 packed-weight slices (0, 4) and (4, 8) -- both derived
    from `gate_start/block_size` and `gate_end/block_size`."""
    T, HID, INT_, OUT = 16, 256, 128, 256

    def mlp(x, merged_w, down_w):
        x_3d = torch.ops.aten.unsqueeze.default(x, 0)
        gate_up_3d = torch.ops.pace.libxsmmlinear_plain.default(x_3d, merged_w, None)
        gate_up_2d = torch.ops.aten.reshape.default(gate_up_3d, [T, 2 * INT_])
        gate = torch.ops.aten.slice.Tensor(gate_up_2d, 1, 0, INT_)
        up = torch.ops.aten.slice.Tensor(gate_up_2d, 1, INT_, 2 * INT_)
        gate_f32 = torch.ops.prims.convert_element_type.default(gate, torch.float32)
        sig = torch.ops.aten.sigmoid.default(gate_f32)
        silu_f32 = torch.ops.aten.mul.Tensor(gate_f32, sig)
        silu_bf16 = torch.ops.prims.convert_element_type.default(
            silu_f32, torch.bfloat16
        )
        prod = torch.ops.aten.mul.Tensor(silu_bf16, up)
        prod_3d = torch.ops.aten.unsqueeze.default(prod, 0)
        down_3d = torch.ops.pace.libxsmmlinear_plain.default(prod_3d, down_w, None)
        return torch.ops.aten.reshape.default(down_3d, [T, OUT])

    with FakeTensorMode():
        x = torch.empty(T, HID, dtype=torch.bfloat16)
        merged_w = torch.empty(
            (2 * INT_) // 32, HID // 64, 32, 32, 2, dtype=torch.bfloat16
        )
        down_w = torch.empty(OUT // 32, INT_ // 64, 32, 32, 2, dtype=torch.bfloat16)
        gm = make_fx(mlp)(x, merged_w, down_w)

    _restore_reshape(gm)
    return gm


class TestGatedSplitDerivedFromMatch(unittest.TestCase):
    """The replacement must translate the *matched* dim=1 slice offsets
    on the 2D activation into dim=0 ranges on the 5D-packed weight by
    dividing by `block_size`. This guards against regressing back to the
    `n_blocks // 2` midpoint heuristic, which produced wrong rewrites
    for non-canonical splits and odd `n_blocks`."""

    def test_dim0_slices_are_activation_offsets_over_block_size(self) -> None:
        gm = _build_small_gated_swiglu_module()
        pass_obj = FusedMLPPass(_fake_vllm_config())

        pass_obj(gm.graph)

        self.assertEqual(pass_obj.matched_count, 1)
        # The replacement emits two dim-0 slices on the packed weight
        # (gate half, up half). For intermediate=128 / block_size=32 the
        # ranges must be (0, 4) and (4, 8): activation offsets 0/128/256
        # divided by block_size 32.
        dim0_slices = [
            n
            for n in gm.graph.nodes
            if n.target is torch.ops.aten.slice.Tensor and n.args[1] == 0
        ]
        self.assertEqual(
            len(dim0_slices),
            2,
            "expected exactly two dim-0 slices in the replacement",
        )
        ranges = sorted((int(n.args[2]), int(n.args[3])) for n in dim0_slices)
        self.assertEqual(ranges, [(0, 4), (4, 8)])

    def test_replacement_does_not_use_midpoint_heuristic(self) -> None:
        # Sanity: the rewritten graph must not contain `n_blocks // 2`
        # (i.e. an aten.floordiv with a sym_size + literal 2). This
        # check is structural -- if anyone reintroduces the midpoint
        # path the test will catch it.
        gm = _build_small_gated_swiglu_module()
        pass_obj = FusedMLPPass(_fake_vllm_config())
        pass_obj(gm.graph)

        for node in gm.graph.nodes:
            if node.target is torch.ops.aten.floordiv.default:
                self.fail(
                    f"unexpected aten.floordiv in rewritten graph: {node.format_node()}"
                )


def _build_gated_swiglu_div_view_module() -> torch.fx.GraphModule:
    """Same SwiGLU shape as `_build_gated_swiglu_div_module`, but the
    `_restore_reshape` shim is *not* applied. The graph carries
    `aten.view.default` everywhere `make_fx` emitted it, simulating a
    future Inductor that normalises `reshape -> view` at the
    libxsmmlinear_plain boundary."""
    T, HID, INT_, OUT = 16, 3072, 8192, 3072

    def mlp(x, merged_w, down_w):
        x_3d = torch.ops.aten.unsqueeze.default(x, 0)
        gate_up_3d = torch.ops.pace.libxsmmlinear_plain.default(x_3d, merged_w, None)
        gate_up_2d = torch.ops.aten.reshape.default(gate_up_3d, [T, 2 * INT_])
        gate = torch.ops.aten.slice.Tensor(gate_up_2d, 1, 0, INT_)
        up = torch.ops.aten.slice.Tensor(gate_up_2d, 1, INT_, _INT64_MAX)
        gate_f32 = torch.ops.prims.convert_element_type.default(gate, torch.float32)
        neg = torch.ops.aten.neg.default(gate_f32)
        exp_neg = torch.ops.aten.exp.default(neg)
        denom = torch.ops.aten.add.Tensor(exp_neg, 1)
        silu_f32 = torch.ops.aten.div.Tensor(gate_f32, denom)
        silu_bf16 = torch.ops.prims.convert_element_type.default(
            silu_f32, torch.bfloat16
        )
        prod = torch.ops.aten.mul.Tensor(silu_bf16, up)
        prod_3d = torch.ops.aten.unsqueeze.default(prod, 0)
        down_3d = torch.ops.pace.libxsmmlinear_plain.default(prod_3d, down_w, None)
        return torch.ops.aten.reshape.default(down_3d, [T, OUT])

    with FakeTensorMode():
        x = torch.empty(T, HID, dtype=torch.bfloat16)
        merged_w = torch.empty(
            (2 * INT_) // 32, HID // 64, 32, 32, 2, dtype=torch.bfloat16
        )
        down_w = torch.empty(OUT // 32, INT_ // 64, 32, 32, 2, dtype=torch.bfloat16)
        gm = make_fx(mlp)(x, merged_w, down_w)

    # Deliberately NO _restore_reshape() call -- exercise the matcher's
    # ability to handle aten.view.default that make_fx left in place.
    return gm


class TestPatternMatchesUnmassagedMakeFxGraph(unittest.TestCase):
    """The pattern must match both `aten.reshape.default` and
    `aten.view.default` at the libxsmmlinear_plain boundary. Older
    tests rewrite view -> reshape via `_restore_reshape` to mirror
    today's torch 2.11 Inductor; this test deliberately skips that
    shim so a future Inductor that emits `view` instead of `reshape`
    fails CI here before going silent in production."""

    def test_gated_swiglu_div_matches_view_form(self) -> None:
        gm = _build_gated_swiglu_div_view_module()
        # Sanity: the graph really does carry view, not reshape, at
        # the libxsmmlinear_plain boundary.
        view_count = sum(
            1 for n in gm.graph.nodes if n.target is torch.ops.aten.view.default
        )
        reshape_count = sum(
            1 for n in gm.graph.nodes if n.target is torch.ops.aten.reshape.default
        )
        self.assertGreater(
            view_count,
            0,
            "fixture invariant: make_fx should leave aten.view.default in the graph",
        )
        self.assertEqual(
            reshape_count,
            0,
            "fixture invariant: no _restore_reshape rewrite should have run",
        )

        pass_obj = FusedMLPPass(_fake_vllm_config())
        pass_obj(gm.graph)
        self.assertEqual(
            pass_obj.matched_count,
            1,
            "FusedMLPPass must match the view form too; failing here means "
            "_RESHAPE_OR_VIEW regressed back to a single target.",
        )


if __name__ == "__main__":
    unittest.main()
