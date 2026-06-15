# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Fused MLP post-grad pattern matcher pass.

Rewrites pace `libxsmmlinear_plain` + activation glue +
`libxsmmlinear_plain` MLP blocks into one `pace::libxsmm_fused_mlp`
call. Covers all C++ kernel activations (silu / gelu-tanh / gelu-exact
/ relu) across gated (SwiGLU) and ungated (fc1 -> act -> fc2)
topologies, with separate bias / no-bias variants (the FX node for
`bias=None` differs from one that captures a tensor).

Registration is declarative: `_ACTIVATIONS` maps an activation key to
(FX inner-section builder, C++ enum string); `_Topology` (gated /
ungated) owns everything around the activation; `_Variant` rows in
`_VARIANTS` produce one `register_replacement(...)` call each. Adding
a new activation = 1 row in `_ACTIVATIONS` + per-topology rows in
`_VARIANTS`.

Search patterns are hand-built via `CallFunction` because tracing them
through `fwd_only` decomposes `aten.reshape` to `aten.view`, while
Inductor's post-grad keeps `aten.reshape.default`. Replacements are
plain Python functions traced via `fwd_only` so emitted nodes carry
`meta['val']`.

The pass is installed into `compilation_config.inductor_compile_config
["post_grad_custom_post_pass"]` from `PaceWorker.init_device` because
it captures `torch.ops.pace.*` `OpOverload` objects whose pybind11
records aren't pickleable across the worker spawn.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.fx as fx
from torch._inductor.pattern_matcher import (
    CallFunction,
    Ignored,
    KeywordArg,
    PatternMatcherPass,
    fwd_only,
    register_replacement,
)
from vllm.compilation.passes.vllm_inductor_pass import (
    VllmInductorPass,
    VllmPatternMatcherPass,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger("pace_vllm.compilation.fused_mlp_pass")

_aten = torch.ops.aten
_prims = torch.ops.prims
_pace = torch.ops.pace

# Inductor emits `reshape` on torch 2.11 today; accept `view` too so a future
# normalisation flip doesn't silently disable fusion.
_RESHAPE_OR_VIEW = [_aten.reshape.default, _aten.view.default]

_GELU_TANH_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_GELU_EXACT_INV_SQRT_2 = 1.0 / math.sqrt(2.0)

_FUSED_MLP_DISABLE_ENV = "PACE_VLLM_FUSED_MLP"
_DISABLE_TOKENS = frozenset({"0", "false", "off", "no", "disable", "disabled"})


def _resolve_disabled_from_env() -> bool:
    raw = os.environ.get(_FUSED_MLP_DISABLE_ENV)
    return raw is not None and raw.strip().lower() in _DISABLE_TOKENS


def _silu_inner(gate_bf16: Any) -> Any:
    # torch <= 2.10 / vLLM <= 0.19 shape: x * sigmoid(x).
    gate_f32 = CallFunction(
        _prims.convert_element_type.default, gate_bf16, Ignored(), _users=2
    )
    sig = CallFunction(_aten.sigmoid.default, gate_f32)
    silu_f32 = CallFunction(_aten.mul.Tensor, gate_f32, sig)
    return CallFunction(_prims.convert_element_type.default, silu_f32, Ignored())


def _silu_inner_div(gate_bf16: Any) -> Any:
    # torch >= 2.11 / vLLM >= 0.20 shape: x / (1 + exp(-x)). Constants
    # pinned so a non-silu lookalike cannot match the canonical pattern.
    gate_f32 = CallFunction(
        _prims.convert_element_type.default, gate_bf16, Ignored(), _users=2
    )
    neg_f32 = CallFunction(_aten.neg.default, gate_f32)
    exp_neg = CallFunction(_aten.exp.default, neg_f32)
    denom = CallFunction(_aten.add.Tensor, exp_neg, 1)
    silu_f32 = CallFunction(_aten.div.Tensor, gate_f32, denom)
    return CallFunction(_prims.convert_element_type.default, silu_f32, Ignored())


def _gelu_tanh_inner(gate_bf16: Any) -> Any:
    # gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    gate_f32 = CallFunction(
        _prims.convert_element_type.default, gate_bf16, Ignored(), _users=4
    )
    half_x = CallFunction(_aten.mul.Tensor, gate_f32, 0.5)
    x_sq = CallFunction(_aten.mul.Tensor, gate_f32, gate_f32)
    x_cube = CallFunction(_aten.mul.Tensor, x_sq, gate_f32)
    scaled_c = CallFunction(_aten.mul.Tensor, x_cube, 0.044715)
    inner = CallFunction(_aten.add.Tensor, gate_f32, scaled_c)
    inner_sc = CallFunction(_aten.mul.Tensor, inner, _GELU_TANH_SQRT_2_OVER_PI)
    tanh_val = CallFunction(_aten.tanh.default, inner_sc)
    tanh_p1 = CallFunction(_aten.add.Tensor, tanh_val, 1.0)
    gelu_f32 = CallFunction(_aten.mul.Tensor, half_x, tanh_p1)
    return CallFunction(_prims.convert_element_type.default, gelu_f32, Ignored())


def _gelu_exact_inner(gate_bf16: Any) -> Any:
    # gelu_exact(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    gate_f32 = CallFunction(
        _prims.convert_element_type.default, gate_bf16, Ignored(), _users=2
    )
    half_x = CallFunction(_aten.mul.Tensor, gate_f32, 0.5)
    scaled = CallFunction(_aten.mul.Tensor, gate_f32, _GELU_EXACT_INV_SQRT_2)
    erf_val = CallFunction(_aten.erf.default, scaled)
    plus_one = CallFunction(_aten.add.Tensor, erf_val, 1.0)
    gelu_f32 = CallFunction(_aten.mul.Tensor, half_x, plus_one)
    return CallFunction(_prims.convert_element_type.default, gelu_f32, Ignored())


def _relu_inner(src_bf16: Any) -> Any:
    # ReLU stays in bf16; no f32 promotion from Inductor on this path.
    return CallFunction(_aten.relu.default, src_bf16)


# `silu` and `silu_div` map to the same C++ enum -- different Inductor
# shapes (pre / post 2.11) both reduce to the kernel's silu.
_ACTIVATIONS: dict[str, tuple[Callable[[Any], Any], str]] = {
    "silu": (_silu_inner, "silu"),
    "silu_div": (_silu_inner_div, "silu"),
    "gelu_tanh": (_gelu_tanh_inner, "gelu"),
    "gelu_exact": (_gelu_exact_inner, "gelu"),
    "relu": (_relu_inner, "relu"),
}


# A topology owns build_pattern / stub_for / make_replacement /
# example_inputs for one MLP shape (gated vs ungated). The same
# topology is reused across all activation keys.
@dataclass(frozen=True)
class _Topology:
    name: str
    build_pattern: Callable[..., Any]
    stub_for: Callable[[bool], Callable[..., torch.Tensor]]
    make_replacement: Callable[[str, bool], Callable[..., torch.Tensor]]
    example_inputs: Callable[[bool], list[torch.Tensor]]


def _gated_pattern(inner_fn: Callable[[Any], Any], with_bias: bool) -> Any:
    x = KeywordArg("x")
    merged_w = KeywordArg("merged_gate_up_w")
    down_w = KeywordArg("down_w")
    # `KeywordArg` vs literal `None` produce different FX node shapes, so
    # with-bias and no-bias must be registered as separate variants.
    merged_bias = KeywordArg("merged_gate_up_bias") if with_bias else None
    down_bias = KeywordArg("down_bias") if with_bias else None
    # KeywordArg the slice offsets so the replacement uses the matched
    # split, not a midpoint assumption. dim=1 is a literal so a graph
    # slicing a different dim does not match.
    gate_start = KeywordArg("gate_start")
    gate_end = KeywordArg("gate_end")
    up_start = KeywordArg("up_start")
    up_end = KeywordArg("up_end")

    unsq = CallFunction(_aten.unsqueeze.default, x, Ignored())
    gate_up_3d = CallFunction(
        _pace.libxsmmlinear_plain.default, unsq, merged_w, merged_bias
    )
    gate_up_2d = CallFunction(_RESHAPE_OR_VIEW, gate_up_3d, Ignored(), _users=2)

    gate = CallFunction(_aten.slice.Tensor, gate_up_2d, 1, gate_start, gate_end)
    up = CallFunction(_aten.slice.Tensor, gate_up_2d, 1, up_start, up_end)
    act = inner_fn(gate)
    prod = CallFunction(_aten.mul.Tensor, act, up)

    unsq_down = CallFunction(_aten.unsqueeze.default, prod, Ignored())
    down_3d = CallFunction(
        _pace.libxsmmlinear_plain.default, unsq_down, down_w, down_bias
    )
    return CallFunction(_RESHAPE_OR_VIEW, down_3d, Ignored())


def _gated_stub(with_bias: bool) -> Callable[..., torch.Tensor]:
    if with_bias:

        def stub(
            x: torch.Tensor,
            merged_gate_up_w: torch.Tensor,
            merged_gate_up_bias: torch.Tensor,
            down_w: torch.Tensor,
            down_bias: torch.Tensor,
            gate_start: int,
            gate_end: int,
            up_start: int,
            up_end: int,
        ) -> torch.Tensor:
            out = torch.empty(
                x.shape[0],
                down_w.shape[0] * down_w.shape[3],
                dtype=x.dtype,
                device=x.device,
            )
            _ = (
                merged_gate_up_w,
                merged_gate_up_bias,
                down_bias,
                gate_start,
                gate_end,
                up_start,
                up_end,
            )
            return out

    else:

        def stub(
            x: torch.Tensor,
            merged_gate_up_w: torch.Tensor,
            down_w: torch.Tensor,
            gate_start: int,
            gate_end: int,
            up_start: int,
            up_end: int,
        ) -> torch.Tensor:
            out = torch.empty(
                x.shape[0],
                down_w.shape[0] * down_w.shape[3],
                dtype=x.dtype,
                device=x.device,
            )
            _ = merged_gate_up_w, gate_start, gate_end, up_start, up_end
            return out

    return stub


def _gated_replacement(op_str: str, with_bias: bool) -> Callable[..., torch.Tensor]:
    if with_bias:

        def repl(
            x,
            merged_gate_up_w,
            merged_gate_up_bias,
            down_w,
            down_bias,
            gate_start,
            gate_end,
            up_start,
            up_end,
        ):
            # Bias is 1D [2*I] so it splits on its only axis with the
            # raw activation offsets; weight slicing follows the no-bias
            # branch's block_size derivation.
            block_size = torch.ops.aten.sym_size.int(merged_gate_up_w, 3)
            gate_w_start = gate_start // block_size
            gate_w_end = gate_end // block_size
            up_w_start = up_start // block_size
            up_w_end = up_end // block_size
            gate_w = torch.ops.aten.slice.Tensor(
                merged_gate_up_w, 0, gate_w_start, gate_w_end
            )
            up_w = torch.ops.aten.slice.Tensor(
                merged_gate_up_w, 0, up_w_start, up_w_end
            )
            gate_bias = torch.ops.aten.slice.Tensor(
                merged_gate_up_bias, 0, gate_start, gate_end
            )
            up_bias = torch.ops.aten.slice.Tensor(
                merged_gate_up_bias, 0, up_start, up_end
            )

            x_3d = torch.ops.aten.unsqueeze.default(x, 0)
            out_3d = torch.ops.pace.libxsmm_fused_mlp.default(
                x_3d, gate_w, up_w, down_w, gate_bias, up_bias, down_bias, op_str
            )
            t = torch.ops.aten.sym_size.int(x, 0)
            return torch.ops.aten.reshape.default(out_3d, [t, -1])

    else:

        def repl(
            x,
            merged_gate_up_w,
            down_w,
            gate_start,
            gate_end,
            up_start,
            up_end,
        ):
            # Translate dim=1 activation offsets into dim=0 packed-weight
            # offsets. TPP layout is [out/block, in/64, 32, block, 2], so
            # block_size lives at dim=3.
            block_size = torch.ops.aten.sym_size.int(merged_gate_up_w, 3)
            gate_w_start = gate_start // block_size
            gate_w_end = gate_end // block_size
            up_w_start = up_start // block_size
            up_w_end = up_end // block_size
            gate_w = torch.ops.aten.slice.Tensor(
                merged_gate_up_w, 0, gate_w_start, gate_w_end
            )
            up_w = torch.ops.aten.slice.Tensor(
                merged_gate_up_w, 0, up_w_start, up_w_end
            )

            x_3d = torch.ops.aten.unsqueeze.default(x, 0)
            out_3d = torch.ops.pace.libxsmm_fused_mlp.default(
                x_3d, gate_w, up_w, down_w, None, None, None, op_str
            )
            t = torch.ops.aten.sym_size.int(x, 0)
            return torch.ops.aten.reshape.default(out_3d, [t, -1])

    return repl


def _gated_example_inputs(with_bias: bool) -> list:
    # Llama-3.1-8B shapes; pattern is shape-agnostic (every non-KeywordArg
    # slot is `Ignored()`). Trailing ints are the canonical midpoint split.
    t, hidden_in, intermediate, hidden_out = 16, 4096, 14336, 4096
    inputs: list = [
        torch.empty(t, hidden_in, dtype=torch.bfloat16),
        torch.empty(
            (2 * intermediate) // 32, hidden_in // 64, 32, 32, 2, dtype=torch.bfloat16
        ),
    ]
    if with_bias:
        inputs.append(torch.empty(2 * intermediate, dtype=torch.bfloat16))
    inputs.append(
        torch.empty(
            hidden_out // 32, intermediate // 64, 32, 32, 2, dtype=torch.bfloat16
        )
    )
    if with_bias:
        inputs.append(torch.empty(hidden_out, dtype=torch.bfloat16))
    inputs.extend([0, intermediate, intermediate, 2 * intermediate])
    return inputs


def _ungated_pattern(inner_fn: Callable[[Any], Any], with_bias: bool) -> Any:
    x = KeywordArg("x")
    fc1_w = KeywordArg("fc1_w")
    fc2_w = KeywordArg("fc2_w")
    # See _gated_pattern: with-bias / no-bias are distinct FX shapes.
    fc1_bias = KeywordArg("fc1_bias") if with_bias else None
    fc2_bias = KeywordArg("fc2_bias") if with_bias else None

    unsq1 = CallFunction(_aten.unsqueeze.default, x, Ignored())
    fc1_3d = CallFunction(_pace.libxsmmlinear_plain.default, unsq1, fc1_w, fc1_bias)
    fc1_2d = CallFunction(_RESHAPE_OR_VIEW, fc1_3d, Ignored())

    act = inner_fn(fc1_2d)

    unsq2 = CallFunction(_aten.unsqueeze.default, act, Ignored())
    fc2_3d = CallFunction(_pace.libxsmmlinear_plain.default, unsq2, fc2_w, fc2_bias)
    return CallFunction(_RESHAPE_OR_VIEW, fc2_3d, Ignored())


def _ungated_stub(with_bias: bool) -> Callable[..., torch.Tensor]:
    if with_bias:

        def stub(
            x: torch.Tensor,
            fc1_w: torch.Tensor,
            fc1_bias: torch.Tensor,
            fc2_w: torch.Tensor,
            fc2_bias: torch.Tensor,
        ) -> torch.Tensor:
            out = torch.empty(
                x.shape[0],
                fc2_w.shape[0] * fc2_w.shape[3],
                dtype=x.dtype,
                device=x.device,
            )
            _ = fc1_w
            _ = fc1_bias
            _ = fc2_bias
            return out

    else:

        def stub(
            x: torch.Tensor,
            fc1_w: torch.Tensor,
            fc2_w: torch.Tensor,
        ) -> torch.Tensor:
            out = torch.empty(
                x.shape[0],
                fc2_w.shape[0] * fc2_w.shape[3],
                dtype=x.dtype,
                device=x.device,
            )
            _ = fc1_w
            return out

    return stub


def _ungated_replacement(op_str: str, with_bias: bool) -> Callable[..., torch.Tensor]:
    if with_bias:

        def repl(x, fc1_w, fc1_bias, fc2_w, fc2_bias):
            x_3d = torch.ops.aten.unsqueeze.default(x, 0)
            out_3d = torch.ops.pace.libxsmm_fused_mlp.default(
                x_3d, None, fc1_w, fc2_w, None, fc1_bias, fc2_bias, op_str
            )
            t = torch.ops.aten.sym_size.int(x, 0)
            return torch.ops.aten.reshape.default(out_3d, [t, -1])

    else:

        def repl(x, fc1_w, fc2_w):
            x_3d = torch.ops.aten.unsqueeze.default(x, 0)
            out_3d = torch.ops.pace.libxsmm_fused_mlp.default(
                x_3d, None, fc1_w, fc2_w, None, None, None, op_str
            )
            t = torch.ops.aten.sym_size.int(x, 0)
            return torch.ops.aten.reshape.default(out_3d, [t, -1])

    return repl


def _ungated_example_inputs(with_bias: bool) -> list[torch.Tensor]:
    t, hidden_in, intermediate, hidden_out = 16, 4096, 16384, 4096
    inputs = [
        torch.empty(t, hidden_in, dtype=torch.bfloat16),
        torch.empty(
            intermediate // 32, hidden_in // 64, 32, 32, 2, dtype=torch.bfloat16
        ),
    ]
    if with_bias:
        inputs.append(torch.empty(intermediate, dtype=torch.bfloat16))
    inputs.append(
        torch.empty(
            hidden_out // 32, intermediate // 64, 32, 32, 2, dtype=torch.bfloat16
        )
    )
    if with_bias:
        inputs.append(torch.empty(hidden_out, dtype=torch.bfloat16))
    return inputs


_GATED = _Topology(
    name="gated",
    build_pattern=_gated_pattern,
    stub_for=_gated_stub,
    make_replacement=_gated_replacement,
    example_inputs=_gated_example_inputs,
)

_UNGATED = _Topology(
    name="ungated",
    build_pattern=_ungated_pattern,
    stub_for=_ungated_stub,
    make_replacement=_ungated_replacement,
    example_inputs=_ungated_example_inputs,
)


@dataclass(frozen=True)
class _Variant:
    topology: _Topology
    activation: str  # key into _ACTIVATIONS
    with_bias: bool


# Full cross product so the kernel surface is reachable from compile
# mode without a follow-up patch when a new model lands.
_VARIANTS: tuple[_Variant, ...] = (
    _Variant(_GATED, "silu", False),  # Llama / Qwen2 / Phi3 / Phi-4 (torch <= 2.10)
    _Variant(_GATED, "silu", True),
    _Variant(_GATED, "silu_div", False),  # Llama / Qwen2 / Phi3 / Phi-4 (torch >= 2.11)
    _Variant(_GATED, "silu_div", True),
    _Variant(_GATED, "gelu_tanh", False),  # Gemma-3
    _Variant(_GATED, "gelu_tanh", True),
    _Variant(_GATED, "gelu_exact", False),
    _Variant(_GATED, "gelu_exact", True),
    _Variant(_GATED, "relu", False),
    _Variant(_GATED, "relu", True),
    _Variant(_UNGATED, "silu", True),
    _Variant(_UNGATED, "silu", False),
    _Variant(_UNGATED, "silu_div", True),
    _Variant(_UNGATED, "silu_div", False),
    _Variant(_UNGATED, "gelu_tanh", True),  # Phi-2 / gelu_new
    _Variant(_UNGATED, "gelu_tanh", False),
    _Variant(_UNGATED, "gelu_exact", True),  # BERT-style
    _Variant(_UNGATED, "gelu_exact", False),
    _Variant(_UNGATED, "relu", True),  # OPT
    _Variant(_UNGATED, "relu", False),
)


def _register_variant(patterns: PatternMatcherPass, v: _Variant) -> None:
    inner_fn, op_str = _ACTIVATIONS[v.activation]
    register_replacement(
        v.topology.stub_for(v.with_bias),
        v.topology.make_replacement(op_str, v.with_bias),
        v.topology.example_inputs(v.with_bias),
        fwd_only,
        patterns,
        search_fn_pattern=v.topology.build_pattern(inner_fn, v.with_bias),
    )


class FusedMLPPass(VllmPatternMatcherPass):
    """Replace MLP blocks with `pace::libxsmm_fused_mlp`."""

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.disabled = _resolve_disabled_from_env()
        if self.disabled:
            logger.info(
                "pace-vllm: FusedMLPPass disabled via %s; MLPs will run "
                "as three separate libxsmmlinear_plain calls.",
                _FUSED_MLP_DISABLE_ENV,
            )
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="pace_fused_mlp_pass"
        )
        for variant in _VARIANTS:
            _register_variant(self.patterns, variant)
        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        if self.disabled:
            return
        self.matched_count = self.patterns.apply(graph)
        logger.info("pace-vllm: FusedMLPPass matched %d MLP sites.", self.matched_count)
