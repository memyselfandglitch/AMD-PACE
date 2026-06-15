# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Structural tests for `pace_vllm.compilation.fused_mlp_pass`.

Confirms `_ACTIVATIONS` / `_VARIANTS` / topology metadata stays in sync with
the C++ kernel surface, and that `FusedMLPPass(config)` constructs without
side effects when handed a minimal `VllmConfig`-shaped stub.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.fx as fx

import pace_vllm

pace_vllm._load_pace_native()

from pace_vllm.compilation.fused_mlp_pass import (  # noqa: E402
    _ACTIVATIONS,
    _GATED,
    _UNGATED,
    _VARIANTS,
    FusedMLPPass,
    _Topology,
)


def _fake_vllm_config() -> SimpleNamespace:
    """Minimal stand-in for `vllm.config.VllmConfig` covering the fields
    `VllmInductorPass.__init__` and `dump_patterns` actually touch."""
    compilation_config = SimpleNamespace(
        splitting_ops=[],
        use_inductor_graph_partition=False,
        pass_config=SimpleNamespace(),
        inductor_compile_config={},
    )
    return SimpleNamespace(
        compilation_config=compilation_config,
        model_config=None,
        device_config=None,
        compile_debug_dump_path=lambda: None,
    )


class TestActivations(unittest.TestCase):
    def test_activation_keys_registered(self) -> None:
        # `silu` and `silu_div` are two FX shapes for the same C++ enum;
        # both registered so the matcher walks Inductor's torch <= 2.10
        # decomposition (`x * sigmoid(x)`) AND the torch >= 2.11 one
        # (`x / (1 + exp(-x))`). Drift here means a future Inductor
        # decomposition shift slipped through unnoticed.
        self.assertEqual(
            set(_ACTIVATIONS.keys()),
            {"silu", "silu_div", "gelu_tanh", "gelu_exact", "relu"},
        )

    def test_op_strings_match_c_kernel_enums(self) -> None:
        # The C++ libxsmm_fused_mlp kernel accepts exactly three
        # activation enum strings; gelu_tanh / gelu_exact both map to
        # "gelu", and silu / silu_div both map to "silu".
        for key, expected_op_str in (
            ("silu", "silu"),
            ("silu_div", "silu"),
            ("gelu_tanh", "gelu"),
            ("gelu_exact", "gelu"),
            ("relu", "relu"),
        ):
            with self.subTest(key=key):
                _, op_str = _ACTIVATIONS[key]
                self.assertEqual(op_str, expected_op_str)


class TestTopologies(unittest.TestCase):
    def test_topology_instances(self) -> None:
        self.assertIsInstance(_GATED, _Topology)
        self.assertIsInstance(_UNGATED, _Topology)

    def test_topology_callables_are_set(self) -> None:
        for topo in (_GATED, _UNGATED):
            with self.subTest(topo=topo.name):
                self.assertTrue(callable(topo.build_pattern))
                self.assertTrue(callable(topo.stub_for))
                self.assertTrue(callable(topo.make_replacement))
                self.assertTrue(callable(topo.example_inputs))


class TestVariants(unittest.TestCase):
    def test_full_kernel_cross_product_registered(self) -> None:
        # 2 topologies x 5 activation FX shapes (silu, silu_div,
        # gelu_tanh, gelu_exact, relu) x 2 bias states = 20 variants.
        # The kernel surface (gated/ungated x silu/gelu/relu x bias)
        # is fully reachable from compile mode; the silu_div rows
        # cover the torch >= 2.11 Inductor decomposition. Drift here
        # means a row was dropped or duplicated.
        self.assertEqual(len(_VARIANTS), 20)

    def test_variants_cover_expected_combinations(self) -> None:
        expected: set[tuple[str, str, bool]] = {
            (topo, act, bias)
            for topo in ("gated", "ungated")
            for act in ("silu", "silu_div", "gelu_tanh", "gelu_exact", "relu")
            for bias in (False, True)
        }
        actual = {(v.topology.name, v.activation, v.with_bias) for v in _VARIANTS}
        self.assertEqual(actual, expected)

    def test_every_variant_activation_resolves(self) -> None:
        for v in _VARIANTS:
            with self.subTest(variant=v):
                self.assertIn(v.activation, _ACTIVATIONS)

    def test_variant_is_frozen(self) -> None:
        v = _VARIANTS[0]
        with self.assertRaises(Exception):
            v.activation = "relu"  # type: ignore[misc]


class TestFusedMLPPassConstruction(unittest.TestCase):
    def test_construct_registers_patterns_and_logs_no_errors(self) -> None:
        cfg = _fake_vllm_config()
        pass_obj = FusedMLPPass(cfg)
        self.assertFalse(pass_obj.disabled)
        self.assertIsNotNone(pass_obj.patterns)
        # PatternMatcherPass.patterns is a dict[OpOverload, list[...]];
        # we registered 20 variants but multiple variants can land under
        # the same anchor op, so just sanity-check it's non-empty.
        self.assertGreater(len(pass_obj.patterns.patterns), 0)

    def test_empty_graph_matches_zero_sites(self) -> None:
        # `__call__` is annotated `fx.Graph` and vLLM hands it the graph
        # (with `owning_module` pointing at the post-grad GraphModule);
        # `dump_graph` reads `graph.owning_module` so the Graph must be
        # owned by a real GraphModule.
        cfg = _fake_vllm_config()
        pass_obj = FusedMLPPass(cfg)
        graph = _empty_graph()
        pass_obj(graph)
        self.assertEqual(pass_obj.matched_count, 0)

    def test_disabled_short_circuits_call(self) -> None:
        cfg = _fake_vllm_config()
        pass_obj = FusedMLPPass(cfg)
        pass_obj.disabled = True
        pass_obj.matched_count = 7  # sentinel: must not be overwritten
        graph = _empty_graph()
        pass_obj(graph)
        self.assertEqual(pass_obj.matched_count, 7)


def _empty_graph() -> fx.Graph:
    g = fx.Graph()
    x = g.placeholder("x")
    g.output(x)
    fx.GraphModule(torch.nn.Module(), g)  # wires `g.owning_module`
    return g


if __name__ == "__main__":
    unittest.main()
