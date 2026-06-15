# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Tests for `pace_vllm.v1.worker.cpu_worker._install_fused_mlp_pass`.

Exercises the three branches: eager-mode early return, already-installed
bypass, and happy-path install.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import pace_vllm

pace_vllm._load_pace_native()

from vllm.config.compilation import CompilationMode  # noqa: E402

from pace_vllm.compilation.fused_mlp_pass import FusedMLPPass  # noqa: E402
from pace_vllm.v1.worker.cpu_worker import _install_fused_mlp_pass  # noqa: E402

_PASS_KEY = "post_grad_custom_post_pass"


def _vllm_config(mode: CompilationMode, custom_pass: object = None) -> SimpleNamespace:
    """Minimal `vllm_config` stand-in covering exactly what
    `_install_fused_mlp_pass` and `FusedMLPPass.__init__` reach for."""
    inductor_compile_config: dict[str, object] = {}
    if custom_pass is not None:
        inductor_compile_config[_PASS_KEY] = custom_pass

    compilation_config = SimpleNamespace(
        mode=mode,
        splitting_ops=[],
        use_inductor_graph_partition=False,
        pass_config=SimpleNamespace(),
        inductor_compile_config=inductor_compile_config,
    )
    return SimpleNamespace(
        compilation_config=compilation_config,
        model_config=None,
        device_config=None,
        compile_debug_dump_path=lambda: None,
    )


class TestInstallFusedMlpPass(unittest.TestCase):
    def test_eager_mode_is_a_noop(self) -> None:
        cfg = _vllm_config(CompilationMode.NONE)
        _install_fused_mlp_pass(cfg)
        self.assertNotIn(_PASS_KEY, cfg.compilation_config.inductor_compile_config)

    def test_existing_pass_is_left_in_place(self) -> None:
        sentinel = object()
        cfg = _vllm_config(CompilationMode.VLLM_COMPILE, custom_pass=sentinel)
        _install_fused_mlp_pass(cfg)
        self.assertIs(
            cfg.compilation_config.inductor_compile_config[_PASS_KEY],
            sentinel,
            "existing pass must not be overwritten",
        )

    def test_compile_mode_installs_fused_mlp_pass(self) -> None:
        cfg = _vllm_config(CompilationMode.VLLM_COMPILE)
        _install_fused_mlp_pass(cfg)
        installed = cfg.compilation_config.inductor_compile_config[_PASS_KEY]
        self.assertIsInstance(installed, FusedMLPPass)


class TestInstallDecisionPerCompilationMode(unittest.TestCase):
    """One row per CompilationMode value, plus a guard that fails when
    a future vLLM adds a new mode -- forces an explicit decision rather
    than silent install-on-non-NONE behaviour. NONE is no-op; the three
    compile modes (STOCK_TORCH_COMPILE, DYNAMO_TRACE_ONCE, VLLM_COMPILE)
    all install the pass."""

    def test_none_is_a_noop(self) -> None:
        cfg = _vllm_config(CompilationMode.NONE)
        _install_fused_mlp_pass(cfg)
        self.assertNotIn(_PASS_KEY, cfg.compilation_config.inductor_compile_config)

    def test_stock_torch_compile_installs_pass(self) -> None:
        cfg = _vllm_config(CompilationMode.STOCK_TORCH_COMPILE)
        _install_fused_mlp_pass(cfg)
        self.assertIsInstance(
            cfg.compilation_config.inductor_compile_config.get(_PASS_KEY),
            FusedMLPPass,
        )

    def test_dynamo_trace_once_installs_pass(self) -> None:
        # The mode the live serving smoke run actually exercises today.
        cfg = _vllm_config(CompilationMode.DYNAMO_TRACE_ONCE)
        _install_fused_mlp_pass(cfg)
        self.assertIsInstance(
            cfg.compilation_config.inductor_compile_config.get(_PASS_KEY),
            FusedMLPPass,
        )

    def test_vllm_compile_installs_pass(self) -> None:
        cfg = _vllm_config(CompilationMode.VLLM_COMPILE)
        _install_fused_mlp_pass(cfg)
        self.assertIsInstance(
            cfg.compilation_config.inductor_compile_config.get(_PASS_KEY),
            FusedMLPPass,
        )

    def test_all_known_modes_are_covered(self) -> None:
        # Forces a conscious decision when vLLM adds a new mode (the
        # dispatcher in _install_fused_mlp_pass currently treats anything
        # other than NONE as a compile mode).
        covered = {
            CompilationMode.NONE,
            CompilationMode.STOCK_TORCH_COMPILE,
            CompilationMode.DYNAMO_TRACE_ONCE,
            CompilationMode.VLLM_COMPILE,
        }
        new_modes = set(CompilationMode) - covered
        self.assertEqual(
            new_modes,
            set(),
            f"new CompilationMode(s) appeared in vLLM: {new_modes}; "
            "decide whether they install the FusedMLPPass and add a row above.",
        )


if __name__ == "__main__":
    unittest.main()
