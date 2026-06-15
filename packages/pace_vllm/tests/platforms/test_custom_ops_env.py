# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Tests for the `PACE_VLLM_CUSTOM_OPS` env-var resolver and the
`_enable_pace_custom_ops` helper that mutates `compilation_config.custom_ops`."""

from __future__ import annotations

import os
import types
import unittest
from unittest.mock import patch

import pace_vllm.platform as platform_mod
from pace_vllm.platform import (
    _PACE_CUSTOM_OP_GROUPS,
    _enable_pace_custom_ops,
    _resolve_pace_custom_op_names,
)

_ENV = "PACE_VLLM_CUSTOM_OPS"


class TestResolvePaceCustomOpNames(unittest.TestCase):
    def test_unset_returns_all_groups(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != _ENV}
        with patch.dict(os.environ, env, clear=True):
            names = _resolve_pace_custom_op_names()
        expected = [n for grp in _PACE_CUSTOM_OP_GROUPS.values() for n in grp]
        self.assertEqual(names, expected)

    def test_all_keyword_returns_everything(self) -> None:
        for raw in ("all", "ALL", "true", "1"):
            with self.subTest(raw=raw):
                with patch.dict(os.environ, {_ENV: raw}, clear=False):
                    names = _resolve_pace_custom_op_names()
                expected = [n for grp in _PACE_CUSTOM_OP_GROUPS.values() for n in grp]
                self.assertEqual(names, expected)

    def test_none_returns_empty_list(self) -> None:
        for raw in ("none", "", "0", "false", "  none  "):
            with self.subTest(raw=raw):
                with patch.dict(os.environ, {_ENV: raw}, clear=False):
                    self.assertEqual(_resolve_pace_custom_op_names(), [])

    def test_single_group_selection(self) -> None:
        with patch.dict(os.environ, {_ENV: "rms_norm"}, clear=False):
            names = _resolve_pace_custom_op_names()
        self.assertEqual(names, _PACE_CUSTOM_OP_GROUPS["rms_norm"])

    def test_unknown_group_warns_and_skips(self) -> None:
        # Verify the warning actually fires (not just the resolved set).
        with (
            patch.dict(os.environ, {_ENV: "rms_norm,not_a_group"}, clear=False),
            self.assertLogs(platform_mod.logger, level="WARNING") as cm,
        ):
            names = _resolve_pace_custom_op_names()
        self.assertEqual(names, _PACE_CUSTOM_OP_GROUPS["rms_norm"])
        self.assertTrue(
            any("not_a_group" in m for m in cm.output),
            f"warning for unknown group missing from logs: {cm.output}",
        )


class _FakeVllmConfig:
    """Tiny stand-in for vllm.config.VllmConfig with the fields we touch."""

    def __init__(self, custom_ops: list[str]):
        self.compilation_config = types.SimpleNamespace(custom_ops=custom_ops)


class TestEnablePaceCustomOps(unittest.TestCase):
    def test_appends_plus_entries_for_each_resolved_name(self) -> None:
        # Returns the *group keys* (env-var-shape strings) so users can
        # copy back into PACE_VLLM_CUSTOM_OPS, while still mutating
        # custom_ops with the underlying class names that vLLM expects.
        cfg = _FakeVllmConfig(custom_ops=["none"])
        with patch.dict(os.environ, {_ENV: "rms_norm"}, clear=False):
            enabled_groups = _enable_pace_custom_ops(cfg)
        self.assertEqual(enabled_groups, ["rms_norm"])
        for name in _PACE_CUSTOM_OP_GROUPS["rms_norm"]:
            self.assertIn(f"+{name}", cfg.compilation_config.custom_ops)

    def test_does_not_clobber_user_plus_entries(self) -> None:
        # User already +listed RMSNorm explicitly -- we must not duplicate.
        cfg = _FakeVllmConfig(custom_ops=["none", "+RMSNorm"])
        with patch.dict(os.environ, {_ENV: "rms_norm"}, clear=False):
            _enable_pace_custom_ops(cfg)
        self.assertEqual(cfg.compilation_config.custom_ops.count("+RMSNorm"), 1)

    def test_respects_user_minus_entries(self) -> None:
        # User -listed GemmaRMSNorm -- our +<Name> must not override it.
        cfg = _FakeVllmConfig(custom_ops=["none", "-GemmaRMSNorm"])
        with patch.dict(os.environ, {_ENV: "rms_norm"}, clear=False):
            _enable_pace_custom_ops(cfg)
        self.assertNotIn("+GemmaRMSNorm", cfg.compilation_config.custom_ops)
        # Other rms_norm entries still added.
        self.assertIn("+RMSNorm", cfg.compilation_config.custom_ops)

    def test_adds_none_baseline_if_missing(self) -> None:
        # CustomOp.default_on() requires exactly one of "none" / "all".
        cfg = _FakeVllmConfig(custom_ops=[])
        with patch.dict(os.environ, {_ENV: "rms_norm"}, clear=False):
            _enable_pace_custom_ops(cfg)
        self.assertIn("none", cfg.compilation_config.custom_ops)

    def test_does_not_add_none_if_all_present(self) -> None:
        cfg = _FakeVllmConfig(custom_ops=["all"])
        with patch.dict(os.environ, {_ENV: "rms_norm"}, clear=False):
            _enable_pace_custom_ops(cfg)
        self.assertNotIn("none", cfg.compilation_config.custom_ops)

    def test_disabled_via_env_returns_no_groups(self) -> None:
        cfg = _FakeVllmConfig(custom_ops=["none"])
        with patch.dict(os.environ, {_ENV: "none"}, clear=False):
            enabled_groups = _enable_pace_custom_ops(cfg)
        self.assertEqual(enabled_groups, [])
        # Existing ops untouched.
        self.assertEqual(cfg.compilation_config.custom_ops, ["none"])


class TestCustomOpGroups(unittest.TestCase):
    """Sanity-check the constants don't drift silently."""

    def test_groups_exist(self) -> None:
        self.assertIn("rms_norm", _PACE_CUSTOM_OP_GROUPS)

    def test_rms_norm_group(self) -> None:
        self.assertEqual(
            _PACE_CUSTOM_OP_GROUPS["rms_norm"], ["RMSNorm", "GemmaRMSNorm"]
        )

    def test_env_var_name_constant(self) -> None:
        self.assertEqual(platform_mod._PACE_CUSTOM_OPS_ENV, "PACE_VLLM_CUSTOM_OPS")


if __name__ == "__main__":
    unittest.main()
