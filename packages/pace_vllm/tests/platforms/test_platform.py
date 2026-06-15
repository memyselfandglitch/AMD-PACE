# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Tests for `PacePlatform` -- AVX512_BF16 host detection, inheritance from
`CpuPlatform`, and the TP=1 UniProcExecutor preservation."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from pace_vllm.platform import (
    _PACE_WORKER_CLS,
    PacePlatform,
    _cpu_supports_avx512_bf16,
)


class TestPlatform(unittest.TestCase):
    def test_cpu_probe_returns_bool(self) -> None:
        self.assertIsInstance(_cpu_supports_avx512_bf16(), bool)

    def test_is_available_agrees_with_cpu_probe(self) -> None:
        self.assertEqual(PacePlatform.is_available(), _cpu_supports_avx512_bf16())

    def test_inherits_from_cpu_platform(self) -> None:
        from vllm.platforms.cpu import CpuPlatform

        self.assertTrue(issubclass(PacePlatform, CpuPlatform))

    def test_enum_stays_cpu(self) -> None:
        """`_enum = PlatformEnum.CPU` is load-bearing -- if it changes, every
        `CustomOp` dispatch stops routing to `forward_cpu`."""
        from vllm.platforms.interface import PlatformEnum

        self.assertEqual(PacePlatform._enum, PlatformEnum.CPU)


def _fake_cpu_check_and_update_config(cls, vllm_config):
    """Stand-in for `vllm.platforms.cpu.CpuPlatform.check_and_update_config`
    that performs only the bits the UniProc tests care about: rewrite the
    executor `'uni' -> 'mp'` (vLLM 0.20+ does this unconditionally) and
    resolve `worker_cls == 'auto'` to the stock CPU worker. Everything
    else the real parent does is irrelevant here."""
    pc = vllm_config.parallel_config
    if pc.distributed_executor_backend == "uni":
        pc.distributed_executor_backend = "mp"
    if pc.worker_cls == "auto":
        pc.worker_cls = "vllm.v1.worker.cpu_worker.CPUWorker"


def _make_config(world_size: int, executor, dtype: torch.dtype = torch.bfloat16):
    """Minimal `vllm_config`-shaped stub covering the attributes
    PacePlatform.check_and_update_config touches: parallel config (TP /
    executor / worker_cls), cache config (prefix caching), compilation
    config (custom_ops), model config (dtype -- the bf16-only check
    runs first)."""
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            world_size=world_size,
            distributed_executor_backend=executor,
            worker_cls="auto",
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        compilation_config=SimpleNamespace(custom_ops=[]),
        model_config=SimpleNamespace(dtype=dtype),
    )


def _run_with_fake_super(cfg) -> None:
    """Invoke `PacePlatform.check_and_update_config(cfg)` with the
    `CpuPlatform.check_and_update_config` super replaced by the
    `_fake_cpu_check_and_update_config` stub above."""
    from vllm.platforms.cpu import CpuPlatform

    with patch.object(
        CpuPlatform,
        "check_and_update_config",
        new=classmethod(_fake_cpu_check_and_update_config),
    ):
        PacePlatform.check_and_update_config(cfg)


class TestCheckAndUpdateConfigUniProc(unittest.TestCase):
    """vLLM 0.20+ `CpuPlatform.check_and_update_config` rewrites
    `distributed_executor_backend == 'uni'` to `'mp'` unconditionally;
    pace-vllm restores `'uni'` for `world_size == 1` because the
    OMP-bind concern that motivates 'mp' doesn't apply when the launch
    script (`OMP_NUM_THREADS=... numactl ... python`) sets the env in
    the parent shell. The MP executor costs ~20% on BS=1 decode."""

    def test_tp1_uni_executor_is_restored_after_super_flip(self) -> None:
        # Canonical regression: TP=1 + user picked 'uni' (or default).
        # Super flips to 'mp'; we restore to 'uni'.
        cfg = _make_config(world_size=1, executor="uni")
        _run_with_fake_super(cfg)
        self.assertEqual(cfg.parallel_config.distributed_executor_backend, "uni")

    def test_tp1_none_executor_left_alone_when_super_does_not_flip(self) -> None:
        # vLLM 0.20 only flips the literal 'uni' string, not None. We
        # must not spuriously rewrite None -> 'uni': that would be a
        # behavior change and could mask vLLM's own default resolution.
        cfg = _make_config(world_size=1, executor=None)
        _run_with_fake_super(cfg)
        self.assertIsNone(cfg.parallel_config.distributed_executor_backend)

    def test_tp1_explicit_mp_is_left_alone(self) -> None:
        # User explicitly asked for 'mp' -> we honor it; keep_uniproc
        # is False because prev_executor not in {None, 'uni'}.
        cfg = _make_config(world_size=1, executor="mp")
        _run_with_fake_super(cfg)
        self.assertEqual(cfg.parallel_config.distributed_executor_backend, "mp")

    def test_tp1_explicit_other_backend_is_left_alone(self) -> None:
        # 'ray' or any other explicit backend stays put.
        cfg = _make_config(world_size=1, executor="ray")
        _run_with_fake_super(cfg)
        self.assertEqual(cfg.parallel_config.distributed_executor_backend, "ray")

    def test_tp_gt_1_uni_to_mp_flip_is_not_undone(self) -> None:
        # When world_size > 1 we genuinely need MP (or whatever super
        # picked) -- the OMP shim DOES matter for distributed execution.
        cfg = _make_config(world_size=2, executor="uni")
        _run_with_fake_super(cfg)
        self.assertEqual(cfg.parallel_config.distributed_executor_backend, "mp")

    def test_worker_cls_always_swapped_to_pace(self) -> None:
        # Plan 1's basic invariant: worker_cls is always pace's
        # regardless of executor backend.
        cfg = _make_config(world_size=1, executor="uni")
        _run_with_fake_super(cfg)
        self.assertEqual(cfg.parallel_config.worker_cls, _PACE_WORKER_CLS)


class TestCheckAndUpdateConfigBf16Guard(unittest.TestCase):
    """Pace's slab attention C++ kernel is bf16-only.
    check_and_update_config must fail loudly at startup with an
    actionable message rather than letting the user wait through
    model load + profiling before PaceKVCache rejects the dtype."""

    def test_bf16_dtype_passes_through(self) -> None:
        cfg = _make_config(world_size=1, executor="uni", dtype=torch.bfloat16)
        _run_with_fake_super(cfg)
        # Got past the guard; worker was swapped as the normal path does.
        self.assertEqual(cfg.parallel_config.worker_cls, _PACE_WORKER_CLS)

    def test_fp16_dtype_raises_with_actionable_message(self) -> None:
        cfg = _make_config(world_size=1, executor="uni", dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "bfloat16") as cm:
            PacePlatform.check_and_update_config(cfg)
        msg = str(cm.exception)
        # Message must name the knob (dtype=bfloat16) and the bypass
        # (unset VLLM_PLUGINS) so the user knows their two options
        # without reading pace's source.
        self.assertIn("dtype='bfloat16'", msg)
        self.assertIn("VLLM_PLUGINS", msg)

    def test_fp32_dtype_raises(self) -> None:
        cfg = _make_config(world_size=1, executor="uni", dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            PacePlatform.check_and_update_config(cfg)

    def test_guard_runs_before_super(self) -> None:
        # If the bf16 guard runs first, super().check_and_update_config
        # never executes -- our fake super would have run and flipped
        # worker_cls / executor, but here neither should happen because
        # we raised. This catches a regression that moves the guard
        # after super().
        cfg = _make_config(world_size=1, executor="uni", dtype=torch.float16)
        with self.assertRaises(ValueError):
            _run_with_fake_super(cfg)
        # worker_cls still "auto" -> the fake super never ran.
        self.assertEqual(cfg.parallel_config.worker_cls, "auto")


if __name__ == "__main__":
    unittest.main()
