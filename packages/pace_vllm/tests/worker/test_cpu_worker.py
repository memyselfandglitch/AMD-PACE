# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Tests for `PaceWorker.determine_available_memory` -- the override that
mirrors vLLM's auto-computed CPU KV cache budget back onto cache_config so
`PaceKVCache.from_kv_cache_config` reads a populated field whether the user
set the env var / CLI flag or not."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pace_vllm

pace_vllm._load_pace_native()

from pace_vllm.v1.worker.cpu_worker import PaceWorker  # noqa: E402


def _stub_worker(
    kv_cache_memory_bytes: "int | None" = None,
) -> PaceWorker:
    """Build a barely-initialised PaceWorker via __new__ so the test
    avoids CPUWorker's NUMA / OMP / model-load init chain. Only the
    fields the override touches are populated."""
    worker = PaceWorker.__new__(PaceWorker)
    worker.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(kv_cache_memory_bytes=kv_cache_memory_bytes)
    )
    return worker


class TestPaceWorkerDetermineAvailableMemory(unittest.TestCase):
    def test_super_value_is_stashed_on_cache_config(self) -> None:
        worker = _stub_worker()
        with patch(
            "vllm.v1.worker.cpu_worker.CPUWorker.determine_available_memory",
            return_value=8 * 1024**3,
        ):
            ret = worker.determine_available_memory()
        self.assertEqual(ret, 8 * 1024**3)
        self.assertEqual(
            worker.vllm_config.cache_config.kv_cache_memory_bytes, 8 * 1024**3
        )

    def test_explicit_budget_is_passed_through(self) -> None:
        # When the user set --kv-cache-memory-bytes, vLLM's super() returns
        # the same value; our override must mirror it (no double-counting).
        worker = _stub_worker(kv_cache_memory_bytes=4 * 1024**3)
        with patch(
            "vllm.v1.worker.cpu_worker.CPUWorker.determine_available_memory",
            return_value=4 * 1024**3,
        ):
            ret = worker.determine_available_memory()
        self.assertEqual(ret, 4 * 1024**3)
        self.assertEqual(
            worker.vllm_config.cache_config.kv_cache_memory_bytes, 4 * 1024**3
        )

    def test_non_positive_budget_raises(self) -> None:
        # vLLM's auto path raises on <= 0, but the explicit path doesn't
        # guard `--kv-cache-memory-bytes=0`. Catch here so PaceKVCache
        # never has to reason about a non-positive budget.
        worker = _stub_worker()
        with patch(
            "vllm.v1.worker.cpu_worker.CPUWorker.determine_available_memory",
            return_value=0,
        ):
            with self.assertRaisesRegex(RuntimeError, "non-positive"):
                worker.determine_available_memory()


if __name__ == "__main__":
    unittest.main()
