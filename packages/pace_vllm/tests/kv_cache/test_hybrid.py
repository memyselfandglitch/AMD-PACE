# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Tests for hybrid-model and MLA support gaps.

Two related concerns:

- Hybrid models (Qwen3-Next, Jamba) ship Mamba groups alongside attention
  groups in `KVCacheConfig`. `PaceKVCache.from_kv_cache_config` already
  filters non-AttentionSpec; `PaceModelRunner.initialize_kv_cache_tensors`
  must do the same so `_layer_dtype` never sees a `MambaSpec`.

- MLA (`MLAAttentionSpec`) is a subclass of `FullAttentionSpec`, so the
  existing `isinstance(layer_spec, AttentionSpec)` filter accepts it
  silently. SlabPool's K/V geometry can't service MLA's projection
  layout; we want a loud `NotImplementedError` instead of wrong outputs.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import pace_vllm
from pace_vllm.v1 import kv_cache as kv_cache_module
from pace_vllm.v1.kv_cache import PaceKVCache, _pace_kvcache_budget_bytes
from pace_vllm.v1.worker.cpu_model_runner import PaceModelRunner

pace_vllm._load_pace_native()


# Budget large enough that num_blocks resolves positive for the small
# fixture geometries used here.
_BUDGET_BYTES = 2 * 1024**3


def _vllm_config() -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(kv_cache_memory_bytes=_BUDGET_BYTES)
    )


def _attention_spec(num_kv_heads: int = 8, head_size: int = 128, block_size: int = 16):
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.bfloat16,
        sliding_window=None,
        attention_chunk_size=None,
    )


def _mla_spec():
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    # Standard DeepSeek-V2 shapes: kv_lora_rank=512, qk_rope_head_dim=64
    # -> head_size = 576. The exact numbers don't matter for our reject
    # path; just need the type to be MLAAttentionSpec.
    return MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        sliding_window=None,
        attention_chunk_size=None,
    )


def _mamba_spec():
    from vllm.v1.kv_cache_interface import MambaSpec

    return MambaSpec(
        block_size=16,
        shapes=((128, 64),),
        dtypes=(torch.bfloat16,),
        mamba_type="mamba2",
        mamba_cache_mode="mamba2",
        num_speculative_blocks=0,
    )


def _kv_cache_config(*groups):
    from vllm.v1.kv_cache_interface import KVCacheConfig

    return KVCacheConfig(
        num_blocks=0, kv_cache_tensors=[], kv_cache_groups=list(groups)
    )


def _group(layer_names, spec):
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec

    return KVCacheGroupSpec(layer_names=list(layer_names), kv_cache_spec=spec)


def _uniform_group(name_to_spec):
    """Wrap a dict of `name -> spec` in `UniformTypeKVCacheSpecs` so we
    can exercise the MLA-in-uniform branch."""
    from vllm.v1.kv_cache_interface import (
        KVCacheGroupSpec,
        UniformTypeKVCacheSpecs,
    )

    uniform = UniformTypeKVCacheSpecs(
        block_size=next(iter(name_to_spec.values())).block_size,
        kv_cache_specs=dict(name_to_spec),
    )
    return KVCacheGroupSpec(
        layer_names=list(name_to_spec.keys()), kv_cache_spec=uniform
    )


class TestMambaHybridSkipped(unittest.TestCase):
    """Item 1: PaceModelRunner / PaceKVCache must not crash on
    KVCacheConfig that mixes attention + Mamba groups."""

    def test_initialize_kv_cache_tensors_skips_mamba_group(self) -> None:
        cfg = _kv_cache_config(
            _group(["attn.0", "attn.1"], _attention_spec()),
            _group(["ssm.0", "ssm.1"], _mamba_spec()),
        )
        # Stand in for `PaceModelRunner`-the-instance: only the bound
        # method needs `self` with `shared_kv_cache_layers` and the few
        # attrs `bind_kv_cache` reaches for. We bypass the real ctor
        # (which calls super().__init__ -> CPUModelRunner.__init__,
        # which loads weights) by using a MagicMock and binding the
        # method explicitly.
        runner = MagicMock(spec=PaceModelRunner)
        runner.shared_kv_cache_layers = {}
        runner.model_config = MagicMock()
        runner.model_config.hf_config = MagicMock()
        runner.model_config.hf_config.model_type = "llama"
        runner.compilation_config = MagicMock()
        runner.compilation_config.static_forward_context = {}
        runner.kv_caches = {}

        # Patch out vLLM's bind_kv_cache call -- not exercising that
        # path here.
        import pace_vllm.v1.worker.cpu_model_runner as cmr

        original_bind = cmr.bind_kv_cache
        cmr.bind_kv_cache = lambda *a, **kw: None
        try:
            kv_caches = PaceModelRunner.initialize_kv_cache_tensors(
                runner, cfg, kernel_block_sizes=[16]
            )
        finally:
            cmr.bind_kv_cache = original_bind

        self.assertEqual(set(kv_caches.keys()), {"attn.0", "attn.1"})

    def test_from_kv_cache_config_skips_mamba_group(self) -> None:
        cfg = _kv_cache_config(
            _group(["attn.0", "attn.1"], _attention_spec()),
            _group(["ssm.0"], _mamba_spec()),
        )
        kv = PaceKVCache.from_kv_cache_config(cfg, _vllm_config())
        self.assertEqual(kv.layer_names, ["attn.0", "attn.1"])


class TestMLARejected(unittest.TestCase):
    """Item 2: MLAAttentionSpec must raise NotImplementedError before
    SlabPool gets constructed against the wrong K/V geometry."""

    def test_from_kv_cache_config_rejects_mla_bare(self) -> None:
        cfg = _kv_cache_config(_group(["attn.0", "attn.1"], _mla_spec()))
        with self.assertRaises(NotImplementedError) as cm:
            PaceKVCache.from_kv_cache_config(cfg, _vllm_config())
        self.assertIn("MLA", str(cm.exception))

    def test_from_kv_cache_config_rejects_mla_in_uniform(self) -> None:
        cfg = _kv_cache_config(
            _uniform_group({"attn.0": _mla_spec(), "attn.1": _mla_spec()})
        )
        with self.assertRaises(NotImplementedError) as cm:
            PaceKVCache.from_kv_cache_config(cfg, _vllm_config())
        self.assertIn("MLA", str(cm.exception))


class TestHeterogeneousGeometry(unittest.TestCase):
    """Item 3: PaceKVCache must accept layers with different
    `(num_kv_heads, head_dim)` and build one independently-sized
    SlabPool per layer. Sizing keeps `num_blocks` equal across layers
    using `num_blocks = budget // sum(page_size_bytes)`."""

    def test_heterogeneous_geometry_accepted(self) -> None:
        # Two layers with different num_kv_heads (8 vs 4); both bf16.
        cfg = _kv_cache_config(
            _group(["attn.wide"], _attention_spec(num_kv_heads=8, head_size=128)),
            _group(["attn.narrow"], _attention_spec(num_kv_heads=4, head_size=128)),
        )
        # Force-deterministic block_size so sizing math is checkable.
        with patch.dict(os.environ, {"PACE_VLLM_SLAB_BLOCK_SIZE": "64"}):
            kv = PaceKVCache.from_kv_cache_config(cfg, _vllm_config())
        self.assertEqual(kv.layer_names, ["attn.wide", "attn.narrow"])
        self.assertEqual(kv.spec.per_layer[0].num_kv_heads, 8)
        self.assertEqual(kv.spec.per_layer[1].num_kv_heads, 4)
        # Both layers got SlabPool instances (distinct objects).
        self.assertIsNot(kv.pool_for_layer(0), kv.pool_for_layer(1))

    def test_heterogeneous_sizing_matches_formula(self) -> None:
        # 2 layers, geometry (num_kv_heads=8, head_dim=128) and
        # (num_kv_heads=4, head_dim=128); env-pinned block_size=64.
        # bf16 -> 2 bytes/elem; K+V -> *2.
        # page0 = 2 * 8 * 128 * 64 * 2 = 262144
        # page1 = 2 * 4 * 128 * 64 * 2 = 131072
        # sum = 393216
        # num_blocks = floor(2 GiB / 393216) = 5461
        cfg = _kv_cache_config(
            _group(["attn.wide"], _attention_spec(num_kv_heads=8, head_size=128)),
            _group(["attn.narrow"], _attention_spec(num_kv_heads=4, head_size=128)),
        )
        with patch.dict(os.environ, {"PACE_VLLM_SLAB_BLOCK_SIZE": "64"}):
            kv = PaceKVCache.from_kv_cache_config(cfg, _vllm_config())
        page0 = 2 * 8 * 128 * 64 * 2
        page1 = 2 * 4 * 128 * 64 * 2
        expected_num_blocks = _BUDGET_BYTES // (page0 + page1)
        self.assertEqual(kv.spec.per_layer[0].num_blocks, expected_num_blocks)
        self.assertEqual(kv.spec.per_layer[1].num_blocks, expected_num_blocks)
        self.assertLessEqual(kv.spec.total_bytes, _BUDGET_BYTES)


class TestSlabBlockSizeAutotune(unittest.TestCase):
    """Item 4: `block_size` per layer must come from the C++ L2
    autotuner (or `PACE_VLLM_SLAB_BLOCK_SIZE` override), NOT from
    vLLM's `AttentionSpec.block_size`."""

    def test_autotune_overrides_vllm_block_size(self) -> None:
        # vLLM's spec.block_size is 16; the autotuner says 128. The
        # SlabPool must be sized with 128.
        cfg = _kv_cache_config(
            _group(["attn.0"], _attention_spec(block_size=16)),
        )
        with patch.object(
            kv_cache_module, "_autotune_slab_block_size", return_value=128
        ):
            with patch.dict(os.environ, {}, clear=False):
                # Make sure the env override is unset for this test.
                os.environ.pop("PACE_VLLM_SLAB_BLOCK_SIZE", None)
                kv = PaceKVCache.from_kv_cache_config(cfg, _vllm_config())
        self.assertEqual(kv.spec.per_layer[0].block_size, 128)

    def test_env_override_skips_autotune(self) -> None:
        # When the env is set, autotune must NOT be called.
        cfg = _kv_cache_config(
            _group(["attn.0", "attn.1"], _attention_spec()),
        )
        with patch.object(
            kv_cache_module, "_autotune_slab_block_size"
        ) as mock_autotune:
            with patch.dict(os.environ, {"PACE_VLLM_SLAB_BLOCK_SIZE": "64"}):
                kv = PaceKVCache.from_kv_cache_config(cfg, _vllm_config())
        mock_autotune.assert_not_called()
        for layer in kv.spec.per_layer:
            self.assertEqual(layer.block_size, 64)

    def test_autotune_per_unique_geometry_only(self) -> None:
        # Three layers, only two distinct (num_kv_heads, head_dim)
        # tuples -> autotune called exactly twice.
        cfg = _kv_cache_config(
            _group(["a"], _attention_spec(num_kv_heads=8, head_size=128)),
            _group(["b"], _attention_spec(num_kv_heads=8, head_size=128)),
            _group(["c"], _attention_spec(num_kv_heads=4, head_size=128)),
        )
        with patch.object(
            kv_cache_module, "_autotune_slab_block_size", return_value=64
        ) as mock_autotune:
            os.environ.pop("PACE_VLLM_SLAB_BLOCK_SIZE", None)
            PaceKVCache.from_kv_cache_config(cfg, _vllm_config())
        # Two unique (num_kv_heads, head_dim) pairs -> two calls.
        self.assertEqual(mock_autotune.call_count, 2)
        called_keys = {tuple(call.args) for call in mock_autotune.call_args_list}
        self.assertEqual(called_keys, {(8, 128), (4, 128)})


class TestKvCacheBudgetReader(unittest.TestCase):
    """`_pace_kvcache_budget_bytes` is a thin reader for
    `cache_config.kv_cache_memory_bytes`. It returns the int when set
    positive, None otherwise. The actual auto-budget computation lives
    in `PaceWorker.determine_available_memory`, which writes the
    auto-computed value back onto `cache_config` so this reader sees it
    regardless of whether the user set the env var / CLI flag."""

    def test_positive_value_is_returned(self) -> None:
        cfg = SimpleNamespace(
            cache_config=SimpleNamespace(kv_cache_memory_bytes=_BUDGET_BYTES)
        )
        self.assertEqual(_pace_kvcache_budget_bytes(cfg), _BUDGET_BYTES)

    def test_unset_returns_none(self) -> None:
        cfg = SimpleNamespace(cache_config=SimpleNamespace())
        self.assertIsNone(_pace_kvcache_budget_bytes(cfg))

    def test_zero_budget_returns_none(self) -> None:
        cfg = SimpleNamespace(cache_config=SimpleNamespace(kv_cache_memory_bytes=0))
        self.assertIsNone(_pace_kvcache_budget_bytes(cfg))

    def test_no_cache_config_returns_none(self) -> None:
        cfg = SimpleNamespace()
        self.assertIsNone(_pace_kvcache_budget_bytes(cfg))

    def test_factory_succeeds_with_kv_cache_memory_bytes(self) -> None:
        # End-to-end: from_kv_cache_config reaches the sizing path when
        # the budget field is populated (whether by user or by
        # PaceWorker.determine_available_memory mirror-write).
        cfg = _kv_cache_config(_group(["attn.0"], _attention_spec()))
        vllm_cfg = SimpleNamespace(
            cache_config=SimpleNamespace(kv_cache_memory_bytes=_BUDGET_BYTES)
        )
        with patch.dict(os.environ, {"PACE_VLLM_SLAB_BLOCK_SIZE": "64"}):
            kv = PaceKVCache.from_kv_cache_config(cfg, vllm_cfg)
        self.assertEqual(kv.layer_names, ["attn.0"])


if __name__ == "__main__":
    unittest.main()
