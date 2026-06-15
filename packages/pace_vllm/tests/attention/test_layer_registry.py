# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Tests for the per-process layer registry that maps `layer_name -> (PaceKVCache,
SlabPool)` for `PaceAttentionImpl.forward` and tracks the active `PaceKVCache`
for the metadata builder."""

from __future__ import annotations

import unittest

from pace_vllm.v1.attention.layer_registry import (
    clear_layer_registry,
    get_current_kv_cache,
    lookup_layer,
    register_layer,
    registered_layer_count,
    set_current_kv_cache,
)


class TestLayerRegistry(unittest.TestCase):
    def setUp(self) -> None:
        clear_layer_registry()

    def tearDown(self) -> None:
        clear_layer_registry()

    def test_register_then_lookup(self) -> None:
        kv = object()
        pool = object()
        register_layer("layer.0", kv, pool)
        self.assertEqual(lookup_layer("layer.0"), (kv, pool))

    def test_lookup_unknown_returns_none(self) -> None:
        self.assertIsNone(lookup_layer("never-registered"))

    def test_register_overwrites_existing(self) -> None:
        register_layer("layer.0", "v1", "p1")
        register_layer("layer.0", "v2", "p2")
        self.assertEqual(lookup_layer("layer.0"), ("v2", "p2"))

    def test_current_kv_cache_round_trip(self) -> None:
        marker = object()
        set_current_kv_cache(marker)
        self.assertIs(get_current_kv_cache(), marker)
        set_current_kv_cache(None)
        self.assertIsNone(get_current_kv_cache())

    def test_clear_empties_layer_map_and_current(self) -> None:
        register_layer("a", "kv", "p")
        register_layer("b", "kv", "p")
        set_current_kv_cache("active")
        clear_layer_registry()
        self.assertIsNone(lookup_layer("a"))
        self.assertIsNone(lookup_layer("b"))
        self.assertIsNone(get_current_kv_cache())
        self.assertEqual(registered_layer_count(), 0)

    def test_registered_layer_count(self) -> None:
        self.assertEqual(registered_layer_count(), 0)
        register_layer("a", "kv", "p")
        register_layer("b", "kv", "p")
        self.assertEqual(registered_layer_count(), 2)

    def test_follower_alias_resolves_to_target_pool(self) -> None:
        # KV-sharing path: PaceModelRunner.initialize_kv_cache registers
        # each unique target layer with its own pool, then re-registers
        # every follower layer to the *target's* pool. lookup_layer for
        # the follower must return the target's (kv_cache, pool) pair so
        # PaceAttentionImpl.forward routes through the shared SlabPool.
        kv = object()
        target_pool = object()
        other_pool = object()

        register_layer("layer_0", kv, target_pool)
        register_layer("layer_1", kv, other_pool)
        # Follower aliases the target's pool.
        register_layer("layer_2", kv, target_pool)

        target_entry = lookup_layer("layer_0")
        follower_entry = lookup_layer("layer_2")
        self.assertIsNotNone(target_entry)
        self.assertIsNotNone(follower_entry)
        # Same kv_cache, same pool object -- the follower transparently
        # shares the target's SlabPool.
        self.assertIs(follower_entry[0], target_entry[0])
        self.assertIs(follower_entry[1], target_entry[1])
        # And the unrelated layer still resolves to its own distinct pool.
        self.assertIsNot(lookup_layer("layer_1")[1], target_pool)


if __name__ == "__main__":
    unittest.main()
