# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Per-process registry bridging pace-vllm's Python ↔ slab state.

- `_LAYER_REGISTRY`: `layer_name -> (PaceKVCache, SlabPool)`, read by
  `PaceAttentionImpl.forward` on its first call and then cached on
  the impl instance.
- `_CURRENT_KV_CACHE`: the single `PaceKVCache` the current worker
  owns. `PaceAttentionMetadataBuilder.build` reads it to resolve
  `req_id -> slab seq_id` once per step.

Plain module-level state -- pace-vllm runs one worker per process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from pace_vllm.v1.kv_cache import PaceKVCache

# `Any` for the pool because `torch.classes.pace.SlabPool` is not a
# Python-visible type.
_LAYER_REGISTRY: dict[str, tuple["PaceKVCache", Any]] = {}

_CURRENT_KV_CACHE: "PaceKVCache | None" = None


def register_layer(layer_name: str, kv_cache: "PaceKVCache", pool: Any) -> None:
    _LAYER_REGISTRY[layer_name] = (kv_cache, pool)


def lookup_layer(layer_name: str) -> tuple["PaceKVCache", Any] | None:
    return _LAYER_REGISTRY.get(layer_name)


def set_current_kv_cache(kv_cache: "PaceKVCache | None") -> None:
    global _CURRENT_KV_CACHE
    _CURRENT_KV_CACHE = kv_cache


def get_current_kv_cache() -> "PaceKVCache | None":
    return _CURRENT_KV_CACHE


def clear_layer_registry() -> None:
    """Drop every entry. Use only at worker teardown."""
    global _CURRENT_KV_CACHE
    _LAYER_REGISTRY.clear()
    _CURRENT_KV_CACHE = None


def registered_layer_count() -> int:
    return len(_LAYER_REGISTRY)
