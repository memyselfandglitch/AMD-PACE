# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""PaceModelRunner: vLLM v1 CPU model runner subclass.

Owns the slab-backed KV state. `initialize_kv_cache_tensors` ships zero-sized
placeholders (real K/V lives in `SlabPool`); `initialize_kv_cache` builds the
`PaceKVCache` and populates the attention backend's layer registry;
`_update_states` fans out request lifecycle events (create / remove / reset
on preempt) to the slab.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.cpu_model_runner import CPUModelRunner
from vllm.v1.worker.utils import bind_kv_cache

from pace_vllm.v1.attention.layer_registry import (
    register_layer,
    set_current_kv_cache,
)
from pace_vllm.v1.kv_cache import PaceKVCache

if TYPE_CHECKING:  # pragma: no cover
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger("pace_vllm.v1.worker.cpu_model_runner")


class PaceModelRunner(CPUModelRunner):
    """PACE-aware CPU model runner."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        super().__init__(vllm_config, device)
        self.pace_kv_cache: PaceKVCache | None = None
        logger.info("pace-vllm: PaceModelRunner active.")

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        super().initialize_kv_cache(kv_cache_config)

        self.pace_kv_cache = PaceKVCache.from_kv_cache_config(
            kv_cache_config,
            self.vllm_config,
            shared_kv_cache_layers=self.shared_kv_cache_layers,
        )
        self.pace_kv_cache.set_owner(self)
        set_current_kv_cache(self.pace_kv_cache)

        for name in self.pace_kv_cache.layer_names:
            pool = self.pace_kv_cache.pool_for_layer(
                self.pace_kv_cache.layer_idx_of(name)
            )
            register_layer(name, self.pace_kv_cache, pool)

        # KV-sharing followers alias the target's pool so
        # `lookup_layer(follower)` returns the shared SlabPool. The
        # factory excluded followers from per-pool allocation.
        for follower, target in self.shared_kv_cache_layers.items():
            # Hybrid models may declare KV-sharing among non-attention
            # layers (Mamba/SSM groups) that SlabPool doesn't service;
            # the factory's existing raises cover the cases that should
            # fail loud (attention follower targeting an unknown layer).
            if target not in self.pace_kv_cache.layer_names:
                continue
            target_pool = self.pace_kv_cache.pool_for_layer(
                self.pace_kv_cache.layer_idx_of(target)
            )
            register_layer(follower, self.pace_kv_cache, target_pool)

    def initialize_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> dict[str, torch.Tensor]:
        from vllm.v1.kv_cache_interface import (
            AttentionSpec,
            UniformTypeKVCacheSpecs,
        )

        def _is_attention_group(group) -> bool:
            """Hybrid models (Qwen3-Next, Jamba) ship Mamba / linear-attn
            groups alongside attention groups; SlabPool only services
            attention. Non-attention layers stay owned by vLLM's stock
            state path -- we just don't bind a placeholder for them."""
            raw = group.kv_cache_spec
            if isinstance(raw, AttentionSpec):
                return True
            if isinstance(raw, UniformTypeKVCacheSpecs):
                # All layers in a UniformType group share a spec class;
                # sample any.
                sample = next(iter(raw.kv_cache_specs.values()), None)
                return isinstance(sample, AttentionSpec)
            return False

        def _layer_dtype(group, layer_name: str) -> torch.dtype:
            raw_spec = group.kv_cache_spec
            if isinstance(raw_spec, UniformTypeKVCacheSpecs):
                return raw_spec.kv_cache_specs[layer_name].dtype
            if not isinstance(raw_spec, AttentionSpec):
                raise TypeError(
                    "pace-vllm: expected AttentionSpec or UniformTypeKVCacheSpecs, "
                    f"got {type(raw_spec).__name__}"
                )
            return raw_spec.dtype

        kv_caches: dict[str, torch.Tensor] = {}
        for group in kv_cache_config.kv_cache_groups:
            if not _is_attention_group(group):
                continue
            for layer_name in group.layer_names:
                kv_caches[layer_name] = torch.empty(
                    (0,), dtype=_layer_dtype(group, layer_name)
                )

        # KV-sharing models map the follower's kv_cache onto the target's.
        for layer_name, target in self.shared_kv_cache_layers.items():
            if layer_name in kv_caches and target in kv_caches:
                kv_caches[layer_name] = kv_caches[target]

        # longcat_flash uses num_attn_module=2; preserved for forward-compat.
        num_attn_module = (
            2 if self.model_config.hf_config.model_type == "longcat_flash" else 1
        )
        bind_kv_cache(
            kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_caches,
            num_attn_module,
        )
        logger.info(
            "pace-vllm: bound %d zero-sized KV placeholders (slab owns all).",
            len(kv_caches),
        )
        return kv_caches

    def _update_states(self, scheduler_output: "SchedulerOutput"):
        if self.pace_kv_cache is not None:
            max_len = self.model_config.max_model_len
            n_new = len(scheduler_output.scheduled_new_reqs)
            n_finished = len(scheduler_output.finished_req_ids)
            preempted = scheduler_output.preempted_req_ids or ()
            n_preempted = len(preempted)
            if n_new or n_finished or n_preempted:
                logger.info(
                    "pace-vllm: slab lifecycle step: +%d new, -%d finished, "
                    "!%d preempted (reset).",
                    n_new,
                    n_finished,
                    n_preempted,
                )
            for new_req in scheduler_output.scheduled_new_reqs:
                self.pace_kv_cache.create_sequence(new_req.req_id, max_len)
            for preempted_rid in preempted:
                self.pace_kv_cache.reset_sequence(preempted_rid)
            for finished_rid in scheduler_output.finished_req_ids:
                self.pace_kv_cache.remove_sequence(finished_rid)
        return super()._update_states(scheduler_output)
