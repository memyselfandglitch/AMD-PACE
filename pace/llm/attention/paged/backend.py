# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""
PagedAttentionBackend: receives pre-built PagedAttentionMetadata via kwargs,
flattens QKV from BSNH to [T,N,H], and calls the vLLM C++ free functions
(reshape_and_cache + paged_attention_with_kv_cache) directly.

Shared metadata (slot_mapping, block_table, seq_lens, query_start_loc) is
built once per step in the generator/server.  The scheduler_metadata (work
item partitioning) is built per-layer here, so each layer's sliding_window
is correctly accounted for — matching the vLLM approach and avoiding empty
KV splits that would corrupt output for models with attention sinks.
"""

import math
from typing import Optional

import torch

from pace.llm.attention.base import AttentionBackend
from pace.llm.attention.paged.ops import get_paged_attention_scheduler_metadata


class PagedAttentionBackend(AttentionBackend):

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: int = 0,
        sinks: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = scale if scale is not None else 1.0 / math.sqrt(head_dim)
        self.sliding_window = sliding_window
        self.sinks = sinks

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        kv_cache,
        positions,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            Q: [B, S, num_heads, head_dim] — already RoPE'd, BSNH
            K: [B, S, num_kv_heads, head_dim]
            V: [B, S, num_kv_heads, head_dim]
            kv_cache: PagedKVCache (single layer cache with get_cache_tensors)
            positions: ignored (paged attention uses causal masking internally)
            **kwargs: must contain paged_attn_metadata
        Returns:
            [B, S, num_heads, head_dim]
        """
        paged_attn_metadata = kwargs["paged_attn_metadata"]

        bsz, q_len = Q.shape[0], Q.shape[1]
        num_tokens = bsz * q_len

        # Flatten BSNH [B, S, N, H] -> [T, N, H]
        Q_flat = Q.reshape(num_tokens, self.num_heads, self.head_dim)
        K_flat = K.reshape(num_tokens, self.num_kv_heads, self.head_dim)
        V_flat = V.reshape(num_tokens, self.num_kv_heads, self.head_dim)

        # Get cache tensors from the per-layer PagedKVCache
        key_cache, value_cache = kv_cache.get_cache_tensors()

        # Write K/V into paged cache
        torch.ops.pace.paged_attention_reshape_and_cache(
            K_flat.contiguous(),
            V_flat.contiguous(),
            key_cache,
            value_cache,
            paged_attn_metadata.slot_mapping,
            paged_attn_metadata.isa,
        )

        sw = self.sliding_window
        sw_left = sw - 1 if sw > 0 else -1
        sw_right = 0

        # Build scheduler metadata per-layer with the correct sliding window.
        # This matches the vLLM approach: the scheduler partitions KV work
        # items using the layer's actual sliding_window_size, so it never
        # creates splits outside the window — avoiding empty KV splits that
        # would corrupt output for models with attention sinks (e.g. GPT-OSS).
        scheduler_metadata = get_paged_attention_scheduler_metadata(
            num_reqs=paged_attn_metadata.seq_lens.shape[0],
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            seq_lens=paged_attn_metadata.seq_lens,
            dtype=Q.dtype,
            query_start_loc=paged_attn_metadata.query_start_loc,
            causal=paged_attn_metadata.causal,
            sliding_window_size=self.sliding_window if self.sliding_window > 0 else -1,
            isa=paged_attn_metadata.isa,
            enable_kv_split=True,
        )

        s_aux = self.sinks
        if s_aux is not None:
            if s_aux.dtype != torch.bfloat16:
                s_aux = s_aux.to(torch.bfloat16)
            remainder = s_aux.numel() % 16
            if remainder != 0:
                s_aux = torch.nn.functional.pad(s_aux, (0, 16 - remainder))
            s_aux = s_aux.contiguous()

        output = torch.empty_like(Q_flat)
        torch.ops.pace.paged_attention_with_kv_cache(
            Q_flat.contiguous(),
            key_cache,
            value_cache,
            output,
            paged_attn_metadata.query_start_loc,
            paged_attn_metadata.seq_lens,
            self.scaling,
            paged_attn_metadata.causal,
            None,
            sw_left,
            sw_right,
            paged_attn_metadata.block_table,
            0.0,
            scheduler_metadata,
            s_aux,
        )

        # Packed (ragged) inputs arrive as [1, T, N, H] — the kernel uses
        # query_start_loc to dispatch per-sequence, so this view round-trips correctly.
        return output.view(bsz, q_len, self.num_heads, self.head_dim)
