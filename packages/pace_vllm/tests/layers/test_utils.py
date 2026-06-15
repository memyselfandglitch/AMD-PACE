# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Tests for the TPP weight-packing helper."""

from __future__ import annotations

import unittest

import torch

from pace_vllm.model_executor.layers.utils import tpp_prepack


class TestTppPrepack(unittest.TestCase):
    def test_packs_valid_bf16_shape(self) -> None:
        # out_features divisible by block_size (32), in_features by 64.
        weight = torch.zeros(128, 192, dtype=torch.bfloat16)
        packed = tpp_prepack(weight, block_size=32)
        self.assertIsNotNone(packed)
        # Expected 5D layout: (out//block_size, in//64, 32, block_size, 2).
        self.assertEqual(packed.shape, (128 // 32, 192 // 64, 32, 32, 2))
        self.assertEqual(packed.dtype, torch.bfloat16)
        self.assertTrue(packed.is_contiguous())

    def test_round_trips_to_original_shape(self) -> None:
        # Packing is a reshape + permute, so total element count is preserved.
        weight = torch.randn(64, 128, dtype=torch.bfloat16)
        packed = tpp_prepack(weight, block_size=32)
        self.assertEqual(packed.numel(), weight.numel())

    def test_returns_none_for_non_bf16(self) -> None:
        weight = torch.zeros(128, 64, dtype=torch.float32)
        self.assertIsNone(tpp_prepack(weight, block_size=32))

    def test_returns_none_for_non_2d(self) -> None:
        weight = torch.zeros(128, 64, 4, dtype=torch.bfloat16)
        self.assertIsNone(tpp_prepack(weight, block_size=32))

    def test_returns_none_for_meta_tensor(self) -> None:
        # CPU weight loading sometimes leaves a meta tensor behind during
        # init; packing one would crash opaquely inside the C++ kernel.
        weight = torch.empty(128, 64, dtype=torch.bfloat16, device="meta")
        self.assertIsNone(tpp_prepack(weight, block_size=32))

    def test_rejects_out_features_not_divisible_by_block_size(self) -> None:
        # 100 not divisible by 32.
        weight = torch.zeros(100, 64, dtype=torch.bfloat16)
        self.assertIsNone(tpp_prepack(weight, block_size=32))

    def test_rejects_in_features_not_divisible_by_64(self) -> None:
        # 100 not divisible by 64.
        weight = torch.zeros(128, 100, dtype=torch.bfloat16)
        self.assertIsNone(tpp_prepack(weight, block_size=32))

    def test_explicit_block_size_argument(self) -> None:
        # block_size=64 also works as long as out_features % 64 == 0 and
        # in_features % 64 == 0.
        weight = torch.zeros(128, 64, dtype=torch.bfloat16)
        packed = tpp_prepack(weight, block_size=64)
        self.assertIsNotNone(packed)
        self.assertEqual(packed.shape, (128 // 64, 64 // 64, 32, 64, 2))


if __name__ == "__main__":
    unittest.main()
