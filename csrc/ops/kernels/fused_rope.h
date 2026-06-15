/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#ifndef PACE_FUSED_ROPE_H
#define PACE_FUSED_ROPE_H

#include <ATen/ATen.h>

namespace pace {
namespace kernels {

std::tuple<at::Tensor, at::Tensor> fused_rope_forward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const int64_t unsqueeze_dim);

} // namespace kernels
} // namespace pace

#endif // PACE_FUSED_ROPE_H
