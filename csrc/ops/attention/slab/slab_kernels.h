/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 *
 * SLAB attention kernel declarations.
 *
 * KernelCtx and function declarations only. BRGeMM infrastructure
 * (BrgemmCache, init_brgemm_cache) lives entirely in slab_prefill_avx512.cpp.
 *
 * Implementations in separate compilation units:
 *   - slab_decode_avx512.cpp:   decode_one_head, decode_gqa_group, mtd_one_row
 *   - slab_prefill_avx512.cpp:  prefill_tile (+ BrgemmCache internally)
 *   - slab_attention_avx512.cpp: SlabPool::attention (4-way OMP dispatch)
 ******************************************************************************/

#ifndef PACE_SLAB_KERNELS_H
#define PACE_SLAB_KERNELS_H

#include <ATen/ATen.h>
#include <cstdint>
#include <vector>

namespace pace {
namespace kernels {

// Constants
constexpr int64_t SLAB_DEFAULT_Q_TILE = 64;
constexpr int64_t SLAB_MTD_MAX_Q = 64;
constexpr int64_t SLAB_N_MAX = 64;
constexpr int64_t SLAB_MAX_REP = 16;
constexpr int64_t SLAB_MAX_SPLITS = 16;

struct PartialSoftmax {
  float max_score;
  float sum_exp;
  float output_fp32[512];
};

// KernelCtx: per-call constants shared across all work items.
struct KernelCtx {
  const at::BFloat16* pool_ptr;
  const at::BFloat16* q_ptr;
  at::BFloat16* o_ptr;
  int64_t num_q_heads, head_dim, num_kv, n_rep, blk_size;
  int64_t ph, pb, pvo; // pool strides
  int64_t token_stride; // num_q_heads * head_dim
  float scale_f;
  int64_t sliding_window;
  bool has_sinks;
  const float* sinks_ptr;
};

// Thread-local diagnostic counters for one prefill attention call. Timings are
// CPU nanoseconds summed across work items, not wall-clock durations.
struct alignas(64) PrefillStageProfile {
  uint64_t cache_init_ns = 0, q_prepare_ns = 0, k_pack_ns = 0, qk_ns = 0;
  uint64_t softmax_ns = 0, v_pack_ns = 0, pv_ns = 0, normalize_ns = 0;
  uint64_t cache_init_pairs = 0, q_prepare_pairs = 0, k_pack_pairs = 0;
  uint64_t qk_pairs = 0, softmax_pairs = 0, v_pack_pairs = 0, pv_pairs = 0;
  uint64_t normalize_pairs = 0, work_items = 0, kv_blocks = 0;
};

// Function declarations
// BRGeMM tiled prefill for one (kv_head, q_tile) work item.
// Manages its own thread-local BrgemmCache internally.
void prefill_tile(
    const KernelCtx& ctx,
    int64_t kv_len,
    int64_t q_len,
    int64_t q_offset,
    const std::vector<int64_t>& block_indices,
    int64_t kv_h,
    int64_t qt,
    int64_t query_tile,
    PrefillStageProfile* profile = nullptr);

double prefill_timer_pair_cost_ns();
double prefill_timer_empty_interval_ns();

namespace impl {

void decode_one_head(
    const KernelCtx& ctx,
    const at::BFloat16* q_ptr,
    const std::vector<int64_t>& block_indices,
    int64_t seq_len,
    int64_t kv_h,
    float sink_bias,
    at::BFloat16* out_ptr);

void decode_gqa_group(
    const KernelCtx& ctx,
    const at::BFloat16* const* q_ptrs,
    int64_t n_rep,
    const std::vector<int64_t>& block_indices,
    int64_t seq_len,
    int64_t kv_h,
    const float* sink_biases,
    at::BFloat16* const* out_ptrs);

void decode_gqa_group_partial(
    const KernelCtx& ctx,
    const at::BFloat16* const* q_ptrs,
    int64_t n_rep,
    const std::vector<int64_t>& block_indices,
    int64_t seq_len,
    int64_t kv_h,
    int64_t blk_start,
    int64_t blk_end,
    PartialSoftmax* partials);

void reduce_gqa_partials(
    const PartialSoftmax* partials,
    int64_t num_splits,
    int64_t n_rep,
    int64_t head_dim,
    at::BFloat16* const* out_ptrs);

void multi_token_decode_one_row(
    const KernelCtx& ctx,
    const at::BFloat16* q_row,
    const std::vector<int64_t>& block_indices,
    int64_t kv_len,
    int64_t q_pos,
    int64_t kv_h,
    float sink_bias,
    at::BFloat16* out_ptr);

} // namespace impl
} // namespace kernels
} // namespace pace

#endif // PACE_SLAB_KERNELS_H
