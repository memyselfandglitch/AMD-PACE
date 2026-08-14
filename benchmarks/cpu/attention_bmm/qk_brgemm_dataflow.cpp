/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 *
 * Compare PACE's full-K QK^T BRGeMM with tiled IKJ and KIJ reductions.
 ******************************************************************************/

#include "oneapi/dnnl/dnnl_ukernel.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using BFloat16 = uint16_t;
using dnnl::memory;
using dnnl::ukernel::brgemm;
using dnnl::ukernel::pack_type;
using dnnl::ukernel::transform;

constexpr int64_t kQueryTile = 64;
constexpr int64_t kKvTile = 64;

struct Options {
  std::vector<int64_t> query_lengths{64, 128, 256, 512};
  std::vector<int64_t> kv_lengths{2048, 8192, 16384};
  std::vector<int64_t> head_dims{64, 128};
  std::vector<uint64_t> data_seeds{11, 29, 47};
  int64_t k_chunk = 32;
  int warmups = 2;
  int repeats = 20;
  uint64_t order_seed = 20260814;
  std::string out = "qk_brgemm_dataflow_trials.csv";
};

std::vector<std::string> split(const std::string& text) {
  std::vector<std::string> values;
  size_t begin = 0;
  while (begin < text.size()) {
    const size_t end = text.find(',', begin);
    values.push_back(text.substr(begin, end - begin));
    if (end == std::string::npos)
      break;
    begin = end + 1;
  }
  return values;
}

std::vector<int64_t> dimension_list(const std::string& text) {
  std::vector<int64_t> values;
  for (const auto& token : split(text)) {
    const int64_t value = std::stoll(token);
    if (value <= 0 || value % 64 != 0)
      throw std::invalid_argument("dimensions must be positive multiples of 64");
    values.push_back(value);
  }
  return values;
}

std::vector<uint64_t> seed_list(const std::string& text) {
  std::vector<uint64_t> values;
  for (const auto& token : split(text))
    values.push_back(std::stoull(token));
  return values;
}

Options options(int argc, char** argv) {
  Options result;
  for (int index = 1; index < argc; ++index) {
    const std::string name = argv[index];
    if (index + 1 >= argc)
      throw std::invalid_argument("missing value for " + name);
    const std::string value = argv[++index];
    if (name == "--query-lens")
      result.query_lengths = dimension_list(value);
    else if (name == "--kv-lens")
      result.kv_lengths = dimension_list(value);
    else if (name == "--head-dims")
      result.head_dims = dimension_list(value);
    else if (name == "--data-seeds")
      result.data_seeds = seed_list(value);
    else if (name == "--k-chunk")
      result.k_chunk = std::stoll(value);
    else if (name == "--warmups")
      result.warmups = std::stoi(value);
    else if (name == "--repeats")
      result.repeats = std::stoi(value);
    else if (name == "--order-seed")
      result.order_seed = std::stoull(value);
    else if (name == "--out")
      result.out = value;
    else
      throw std::invalid_argument("unknown option: " + name);
  }
  if (result.warmups < 0 || result.repeats <= 0 || result.k_chunk <= 0 ||
      result.k_chunk % 16 != 0 || result.query_lengths.empty() ||
      result.kv_lengths.empty() || result.head_dims.empty() ||
      result.data_seeds.empty())
    throw std::invalid_argument("invalid list, K chunk, or repeat count");
  for (int64_t head_dim : result.head_dims)
    if (head_dim % result.k_chunk != 0)
      throw std::invalid_argument("K chunk must divide every head dimension");
  return result;
}

BFloat16 to_bfloat16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t rounding = 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<BFloat16>((bits + rounding) >> 16U);
}

struct Workload {
  int64_t query_len;
  int64_t kv_len;
  int64_t head_dim;
  int64_t k_chunk;
  int64_t query_tiles;
  int64_t kv_tiles;
  int64_t k_chunks;
  std::vector<BFloat16> query;
  std::vector<BFloat16> key;
  std::vector<BFloat16> key_full_packed;
  std::vector<BFloat16> key_chunk_packed;

  Workload(
      int64_t query_len,
      int64_t kv_len,
      int64_t head_dim,
      int64_t k_chunk,
      uint64_t seed)
      : query_len(query_len),
        kv_len(kv_len),
        head_dim(head_dim),
        k_chunk(k_chunk),
        query_tiles(query_len / kQueryTile),
        kv_tiles(kv_len / kKvTile),
        k_chunks(head_dim / k_chunk),
        query(static_cast<size_t>(query_len * head_dim)),
        key(static_cast<size_t>(kv_len * head_dim)),
        key_full_packed(static_cast<size_t>(kv_len * head_dim)),
        key_chunk_packed(static_cast<size_t>(kv_len * head_dim)) {
    std::mt19937_64 random(seed);
    std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
    for (auto& value : query)
      value = to_bfloat16(distribution(random));
    for (auto& value : key)
      value = to_bfloat16(distribution(random));
  }

  size_t full_key_offset(int64_t kv_tile) const {
    return static_cast<size_t>(kv_tile * head_dim * kKvTile);
  }

  size_t chunk_key_offset(int64_t k_index, int64_t kv_tile) const {
    return static_cast<size_t>(
        (k_index * kv_tiles + kv_tile) * k_chunk * kKvTile);
  }

  size_t score_offset(int64_t query_tile, int64_t kv_tile) const {
    return static_cast<size_t>(
        (query_tile * kv_tiles + kv_tile) * kQueryTile * kKvTile);
  }
};

class QkKernel {
 public:
  QkKernel(int64_t head_dim, int64_t k_chunk)
      : full_kernel_(
            kQueryTile,
            kKvTile,
            head_dim,
            1,
            head_dim,
            kKvTile,
            kKvTile,
            memory::data_type::bf16,
            memory::data_type::bf16,
            memory::data_type::f32),
        chunk_kernel_(
            kQueryTile,
            kKvTile,
            k_chunk,
            1,
            head_dim,
            kKvTile,
            kKvTile,
            memory::data_type::bf16,
            memory::data_type::bf16,
            memory::data_type::f32),
        full_pack_(
            head_dim,
            kKvTile,
            pack_type::trans,
            head_dim,
            kKvTile,
            memory::data_type::bf16,
            memory::data_type::bf16),
        chunk_pack_(
            k_chunk,
            kKvTile,
            pack_type::trans,
            head_dim,
            kKvTile,
            memory::data_type::bf16,
            memory::data_type::bf16),
        offsets_{{0, 0}} {
    full_kernel_.set_add_C(false);
    full_kernel_.finalize();
    full_kernel_.generate();
    chunk_kernel_.set_add_C(true);
    chunk_kernel_.finalize();
    chunk_kernel_.generate();
    full_pack_.generate();
    chunk_pack_.generate();
    const size_t scratch_size = std::max(
        full_kernel_.get_scratchpad_size(),
        chunk_kernel_.get_scratchpad_size());
    scratch_.resize(scratch_size + 64);
  }

  void prepare_keys(Workload& workload) {
    for (int64_t kv_tile = 0; kv_tile < workload.kv_tiles; ++kv_tile) {
      const BFloat16* key_block = workload.key.data() +
          kv_tile * kKvTile * workload.head_dim;
      full_pack_.execute(
          key_block,
          workload.key_full_packed.data() +
              workload.full_key_offset(kv_tile));
      for (int64_t k_index = 0; k_index < workload.k_chunks; ++k_index)
        chunk_pack_.execute(
            key_block + k_index * workload.k_chunk,
            workload.key_chunk_packed.data() +
                workload.chunk_key_offset(k_index, kv_tile));
    }
  }

  void run_pace_fullk(const Workload& workload, float* scores) {
    full_kernel_.set_hw_context();
    for (int64_t query_tile = 0; query_tile < workload.query_tiles;
         ++query_tile)
      for (int64_t kv_tile = 0; kv_tile < workload.kv_tiles; ++kv_tile)
        execute_full(workload, scores, query_tile, kv_tile);
    brgemm::release_hw_context();
  }

  void run_ikj(const Workload& workload, float* scores) {
    chunk_kernel_.set_hw_context();
    for (int64_t query_tile = 0; query_tile < workload.query_tiles;
         ++query_tile)
      for (int64_t k_index = 0; k_index < workload.k_chunks; ++k_index)
        for (int64_t kv_tile = 0; kv_tile < workload.kv_tiles; ++kv_tile)
          execute_chunk(
              workload, scores, query_tile, k_index, kv_tile);
    brgemm::release_hw_context();
  }

  void run_kij(const Workload& workload, float* scores) {
    chunk_kernel_.set_hw_context();
    for (int64_t k_index = 0; k_index < workload.k_chunks; ++k_index)
      for (int64_t query_tile = 0; query_tile < workload.query_tiles;
           ++query_tile)
        for (int64_t kv_tile = 0; kv_tile < workload.kv_tiles; ++kv_tile)
          execute_chunk(
              workload, scores, query_tile, k_index, kv_tile);
    brgemm::release_hw_context();
  }

 private:
  void execute_full(
      const Workload& workload,
      float* scores,
      int64_t query_tile,
      int64_t kv_tile) {
    full_kernel_.execute(
        workload.query.data() +
            query_tile * kQueryTile * workload.head_dim,
        workload.key_full_packed.data() +
            workload.full_key_offset(kv_tile),
        offsets_,
        scores + workload.score_offset(query_tile, kv_tile),
        scratch_.data());
  }

  void execute_chunk(
      const Workload& workload,
      float* scores,
      int64_t query_tile,
      int64_t k_index,
      int64_t kv_tile) {
    chunk_kernel_.execute(
        workload.query.data() +
            query_tile * kQueryTile * workload.head_dim +
            k_index * workload.k_chunk,
        workload.key_chunk_packed.data() +
            workload.chunk_key_offset(k_index, kv_tile),
        offsets_,
        scores + workload.score_offset(query_tile, kv_tile),
        scratch_.data());
  }

  brgemm full_kernel_;
  brgemm chunk_kernel_;
  transform full_pack_;
  transform chunk_pack_;
  std::vector<std::pair<memory::dim, memory::dim>> offsets_;
  std::vector<uint8_t> scratch_;
};

double max_error(const std::vector<float>& left, const std::vector<float>& right) {
  double result = 0.0;
  for (size_t index = 0; index < left.size(); ++index)
    result = std::max(result, std::abs(double(left[index] - right[index])));
  return result;
}

void run_candidate(
    QkKernel& kernel,
    const Workload& workload,
    const std::string& candidate,
    float* scores) {
  if (candidate == "pace_fullk")
    kernel.run_pace_fullk(workload, scores);
  else if (candidate == "ikj")
    kernel.run_ikj(workload, scores);
  else if (candidate == "kij")
    kernel.run_kij(workload, scores);
  else
    throw std::invalid_argument("unknown candidate: " + candidate);
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Options config = options(argc, argv);
    std::ofstream output(config.out);
    if (!output)
      throw std::runtime_error("cannot open output: " + config.out);
    output << "head_dim,query_len,kv_len,k_chunk,data_seed,round,"
              "order_position,dataflow,elapsed_ms,gflops,reference_error,"
              "max_abs_error,correct,query_tile,kv_tile,order_seed\n";
    output << std::setprecision(10);
    std::mt19937_64 order_random(config.order_seed);
    int64_t trial_count = 0;

    for (int64_t head_dim : config.head_dims) {
      QkKernel kernel(head_dim, config.k_chunk);
      for (int64_t query_len : config.query_lengths) {
        for (int64_t kv_len : config.kv_lengths) {
          for (uint64_t data_seed : config.data_seeds) {
            Workload workload(
                query_len, kv_len, head_dim, config.k_chunk, data_seed);
            kernel.prepare_keys(workload);
            const size_t score_elements =
                static_cast<size_t>(query_len * kv_len);
            std::vector<float> actual(score_elements);
            std::vector<float> pace_reference(score_elements, 0.0f);
            std::vector<float> ikj_reference(score_elements, 0.0f);
            std::vector<float> kij_reference(score_elements, 0.0f);
            kernel.run_pace_fullk(workload, pace_reference.data());
            kernel.run_ikj(workload, ikj_reference.data());
            kernel.run_kij(workload, kij_reference.data());
            const double ikj_kij_error =
                max_error(ikj_reference, kij_reference);
            const double ikj_pace_error =
                max_error(ikj_reference, pace_reference);
            const double kij_pace_error =
                max_error(kij_reference, pace_reference);
            if (ikj_kij_error > 1.0e-5 || ikj_pace_error > 2.0e-3 ||
                kij_pace_error > 2.0e-3)
              throw std::runtime_error("QK^T candidate correctness mismatch");

            std::array<std::string, 3> order{{"pace_fullk", "ikj", "kij"}};
            const int rounds = config.warmups + config.repeats;
            for (int round = 0; round < rounds; ++round) {
              std::shuffle(order.begin(), order.end(), order_random);
              for (size_t position = 0; position < order.size(); ++position) {
                std::fill(actual.begin(), actual.end(), 0.0f);
                const auto start = Clock::now();
                run_candidate(
                    kernel, workload, order[position], actual.data());
                const auto end = Clock::now();
                const double elapsed_ms =
                    std::chrono::duration<double, std::milli>(end - start)
                        .count();
                const std::vector<float>* reference = &pace_reference;
                double reference_error = 0.0;
                if (order[position] == "ikj") {
                  reference = &ikj_reference;
                  reference_error = ikj_pace_error;
                } else if (order[position] == "kij") {
                  reference = &kij_reference;
                  reference_error = kij_pace_error;
                }
                const double error = max_error(actual, *reference);
                const bool correct = error <= 1.0e-5;
                if (!correct)
                  throw std::runtime_error("measured output failed correctness");
                if (round < config.warmups)
                  continue;
                const double operations =
                    2.0 * query_len * kv_len * head_dim;
                output << head_dim << ',' << query_len << ',' << kv_len << ','
                       << config.k_chunk << ',' << data_seed << ','
                       << round - config.warmups << ',' << position << ','
                       << order[position] << ',' << elapsed_ms << ','
                       << operations / (elapsed_ms * 1.0e6) << ','
                       << reference_error << ',' << error << ','
                       << (correct ? "true" : "false") << ',' << kQueryTile
                       << ',' << kKvTile << ',' << config.order_seed << '\n';
                ++trial_count;
              }
            }
          }
        }
      }
    }
    std::cout << "Wrote " << trial_count << " trials to " << config.out
              << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
