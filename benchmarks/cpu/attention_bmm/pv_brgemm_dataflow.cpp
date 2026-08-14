/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 *
 * Compare tiled IKJ and KIJ traversal with the same oneDNN BF16 BRGeMM.
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

constexpr int64_t kTile = 64;

struct Options {
  std::vector<int64_t> query_lengths{64, 128, 256, 512};
  std::vector<int64_t> kv_lengths{2048, 8192, 16384};
  std::vector<int64_t> head_dims{64, 128};
  std::vector<uint64_t> data_seeds{11, 29, 47};
  int warmups = 2;
  int repeats = 20;
  uint64_t order_seed = 20260813;
  std::string out = "pv_brgemm_dataflow_trials.csv";
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

std::vector<int64_t> integer_list(const std::string& text) {
  std::vector<int64_t> values;
  for (const auto& token : split(text)) {
    const int64_t value = std::stoll(token);
    if (value <= 0 || value % kTile != 0)
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
      result.query_lengths = integer_list(value);
    else if (name == "--kv-lens")
      result.kv_lengths = integer_list(value);
    else if (name == "--head-dims")
      result.head_dims = integer_list(value);
    else if (name == "--data-seeds")
      result.data_seeds = seed_list(value);
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
  if (result.warmups < 0 || result.repeats <= 0 ||
      result.query_lengths.empty() || result.kv_lengths.empty() ||
      result.head_dims.empty() || result.data_seeds.empty())
    throw std::invalid_argument("invalid empty list or repeat count");
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
  int64_t query_tiles;
  int64_t kv_blocks;
  int64_t dim_tiles;
  std::vector<BFloat16> probability_tiles;
  std::vector<BFloat16> value_tiles;

  Workload(int64_t query_len, int64_t kv_len, int64_t head_dim, uint64_t seed)
      : query_len(query_len),
        kv_len(kv_len),
        head_dim(head_dim),
        query_tiles(query_len / kTile),
        kv_blocks(kv_len / kTile),
        dim_tiles(head_dim / kTile),
        probability_tiles(
            static_cast<size_t>(query_len * kv_len)),
        value_tiles(static_cast<size_t>(kv_len * head_dim)) {
    std::mt19937_64 random(seed);
    std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);

    // Both candidates consume the same pre-tiled operands. This isolates
    // traversal/locality from conversion and packing costs.
    for (int64_t qi = 0; qi < query_tiles; ++qi)
      for (int64_t kb = 0; kb < kv_blocks; ++kb)
        for (int64_t i = 0; i < kTile; ++i)
          for (int64_t k = 0; k < kTile; ++k)
            probability_tiles[probability_offset(qi, kb) + i * kTile + k] =
                to_bfloat16(distribution(random));
  }

  size_t probability_offset(int64_t qi, int64_t kb) const {
    return static_cast<size_t>((qi * kv_blocks + kb) * kTile * kTile);
  }

  size_t value_offset(int64_t kb, int64_t dj) const {
    return static_cast<size_t>((kb * dim_tiles + dj) * kTile * kTile);
  }
};

class PvKernel {
 public:
  explicit PvKernel(int64_t head_dim)
      : head_dim_(head_dim),
        kernel_(
            kTile,
            kTile,
            kTile,
            1,
            kTile,
            kTile,
            head_dim,
            memory::data_type::bf16,
            memory::data_type::bf16,
            memory::data_type::f32),
        pack_value_(
            kTile,
            kTile,
            pack_type::no_trans,
            head_dim,
            kTile,
            memory::data_type::bf16,
            memory::data_type::bf16),
        offsets_{{0, 0}} {
    kernel_.set_add_C(true);
    kernel_.finalize();
    kernel_.generate();
    pack_value_.generate();
    scratch_.resize(kernel_.get_scratchpad_size() + 64);
  }

  void prepare_values(Workload& workload, uint64_t seed) {
    std::mt19937_64 random(seed ^ 0x9e3779b97f4a7c15ULL);
    std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
    std::vector<BFloat16> row_major(
        static_cast<size_t>(workload.kv_len * workload.head_dim));
    for (auto& value : row_major)
      value = to_bfloat16(distribution(random));
    for (int64_t kb = 0; kb < workload.kv_blocks; ++kb)
      for (int64_t dj = 0; dj < workload.dim_tiles; ++dj)
        pack_value_.execute(
            row_major.data() + kb * kTile * workload.head_dim + dj * kTile,
            workload.value_tiles.data() + workload.value_offset(kb, dj));
  }

  void run_ikj(const Workload& workload, float* output) {
    kernel_.set_hw_context();
    for (int64_t qi = 0; qi < workload.query_tiles; ++qi)
      for (int64_t kb = 0; kb < workload.kv_blocks; ++kb)
        for (int64_t dj = 0; dj < workload.dim_tiles; ++dj)
          execute(workload, output, qi, kb, dj);
    brgemm::release_hw_context();
  }

  void run_kij(const Workload& workload, float* output) {
    kernel_.set_hw_context();
    for (int64_t kb = 0; kb < workload.kv_blocks; ++kb)
      for (int64_t qi = 0; qi < workload.query_tiles; ++qi)
        for (int64_t dj = 0; dj < workload.dim_tiles; ++dj)
          execute(workload, output, qi, kb, dj);
    brgemm::release_hw_context();
  }

 private:
  void execute(
      const Workload& workload,
      float* output,
      int64_t qi,
      int64_t kb,
      int64_t dj) {
    kernel_.execute(
        workload.probability_tiles.data() +
            workload.probability_offset(qi, kb),
        workload.value_tiles.data() + workload.value_offset(kb, dj),
        offsets_,
        output + qi * kTile * head_dim_ + dj * kTile,
        scratch_.data());
  }

  int64_t head_dim_;
  brgemm kernel_;
  transform pack_value_;
  std::vector<std::pair<memory::dim, memory::dim>> offsets_;
  std::vector<uint8_t> scratch_;
};

double max_error(const std::vector<float>& left, const std::vector<float>& right) {
  double result = 0.0;
  for (size_t index = 0; index < left.size(); ++index)
    result = std::max(result, std::abs(double(left[index] - right[index])));
  return result;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Options config = options(argc, argv);
    std::ofstream output(config.out);
    if (!output)
      throw std::runtime_error("cannot open output: " + config.out);
    output << "head_dim,query_len,kv_len,data_seed,round,order_position,"
              "dataflow,elapsed_ms,gflops,max_abs_error,correct,query_tile,"
              "kv_block,order_seed\n";
    output << std::setprecision(10);
    std::mt19937_64 order_random(config.order_seed);
    int64_t trial_count = 0;

    for (int64_t head_dim : config.head_dims) {
      for (int64_t query_len : config.query_lengths) {
        for (int64_t kv_len : config.kv_lengths) {
          PvKernel kernel(head_dim);
          for (uint64_t data_seed : config.data_seeds) {
            Workload workload(query_len, kv_len, head_dim, data_seed);
            kernel.prepare_values(workload, data_seed);
            const size_t output_elements =
                static_cast<size_t>(query_len * head_dim);
            std::vector<float> actual(output_elements);
            std::vector<float> ikj_reference(output_elements, 0.0f);
            std::vector<float> kij_reference(output_elements, 0.0f);
            kernel.run_ikj(workload, ikj_reference.data());
            kernel.run_kij(workload, kij_reference.data());
            const double reference_error =
                max_error(ikj_reference, kij_reference);
            if (reference_error > 1.0e-5)
              throw std::runtime_error("IKJ/KIJ correctness mismatch");

            std::array<std::string, 2> order{{"ikj", "kij"}};
            const int rounds = config.warmups + config.repeats;
            for (int round = 0; round < rounds; ++round) {
              std::shuffle(order.begin(), order.end(), order_random);
              for (size_t position = 0; position < order.size(); ++position) {
                std::fill(actual.begin(), actual.end(), 0.0f);
                const auto start = Clock::now();
                if (order[position] == "ikj")
                  kernel.run_ikj(workload, actual.data());
                else
                  kernel.run_kij(workload, actual.data());
                const auto end = Clock::now();
                const double elapsed_ms =
                    std::chrono::duration<double, std::milli>(end - start)
                        .count();
                const auto& reference = order[position] == "ikj"
                    ? ikj_reference
                    : kij_reference;
                const double error = max_error(actual, reference);
                const bool correct = error <= 1.0e-5;
                if (!correct)
                  throw std::runtime_error("measured output failed correctness");
                if (round < config.warmups)
                  continue;
                const double operations =
                    2.0 * query_len * kv_len * head_dim;
                output << head_dim << ',' << query_len << ',' << kv_len << ','
                       << data_seed << ',' << round - config.warmups << ','
                       << position << ',' << order[position] << ','
                       << elapsed_ms << ','
                       << operations / (elapsed_ms * 1.0e6) << ',' << error
                       << ',' << (correct ? "true" : "false") << ',' << kTile
                       << ',' << kTile << ',' << config.order_seed << '\n';
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
