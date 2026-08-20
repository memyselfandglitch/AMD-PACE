/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 *
 * Isolate physical KV layout from decode traversal order using PACE's fused
 * blockwise online-softmax dataflow and AVX-512 BF16 primitives.
 ******************************************************************************/

#include <ops/exp_approx.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using BFloat16 = uint16_t;

constexpr int64_t kMaxBlockSize = 256;
constexpr size_t kAlignmentBytes = 64;

template <typename T, size_t Alignment>
class AlignedAllocator {
 public:
  using value_type = T;

  AlignedAllocator() noexcept = default;

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

  T* allocate(size_t count) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T))
      throw std::bad_alloc();
    void* pointer = nullptr;
    if (posix_memalign(&pointer, Alignment, count * sizeof(T)) != 0)
      throw std::bad_alloc();
    return static_cast<T*>(pointer);
  }

  void deallocate(T* pointer, size_t) noexcept {
    std::free(pointer);
  }

  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };
};

template <typename T, typename U, size_t Alignment>
bool operator==(
    const AlignedAllocator<T, Alignment>&,
    const AlignedAllocator<U, Alignment>&) noexcept {
  return true;
}

template <typename T, typename U, size_t Alignment>
bool operator!=(
    const AlignedAllocator<T, Alignment>&,
    const AlignedAllocator<U, Alignment>&) noexcept {
  return false;
}

using BFloat16Vector =
    std::vector<BFloat16, AlignedAllocator<BFloat16, kAlignmentBytes>>;
using FloatVector = std::vector<float, AlignedAllocator<float, kAlignmentBytes>>;

enum class Layout { HeadMajor, BlockMajor };
enum class Traversal { HeadFirst, BlockFirst };
enum class QueryProcessing { PerQuery, Grouped };

struct Candidate {
  const char* name;
  Layout layout;
  Traversal traversal;
  QueryProcessing query_processing;
};

constexpr std::array<Candidate, 4> kCandidates{{
    {"head_major_head_first", Layout::HeadMajor, Traversal::HeadFirst,
     QueryProcessing::PerQuery},
    {"block_major_head_first", Layout::BlockMajor, Traversal::HeadFirst,
     QueryProcessing::PerQuery},
    {"head_major_block_first", Layout::HeadMajor, Traversal::BlockFirst,
     QueryProcessing::PerQuery},
    {"block_major_block_first", Layout::BlockMajor, Traversal::BlockFirst,
     QueryProcessing::PerQuery},
}};

constexpr std::array<Candidate, 4> kGqaCandidates{{
    {"head_major_head_first", Layout::HeadMajor, Traversal::HeadFirst,
     QueryProcessing::PerQuery},
    {"block_major_block_first", Layout::BlockMajor, Traversal::BlockFirst,
     QueryProcessing::PerQuery},
    {"head_major_head_first_grouped", Layout::HeadMajor, Traversal::HeadFirst,
     QueryProcessing::Grouped},
    {"block_major_block_first_grouped", Layout::BlockMajor,
     Traversal::BlockFirst, QueryProcessing::Grouped},
}};

struct Shape {
  std::string family;
  std::string name;
  int64_t num_q_heads;
  int64_t num_kv_heads;
  int64_t head_dim;
};

struct Case {
  std::string family;
  std::string name;
  Shape shape;
  int64_t batch_size;
  int64_t sequence_length;
  int64_t block_size;
  double target_kv_mib;
};

const std::array<Shape, 3> kShapes{{
    {"default", "slm_gqa", 8, 4, 64},
    {"default", "llama_gqa", 32, 8, 128},
    {"default", "mha", 8, 8, 64},
}};

struct Options {
  std::vector<std::string> shapes{"slm_gqa", "llama_gqa", "mha"};
  std::string shape_specs;
  std::vector<int64_t> sequence_lengths{512, 2048, 8192, 16384};
  std::vector<int64_t> batch_sizes{1, 4};
  std::vector<uint64_t> data_seeds{11, 29, 47};
  std::vector<int64_t> block_sizes{64};
  int warmups = 2;
  int repeats = 20;
  uint64_t order_seed = 20260816;
  std::string candidate_set = "layout";
  std::string cases;
  std::string process_launch = "0";
  std::string out = "decode_layout_traversal_trials.csv";
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
    if (value <= 0)
      throw std::invalid_argument("integer lists require positive values");
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
    if (name == "--shapes")
      result.shapes = split(value);
    else if (name == "--shape-specs")
      result.shape_specs = value;
    else if (name == "--seq-lens")
      result.sequence_lengths = integer_list(value);
    else if (name == "--batch-sizes")
      result.batch_sizes = integer_list(value);
    else if (name == "--data-seeds")
      result.data_seeds = seed_list(value);
    else if (name == "--block-size")
      result.block_sizes = {std::stoll(value)};
    else if (name == "--block-sizes")
      result.block_sizes = integer_list(value);
    else if (name == "--warmups")
      result.warmups = std::stoi(value);
    else if (name == "--repeats")
      result.repeats = std::stoi(value);
    else if (name == "--order-seed")
      result.order_seed = std::stoull(value);
    else if (name == "--candidate-set")
      result.candidate_set = value;
    else if (name == "--cases")
      result.cases = value;
    else if (name == "--process-launch")
      result.process_launch = value;
    else if (name == "--out")
      result.out = value;
    else
      throw std::invalid_argument("unknown option: " + name);
  }
  if (result.data_seeds.empty() || result.warmups < 0 || result.repeats <= 0 ||
      (result.candidate_set != "layout" &&
       result.candidate_set != "gqa_grouped") ||
      (result.cases.empty() &&
       (result.shapes.empty() || result.sequence_lengths.empty() ||
        result.batch_sizes.empty() || result.block_sizes.empty())))
    throw std::invalid_argument("invalid empty list, block size, or repeat count");
  for (const int64_t block_size : result.block_sizes) {
    if (block_size <= 0 || block_size > kMaxBlockSize || block_size % 16 != 0)
      throw std::invalid_argument(
          "block sizes must be multiples of 16 no larger than 256");
  }
  return result;
}

const std::array<Candidate, 4>& candidates(const Options& opts) {
  return opts.candidate_set == "gqa_grouped" ? kGqaCandidates : kCandidates;
}

std::vector<Case> read_cases(const std::string& path) {
  std::ifstream stream(path);
  if (!stream)
    throw std::runtime_error("could not open case manifest: " + path);

  std::string line;
  if (!std::getline(stream, line))
    throw std::runtime_error("case manifest is empty: " + path);
  if (!line.empty() && line.back() == '\r')
    line.pop_back();
  const auto header = split(line);
  std::unordered_map<std::string, size_t> columns;
  for (size_t index = 0; index < header.size(); ++index)
    columns[header[index]] = index;
  const std::array<const char*, 10> required{{
      "case_family", "case_name", "shape", "num_q_heads", "num_kv_heads",
      "head_dim", "batch_size", "seq_len", "block_size", "target_kv_mib"}};
  for (const char* name : required) {
    if (columns.count(name) == 0)
      throw std::runtime_error("case manifest missing column: " + std::string(name));
  }

  std::vector<Case> cases;
  while (std::getline(stream, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (line.empty())
      continue;
    const auto fields = split(line);
    if (fields.size() != header.size())
      throw std::runtime_error("case manifest rows must not contain quoted commas");
    const auto field = [&](const char* name) -> const std::string& {
      return fields[columns.at(name)];
    };
    Case item{
        field("case_family"),
        field("case_name"),
        {field("case_family"),
         field("shape"),
         std::stoll(field("num_q_heads")),
         std::stoll(field("num_kv_heads")),
         std::stoll(field("head_dim"))},
        std::stoll(field("batch_size")),
        std::stoll(field("seq_len")),
        std::stoll(field("block_size")),
        std::stod(field("target_kv_mib"))};
    if (item.family.empty() || item.name.empty() || item.shape.name.empty() ||
        item.shape.num_q_heads <= 0 || item.shape.num_kv_heads <= 0 ||
        item.shape.num_q_heads % item.shape.num_kv_heads != 0 ||
        (item.shape.head_dim != 64 && item.shape.head_dim != 128 &&
         item.shape.head_dim != 256) ||
        item.batch_size <= 0 || item.sequence_length <= 0 ||
        item.block_size <= 0 || item.block_size > kMaxBlockSize ||
        item.block_size % 16 != 0)
      throw std::runtime_error("invalid case manifest row: " + item.name);
    cases.push_back(std::move(item));
  }
  if (cases.empty())
    throw std::runtime_error("case manifest has no cases: " + path);
  return cases;
}

const Shape& find_shape(const std::string& name) {
  const auto it = std::find_if(
      kShapes.begin(), kShapes.end(), [&](const Shape& shape) {
        return shape.name == name;
      });
  if (it == kShapes.end())
    throw std::invalid_argument("unknown shape: " + name);
  return *it;
}

std::vector<Shape> selected_shapes(const Options& opts) {
  if (opts.shape_specs.empty()) {
    std::vector<Shape> result;
    for (const auto& name : opts.shapes)
      result.push_back(find_shape(name));
    return result;
  }

  std::vector<Shape> result;
  for (const auto& spec : split(opts.shape_specs)) {
    std::vector<std::string> colon_fields;
    size_t begin = 0;
    while (begin < spec.size()) {
      const size_t end = spec.find(':', begin);
      colon_fields.push_back(spec.substr(begin, end - begin));
      if (end == std::string::npos)
        break;
      begin = end + 1;
    }
    if (colon_fields.size() != 4)
      throw std::invalid_argument(
          "shape specs use family/name:num_q_heads:num_kv_heads:head_dim");
    const std::string family_name = colon_fields[0];
    const size_t slash = family_name.find('/');
    const std::string family =
        slash == std::string::npos ? "custom" : family_name.substr(0, slash);
    const std::string name = slash == std::string::npos
        ? family_name
        : family_name.substr(slash + 1);
    const int64_t num_q_heads = std::stoll(colon_fields[1]);
    const int64_t num_kv_heads = std::stoll(colon_fields[2]);
    const int64_t head_dim = std::stoll(colon_fields[3]);
    if (family.empty() || name.empty() || num_q_heads <= 0 ||
        num_kv_heads <= 0 || num_q_heads % num_kv_heads != 0 ||
        (head_dim != 64 && head_dim != 128 && head_dim != 256))
      throw std::invalid_argument(
          "invalid shape spec; head_dim must be 64, 128, or 256");
    result.push_back({family, name, num_q_heads, num_kv_heads, head_dim});
  }
  return result;
}

std::vector<Case> selected_cases(const Options& opts) {
  if (!opts.cases.empty())
    return read_cases(opts.cases);

  std::vector<Case> result;
  for (const auto& shape : selected_shapes(opts)) {
    for (const int64_t batch_size : opts.batch_sizes) {
      for (const int64_t sequence_length : opts.sequence_lengths) {
        for (const int64_t block_size : opts.block_sizes) {
          std::ostringstream name;
          name << shape.name << "_b" << batch_size << "_s" << sequence_length
               << "_bs" << block_size;
          const double target_kv_mib =
              static_cast<double>(batch_size * sequence_length *
                                  shape.num_kv_heads * shape.head_dim * 4) /
              static_cast<double>(1ULL << 20);
          result.push_back({
              shape.family,
              name.str(),
              shape,
              batch_size,
              sequence_length,
              block_size,
              target_kv_mib});
        }
      }
    }
  }
  return result;
}

BFloat16 to_bfloat16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t rounding = 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<BFloat16>((bits + rounding) >> 16U);
}

float from_bfloat16(BFloat16 value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline __m512 bf16_to_fp32(__m256i value) {
  return _mm512_cvtpbh_ps(reinterpret_cast<__m256bh>(value));
}

inline float dot_product(const BFloat16* a, const BFloat16* b, int64_t n) {
  __m512 acc = _mm512_setzero_ps();
  int64_t index = 0;
  for (; index + 32 <= n; index += 32) {
    const __m512bh av =
        (__m512bh)_mm512_loadu_si512(reinterpret_cast<const void*>(a + index));
    const __m512bh bv =
        (__m512bh)_mm512_loadu_si512(reinterpret_cast<const void*>(b + index));
    acc = _mm512_dpbf16_ps(acc, av, bv);
  }
  if (index + 16 <= n) {
    const __m512 af = bf16_to_fp32(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + index)));
    const __m512 bf = bf16_to_fp32(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + index)));
    acc = _mm512_fmadd_ps(af, bf, acc);
    index += 16;
  }
  float result = _mm512_reduce_add_ps(acc);
  for (; index < n; ++index)
    result += from_bfloat16(a[index]) * from_bfloat16(b[index]);
  return result;
}

template <int Chunks>
inline void accumulate_weighted(
    float* output,
    const BFloat16* values,
    const float* weights,
    int64_t tokens,
    int64_t head_dim) {
  __m512 accumulators[Chunks];
  for (int chunk = 0; chunk < Chunks; ++chunk)
    accumulators[chunk] = _mm512_loadu_ps(output + chunk * 16);
  for (int64_t token = 0; token < tokens; ++token) {
    const __m512 weight = _mm512_set1_ps(weights[token]);
    const BFloat16* row = values + token * head_dim;
    for (int chunk = 0; chunk < Chunks; ++chunk) {
      const __m512 value = bf16_to_fp32(_mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(row + chunk * 16)));
      accumulators[chunk] =
          _mm512_fmadd_ps(weight, value, accumulators[chunk]);
    }
  }
  for (int chunk = 0; chunk < Chunks; ++chunk)
    _mm512_storeu_ps(output + chunk * 16, accumulators[chunk]);
}

void accumulate_weighted(
    float* output,
    const BFloat16* values,
    const float* weights,
    int64_t tokens,
    int64_t head_dim) {
  if (head_dim == 64)
    accumulate_weighted<4>(output, values, weights, tokens, head_dim);
  else if (head_dim == 128)
    accumulate_weighted<8>(output, values, weights, tokens, head_dim);
  else if (head_dim == 256)
    accumulate_weighted<16>(output, values, weights, tokens, head_dim);
  else
    throw std::invalid_argument("benchmark supports head dimensions 64, 128, and 256");
}

struct Workload {
  Shape shape;
  int64_t batch_size;
  int64_t sequence_length;
  int64_t block_size;
  int64_t blocks_per_sequence;
  int64_t total_blocks;
  int64_t repetitions;
  float scale;
  BFloat16Vector queries;
  BFloat16Vector head_major_pool;
  BFloat16Vector block_major_pool;

  Workload(
      const Shape& shape,
      int64_t batch_size,
      int64_t sequence_length,
      int64_t block_size,
      uint64_t seed)
      : shape(shape),
        batch_size(batch_size),
        sequence_length(sequence_length),
        block_size(block_size),
        blocks_per_sequence((sequence_length + block_size - 1) / block_size),
        total_blocks(batch_size * blocks_per_sequence),
        repetitions(shape.num_q_heads / shape.num_kv_heads),
        scale(1.0f / std::sqrt(static_cast<float>(shape.head_dim))),
        queries(static_cast<size_t>(batch_size * shape.num_q_heads * shape.head_dim)),
        head_major_pool(static_cast<size_t>(
            shape.num_kv_heads * total_blocks * 2 * block_size * shape.head_dim)),
        block_major_pool(head_major_pool.size()) {
    if (shape.num_q_heads % shape.num_kv_heads != 0)
      throw std::invalid_argument("query heads must be divisible by KV heads");
    std::mt19937_64 random(seed);
    std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
    for (auto& value : queries)
      value = to_bfloat16(distribution(random));
    for (int64_t block = 0; block < total_blocks; ++block) {
      for (int64_t head = 0; head < shape.num_kv_heads; ++head) {
        for (int64_t kv = 0; kv < 2; ++kv) {
          for (int64_t token = 0; token < block_size; ++token) {
            for (int64_t dim = 0; dim < shape.head_dim; ++dim) {
              const BFloat16 value = to_bfloat16(distribution(random));
              head_major_pool[offset(
                  Layout::HeadMajor, block, head, kv, token, dim)] = value;
              block_major_pool[offset(
                  Layout::BlockMajor, block, head, kv, token, dim)] = value;
            }
          }
        }
      }
    }
  }

  size_t offset(
      Layout layout,
      int64_t block,
      int64_t head,
      int64_t kv,
      int64_t token,
      int64_t dim) const {
    int64_t outer;
    if (layout == Layout::HeadMajor)
      outer = head * total_blocks + block;
    else
      outer = block * shape.num_kv_heads + head;
    return static_cast<size_t>(
        ((((outer * 2) + kv) * block_size + token) * shape.head_dim) + dim);
  }

  const BFloat16* query(int64_t batch, int64_t q_head) const {
    return queries.data() +
        (batch * shape.num_q_heads + q_head) * shape.head_dim;
  }

  const BFloat16* block(
      Layout layout, int64_t physical_block, int64_t kv_head, int64_t kv) const {
    const auto& pool = layout == Layout::HeadMajor ? head_major_pool
                                                   : block_major_pool;
    return pool.data() + offset(layout, physical_block, kv_head, kv, 0, 0);
  }
};

struct State {
  FloatVector maximum;
  FloatVector sum;
  FloatVector output;
  FloatVector scratch_scores;
  FloatVector scratch_weights;
  FloatVector scratch_block_maximum;
  FloatVector scratch_token_scores;

  explicit State(const Workload& workload, bool allocate_grouped_scratch)
      : maximum(static_cast<size_t>(workload.batch_size * workload.shape.num_q_heads)),
        sum(maximum.size()),
        output(maximum.size() * static_cast<size_t>(workload.shape.head_dim)),
        scratch_scores(allocate_grouped_scratch
                ? static_cast<size_t>(
                      workload.repetitions * workload.block_size)
                : 0),
        scratch_weights(scratch_scores.size()),
        scratch_block_maximum(allocate_grouped_scratch
                ? static_cast<size_t>(workload.repetitions)
                : 0),
        scratch_token_scores(scratch_block_maximum.size()) {}

  void reset() {
    std::fill(maximum.begin(), maximum.end(), -std::numeric_limits<float>::infinity());
    std::fill(sum.begin(), sum.end(), 0.0f);
    std::fill(output.begin(), output.end(), 0.0f);
  }
};

void process_query_head_block(
    const Workload& workload,
    int64_t batch,
    int64_t q_head,
    const BFloat16* keys,
    const BFloat16* values,
    int64_t tokens,
    State& state) {
  const size_t state_index =
      static_cast<size_t>(batch * workload.shape.num_q_heads + q_head);
  float& running_max = state.maximum[state_index];
  float& running_sum = state.sum[state_index];
  float* output =
      state.output.data() + state_index * workload.shape.head_dim;
  const BFloat16* query = workload.query(batch, q_head);

  alignas(64) float scores[kMaxBlockSize];
  alignas(64) float weights[kMaxBlockSize];
  float block_max = -std::numeric_limits<float>::infinity();
  for (int64_t token = 0; token < tokens; ++token) {
    const float score = dot_product(
                            query,
                            keys + token * workload.shape.head_dim,
                            workload.shape.head_dim) *
        workload.scale;
    scores[token] = score;
    block_max = std::max(block_max, score);
  }

  const float new_max = std::max(running_max, block_max);
  if (new_max > running_max) {
    float correction;
    EXP_APPROX_SCALAR(running_max - new_max, correction);
    running_sum *= correction;
    const __m512 factor = _mm512_set1_ps(correction);
    for (int64_t dim = 0; dim < workload.shape.head_dim; dim += 16) {
      const __m512 value = _mm512_loadu_ps(output + dim);
      _mm512_storeu_ps(output + dim, _mm512_mul_ps(value, factor));
    }
  }

  int64_t token = 0;
  __m512 vector_sum = _mm512_setzero_ps();
  const __m512 negative_max = _mm512_set1_ps(-new_max);
  {
    EXP_APPROX_AVX512_CONSTANTS();
    for (; token + 16 <= tokens; token += 16) {
      const __m512 score = _mm512_loadu_ps(scores + token);
      __m512 weight;
      EXP_APPROX_AVX512(_mm512_add_ps(score, negative_max), weight);
      _mm512_storeu_ps(weights + token, weight);
      vector_sum = _mm512_add_ps(vector_sum, weight);
    }
  }
  float local_sum = _mm512_reduce_add_ps(vector_sum);
  for (; token < tokens; ++token) {
    EXP_APPROX_SCALAR(scores[token] - new_max, weights[token]);
    local_sum += weights[token];
  }
  running_sum += local_sum;
  accumulate_weighted(
      output, values, weights, tokens, workload.shape.head_dim);
  running_max = new_max;
}

void process_group_block(
    const Workload& workload,
    Layout layout,
    int64_t batch,
    int64_t logical_block,
    int64_t kv_head,
    State& state) {
  const int64_t physical_block =
      batch * workload.blocks_per_sequence + logical_block;
  const int64_t tokens =
      logical_block + 1 == workload.blocks_per_sequence
      ? workload.sequence_length - logical_block * workload.block_size
      : workload.block_size;
  const BFloat16* keys = workload.block(layout, physical_block, kv_head, 0);
  const BFloat16* values = workload.block(layout, physical_block, kv_head, 1);
  const int64_t first_q_head = kv_head * workload.repetitions;
  for (int64_t rep = 0; rep < workload.repetitions; ++rep)
    process_query_head_block(
        workload,
        batch,
        first_q_head + rep,
        keys,
        values,
        tokens,
        state);
}

void grouped_dot_products(
    const BFloat16* queries,
    const BFloat16* key,
    int64_t query_count,
    int64_t head_dim,
    float* results) {
  constexpr int64_t kQueryTile = 4;
  if (head_dim % 32 != 0)
    throw std::invalid_argument("grouped query head dimension must divide 32");
  for (int64_t query_base = 0; query_base < query_count;
       query_base += kQueryTile) {
    const int64_t count = std::min(kQueryTile, query_count - query_base);
    __m512 accumulators[kQueryTile];
    for (int64_t query = 0; query < count; ++query)
      accumulators[query] = _mm512_setzero_ps();
    for (int64_t dim = 0; dim < head_dim; dim += 32) {
      const __m512bh key_vector = (__m512bh)_mm512_loadu_si512(
          reinterpret_cast<const void*>(key + dim));
      for (int64_t query = 0; query < count; ++query) {
        const BFloat16* query_data =
            queries + (query_base + query) * head_dim + dim;
        const __m512bh query_vector = (__m512bh)_mm512_loadu_si512(
            reinterpret_cast<const void*>(query_data));
        accumulators[query] = _mm512_dpbf16_ps(
            accumulators[query], query_vector, key_vector);
      }
    }
    for (int64_t query = 0; query < count; ++query)
      results[query_base + query] =
          _mm512_reduce_add_ps(accumulators[query]);
  }
}

template <int Chunks, int QueryTile>
void accumulate_weighted_grouped(
    float* outputs,
    const BFloat16* values,
    const float* weights,
    int64_t query_count,
    int64_t tokens,
    int64_t head_dim,
    int64_t weight_stride) {
  for (int64_t query_base = 0; query_base < query_count;
       query_base += QueryTile) {
    const int64_t count = std::min<int64_t>(
        QueryTile, query_count - query_base);
    __m512 accumulators[QueryTile][Chunks];
    for (int64_t query = 0; query < count; ++query) {
      float* output = outputs + (query_base + query) * head_dim;
      for (int chunk = 0; chunk < Chunks; ++chunk)
        accumulators[query][chunk] =
            _mm512_loadu_ps(output + chunk * 16);
    }
    for (int64_t token = 0; token < tokens; ++token) {
      __m512 query_weights[QueryTile];
      for (int64_t query = 0; query < count; ++query) {
        query_weights[query] = _mm512_set1_ps(
            weights[(query_base + query) * weight_stride + token]);
      }
      const BFloat16* row = values + token * head_dim;
      for (int chunk = 0; chunk < Chunks; ++chunk) {
        const __m512 value = bf16_to_fp32(_mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(row + chunk * 16)));
        for (int64_t query = 0; query < count; ++query) {
          accumulators[query][chunk] = _mm512_fmadd_ps(
              query_weights[query], value, accumulators[query][chunk]);
        }
      }
    }
    for (int64_t query = 0; query < count; ++query) {
      float* output = outputs + (query_base + query) * head_dim;
      for (int chunk = 0; chunk < Chunks; ++chunk)
        _mm512_storeu_ps(
            output + chunk * 16, accumulators[query][chunk]);
    }
  }
}

void process_group_block_grouped(
    const Workload& workload,
    Layout layout,
    int64_t batch,
    int64_t logical_block,
    int64_t kv_head,
    State& state) {
  const int64_t physical_block =
      batch * workload.blocks_per_sequence + logical_block;
  const int64_t tokens =
      logical_block + 1 == workload.blocks_per_sequence
      ? workload.sequence_length - logical_block * workload.block_size
      : workload.block_size;
  const BFloat16* keys = workload.block(layout, physical_block, kv_head, 0);
  const BFloat16* values = workload.block(layout, physical_block, kv_head, 1);
  const int64_t first_q_head = kv_head * workload.repetitions;
  const BFloat16* queries = workload.query(batch, first_q_head);
  float* scores = state.scratch_scores.data();
  float* weights = state.scratch_weights.data();
  float* block_maximum = state.scratch_block_maximum.data();
  float* token_scores = state.scratch_token_scores.data();
  std::fill(
      block_maximum,
      block_maximum + workload.repetitions,
      -std::numeric_limits<float>::infinity());

  // Reuse each key row while computing every query that shares this KV head.
  for (int64_t token = 0; token < tokens; ++token) {
    grouped_dot_products(
        queries,
        keys + token * workload.shape.head_dim,
        workload.repetitions,
        workload.shape.head_dim,
        token_scores);
    for (int64_t rep = 0; rep < workload.repetitions; ++rep) {
      const float score = token_scores[rep] * workload.scale;
      scores[rep * workload.block_size + token] = score;
      block_maximum[rep] = std::max(block_maximum[rep], score);
    }
  }

  for (int64_t rep = 0; rep < workload.repetitions; ++rep) {
    const size_t state_index = static_cast<size_t>(
        batch * workload.shape.num_q_heads + first_q_head + rep);
    float& running_max = state.maximum[state_index];
    float& running_sum = state.sum[state_index];
    float* output =
        state.output.data() + state_index * workload.shape.head_dim;
    const float new_max = std::max(running_max, block_maximum[rep]);
    if (new_max > running_max) {
      float correction;
      EXP_APPROX_SCALAR(running_max - new_max, correction);
      running_sum *= correction;
      const __m512 factor = _mm512_set1_ps(correction);
      for (int64_t dim = 0; dim < workload.shape.head_dim; dim += 16) {
        const __m512 value = _mm512_loadu_ps(output + dim);
        _mm512_storeu_ps(output + dim, _mm512_mul_ps(value, factor));
      }
    }

    float* query_scores = scores + rep * workload.block_size;
    float* query_weights = weights + rep * workload.block_size;
    int64_t token = 0;
    __m512 vector_sum = _mm512_setzero_ps();
    const __m512 negative_max = _mm512_set1_ps(-new_max);
    {
      EXP_APPROX_AVX512_CONSTANTS();
      for (; token + 16 <= tokens; token += 16) {
        const __m512 score = _mm512_loadu_ps(query_scores + token);
        __m512 weight;
        EXP_APPROX_AVX512(_mm512_add_ps(score, negative_max), weight);
        _mm512_storeu_ps(query_weights + token, weight);
        vector_sum = _mm512_add_ps(vector_sum, weight);
      }
    }
    float local_sum = _mm512_reduce_add_ps(vector_sum);
    for (; token < tokens; ++token) {
      EXP_APPROX_SCALAR(query_scores[token] - new_max, query_weights[token]);
      local_sum += query_weights[token];
    }
    running_sum += local_sum;
    running_max = new_max;
  }

  float* outputs = state.output.data() +
      static_cast<size_t>(batch * workload.shape.num_q_heads + first_q_head) *
          workload.shape.head_dim;
  if (workload.shape.head_dim == 64) {
    accumulate_weighted_grouped<4, 4>(
        outputs, values, weights, workload.repetitions, tokens,
        workload.shape.head_dim, workload.block_size);
  } else if (workload.shape.head_dim == 128) {
    accumulate_weighted_grouped<8, 2>(
        outputs, values, weights, workload.repetitions, tokens,
        workload.shape.head_dim, workload.block_size);
  } else {
    accumulate_weighted_grouped<16, 1>(
        outputs, values, weights, workload.repetitions, tokens,
        workload.shape.head_dim, workload.block_size);
  }
}

void run(const Workload& workload, const Candidate& candidate, State& state) {
  state.reset();
  const auto process = candidate.query_processing == QueryProcessing::Grouped
      ? process_group_block_grouped
      : process_group_block;
  for (int64_t batch = 0; batch < workload.batch_size; ++batch) {
    if (candidate.traversal == Traversal::HeadFirst) {
      for (int64_t kv_head = 0; kv_head < workload.shape.num_kv_heads; ++kv_head)
        for (int64_t block = 0; block < workload.blocks_per_sequence; ++block)
          process(workload, candidate.layout, batch, block, kv_head, state);
    } else {
      for (int64_t block = 0; block < workload.blocks_per_sequence; ++block)
        for (int64_t kv_head = 0; kv_head < workload.shape.num_kv_heads; ++kv_head)
          process(workload, candidate.layout, batch, block, kv_head, state);
    }
  }
  for (size_t index = 0; index < state.maximum.size(); ++index) {
    const float inverse = state.sum[index] > 0.0f ? 1.0f / state.sum[index] : 0.0f;
    float* output =
        state.output.data() + index * workload.shape.head_dim;
    const __m512 factor = _mm512_set1_ps(inverse);
    for (int64_t dim = 0; dim < workload.shape.head_dim; dim += 16) {
      const __m512 value = _mm512_loadu_ps(output + dim);
      _mm512_storeu_ps(output + dim, _mm512_mul_ps(value, factor));
    }
  }
}

double checksum(const State& state) {
  double result = 0.0;
  for (size_t index = 0; index < state.output.size(); ++index)
    result += static_cast<double>(state.output[index]) *
        static_cast<double>((index % 17) + 1);
  return result;
}

std::pair<double, double> error(
    const FloatVector& reference, const FloatVector& candidate) {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
  for (size_t index = 0; index < reference.size(); ++index) {
    const double absolute = std::abs(
        static_cast<double>(reference[index]) - candidate[index]);
    const double denominator =
        std::max(1e-6, std::abs(static_cast<double>(reference[index])));
    maximum_absolute = std::max(maximum_absolute, absolute);
    maximum_relative = std::max(maximum_relative, absolute / denominator);
  }
  return {maximum_absolute, maximum_relative};
}

void write_header(std::ofstream& stream) {
  stream << "case_family,case_name,target_kv_mib,batch_semantics,"
            "kv_bytes_per_sequence,kv_bytes_per_call,"
            "allocated_kv_bytes_per_sequence,allocated_kv_bytes_per_call,"
            "blocks_per_sequence,blocks_per_call,alignment_bytes,order_policy,"
            "process_launch,"
            "shape_family,shape,num_q_heads,num_kv_heads,head_dim,batch_size,seq_len,"
            "block_size,data_seed,round,order_position,candidate,layout,traversal,"
            "query_processing,elapsed_ms,checksum,max_abs_error,max_rel_error,correct\n";
}

void benchmark(const Options& opts) {
  std::ofstream stream(opts.out);
  if (!stream)
    throw std::runtime_error("could not open output: " + opts.out);
  write_header(stream);
  stream << std::setprecision(12);
  std::mt19937_64 order_random(opts.order_seed);
  size_t rows = 0;
  const auto& benchmark_candidates = candidates(opts);

  std::vector<Case> cases = selected_cases(opts);
  std::shuffle(cases.begin(), cases.end(), order_random);
  for (const auto& item : cases) {
    const Shape& shape = item.shape;
    const int64_t batch_size = item.batch_size;
    const int64_t sequence_length = item.sequence_length;
    const int64_t block_size = item.block_size;
    for (const uint64_t data_seed : opts.data_seeds) {
      Workload workload(
          shape, batch_size, sequence_length, block_size, data_seed);
      const bool grouped_scratch = opts.candidate_set == "gqa_grouped";
      std::array<State, 4> states{{
          State(workload, grouped_scratch),
          State(workload, grouped_scratch),
          State(workload, grouped_scratch),
          State(workload, grouped_scratch)}};

      run(workload, benchmark_candidates[0], states[0]);
      const auto reference = states[0].output;
      std::array<std::pair<double, double>, 4> errors{};
      for (size_t index = 0; index < benchmark_candidates.size(); ++index) {
        run(workload, benchmark_candidates[index], states[index]);
        errors[index] = error(reference, states[index].output);
        if (errors[index].first > 1e-4 || errors[index].second > 1e-3)
          throw std::runtime_error(
              std::string("correctness failed for ") +
              benchmark_candidates[index].name);
      }

      for (int warmup = 0; warmup < opts.warmups; ++warmup) {
        std::array<size_t, 4> order{{0, 1, 2, 3}};
        std::shuffle(order.begin(), order.end(), order_random);
        for (const size_t index : order)
          run(workload, benchmark_candidates[index], states[index]);
      }

      std::array<size_t, 4> base_order{{0, 1, 2, 3}};
      for (int round = 0; round < opts.repeats; ++round) {
        if (round % static_cast<int>(benchmark_candidates.size()) == 0)
          std::shuffle(base_order.begin(), base_order.end(), order_random);
        // Rotate each shuffled cycle so every candidate occupies every position.
        std::array<size_t, 4> order{};
        for (size_t position = 0; position < order.size(); ++position) {
          order[position] = base_order[
              (position + static_cast<size_t>(round)) % order.size()];
        }
        for (size_t position = 0; position < order.size(); ++position) {
          const size_t index = order[position];
          const auto start = Clock::now();
          run(workload, benchmark_candidates[index], states[index]);
          const auto stop = Clock::now();
          const double elapsed_ms =
              std::chrono::duration<double, std::milli>(stop - start).count();
          const Candidate& candidate = benchmark_candidates[index];
          const uint64_t kv_bytes_per_sequence =
              static_cast<uint64_t>(sequence_length) *
              static_cast<uint64_t>(shape.num_kv_heads) *
              static_cast<uint64_t>(shape.head_dim) * 4ULL;
          const uint64_t allocated_kv_bytes_per_sequence =
              static_cast<uint64_t>(workload.blocks_per_sequence) *
              static_cast<uint64_t>(block_size) *
              static_cast<uint64_t>(shape.num_kv_heads) *
              static_cast<uint64_t>(shape.head_dim) * 4ULL;
          stream << item.family << ',' << item.name << ','
                 << item.target_kv_mib << ",sequential_outer_loop,"
                 << kv_bytes_per_sequence << ','
                 << kv_bytes_per_sequence * static_cast<uint64_t>(batch_size)
                 << ',' << allocated_kv_bytes_per_sequence << ','
                 << allocated_kv_bytes_per_sequence *
                        static_cast<uint64_t>(batch_size)
                 << ',' << workload.blocks_per_sequence << ','
                 << workload.total_blocks << ',' << kAlignmentBytes << ','
                 << "latin_square_cycle4," << opts.process_launch << ','
                 << shape.family << ',' << shape.name << ','
                 << shape.num_q_heads << ',' << shape.num_kv_heads << ','
                 << shape.head_dim << ',' << batch_size << ',' << sequence_length
                 << ',' << block_size << ',' << data_seed << ',' << round << ','
                 << position << ',' << candidate.name << ','
                 << (candidate.layout == Layout::HeadMajor ? "head_major"
                                                           : "block_major")
                 << ','
                 << (candidate.traversal == Traversal::HeadFirst ? "head_first"
                                                                  : "block_first")
                 << ','
                 << (candidate.query_processing == QueryProcessing::Grouped
                         ? "grouped"
                         : "per_query")
                 << ',' << elapsed_ms << ',' << checksum(states[index]) << ','
                 << errors[index].first << ',' << errors[index].second
                 << ",true\n";
          ++rows;
        }
      }
    }
  }
  std::cout << "Wrote " << rows << " balanced randomized trials to " << opts.out
            << '\n';
}

} // namespace

int main(int argc, char** argv) {
  try {
    benchmark(options(argc, argv));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
