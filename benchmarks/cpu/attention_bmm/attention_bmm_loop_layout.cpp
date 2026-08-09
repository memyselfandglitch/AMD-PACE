/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 *
 * Compare loop order and K/V layout for CPU attention-shaped BMM.
 ******************************************************************************/

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

enum class Operation { QK, PV };
enum class Layout { HeadMajor, TokenMajor };
enum class Order { IJK, IKJ, JIK, JKI, KIJ, KJI };

struct Shape {
  std::string name;
  int64_t num_q_heads;
  int64_t num_kv_heads;
  int64_t head_dim;
};

const std::array<Shape, 2> kShapes{{
    {"slm", 14, 2, 64},
    {"llm", 28, 4, 128},
}};

struct Options {
  std::vector<std::string> shapes{"slm", "llm"};
  std::vector<int64_t> query_lengths{1, 16};
  std::vector<int64_t> kv_lengths{128, 512};
  std::vector<int64_t> batch_sizes{1};
  int warmups = 2;
  int repeats = 10;
  uint64_t seed = 20260809;
  std::string raw_csv = "attention_bmm_trials.csv";
  std::string summary_csv = "attention_bmm_summary.csv";
  std::string decisions_csv = "attention_bmm_decisions.csv";
};

struct Problem {
  int64_t batch_size;
  int64_t num_q_heads;
  int64_t num_kv_heads;
  int64_t query_len;
  int64_t kv_len;
  int64_t head_dim;
  const float* query;
  const float* key_head_major;
  const float* key_token_major;
  const float* probability;
  const float* value_head_major;
  const float* value_token_major;
  float* output;
};

using Kernel = void (*)(const Problem&);

struct KernelSpec {
  Operation operation;
  Layout layout;
  Order order;
  Kernel function;
};

struct Workload {
  Shape shape;
  int64_t batch_size;
  int64_t query_len;
  int64_t kv_len;
};

struct Trial {
  std::string operation;
  std::string shape;
  int64_t batch_size;
  int64_t num_q_heads;
  int64_t num_kv_heads;
  int64_t query_len;
  int64_t kv_len;
  int64_t head_dim;
  std::string layout;
  std::string order;
  int round;
  int position;
  int64_t working_set_bytes;
  double elapsed_ms;
  double gflops;
  double max_abs_error;
  bool correct;
};

const char* operation_name(Operation operation) {
  return operation == Operation::QK ? "qk_transpose" : "probability_value";
}

const char* layout_name(Layout layout) {
  return layout == Layout::HeadMajor ? "head_major" : "token_major";
}

const char* order_name(Order order) {
  switch (order) {
    case Order::IJK:
      return "ijk";
    case Order::IKJ:
      return "ikj";
    case Order::JIK:
      return "jik";
    case Order::JKI:
      return "jki";
    case Order::KIJ:
      return "kij";
    case Order::KJI:
      return "kji";
  }
  return "unknown";
}

template <Layout layout>
inline float kv_value(
    const float* head_major,
    const float* token_major,
    int64_t batch,
    int64_t kv_head,
    int64_t token,
    int64_t dim,
    const Problem& problem) {
  if constexpr (layout == Layout::HeadMajor) {
    const int64_t index =
        (((batch * problem.num_kv_heads + kv_head) * problem.kv_len + token) *
             problem.head_dim +
         dim);
    return head_major[index];
  } else {
    const int64_t index =
        (((batch * problem.kv_len + token) * problem.num_kv_heads + kv_head) *
             problem.head_dim +
         dim);
    return token_major[index];
  }
}

template <Operation operation>
inline float left_value(
    const Problem& problem,
    const float* left,
    int64_t i,
    int64_t k) {
  if constexpr (operation == Operation::QK)
    return left[i * problem.head_dim + k];
  else
    return left[i * problem.kv_len + k];
}

template <Operation operation, Layout layout>
inline float right_value(
    const Problem& problem,
    int64_t batch,
    int64_t kv_head,
    int64_t k,
    int64_t j) {
  if constexpr (operation == Operation::QK) {
    return kv_value<layout>(
        problem.key_head_major,
        problem.key_token_major,
        batch,
        kv_head,
        j,
        k,
        problem);
  } else {
    return kv_value<layout>(
        problem.value_head_major,
        problem.value_token_major,
        batch,
        kv_head,
        k,
        j,
        problem);
  }
}

template <Operation operation, Layout layout, Order order>
void attention_gemm(const Problem& problem) {
  const int64_t ni = problem.query_len;
  const int64_t nj =
      operation == Operation::QK ? problem.kv_len : problem.head_dim;
  const int64_t nk =
      operation == Operation::QK ? problem.head_dim : problem.kv_len;
  const int64_t left_matrix_size = ni * nk;
  const int64_t output_matrix_size = ni * nj;
  const int64_t q_per_kv = problem.num_q_heads / problem.num_kv_heads;

  for (int64_t batch = 0; batch < problem.batch_size; ++batch) {
    for (int64_t q_head = 0; q_head < problem.num_q_heads; ++q_head) {
      const int64_t kv_head = q_head / q_per_kv;
      const float* left = operation == Operation::QK
          ? problem.query +
              (batch * problem.num_q_heads + q_head) * left_matrix_size
          : problem.probability +
              (batch * problem.num_q_heads + q_head) * left_matrix_size;
      float* output = problem.output +
          (batch * problem.num_q_heads + q_head) * output_matrix_size;

      if constexpr (order == Order::IJK) {
        for (int64_t i = 0; i < ni; ++i)
          for (int64_t j = 0; j < nj; ++j)
            for (int64_t k = 0; k < nk; ++k)
              output[i * nj + j] += left_value<operation>(problem, left, i, k) *
                  right_value<operation, layout>(
                                       problem, batch, kv_head, k, j);
      } else if constexpr (order == Order::IKJ) {
        for (int64_t i = 0; i < ni; ++i)
          for (int64_t k = 0; k < nk; ++k) {
            const float left_ik = left_value<operation>(problem, left, i, k);
            for (int64_t j = 0; j < nj; ++j)
              output[i * nj + j] += left_ik * right_value<operation, layout>(
                                                      problem,
                                                      batch,
                                                      kv_head,
                                                      k,
                                                      j);
          }
      } else if constexpr (order == Order::JIK) {
        for (int64_t j = 0; j < nj; ++j)
          for (int64_t i = 0; i < ni; ++i)
            for (int64_t k = 0; k < nk; ++k)
              output[i * nj + j] += left_value<operation>(problem, left, i, k) *
                  right_value<operation, layout>(
                                       problem, batch, kv_head, k, j);
      } else if constexpr (order == Order::JKI) {
        for (int64_t j = 0; j < nj; ++j)
          for (int64_t k = 0; k < nk; ++k) {
            const float right_kj = right_value<operation, layout>(
                problem, batch, kv_head, k, j);
            for (int64_t i = 0; i < ni; ++i)
              output[i * nj + j] +=
                  left_value<operation>(problem, left, i, k) * right_kj;
          }
      } else if constexpr (order == Order::KIJ) {
        for (int64_t k = 0; k < nk; ++k)
          for (int64_t i = 0; i < ni; ++i) {
            const float left_ik = left_value<operation>(problem, left, i, k);
            for (int64_t j = 0; j < nj; ++j)
              output[i * nj + j] += left_ik * right_value<operation, layout>(
                                                      problem,
                                                      batch,
                                                      kv_head,
                                                      k,
                                                      j);
          }
      } else {
        for (int64_t k = 0; k < nk; ++k)
          for (int64_t j = 0; j < nj; ++j) {
            const float right_kj = right_value<operation, layout>(
                problem, batch, kv_head, k, j);
            for (int64_t i = 0; i < ni; ++i)
              output[i * nj + j] +=
                  left_value<operation>(problem, left, i, k) * right_kj;
          }
      }
    }
  }
}

#define KERNEL(operation, layout, order) \
  {Operation::operation, Layout::layout, Order::order, \
   attention_gemm<Operation::operation, Layout::layout, Order::order>}

const std::array<KernelSpec, 24> kKernels{{
    KERNEL(QK, HeadMajor, IJK),
    KERNEL(QK, HeadMajor, IKJ),
    KERNEL(QK, HeadMajor, JIK),
    KERNEL(QK, HeadMajor, JKI),
    KERNEL(QK, HeadMajor, KIJ),
    KERNEL(QK, HeadMajor, KJI),
    KERNEL(QK, TokenMajor, IJK),
    KERNEL(QK, TokenMajor, IKJ),
    KERNEL(QK, TokenMajor, JIK),
    KERNEL(QK, TokenMajor, JKI),
    KERNEL(QK, TokenMajor, KIJ),
    KERNEL(QK, TokenMajor, KJI),
    KERNEL(PV, HeadMajor, IJK),
    KERNEL(PV, HeadMajor, IKJ),
    KERNEL(PV, HeadMajor, JIK),
    KERNEL(PV, HeadMajor, JKI),
    KERNEL(PV, HeadMajor, KIJ),
    KERNEL(PV, HeadMajor, KJI),
    KERNEL(PV, TokenMajor, IJK),
    KERNEL(PV, TokenMajor, IKJ),
    KERNEL(PV, TokenMajor, JIK),
    KERNEL(PV, TokenMajor, JKI),
    KERNEL(PV, TokenMajor, KIJ),
    KERNEL(PV, TokenMajor, KJI),
}};

#undef KERNEL

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

std::vector<int64_t> parse_positive_integers(const std::string& text) {
  std::vector<int64_t> values;
  for (const std::string& token : split(text)) {
    const int64_t value = std::stoll(token);
    if (value <= 0)
      throw std::invalid_argument("dimension values must be positive");
    values.push_back(value);
  }
  if (values.empty())
    throw std::invalid_argument("at least one value is required");
  return values;
}

void print_usage(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --shapes slm,llm        Attention shapes (default: slm,llm)\n"
      << "  --query-lens N[,N...]   Query lengths (default: 1,16)\n"
      << "  --kv-lens N[,N...]      KV history lengths (default: 128,512)\n"
      << "  --batch-sizes N[,N...]  Batch sizes (default: 1)\n"
      << "  --warmups N              Randomized warmup rounds (default: 2)\n"
      << "  --repeats N              Measured paired rounds (default: 10)\n"
      << "  --seed N                 Data and shuffle seed\n"
      << "  --raw-csv PATH           Per-trial CSV\n"
      << "  --summary-csv PATH       Per-candidate summary CSV\n"
      << "  --decisions-csv PATH     Best candidate per workload CSV\n";
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--help" || argument == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (i + 1 >= argc)
      throw std::invalid_argument("missing value for " + argument);
    const std::string value = argv[++i];
    if (argument == "--shapes")
      options.shapes = split(value);
    else if (argument == "--query-lens")
      options.query_lengths = parse_positive_integers(value);
    else if (argument == "--kv-lens")
      options.kv_lengths = parse_positive_integers(value);
    else if (argument == "--batch-sizes")
      options.batch_sizes = parse_positive_integers(value);
    else if (argument == "--warmups")
      options.warmups = std::stoi(value);
    else if (argument == "--repeats")
      options.repeats = std::stoi(value);
    else if (argument == "--seed")
      options.seed = std::stoull(value);
    else if (argument == "--raw-csv")
      options.raw_csv = value;
    else if (argument == "--summary-csv")
      options.summary_csv = value;
    else if (argument == "--decisions-csv")
      options.decisions_csv = value;
    else
      throw std::invalid_argument("unknown option: " + argument);
  }
  if (options.warmups < 0 || options.repeats <= 0)
    throw std::invalid_argument("warmups must be >= 0 and repeats must be > 0");
  return options;
}

Shape find_shape(const std::string& name) {
  for (const Shape& shape : kShapes)
    if (shape.name == name)
      return shape;
  throw std::invalid_argument("unknown shape: " + name);
}

double percentile(std::vector<double> values, double fraction) {
  std::sort(values.begin(), values.end());
  const double index = fraction * static_cast<double>(values.size() - 1);
  const size_t lower = static_cast<size_t>(std::floor(index));
  const size_t upper = static_cast<size_t>(std::ceil(index));
  const double weight = index - static_cast<double>(lower);
  return values[lower] * (1.0 - weight) + values[upper] * weight;
}

double max_abs_difference(
    const std::vector<float>& actual,
    const std::vector<float>& expected) {
  double maximum = 0.0;
  for (size_t i = 0; i < actual.size(); ++i)
    maximum = std::max(maximum, std::abs(double(actual[i] - expected[i])));
  return maximum;
}

bool matches(const Trial& trial, const Workload& workload, Operation operation) {
  return trial.operation == operation_name(operation) &&
      trial.shape == workload.shape.name &&
      trial.batch_size == workload.batch_size &&
      trial.query_len == workload.query_len && trial.kv_len == workload.kv_len;
}

std::string candidate_name(const Trial& trial) {
  return trial.layout + ":" + trial.order;
}

int64_t output_elements(const Workload& workload, Operation operation) {
  const int64_t columns = operation == Operation::QK
      ? workload.kv_len
      : workload.shape.head_dim;
  return workload.batch_size * workload.shape.num_q_heads *
      workload.query_len * columns;
}

int64_t working_set_bytes(const Workload& workload, Operation operation) {
  const int64_t left_elements = operation == Operation::QK
      ? workload.batch_size * workload.shape.num_q_heads *
          workload.query_len * workload.shape.head_dim
      : workload.batch_size * workload.shape.num_q_heads *
          workload.query_len * workload.kv_len;
  const int64_t right_elements = workload.batch_size *
      workload.shape.num_kv_heads * workload.kv_len * workload.shape.head_dim;
  return (left_elements + right_elements + output_elements(workload, operation)) *
      int64_t(sizeof(float));
}

void write_raw_csv(const Options& options, const std::vector<Trial>& trials) {
  std::ofstream out(options.raw_csv);
  if (!out)
    throw std::runtime_error("cannot open raw CSV: " + options.raw_csv);
  out << "operation,shape,batch_size,num_q_heads,num_kv_heads,query_len,kv_len,"
         "head_dim,layout,loop_order,round,order_position,working_set_bytes,"
         "elapsed_ms,gflops,max_abs_error,correct,seed\n";
  out << std::setprecision(10);
  for (const Trial& trial : trials) {
    out << trial.operation << ',' << trial.shape << ',' << trial.batch_size << ','
        << trial.num_q_heads << ',' << trial.num_kv_heads << ','
        << trial.query_len << ',' << trial.kv_len << ',' << trial.head_dim << ','
        << trial.layout << ',' << trial.order << ',' << trial.round << ','
        << trial.position << ',' << trial.working_set_bytes << ','
        << trial.elapsed_ms << ',' << trial.gflops << ',' << trial.max_abs_error
        << ',' << (trial.correct ? "true" : "false") << ',' << options.seed
        << '\n';
  }
}

void write_summaries(
    const Options& options,
    const std::vector<Workload>& workloads,
    const std::vector<Trial>& trials) {
  std::ofstream summary(options.summary_csv);
  std::ofstream decisions(options.decisions_csv);
  if (!summary || !decisions)
    throw std::runtime_error("cannot open summary output CSV");

  summary << "operation,shape,batch_size,num_q_heads,num_kv_heads,query_len,"
             "kv_len,head_dim,working_set_bytes,layout,loop_order,warmup_rounds,"
             "measured_rounds,mean_ms,median_ms,p95_ms,min_ms,max_ms,"
             "median_gflops,speedup_vs_head_major_ijk,rank_overall,"
             "rank_within_layout,is_best,max_abs_error,correct\n";
  decisions << "operation,shape,batch_size,num_q_heads,num_kv_heads,query_len,"
               "kv_len,head_dim,working_set_bytes,best_layout,best_loop_order,"
               "best_median_ms,best_p95_ms,best_median_gflops,"
               "baseline_median_ms,speedup_vs_head_major_ijk,runner_up,"
               "winner_margin\n";
  summary << std::setprecision(10);
  decisions << std::setprecision(10);

  for (const Workload& workload : workloads) {
    for (Operation operation : {Operation::QK, Operation::PV}) {
      std::unordered_map<std::string, std::vector<const Trial*>> grouped;
      for (const Trial& trial : trials)
        if (matches(trial, workload, operation))
          grouped[candidate_name(trial)].push_back(&trial);

      std::unordered_map<std::string, double> medians;
      std::vector<std::pair<double, std::string>> overall_ranking;
      std::unordered_map<std::string, std::vector<std::pair<double, std::string>>>
          layout_rankings;
      for (const auto& [candidate, rows] : grouped) {
        std::vector<double> times;
        for (const Trial* row : rows)
          times.push_back(row->elapsed_ms);
        const double median = percentile(times, 0.5);
        medians[candidate] = median;
        overall_ranking.emplace_back(median, candidate);
        layout_rankings[rows.front()->layout].emplace_back(median, candidate);
      }
      std::sort(overall_ranking.begin(), overall_ranking.end());
      for (auto& [layout, ranking] : layout_rankings)
        std::sort(ranking.begin(), ranking.end());

      std::unordered_map<std::string, size_t> overall_ranks;
      std::unordered_map<std::string, size_t> layout_ranks;
      for (size_t i = 0; i < overall_ranking.size(); ++i)
        overall_ranks[overall_ranking[i].second] = i + 1;
      for (const auto& [layout, ranking] : layout_rankings)
        for (size_t i = 0; i < ranking.size(); ++i)
          layout_ranks[ranking[i].second] = i + 1;

      const double baseline = medians.at("head_major:ijk");
      for (const KernelSpec& kernel : kKernels) {
        if (kernel.operation != operation)
          continue;
        const std::string candidate = std::string(layout_name(kernel.layout)) +
            ":" + order_name(kernel.order);
        const auto& rows = grouped.at(candidate);
        std::vector<double> times;
        double maximum_error = 0.0;
        bool correct = true;
        for (const Trial* row : rows) {
          times.push_back(row->elapsed_ms);
          maximum_error = std::max(maximum_error, row->max_abs_error);
          correct = correct && row->correct;
        }
        const double mean =
            std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        const double median = medians.at(candidate);
        const double p95 = percentile(times, 0.95);
        const auto [minimum, maximum] =
            std::minmax_element(times.begin(), times.end());
        const double operations = 2.0 * workload.batch_size *
            workload.shape.num_q_heads * workload.query_len * workload.kv_len *
            workload.shape.head_dim;

        summary << operation_name(operation) << ',' << workload.shape.name << ','
                << workload.batch_size << ',' << workload.shape.num_q_heads << ','
                << workload.shape.num_kv_heads << ',' << workload.query_len << ','
                << workload.kv_len << ',' << workload.shape.head_dim << ','
                << working_set_bytes(workload, operation) << ','
                << layout_name(kernel.layout) << ',' << order_name(kernel.order)
                << ',' << options.warmups << ',' << options.repeats << ',' << mean
                << ',' << median << ',' << p95 << ',' << *minimum << ','
                << *maximum << ',' << operations / (median * 1.0e6) << ','
                << baseline / median << ',' << overall_ranks.at(candidate) << ','
                << layout_ranks.at(candidate) << ','
                << (overall_ranks.at(candidate) == 1 ? "true" : "false") << ','
                << maximum_error << ',' << (correct ? "true" : "false") << '\n';
      }

      const auto& best = overall_ranking[0];
      const auto& runner_up = overall_ranking[1];
      const auto& best_rows = grouped.at(best.second);
      std::vector<double> best_times;
      for (const Trial* row : best_rows)
        best_times.push_back(row->elapsed_ms);
      const double operations = 2.0 * workload.batch_size *
          workload.shape.num_q_heads * workload.query_len * workload.kv_len *
          workload.shape.head_dim;
      const size_t separator = best.second.find(':');
      decisions << operation_name(operation) << ',' << workload.shape.name << ','
                << workload.batch_size << ',' << workload.shape.num_q_heads << ','
                << workload.shape.num_kv_heads << ',' << workload.query_len << ','
                << workload.kv_len << ',' << workload.shape.head_dim << ','
                << working_set_bytes(workload, operation) << ','
                << best.second.substr(0, separator) << ','
                << best.second.substr(separator + 1) << ',' << best.first << ','
                << percentile(best_times, 0.95) << ','
                << operations / (best.first * 1.0e6) << ',' << baseline << ','
                << baseline / best.first << ',' << runner_up.second << ','
                << runner_up.first / best.first << '\n';
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    std::vector<Workload> workloads;
    for (const std::string& shape_name : options.shapes)
      for (int64_t batch_size : options.batch_sizes)
        for (int64_t query_len : options.query_lengths)
          for (int64_t kv_len : options.kv_lengths)
            workloads.push_back(
                {find_shape(shape_name), batch_size, query_len, kv_len});

    std::mt19937_64 random(options.seed);
    std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
    std::vector<Trial> trials;

    for (const Workload& workload : workloads) {
      const Shape& shape = workload.shape;
      const int64_t q_elements = workload.batch_size * shape.num_q_heads *
          workload.query_len * shape.head_dim;
      const int64_t kv_elements = workload.batch_size * shape.num_kv_heads *
          workload.kv_len * shape.head_dim;
      const int64_t probability_elements = workload.batch_size *
          shape.num_q_heads * workload.query_len * workload.kv_len;
      const int64_t qk_output_elements = probability_elements;
      const int64_t pv_output_elements = q_elements;

      std::vector<float> query(q_elements), probability(probability_elements);
      std::vector<float> key_head(kv_elements), key_token(kv_elements);
      std::vector<float> value_head(kv_elements), value_token(kv_elements);
      for (float& value : query)
        value = distribution(random);
      for (float& value : probability)
        value = distribution(random);
      for (float& value : key_head)
        value = distribution(random);
      for (float& value : value_head)
        value = distribution(random);

      for (int64_t batch = 0; batch < workload.batch_size; ++batch)
        for (int64_t head = 0; head < shape.num_kv_heads; ++head)
          for (int64_t token = 0; token < workload.kv_len; ++token)
            for (int64_t dim = 0; dim < shape.head_dim; ++dim) {
              const int64_t head_index =
                  (((batch * shape.num_kv_heads + head) * workload.kv_len +
                    token) *
                       shape.head_dim +
                   dim);
              const int64_t token_index =
                  (((batch * workload.kv_len + token) * shape.num_kv_heads +
                    head) *
                       shape.head_dim +
                   dim);
              key_token[token_index] = key_head[head_index];
              value_token[token_index] = value_head[head_index];
            }

      std::vector<float> qk_output(qk_output_elements);
      std::vector<float> pv_output(pv_output_elements);
      std::vector<float> qk_reference(qk_output_elements, 0.0f);
      std::vector<float> pv_reference(pv_output_elements, 0.0f);
      Problem problem{
          workload.batch_size,
          shape.num_q_heads,
          shape.num_kv_heads,
          workload.query_len,
          workload.kv_len,
          shape.head_dim,
          query.data(),
          key_head.data(),
          key_token.data(),
          probability.data(),
          value_head.data(),
          value_token.data(),
          nullptr,
      };

      problem.output = qk_reference.data();
      attention_gemm<Operation::QK, Layout::HeadMajor, Order::IJK>(problem);
      problem.output = pv_reference.data();
      attention_gemm<Operation::PV, Layout::HeadMajor, Order::IJK>(problem);

      std::array<size_t, kKernels.size()> order{};
      std::iota(order.begin(), order.end(), size_t(0));
      const int total_rounds = options.warmups + options.repeats;
      for (int round = 0; round < total_rounds; ++round) {
        std::shuffle(order.begin(), order.end(), random);
        for (size_t position = 0; position < order.size(); ++position) {
          const KernelSpec& kernel = kKernels[order[position]];
          std::vector<float>& output =
              kernel.operation == Operation::QK ? qk_output : pv_output;
          const std::vector<float>& reference = kernel.operation == Operation::QK
              ? qk_reference
              : pv_reference;
          std::fill(output.begin(), output.end(), 0.0f);
          problem.output = output.data();

          const auto start = Clock::now();
          kernel.function(problem);
          const auto end = Clock::now();
          const double elapsed_ms =
              std::chrono::duration<double, std::milli>(end - start).count();
          const double error = max_abs_difference(output, reference);
          const int64_t reduction_size = kernel.operation == Operation::QK
              ? shape.head_dim
              : workload.kv_len;
          const double tolerance = 1.0e-3 * std::max(int64_t(1), reduction_size);
          const bool correct = error <= tolerance;
          if (!correct) {
            std::cerr << "correctness failure: operation="
                      << operation_name(kernel.operation)
                      << " shape=" << shape.name
                      << " layout=" << layout_name(kernel.layout)
                      << " order=" << order_name(kernel.order)
                      << " query_len=" << workload.query_len
                      << " kv_len=" << workload.kv_len
                      << " max_abs_error=" << error
                      << " tolerance=" << tolerance << '\n';
            return 2;
          }

          if (round >= options.warmups) {
            const double operations = 2.0 * workload.batch_size *
                shape.num_q_heads * workload.query_len * workload.kv_len *
                shape.head_dim;
            trials.push_back({
                operation_name(kernel.operation),
                shape.name,
                workload.batch_size,
                shape.num_q_heads,
                shape.num_kv_heads,
                workload.query_len,
                workload.kv_len,
                shape.head_dim,
                layout_name(kernel.layout),
                order_name(kernel.order),
                round - options.warmups,
                static_cast<int>(position),
                working_set_bytes(workload, kernel.operation),
                elapsed_ms,
                operations / (elapsed_ms * 1.0e6),
                error,
                correct,
            });
          }
        }
      }
    }

    write_raw_csv(options, trials);
    write_summaries(options, workloads, trials);
    std::cout << "Wrote " << trials.size() << " trials to " << options.raw_csv
              << '\n';
    std::cout << "Wrote candidate summary to " << options.summary_csv << '\n';
    std::cout << "Wrote workload decisions to " << options.decisions_csv << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
