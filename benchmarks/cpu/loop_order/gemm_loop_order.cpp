/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 *
 * Compare the six loop permutations of row-major C = A * B on one CPU core.
 ******************************************************************************/

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Kernel = void (*)(const float*, const float*, float*, int64_t);

struct Options {
  std::vector<int64_t> sizes{64, 128, 256, 512};
  int warmups = 2;
  int repeats = 20;
  uint64_t seed = 20260809;
  std::string raw_csv = "gemm_loop_order_trials.csv";
  std::string summary_csv = "gemm_loop_order_summary.csv";
};

struct KernelSpec {
  const char* name;
  Kernel function;
};

struct Trial {
  int64_t size;
  int round;
  int position;
  std::string order;
  double elapsed_ms;
  double gflops;
  double max_abs_error;
  bool correct;
};

void gemm_ijk(const float* a, const float* b, float* c, int64_t n) {
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < n; ++j)
      for (int64_t k = 0; k < n; ++k)
        c[i * n + j] += a[i * n + k] * b[k * n + j];
}

void gemm_ikj(const float* a, const float* b, float* c, int64_t n) {
  for (int64_t i = 0; i < n; ++i)
    for (int64_t k = 0; k < n; ++k) {
      const float aik = a[i * n + k];
      for (int64_t j = 0; j < n; ++j)
        c[i * n + j] += aik * b[k * n + j];
    }
}

void gemm_jik(const float* a, const float* b, float* c, int64_t n) {
  for (int64_t j = 0; j < n; ++j)
    for (int64_t i = 0; i < n; ++i)
      for (int64_t k = 0; k < n; ++k)
        c[i * n + j] += a[i * n + k] * b[k * n + j];
}

void gemm_jki(const float* a, const float* b, float* c, int64_t n) {
  for (int64_t j = 0; j < n; ++j)
    for (int64_t k = 0; k < n; ++k) {
      const float bkj = b[k * n + j];
      for (int64_t i = 0; i < n; ++i)
        c[i * n + j] += a[i * n + k] * bkj;
    }
}

void gemm_kij(const float* a, const float* b, float* c, int64_t n) {
  for (int64_t k = 0; k < n; ++k)
    for (int64_t i = 0; i < n; ++i) {
      const float aik = a[i * n + k];
      for (int64_t j = 0; j < n; ++j)
        c[i * n + j] += aik * b[k * n + j];
    }
}

void gemm_kji(const float* a, const float* b, float* c, int64_t n) {
  for (int64_t k = 0; k < n; ++k)
    for (int64_t j = 0; j < n; ++j) {
      const float bkj = b[k * n + j];
      for (int64_t i = 0; i < n; ++i)
        c[i * n + j] += a[i * n + k] * bkj;
    }
}

const std::array<KernelSpec, 6> kKernels{{
    {"ijk", gemm_ijk},
    {"ikj", gemm_ikj},
    {"jik", gemm_jik},
    {"jki", gemm_jki},
    {"kij", gemm_kij},
    {"kji", gemm_kji},
}};

std::vector<int64_t> parse_sizes(const std::string& text) {
  std::vector<int64_t> sizes;
  size_t begin = 0;
  while (begin < text.size()) {
    const size_t end = text.find(',', begin);
    const std::string token = text.substr(begin, end - begin);
    const int64_t value = std::stoll(token);
    if (value <= 0)
      throw std::invalid_argument("matrix sizes must be positive");
    sizes.push_back(value);
    if (end == std::string::npos)
      break;
    begin = end + 1;
  }
  if (sizes.empty())
    throw std::invalid_argument("at least one matrix size is required");
  return sizes;
}

void print_usage(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --sizes N[,N...]      Square matrix sizes (default: 64,128,256,512)\n"
      << "  --warmups N           Randomized warmup rounds (default: 2)\n"
      << "  --repeats N           Randomized measured rounds (default: 20)\n"
      << "  --seed N              Shuffle/data seed (default: 20260809)\n"
      << "  --raw-csv PATH        Per-trial output CSV\n"
      << "  --summary-csv PATH    Aggregated output CSV\n";
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (i + 1 >= argc)
      throw std::invalid_argument("missing value for " + arg);
    const std::string value = argv[++i];
    if (arg == "--sizes")
      options.sizes = parse_sizes(value);
    else if (arg == "--warmups")
      options.warmups = std::stoi(value);
    else if (arg == "--repeats")
      options.repeats = std::stoi(value);
    else if (arg == "--seed")
      options.seed = std::stoull(value);
    else if (arg == "--raw-csv")
      options.raw_csv = value;
    else if (arg == "--summary-csv")
      options.summary_csv = value;
    else
      throw std::invalid_argument("unknown option: " + arg);
  }
  if (options.warmups < 0 || options.repeats <= 0)
    throw std::invalid_argument("warmups must be >= 0 and repeats must be > 0");
  return options;
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

void write_raw_csv(const Options& options, const std::vector<Trial>& trials) {
  std::ofstream out(options.raw_csv);
  if (!out)
    throw std::runtime_error("cannot open raw CSV: " + options.raw_csv);
  out << "matrix_size,round,order_position,loop_order,elapsed_ms,gflops,"
         "max_abs_error,correct,seed\n";
  out << std::setprecision(10);
  for (const auto& trial : trials) {
    out << trial.size << ',' << trial.round << ',' << trial.position << ','
        << trial.order << ',' << trial.elapsed_ms << ',' << trial.gflops << ','
        << trial.max_abs_error << ',' << (trial.correct ? "true" : "false")
        << ',' << options.seed << '\n';
  }
}

void write_summary_csv(const Options& options, const std::vector<Trial>& trials) {
  std::ofstream out(options.summary_csv);
  if (!out)
    throw std::runtime_error("cannot open summary CSV: " + options.summary_csv);
  out << "matrix_size,loop_order,warmup_rounds,measured_rounds,mean_ms,median_ms,"
         "p95_ms,min_ms,max_ms,median_gflops,speedup_vs_ijk,rank_by_median,"
         "is_best,max_abs_error,correct\n";
  out << std::setprecision(10);

  for (int64_t size : options.sizes) {
    std::unordered_map<std::string, std::vector<const Trial*>> grouped;
    for (const auto& trial : trials)
      if (trial.size == size)
        grouped[trial.order].push_back(&trial);

    std::vector<double> ijk_times;
    for (const Trial* trial : grouped.at("ijk"))
      ijk_times.push_back(trial->elapsed_ms);
    const double ijk_median = percentile(ijk_times, 0.5);

    std::vector<std::pair<double, std::string>> median_ranking;
    for (const auto& kernel : kKernels) {
      std::vector<double> times;
      for (const Trial* trial : grouped.at(kernel.name))
        times.push_back(trial->elapsed_ms);
      median_ranking.emplace_back(percentile(times, 0.5), kernel.name);
    }
    std::sort(median_ranking.begin(), median_ranking.end());
    std::unordered_map<std::string, size_t> ranks;
    for (size_t rank = 0; rank < median_ranking.size(); ++rank)
      ranks[median_ranking[rank].second] = rank + 1;

    for (const auto& kernel : kKernels) {
      const auto& rows = grouped.at(kernel.name);
      std::vector<double> times;
      times.reserve(rows.size());
      double max_error = 0.0;
      bool correct = true;
      for (const Trial* row : rows) {
        times.push_back(row->elapsed_ms);
        max_error = std::max(max_error, row->max_abs_error);
        correct = correct && row->correct;
      }
      const double mean =
          std::accumulate(times.begin(), times.end(), 0.0) / times.size();
      const double median = percentile(times, 0.5);
      const double p95 = percentile(times, 0.95);
      const auto [minimum, maximum] =
          std::minmax_element(times.begin(), times.end());
      const double operations = 2.0 * size * size * size;
      const double median_gflops = operations / (median * 1.0e6);

      out << size << ',' << kernel.name << ',' << options.warmups << ','
          << options.repeats << ',' << mean << ',' << median << ',' << p95
          << ',' << *minimum << ',' << *maximum << ',' << median_gflops << ','
          << ijk_median / median << ',' << ranks.at(kernel.name) << ','
          << (ranks.at(kernel.name) == 1 ? "true" : "false") << ','
          << max_error << ','
          << (correct ? "true" : "false") << '\n';
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    std::mt19937_64 random(options.seed);
    std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
    std::vector<Trial> trials;

    for (int64_t n : options.sizes) {
      const size_t elements = static_cast<size_t>(n * n);
      std::vector<float> a(elements), b(elements), c(elements), reference(elements);
      for (float& value : a)
        value = distribution(random);
      for (float& value : b)
        value = distribution(random);

      std::fill(reference.begin(), reference.end(), 0.0f);
      gemm_ijk(a.data(), b.data(), reference.data(), n);

      std::array<size_t, kKernels.size()> order{};
      std::iota(order.begin(), order.end(), size_t(0));
      const int total_rounds = options.warmups + options.repeats;
      for (int round = 0; round < total_rounds; ++round) {
        std::shuffle(order.begin(), order.end(), random);
        for (size_t position = 0; position < order.size(); ++position) {
          const auto& kernel = kKernels[order[position]];
          std::fill(c.begin(), c.end(), 0.0f);
          const auto start = Clock::now();
          kernel.function(a.data(), b.data(), c.data(), n);
          const auto end = Clock::now();

          const double elapsed_ms =
              std::chrono::duration<double, std::milli>(end - start).count();
          const double error = max_abs_difference(c, reference);
          const double tolerance = 1.0e-3 * std::max(1.0, double(n));
          const bool correct = error <= tolerance;
          if (!correct) {
            std::cerr << "correctness failure: size=" << n
                      << " order=" << kernel.name << " max_abs_error=" << error
                      << " tolerance=" << tolerance << '\n';
            return 2;
          }

          if (round >= options.warmups) {
            const double operations = 2.0 * n * n * n;
            trials.push_back({
                n,
                round - options.warmups,
                static_cast<int>(position),
                kernel.name,
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
    write_summary_csv(options, trials);
    std::cout << "Wrote " << trials.size() << " trials to " << options.raw_csv
              << '\n';
    std::cout << "Wrote summary to " << options.summary_csv << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
