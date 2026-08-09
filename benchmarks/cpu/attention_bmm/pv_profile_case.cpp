/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 *
 * Long-running single-kernel P*V case for process-level hardware counters.
 ******************************************************************************/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::string order;
  int64_t query_len = 0;
  int64_t kv_len = 0;
  int64_t head_dim = 0;
  uint64_t seed = 11;
  int warmups = 2;
  double minimum_seconds = 5.0;
  std::string result_csv;
};

void pv_ikj(
    const float* probability,
    const float* value,
    float* output,
    int64_t query_len,
    int64_t kv_len,
    int64_t head_dim) {
  for (int64_t i = 0; i < query_len; ++i)
    for (int64_t k = 0; k < kv_len; ++k) {
      const float probability_ik = probability[i * kv_len + k];
      for (int64_t j = 0; j < head_dim; ++j)
        output[i * head_dim + j] +=
            probability_ik * value[k * head_dim + j];
    }
}

void pv_kij(
    const float* probability,
    const float* value,
    float* output,
    int64_t query_len,
    int64_t kv_len,
    int64_t head_dim) {
  for (int64_t k = 0; k < kv_len; ++k)
    for (int64_t i = 0; i < query_len; ++i) {
      const float probability_ik = probability[i * kv_len + k];
      for (int64_t j = 0; j < head_dim; ++j)
        output[i * head_dim + j] +=
            probability_ik * value[k * head_dim + j];
    }
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--help") {
      std::cout
          << "Usage: " << argv[0]
          << " --order ikj|kij --query-len M --kv-len N --head-dim D"
             " [--seed N] [--warmups N] [--minimum-seconds F]\n";
      std::exit(0);
    }
    if (i + 1 >= argc)
      throw std::invalid_argument("missing value for " + argument);
    const std::string value = argv[++i];
    if (argument == "--order")
      options.order = value;
    else if (argument == "--query-len")
      options.query_len = std::stoll(value);
    else if (argument == "--kv-len")
      options.kv_len = std::stoll(value);
    else if (argument == "--head-dim")
      options.head_dim = std::stoll(value);
    else if (argument == "--seed")
      options.seed = std::stoull(value);
    else if (argument == "--warmups")
      options.warmups = std::stoi(value);
    else if (argument == "--minimum-seconds")
      options.minimum_seconds = std::stod(value);
    else if (argument == "--result-csv")
      options.result_csv = value;
    else
      throw std::invalid_argument("unknown option: " + argument);
  }
  if ((options.order != "ikj" && options.order != "kij") ||
      options.query_len <= 0 || options.kv_len <= 0 ||
      options.head_dim <= 0 || options.warmups < 0 ||
      options.minimum_seconds <= 0.0)
    throw std::invalid_argument("invalid or incomplete arguments");
  return options;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const int64_t probability_elements = options.query_len * options.kv_len;
    const int64_t value_elements = options.kv_len * options.head_dim;
    const int64_t output_elements = options.query_len * options.head_dim;
    std::vector<float> probability(probability_elements);
    std::vector<float> value(value_elements);
    std::vector<float> output(output_elements);
    std::mt19937_64 random(options.seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (float& element : probability)
      element = distribution(random);
    for (float& element : value)
      element = distribution(random);

    const auto kernel = options.order == "ikj" ? pv_ikj : pv_kij;
    for (int i = 0; i < options.warmups; ++i) {
      std::fill(output.begin(), output.end(), 0.0f);
      kernel(
          probability.data(),
          value.data(),
          output.data(),
          options.query_len,
          options.kv_len,
          options.head_dim);
    }

    int64_t iterations = 0;
    const auto begin = Clock::now();
    double elapsed_seconds = 0.0;
    do {
      std::fill(output.begin(), output.end(), 0.0f);
      kernel(
          probability.data(),
          value.data(),
          output.data(),
          options.query_len,
          options.kv_len,
          options.head_dim);
      ++iterations;
      elapsed_seconds =
          std::chrono::duration<double>(Clock::now() - begin).count();
    } while (elapsed_seconds < options.minimum_seconds);

    double checksum = 0.0;
    for (float element : output)
      checksum += element;
    const double operations = 2.0 * options.query_len * options.kv_len *
        options.head_dim * iterations;
    std::ostringstream result;
    result << std::setprecision(12)
           << "order,query_len,kv_len,head_dim,seed,iterations,elapsed_s,"
              "gflops,checksum\n"
           << options.order << ',' << options.query_len << ','
           << options.kv_len << ',' << options.head_dim << ',' << options.seed
           << ',' << iterations << ',' << elapsed_seconds << ','
           << operations / elapsed_seconds / 1.0e9 << ',' << checksum << '\n';
    std::cout << result.str();
    if (!options.result_csv.empty()) {
      std::ofstream stream(options.result_csv);
      if (!stream)
        throw std::runtime_error("cannot open result CSV: " + options.result_csv);
      stream << result.str();
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
