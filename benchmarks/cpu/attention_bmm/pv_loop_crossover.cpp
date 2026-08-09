/******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 * Portions of this file consist of AI-generated content
 *
 * Test when P*V benefits from reusing each V row across query rows.
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
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Kernel = void (*)(const float*, const float*, float*, int64_t, int64_t, int64_t);

struct Shape {
  std::string name;
  int64_t head_dim;
};

const std::array<Shape, 2> kShapes{{{"slm", 64}, {"llm", 128}}};

struct Options {
  std::vector<std::string> shapes{"slm", "llm"};
  std::vector<int64_t> query_lengths{1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
  std::vector<int64_t> kv_lengths{128, 512, 2048, 8192, 16384};
  std::vector<uint64_t> data_seeds{11, 29, 47};
  int warmups = 2;
  int repeats = 20;
  uint64_t order_seed = 20260809;
  int bootstrap_samples = 10000;
  double minimum_effect = 0.05;
  double minimum_win_rate = 0.80;
  std::string raw_csv = "pv_crossover_trials.csv";
  std::string summary_csv = "pv_crossover_summary.csv";
  std::string report_md = "pv_crossover_report.md";
};

struct Trial {
  std::string shape;
  int64_t head_dim;
  int64_t query_len;
  int64_t kv_len;
  int64_t working_set_bytes;
  int64_t output_bytes;
  uint64_t data_seed;
  int round;
  int position;
  std::string order;
  double elapsed_ms;
  double gflops;
  double max_abs_error;
  bool correct;
};

struct Summary {
  std::string shape;
  int64_t head_dim;
  int64_t query_len;
  int64_t kv_len;
  int64_t working_set_bytes;
  int64_t output_bytes;
  int pairs;
  double ikj_median_ms;
  double ikj_p95_ms;
  double kij_median_ms;
  double kij_p95_ms;
  double kij_speedup;
  double ci_low;
  double ci_high;
  double kij_win_rate;
  std::string decision;
  std::string expected;
  std::string claim_match;
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

const std::array<std::pair<const char*, Kernel>, 2> kKernels{{
    {"ikj", pv_ikj},
    {"kij", pv_kij},
}};

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

std::vector<uint64_t> parse_seeds(const std::string& text) {
  std::vector<uint64_t> values;
  for (const std::string& token : split(text))
    values.push_back(std::stoull(token));
  if (values.empty())
    throw std::invalid_argument("at least one data seed is required");
  return values;
}

void print_usage(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --shapes slm,llm          Head dimensions to test\n"
      << "  --query-lens N[,N...]     Query lengths\n"
      << "  --kv-lens N[,N...]        KV-history lengths\n"
      << "  --data-seeds N[,N...]     Independent random input seeds\n"
      << "  --warmups N                Warmup pairs per seed\n"
      << "  --repeats N                Measured pairs per seed\n"
      << "  --order-seed N             Candidate-order seed\n"
      << "  --bootstrap-samples N      Bootstrap resamples\n"
      << "  --minimum-effect F         Strict effect threshold\n"
      << "  --minimum-win-rate F       Strict paired-win threshold\n"
      << "  --raw-csv PATH             Per-trial output\n"
      << "  --summary-csv PATH         Per-workload decisions\n"
      << "  --report-md PATH           Automatic claim report\n";
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
    else if (argument == "--data-seeds")
      options.data_seeds = parse_seeds(value);
    else if (argument == "--warmups")
      options.warmups = std::stoi(value);
    else if (argument == "--repeats")
      options.repeats = std::stoi(value);
    else if (argument == "--order-seed")
      options.order_seed = std::stoull(value);
    else if (argument == "--bootstrap-samples")
      options.bootstrap_samples = std::stoi(value);
    else if (argument == "--minimum-effect")
      options.minimum_effect = std::stod(value);
    else if (argument == "--minimum-win-rate")
      options.minimum_win_rate = std::stod(value);
    else if (argument == "--raw-csv")
      options.raw_csv = value;
    else if (argument == "--summary-csv")
      options.summary_csv = value;
    else if (argument == "--report-md")
      options.report_md = value;
    else
      throw std::invalid_argument("unknown option: " + argument);
  }
  if (options.warmups < 0 || options.repeats <= 0 ||
      options.bootstrap_samples <= 0)
    throw std::invalid_argument("invalid warmup, repeat, or bootstrap count");
  if (options.minimum_effect < 0.0 || options.minimum_win_rate < 0.5 ||
      options.minimum_win_rate > 1.0)
    throw std::invalid_argument("invalid statistical decision threshold");
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

std::pair<double, double> bootstrap_median_ci(
    const std::vector<double>& values,
    int samples,
    uint64_t seed) {
  std::mt19937_64 random(seed);
  std::uniform_int_distribution<size_t> pick(0, values.size() - 1);
  std::vector<double> medians;
  std::vector<double> sample(values.size());
  medians.reserve(samples);
  for (int iteration = 0; iteration < samples; ++iteration) {
    for (double& value : sample)
      value = values[pick(random)];
    medians.push_back(percentile(sample, 0.5));
  }
  return {percentile(medians, 0.025), percentile(medians, 0.975)};
}

std::string expected_result(int64_t query_len, int64_t kv_len) {
  if (query_len == 1)
    return "tie";
  if (query_len >= 16 && query_len <= 128 && kv_len >= 2048)
    return "kij";
  return "exploratory";
}

void write_raw_csv(const Options& options, const std::vector<Trial>& trials) {
  std::ofstream out(options.raw_csv);
  if (!out)
    throw std::runtime_error("cannot open raw CSV: " + options.raw_csv);
  out << "shape,head_dim,query_len,kv_len,working_set_bytes,output_bytes,"
         "data_seed,round,order_position,loop_order,elapsed_ms,gflops,"
         "max_abs_error,correct,order_seed\n";
  out << std::setprecision(10);
  for (const Trial& trial : trials) {
    out << trial.shape << ',' << trial.head_dim << ',' << trial.query_len << ','
        << trial.kv_len << ',' << trial.working_set_bytes << ','
        << trial.output_bytes << ',' << trial.data_seed << ',' << trial.round
        << ',' << trial.position << ',' << trial.order << ',' << trial.elapsed_ms
        << ',' << trial.gflops << ',' << trial.max_abs_error << ','
        << (trial.correct ? "true" : "false") << ',' << options.order_seed
        << '\n';
  }
}

std::vector<Summary> summarize(
    const Options& options,
    const std::vector<Trial>& trials) {
  using WorkloadKey = std::tuple<std::string, int64_t, int64_t>;
  using PairKey = std::pair<uint64_t, int>;
  std::map<WorkloadKey, std::vector<const Trial*>> groups;
  for (const Trial& trial : trials)
    groups[{trial.shape, trial.query_len, trial.kv_len}].push_back(&trial);

  std::vector<Summary> summaries;
  for (const auto& [key, rows] : groups) {
    std::vector<double> ikj_times;
    std::vector<double> kij_times;
    std::map<PairKey, std::map<std::string, double>> pairs;
    for (const Trial* row : rows) {
      (row->order == "ikj" ? ikj_times : kij_times).push_back(row->elapsed_ms);
      pairs[{row->data_seed, row->round}][row->order] = row->elapsed_ms;
    }

    std::vector<double> paired_speedups;
    int kij_wins = 0;
    for (const auto& [pair_key, times] : pairs) {
      const double speedup = times.at("ikj") / times.at("kij");
      paired_speedups.push_back(speedup);
      kij_wins += speedup > 1.0;
    }
    const double median_speedup = percentile(paired_speedups, 0.5);
    const auto [ci_low, ci_high] = bootstrap_median_ci(
        paired_speedups,
        options.bootstrap_samples,
        options.order_seed + summaries.size());
    const double win_rate = double(kij_wins) / paired_speedups.size();

    std::string decision = "tie";
    if (median_speedup >= 1.0 + options.minimum_effect &&
        win_rate >= options.minimum_win_rate && ci_low > 1.0) {
      decision = "kij";
    } else if (
        median_speedup <= 1.0 / (1.0 + options.minimum_effect) &&
        win_rate <= 1.0 - options.minimum_win_rate && ci_high < 1.0) {
      decision = "ikj";
    }

    const Trial* first = rows.front();
    const std::string expected =
        expected_result(first->query_len, first->kv_len);
    const std::string claim_match =
        expected == "exploratory" ? "na" : (decision == expected ? "yes" : "no");
    summaries.push_back({
        first->shape,
        first->head_dim,
        first->query_len,
        first->kv_len,
        first->working_set_bytes,
        first->output_bytes,
        static_cast<int>(paired_speedups.size()),
        percentile(ikj_times, 0.5),
        percentile(ikj_times, 0.95),
        percentile(kij_times, 0.5),
        percentile(kij_times, 0.95),
        median_speedup,
        ci_low,
        ci_high,
        win_rate,
        decision,
        expected,
        claim_match,
    });
  }
  return summaries;
}

void write_summary_csv(
    const Options& options,
    const std::vector<Summary>& summaries) {
  std::ofstream out(options.summary_csv);
  if (!out)
    throw std::runtime_error("cannot open summary CSV: " + options.summary_csv);
  out << "shape,head_dim,query_len,kv_len,working_set_bytes,output_bytes,pairs,"
         "ikj_median_ms,ikj_p95_ms,kij_median_ms,kij_p95_ms,kij_speedup,"
         "speedup_ci_low,speedup_ci_high,kij_win_rate,decision,expected,"
         "claim_match\n";
  out << std::setprecision(10);
  for (const Summary& row : summaries) {
    out << row.shape << ',' << row.head_dim << ',' << row.query_len << ','
        << row.kv_len << ',' << row.working_set_bytes << ',' << row.output_bytes
        << ',' << row.pairs << ',' << row.ikj_median_ms << ',' << row.ikj_p95_ms
        << ',' << row.kij_median_ms << ',' << row.kij_p95_ms << ','
        << row.kij_speedup << ',' << row.ci_low << ',' << row.ci_high << ','
        << row.kij_win_rate << ',' << row.decision << ',' << row.expected << ','
        << row.claim_match << '\n';
  }
}

void write_report(
    const Options& options,
    const std::vector<Summary>& summaries) {
  std::ofstream out(options.report_md);
  if (!out)
    throw std::runtime_error("cannot open report: " + options.report_md);

  int tested = 0;
  int matched = 0;
  int counterexamples = 0;
  int kij_wins = 0;
  int ikj_wins = 0;
  int ties = 0;
  for (const Summary& row : summaries) {
    kij_wins += row.decision == "kij";
    ikj_wins += row.decision == "ikj";
    ties += row.decision == "tie";
    if (row.claim_match == "na")
      continue;
    ++tested;
    matched += row.claim_match == "yes";
    counterexamples += row.claim_match == "no";
  }

  out << "# P*V Loop-Crossover Hypothesis\n\n"
      << "## Pre-Registered Claim\n\n"
      << "- `M=1`: `ikj` and `kij` are equivalent, so the expected decision is `tie`.\n"
      << "- Long-context `M=16..128, N>=2048`: `kij` is expected to win by reusing each V row across query rows.\n"
      << "- Other cells are exploratory and locate where the locality advantage appears or reverses.\n"
      << "- Evidence against a universal loop order requires at least two strict workload wins for each order.\n\n"
      << "A strict winner requires at least " << options.minimum_effect * 100.0
      << "% paired median improvement, " << options.minimum_win_rate * 100.0
      << "% pair wins, a 95% bootstrap CI excluding 1, and no post-hoc threshold changes.\n\n"
      << "## Result\n\n"
      << "- Claim-scored workloads: `" << tested << "`\n"
      << "- Matches: `" << matched << "`\n"
      << "- Counterexamples: `" << counterexamples << "`\n"
      << "- Strict decisions across the full matrix: `kij=" << kij_wins
      << "`, `ikj=" << ikj_wins << "`, `tie=" << ties << "`\n"
      << "- Universal-order test: `"
      << (kij_wins >= 2 && ikj_wins >= 2
              ? "rejected; the preferred order is workload-dependent"
              : "not rejected; both orders did not earn two strict wins")
      << "`\n\n"
      << "## Trend By Query Length\n\n"
      << "| query_len | workloads | median kij speedup | kij wins | ties | ikj wins |\n"
      << "| ---: | ---: | ---: | ---: | ---: | ---: |\n";

  std::map<int64_t, std::vector<const Summary*>> by_query;
  for (const Summary& row : summaries)
    by_query[row.query_len].push_back(&row);
  for (const auto& [query_len, rows] : by_query) {
    std::vector<double> speedups;
    int kij = 0, tie = 0, ikj = 0;
    for (const Summary* row : rows) {
      speedups.push_back(row->kij_speedup);
      kij += row->decision == "kij";
      tie += row->decision == "tie";
      ikj += row->decision == "ikj";
    }
    out << "| " << query_len << " | " << rows.size() << " | "
        << std::fixed << std::setprecision(3) << percentile(speedups, 0.5)
        << "x | " << kij << " | " << tie << " | " << ikj << " |\n";
  }

  out << "\n## Counterexamples\n\n";
  if (counterexamples == 0) {
    out << "None under the pre-registered decision rule.\n";
  } else {
    out << "| shape | M | N | expected | observed | speedup | 95% CI | win rate |\n"
        << "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |\n";
    for (const Summary& row : summaries) {
      if (row.claim_match != "no")
        continue;
      out << "| " << row.shape << " | " << row.query_len << " | "
          << row.kv_len << " | " << row.expected << " | " << row.decision
          << " | " << row.kij_speedup << "x | " << row.ci_low << "-"
          << row.ci_high << " | " << row.kij_win_rate << " |\n";
    }
  }

  out << "\n## Scope Guardrail\n\n"
      << "This is a single-head FP32 microbenchmark with random dense values. It tests P*V loop locality, not complete attention or end-to-end PACE latency.\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    std::vector<Trial> trials;
    std::mt19937_64 order_random(options.order_seed);

    for (const std::string& shape_name : options.shapes) {
      const Shape shape = find_shape(shape_name);
      for (int64_t query_len : options.query_lengths) {
        for (int64_t kv_len : options.kv_lengths) {
          const int64_t probability_elements = query_len * kv_len;
          const int64_t value_elements = kv_len * shape.head_dim;
          const int64_t output_elements = query_len * shape.head_dim;
          const int64_t working_set =
              (probability_elements + value_elements + output_elements) *
              int64_t(sizeof(float));
          const int64_t output_size = output_elements * int64_t(sizeof(float));
          const double operations =
              2.0 * query_len * kv_len * shape.head_dim;

          for (uint64_t data_seed : options.data_seeds) {
            std::mt19937_64 data_random(data_seed);
            std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
            std::vector<float> probability(probability_elements);
            std::vector<float> value(value_elements);
            std::vector<float> output(output_elements);
            std::vector<float> reference(output_elements, 0.0f);
            for (float& element : probability)
              element = distribution(data_random);
            for (float& element : value)
              element = distribution(data_random);
            pv_ikj(
                probability.data(),
                value.data(),
                reference.data(),
                query_len,
                kv_len,
                shape.head_dim);

            std::array<size_t, 2> order{0, 1};
            const int total_rounds = options.warmups + options.repeats;
            for (int round = 0; round < total_rounds; ++round) {
              std::shuffle(order.begin(), order.end(), order_random);
              for (size_t position = 0; position < order.size(); ++position) {
                const auto& [name, kernel] = kKernels[order[position]];
                std::fill(output.begin(), output.end(), 0.0f);
                const auto start = Clock::now();
                kernel(
                    probability.data(),
                    value.data(),
                    output.data(),
                    query_len,
                    kv_len,
                    shape.head_dim);
                const auto end = Clock::now();
                const double elapsed_ms =
                    std::chrono::duration<double, std::milli>(end - start)
                        .count();
                const double error = max_abs_difference(output, reference);
                const double tolerance = 1.0e-4 * std::max(int64_t(1), kv_len);
                const bool correct = error <= tolerance;
                if (!correct) {
                  std::cerr << "correctness failure: shape=" << shape.name
                            << " M=" << query_len << " N=" << kv_len
                            << " seed=" << data_seed << " order=" << name
                            << " max_abs_error=" << error << '\n';
                  return 2;
                }

                if (round >= options.warmups) {
                  trials.push_back({
                      shape.name,
                      shape.head_dim,
                      query_len,
                      kv_len,
                      working_set,
                      output_size,
                      data_seed,
                      round - options.warmups,
                      static_cast<int>(position),
                      name,
                      elapsed_ms,
                      operations / (elapsed_ms * 1.0e6),
                      error,
                      correct,
                  });
                }
              }
            }
          }
        }
      }
    }

    write_raw_csv(options, trials);
    const std::vector<Summary> summaries = summarize(options, trials);
    write_summary_csv(options, summaries);
    write_report(options, summaries);
    std::cout << "Wrote " << trials.size() << " trials to " << options.raw_csv
              << '\n';
    std::cout << "Wrote " << summaries.size() << " workload summaries to "
              << options.summary_csv << '\n';
    std::cout << "Wrote claim report to " << options.report_md << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
