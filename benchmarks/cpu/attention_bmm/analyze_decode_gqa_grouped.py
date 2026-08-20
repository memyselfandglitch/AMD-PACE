#!/usr/bin/env python3
"""Analyze paired grouped-query GQA decode trials."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


BASELINE = "head_major_head_first"
BM_BF = "block_major_block_first"
HM_HF_GROUPED = "head_major_head_first_grouped"
BM_BF_GROUPED = "block_major_block_first_grouped"
CANDIDATES = (BASELINE, BM_BF, HM_HF_GROUPED, BM_BF_GROUPED)
COMPARISONS = (
    ("existing_codesign", BASELINE, BM_BF),
    ("grouping_head_major", BASELINE, HM_HF_GROUPED),
    ("grouping_block_major", BM_BF, BM_BF_GROUPED),
    ("grouped_codesign", BASELINE, BM_BF_GROUPED),
    ("layout_grouped", HM_HF_GROUPED, BM_BF_GROUPED),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trials", type=Path, nargs="+")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    parser.add_argument("--minimum-effect", type=float, default=0.05)
    parser.add_argument("--minimum-win-rate", type=float, default=0.80)
    parser.add_argument("--minimum-launch-wins", type=int, default=2)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_median_ci(
    values: list[float], samples: int, seed: int
) -> tuple[float, float]:
    generator = random.Random(seed)
    medians = [
        statistics.median(generator.choice(values) for _ in values)
        for _ in range(samples)
    ]
    return percentile(medians, 0.025), percentile(medians, 0.975)


def compare(
    pairs: list[dict[str, float]],
    baseline: str,
    candidate: str,
    p95: dict[str, float],
    args: argparse.Namespace,
    seed: int,
) -> dict[str, object]:
    speedups = [pair[baseline] / pair[candidate] for pair in pairs]
    speedup = statistics.median(speedups)
    ci_low, ci_high = bootstrap_median_ci(
        speedups, args.bootstrap_samples, seed
    )
    win_rate = sum(value > 1.0 for value in speedups) / len(speedups)
    decision = "tie"
    if (
        speedup >= 1.0 + args.minimum_effect
        and win_rate >= args.minimum_win_rate
        and ci_low > 1.0
        and p95[candidate] <= p95[baseline]
    ):
        decision = candidate
    elif (
        speedup <= 1.0 / (1.0 + args.minimum_effect)
        and win_rate <= 1.0 - args.minimum_win_rate
        and ci_high < 1.0
        and p95[baseline] <= p95[candidate]
    ):
        decision = baseline
    return {
        "speedup": speedup,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "win_rate": win_rate,
        "decision": decision,
    }


def truthy(value: object) -> bool:
    return str(value).lower() == "true"


def analyze(args: argparse.Namespace) -> list[dict[str, object]]:
    trials: list[dict[str, str]] = []
    for path in args.trials:
        with path.open() as stream:
            trials.extend(csv.DictReader(stream))
    if not trials:
        raise RuntimeError("trial CSV is empty")
    if not all(truthy(row["correct"]) for row in trials):
        raise RuntimeError("at least one trial failed correctness")
    if set(row["candidate"] for row in trials) != set(CANDIDATES):
        raise RuntimeError("trial CSV does not contain the grouped GQA candidate set")

    workload_fields = (
        "case_family",
        "case_name",
        "shape_family",
        "shape",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "batch_size",
        "seq_len",
        "block_size",
    )
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in trials:
        groups[tuple(row[field] for field in workload_fields)].append(row)

    summaries: list[dict[str, object]] = []
    for group_index, (key, rows) in enumerate(sorted(groups.items())):
        by_candidate: dict[str, list[float]] = defaultdict(list)
        paired: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
        by_launch: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            elapsed = float(row["elapsed_ms"])
            by_candidate[row["candidate"]].append(elapsed)
            paired[(row["process_launch"], row["data_seed"], row["round"])][
                row["candidate"]
            ] = elapsed
            by_launch[row["process_launch"]].append(row)
        if any(set(pair) != set(CANDIDATES) for pair in paired.values()):
            raise RuntimeError(f"incomplete candidate quartet for {key}")

        pair_rows = list(paired.values())
        medians = {
            candidate: statistics.median(by_candidate[candidate])
            for candidate in CANDIDATES
        }
        p95 = {
            candidate: percentile(by_candidate[candidate], 0.95)
            for candidate in CANDIDATES
        }
        row_out: dict[str, object] = dict(zip(workload_fields, key))
        first = rows[0]
        for field in (
            "target_kv_mib",
            "batch_semantics",
            "kv_bytes_per_sequence",
            "kv_bytes_per_call",
            "blocks_per_sequence",
            "blocks_per_call",
            "alignment_bytes",
            "order_policy",
        ):
            row_out[field] = first[field]
        row_out["gqa_ratio"] = int(first["num_q_heads"]) // int(
            first["num_kv_heads"]
        )
        row_out["independent_launches"] = len(by_launch)
        row_out["paired_quartets"] = len(pair_rows)
        for candidate in CANDIDATES:
            row_out[f"{candidate}_median_ms"] = medians[candidate]
            row_out[f"{candidate}_p95_ms"] = p95[candidate]

        results: dict[str, dict[str, object]] = {}
        for comparison_index, (label, baseline, candidate) in enumerate(COMPARISONS):
            result = compare(
                pair_rows,
                baseline,
                candidate,
                p95,
                args,
                args.bootstrap_seed + group_index * 100 + comparison_index,
            )
            launch_results = []
            for launch_index, launch in enumerate(sorted(by_launch)):
                launch_rows = by_launch[launch]
                launch_pairs: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
                launch_values: dict[str, list[float]] = defaultdict(list)
                for launch_row in launch_rows:
                    elapsed = float(launch_row["elapsed_ms"])
                    launch_pairs[(launch_row["data_seed"], launch_row["round"])][
                        launch_row["candidate"]
                    ] = elapsed
                    launch_values[launch_row["candidate"]].append(elapsed)
                launch_p95 = {
                    name: percentile(launch_values[name], 0.95)
                    for name in CANDIDATES
                }
                launch_results.append(
                    compare(
                        list(launch_pairs.values()),
                        baseline,
                        candidate,
                        launch_p95,
                        args,
                        args.bootstrap_seed
                        + 1_000_000
                        + group_index * 100
                        + comparison_index * 10
                        + launch_index,
                    )
                )
            candidate_wins = sum(
                item["decision"] == candidate for item in launch_results
            )
            baseline_wins = sum(
                item["decision"] == baseline for item in launch_results
            )
            result["launch_candidate_wins"] = candidate_wins
            result["launch_baseline_wins"] = baseline_wins
            result["repeatable_candidate"] = (
                result["decision"] == candidate
                and candidate_wins >= args.minimum_launch_wins
            )
            result["repeatable_baseline"] = (
                result["decision"] == baseline
                and baseline_wins >= args.minimum_launch_wins
            )
            result["launch_speedups"] = ";".join(
                f"{float(item['speedup']):.6f}" for item in launch_results
            )
            results[label] = result
            for name, value in result.items():
                row_out[f"{label}_{name}"] = value

        eligible = [BASELINE]
        baseline_labels = {
            BM_BF: "existing_codesign",
            HM_HF_GROUPED: "grouping_head_major",
            BM_BF_GROUPED: "grouped_codesign",
        }
        for candidate, label in baseline_labels.items():
            if truthy(results[label]["repeatable_candidate"]):
                eligible.append(candidate)
        row_out["recommendation"] = min(eligible, key=medians.get)
        summaries.append(row_out)
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def median_speedup(rows: list[dict[str, object]], label: str) -> float:
    return statistics.median(float(row[f"{label}_speedup"]) for row in rows)


def repeatable_wins(rows: list[dict[str, object]], label: str) -> int:
    return sum(truthy(row[f"{label}_repeatable_candidate"]) for row in rows)


def write_report(
    path: Path, rows: list[dict[str, object]], args: argparse.Namespace
) -> None:
    recommendations = Counter(str(row["recommendation"]) for row in rows)
    ratios = sorted({int(row["gqa_ratio"]) for row in rows})
    sequence_lengths = sorted({int(row["seq_len"]) for row in rows})
    payloads = sorted({float(row["target_kv_mib"]) for row in rows})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        stream.write("# Grouped-Query GQA Decode Summary\n\n")
        stream.write("## Experiment\n\n")
        stream.write(
            "- Baseline: head-major/head-first, one query head at a time.\n"
        )
        stream.write(
            "- Grouped mode reuses each K vector and V vector across query heads "
            "sharing one KV head.\n"
        )
        stream.write(
            "- Four candidates isolate BM/BF alone, grouping alone, and their "
            "combination.\n"
        )
        stream.write(
            f"- Strict win: >={args.minimum_effect:.0%} paired median effect, "
            f">={args.minimum_win_rate:.0%} pair wins, 95% CI excluding one, "
            "and non-regressing p95.\n"
        )
        stream.write(
            f"- Repeatability requires >={args.minimum_launch_wins} independent "
            "launch wins.\n\n"
        )
        stream.write("## Main Result\n\n")
        stream.write(f"- Workloads: `{len(rows)}`\n")
        for label, description in (
            ("existing_codesign", "Existing BM/BF versus baseline"),
            ("grouping_head_major", "Grouping under HM/HF versus baseline"),
            ("grouping_block_major", "Grouping added under BM/BF"),
            ("grouped_codesign", "Grouped BM/BF versus baseline"),
        ):
            stream.write(
                f"- {description}: median `{median_speedup(rows, label):.3f}x`, "
                f"repeatable wins `{repeatable_wins(rows, label)}/{len(rows)}`\n"
            )
        stream.write("\n### Recommendations\n\n")
        for candidate in CANDIDATES:
            stream.write(
                f"- `{candidate}`: `{recommendations[candidate]}` workloads\n"
            )

        stream.write("\n## By GQA Ratio\n\n")
        stream.write(
            "| ratio | workloads | existing BM/BF | grouped HM/HF | grouped BM/BF | "
            "grouped BM/BF wins |\n"
        )
        stream.write("| ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for ratio in ratios:
            group = [row for row in rows if int(row["gqa_ratio"]) == ratio]
            stream.write(
                f"| {ratio} | {len(group)} | "
                f"{median_speedup(group, 'existing_codesign'):.3f}x | "
                f"{median_speedup(group, 'grouping_head_major'):.3f}x | "
                f"{median_speedup(group, 'grouped_codesign'):.3f}x | "
                f"{repeatable_wins(group, 'grouped_codesign')}/{len(group)} |\n"
            )

        stream.write("\n## By Sequence Length\n\n")
        stream.write(
            "| sequence | workloads | existing BM/BF | grouped HM/HF | "
            "grouped BM/BF | grouped BM/BF wins |\n"
        )
        stream.write("| ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for sequence in sequence_lengths:
            group = [row for row in rows if int(row["seq_len"]) == sequence]
            stream.write(
                f"| {sequence} | {len(group)} | "
                f"{median_speedup(group, 'existing_codesign'):.3f}x | "
                f"{median_speedup(group, 'grouping_head_major'):.3f}x | "
                f"{median_speedup(group, 'grouped_codesign'):.3f}x | "
                f"{repeatable_wins(group, 'grouped_codesign')}/{len(group)} |\n"
            )

        stream.write("\n## By Logical K+V Payload\n\n")
        stream.write(
            "| payload MiB | workloads | existing BM/BF | grouped HM/HF | "
            "grouped BM/BF | grouped BM/BF wins |\n"
        )
        stream.write("| ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for payload in payloads:
            group = [
                row for row in rows if float(row["target_kv_mib"]) == payload
            ]
            stream.write(
                f"| {payload:g} | {len(group)} | "
                f"{median_speedup(group, 'existing_codesign'):.3f}x | "
                f"{median_speedup(group, 'grouping_head_major'):.3f}x | "
                f"{median_speedup(group, 'grouped_codesign'):.3f}x | "
                f"{repeatable_wins(group, 'grouped_codesign')}/{len(group)} |\n"
            )

        stream.write("\n## Guardrail\n\n")
        stream.write(
            "This is a single-threaded synthetic fused-decode mechanism benchmark. "
            "It does not yet modify production SlabPool, use fragmented block "
            "tables, or test OpenMP scaling. Logical K+V payload is not measured "
            "DRAM traffic.\n"
        )


def main() -> None:
    args = arguments()
    rows = analyze(args)
    write_csv(args.summary, rows)
    write_report(args.report, rows, args)
    print(f"Wrote {len(rows)} workload summaries to {args.summary}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
