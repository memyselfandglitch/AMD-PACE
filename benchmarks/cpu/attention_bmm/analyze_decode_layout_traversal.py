#!/usr/bin/env python3
"""Analyze paired KV-layout and decode-traversal trials."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


CANDIDATES = (
    "head_major_head_first",
    "block_major_head_first",
    "head_major_block_first",
    "block_major_block_first",
)

COMPARISONS = (
    ("layout_only_head_first", CANDIDATES[0], CANDIDATES[1]),
    ("traversal_only_head_major", CANDIDATES[0], CANDIDATES[2]),
    ("co_designed_vs_current", CANDIDATES[0], CANDIDATES[3]),
    ("traversal_under_block_major", CANDIDATES[1], CANDIDATES[3]),
    ("layout_under_block_first", CANDIDATES[2], CANDIDATES[3]),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trials", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260816)
    parser.add_argument("--minimum-effect", type=float, default=0.05)
    parser.add_argument("--minimum-win-rate", type=float, default=0.80)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_median_ci(
    values: list[float], samples: int, seed: int
) -> tuple[float, float]:
    generator = random.Random(seed)
    medians = []
    for _ in range(samples):
        medians.append(
            statistics.median(generator.choice(values) for _ in values)
        )
    return percentile(medians, 0.025), percentile(medians, 0.975)


def compare(
    pairs: list[dict[str, float]],
    baseline: str,
    candidate: str,
    p95: dict[str, float],
    args: argparse.Namespace,
    seed: int,
) -> dict[str, float | str]:
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


def analyze(args: argparse.Namespace) -> list[dict[str, object]]:
    with args.trials.open() as stream:
        trials = list(csv.DictReader(stream))
    if not trials:
        raise RuntimeError("trial CSV is empty")
    if not all(row["correct"] == "true" for row in trials):
        raise RuntimeError("at least one trial failed correctness")

    workload_fields = (
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
    expected = set(CANDIDATES)
    for group_index, (key, rows) in enumerate(sorted(groups.items())):
        by_candidate: dict[str, list[float]] = defaultdict(list)
        paired: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
        for row in rows:
            elapsed = float(row["elapsed_ms"])
            by_candidate[row["candidate"]].append(elapsed)
            paired[(row["data_seed"], row["round"])][row["candidate"]] = elapsed
        if any(set(pair) != expected for pair in paired.values()):
            raise RuntimeError(f"incomplete randomized quadruple for {key}")

        pair_rows = list(paired.values())
        medians = {
            name: statistics.median(by_candidate[name]) for name in CANDIDATES
        }
        p95 = {
            name: percentile(by_candidate[name], 0.95) for name in CANDIDATES
        }
        results = {}
        for comparison_index, (label, baseline, candidate) in enumerate(COMPARISONS):
            results[label] = compare(
                pair_rows,
                baseline,
                candidate,
                p95,
                args,
                args.bootstrap_seed + group_index * len(COMPARISONS) + comparison_index,
            )

        strict_candidates = []
        for name in CANDIDATES[1:]:
            result = compare(
                pair_rows,
                CANDIDATES[0],
                name,
                p95,
                args,
                args.bootstrap_seed + 100000 + group_index * 3 + CANDIDATES.index(name),
            )
            if result["decision"] == name:
                strict_candidates.append(name)
        recommendation = CANDIDATES[0]
        if strict_candidates:
            recommendation = min(strict_candidates, key=medians.get)

        row: dict[str, object] = dict(zip(workload_fields, key))
        row["paired_quadruples"] = len(pair_rows)
        for name in CANDIDATES:
            row[f"{name}_median_ms"] = medians[name]
            row[f"{name}_p95_ms"] = p95[name]
        for label, _, _ in COMPARISONS:
            result = results[label]
            row[f"{label}_speedup"] = result["speedup"]
            row[f"{label}_ci_low"] = result["ci_low"]
            row[f"{label}_ci_high"] = result["ci_high"]
            row[f"{label}_win_rate"] = result["win_rate"]
            row[f"{label}_decision"] = result["decision"]
        row["recommendation"] = recommendation
        summaries.append(row)
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path, args: argparse.Namespace, rows: list[dict[str, object]]
) -> None:
    co_design_wins = sum(
        row["co_designed_vs_current_decision"] == "block_major_block_first"
        for row in rows
    )
    current_wins = sum(
        row["co_designed_vs_current_decision"] == "head_major_head_first"
        for row in rows
    )
    recommendations = defaultdict(int)
    for row in rows:
        recommendations[str(row["recommendation"])] += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        stream.write("# Decode KV Layout and Traversal Summary\n\n")
        stream.write("## Experiment\n\n")
        stream.write("- Current baseline: head-major storage plus head-first traversal.\n")
        stream.write("- Layout-only control: block-major storage plus head-first traversal.\n")
        stream.write("- Traversal-only control: head-major storage plus block-first traversal.\n")
        stream.write("- Co-designed candidate: block-major storage plus block-first traversal.\n")
        stream.write(
            "- Each candidate performs the same fused BF16 decode: QK dot products, "
            "blockwise online softmax, and weighted V accumulation.\n"
        )
        stream.write(
            "- Candidate order is randomized within paired repeats and all outputs "
            "must pass correctness before timing.\n"
        )
        stream.write(
            f"- Strict winner: >={args.minimum_effect:.0%} paired median effect, "
            f">={args.minimum_win_rate:.0%} pair wins, 95% CI excluding one, "
            "and non-regressing p95.\n\n"
        )
        stream.write("## Main Result\n\n")
        stream.write(f"- Workloads: `{len(rows)}`\n")
        stream.write(
            "- Strict block-major/block-first wins over current baseline: "
            f"`{co_design_wins}/{len(rows)}`\n"
        )
        stream.write(
            "- Strict current-baseline wins over block-major/block-first: "
            f"`{current_wins}/{len(rows)}`\n"
        )
        for candidate in CANDIDATES:
            stream.write(
                f"- Recommended `{candidate}`: `{recommendations[candidate]}` workloads\n"
            )
        stream.write("\n## Workloads\n\n")
        stream.write(
            "| shape | batch | seq | current ms | layout-only | traversal-only | "
            "co-designed | recommendation |\n"
        )
        stream.write(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
        )
        for row in rows:
            stream.write(
                f"| {row['shape']} | {row['batch_size']} | {row['seq_len']} | "
                f"{float(row['head_major_head_first_median_ms']):.4f} | "
                f"{float(row['layout_only_head_first_speedup']):.3f}x "
                f"[{row['layout_only_head_first_decision']}] | "
                f"{float(row['traversal_only_head_major_speedup']):.3f}x "
                f"[{row['traversal_only_head_major_decision']}] | "
                f"{float(row['co_designed_vs_current_speedup']):.3f}x "
                f"[{row['co_designed_vs_current_decision']}] | "
                f"{row['recommendation']} |\n"
            )
        stream.write("\n## Scope Guardrail\n\n")
        stream.write(
            "This is a single-threaded decode-mechanism benchmark. It mirrors PACE's "
            "GQA blockwise online-softmax dataflow and AVX-512 BF16 arithmetic, but it "
            "does not yet modify SlabPool, exercise its allocator's non-contiguous "
            "physical block mapping, use Split-K, or include OpenMP scheduling. A "
            "production prototype is justified only if a repeatable co-design signal "
            "survives this controlled test.\n"
        )


def main() -> None:
    args = arguments()
    rows = analyze(args)
    write_csv(args.summary, rows)
    write_report(args.report, args, rows)
    print(f"Wrote {len(rows)} workload summaries to {args.summary}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
