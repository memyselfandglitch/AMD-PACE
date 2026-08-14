#!/usr/bin/env python3
"""Summarize paired tiled IKJ/KIJ BF16 BRGeMM trials."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trials", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260813)
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


def analyze(args: argparse.Namespace) -> list[dict[str, object]]:
    with args.trials.open() as stream:
        trials = list(csv.DictReader(stream))
    if not trials:
        raise RuntimeError("trial CSV is empty")
    if not all(row["correct"] == "true" for row in trials):
        raise RuntimeError("at least one trial failed correctness")

    workload_fields = ("head_dim", "query_len", "kv_len")
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in trials:
        groups[tuple(row[field] for field in workload_fields)].append(row)

    summaries: list[dict[str, object]] = []
    ordered_groups = sorted(
        groups.items(), key=lambda item: tuple(map(int, item[0]))
    )
    for index, (key, rows) in enumerate(ordered_groups):
        by_order: dict[str, list[float]] = defaultdict(list)
        pairs: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
        for row in rows:
            elapsed = float(row["elapsed_ms"])
            by_order[row["dataflow"]].append(elapsed)
            pairs[(row["data_seed"], row["round"])][row["dataflow"]] = elapsed
        if any(set(pair) != {"ikj", "kij"} for pair in pairs.values()):
            raise RuntimeError(f"incomplete randomized pair for workload {key}")

        speedups = [pair["ikj"] / pair["kij"] for pair in pairs.values()]
        speedup = statistics.median(speedups)
        ci_low, ci_high = bootstrap_median_ci(
            speedups, args.bootstrap_samples, args.bootstrap_seed + index
        )
        kij_win_rate = sum(value > 1.0 for value in speedups) / len(speedups)
        ikj_p95 = percentile(by_order["ikj"], 0.95)
        kij_p95 = percentile(by_order["kij"], 0.95)
        decision = "tie"
        if (
            speedup >= 1.0 + args.minimum_effect
            and kij_win_rate >= args.minimum_win_rate
            and ci_low > 1.0
            and kij_p95 <= ikj_p95
        ):
            decision = "kij"
        elif (
            speedup <= 1.0 / (1.0 + args.minimum_effect)
            and kij_win_rate <= 1.0 - args.minimum_win_rate
            and ci_high < 1.0
            and ikj_p95 <= kij_p95
        ):
            decision = "ikj"

        head_dim, query_len, kv_len = map(int, key)
        summaries.append(
            {
                "head_dim_class": f"hd{head_dim}",
                "head_dim": head_dim,
                "query_len": query_len,
                "query_tiles": query_len // 64,
                "kv_len": kv_len,
                "kv_blocks": kv_len // 64,
                "pairs": len(speedups),
                "ikj_median_ms": statistics.median(by_order["ikj"]),
                "ikj_p95_ms": ikj_p95,
                "kij_median_ms": statistics.median(by_order["kij"]),
                "kij_p95_ms": kij_p95,
                "kij_speedup": speedup,
                "speedup_ci_low": ci_low,
                "speedup_ci_high": ci_high,
                "kij_win_rate": kij_win_rate,
                "decision": decision,
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path, args: argparse.Namespace, rows: list[dict[str, object]]
) -> None:
    counts = {
        decision: sum(row["decision"] == decision for row in rows)
        for decision in ("ikj", "kij", "tie")
    }
    with path.open("w") as stream:
        stream.write("# Tiled BF16 BRGeMM Dataflow Comparison\n\n")
        stream.write("## Experiment\n\n")
        stream.write("- Candidate `IKJ`: query tile -> KV block -> dimension tile.\n")
        stream.write("- Candidate `KIJ`: KV block -> query tile -> dimension tile.\n")
        stream.write("- Query tile and KV block are fixed at `64`.\n")
        stream.write("- Both candidates use the same oneDNN BF16 BRGeMM ukernel.\n")
        stream.write("- Operands are pre-tiled identically outside the timed region.\n")
        stream.write("- Execution is single-threaded and candidate order is randomized.\n")
        stream.write(
            f"- Strict winner: >={args.minimum_effect:.0%} paired median effect, "
            f">={args.minimum_win_rate:.0%} pair wins, 95% CI excluding one, "
            "and non-regressing p95.\n\n"
        )
        stream.write("## Result\n\n")
        stream.write(f"- Workloads: `{len(rows)}`\n")
        stream.write(
            f"- Decisions: `IKJ={counts['ikj']}`, `KIJ={counts['kij']}`, "
            f"`tie={counts['tie']}`\n"
        )
        stream.write(
            "- Proceed to batched/OpenMP validation only if KIJ earns at least "
            "two strict wins.\n\n"
        )
        stream.write(
            "| head dim (N) | query len (M) | query tiles | KV len (K) | "
            "KV blocks | IKJ ms | KIJ ms | "
            "KIJ speedup (95% CI) | win rate | decision |\n"
        )
        stream.write(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
        )
        for row in rows:
            stream.write(
                f"| {row['head_dim']} | {row['query_len']} | "
                f"{row['query_tiles']} | {row['kv_len']} | "
                f"{row['kv_blocks']} | "
                f"{float(row['ikj_median_ms']):.4f} | "
                f"{float(row['kij_median_ms']):.4f} | "
                f"{float(row['kij_speedup']):.3f}x "
                f"({float(row['speedup_ci_low']):.3f}-"
                f"{float(row['speedup_ci_high']):.3f}) | "
                f"{float(row['kij_win_rate']):.1%} | {row['decision']} |\n"
            )
        stream.write("\n## Scope Guardrail\n\n")
        stream.write(
            "This isolates the P*V traversal of pre-tiled operands. It does not "
            "include QK^T, online softmax, K/V packing, physical SlabPool layout, "
            "GQA batching, or OpenMP scheduling. A KIJ win is evidence to build "
            "the next prototype, not yet evidence of faster complete PACE attention.\n"
        )
        stream.write(
            "Head dimensions 64 and 128 are per-head attention dimensions, not "
            "complete SLM and LLM model shapes by themselves.\n"
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
