#!/usr/bin/env python3
"""Analyze SlabPool block-size sweeps and derive a simple tuning rule."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


INT_FIELDS = {
    "block_size",
    "batch_size",
    "seq_len",
    "query_len",
    "active_batch_size",
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
    "tokens",
    "total_blocks",
    "heuristic_block_size",
    "l2_bytes",
    "bytes_per_token",
    "block_bytes",
    "kv_history_bytes",
}

FLOAT_FIELDS = {
    "exit_fraction",
    "mean_ms",
    "median_ms",
    "p95_ms",
    "min_ms",
    "max_ms",
    "tokens_per_second",
}

BLOCK_SIZES = (16, 32, 64, 128, 256)


def read_csvs(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for field in INT_FIELDS:
                    if row.get(field) not in (None, ""):
                        row[field] = int(float(row[field]))
                for field in FLOAT_FIELDS:
                    if row.get(field) not in (None, ""):
                        row[field] = float(row[field])
                row["_source_csv"] = path
                rows.append(row)
    return rows


def heuristic_block_size(num_kv_heads: int, head_dim: int, l2_bytes: int) -> int:
    if l2_bytes <= 0:
        l2_bytes = 1024 * 1024
    bytes_per_token = 2 * num_kv_heads * head_dim * 2
    target = l2_bytes // 4
    for block_size in (256, 128, 64, 32):
        if block_size * bytes_per_token <= target:
            return block_size
    return 32


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def pct(count: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{100.0 * count / total:.1f}%"


def fmt_speed(value: float) -> str:
    return f"{value:.2f}x"


def seq_bucket(seq_len: int) -> str:
    if seq_len <= 512:
        return "short"
    if seq_len <= 2048:
        return "medium"
    return "long"


def workload_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["phase"],
        row["layout"],
        row["batch_size"],
        row["seq_len"],
        row["shape"],
        row["exit_fraction"],
    )


def grouping_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (row["phase"], row["shape"], seq_bucket(row["seq_len"]))


def shape_heuristic(row: dict[str, Any], l2_bytes: int) -> int:
    if row.get("heuristic_block_size"):
        return int(row["heuristic_block_size"])
    return heuristic_block_size(row["num_kv_heads"], row["head_dim"], l2_bytes)


def build_workloads(rows: list[dict[str, Any]], layout: str) -> dict[tuple[Any, ...], dict[int, dict[str, Any]]]:
    groups: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["layout"] != layout:
            continue
        groups[workload_key(row)][row["block_size"]] = row
    return groups


def best_row(rows_by_block: dict[int, dict[str, Any]], metric: str) -> dict[str, Any]:
    return min(rows_by_block.values(), key=lambda row: row[metric])


def derive_decisions(groups: dict[tuple[Any, ...], dict[int, dict[str, Any]]], metric: str, l2_bytes: int) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for key, rows_by_block in sorted(groups.items()):
        sample = next(iter(rows_by_block.values()))
        baseline_block = shape_heuristic(sample, l2_bytes)
        if baseline_block not in rows_by_block:
            continue

        baseline = rows_by_block[baseline_block]
        best = best_row(rows_by_block, metric)
        block16 = rows_by_block.get(16)
        speedup = baseline[metric] / best[metric]
        p95_speedup = baseline["p95_ms"] / best["p95_ms"]

        decisions.append(
            {
                "phase": sample["phase"],
                "layout": sample["layout"],
                "batch_size": sample["batch_size"],
                "seq_len": sample["seq_len"],
                "seq_bucket": seq_bucket(sample["seq_len"]),
                "shape": sample["shape"],
                "num_q_heads": sample["num_q_heads"],
                "num_kv_heads": sample["num_kv_heads"],
                "head_dim": sample["head_dim"],
                "exit_fraction": sample["exit_fraction"],
                "baseline_block_size": baseline_block,
                "best_block_size": best["block_size"],
                "baseline_ms": baseline[metric],
                "best_ms": best[metric],
                "speedup": speedup,
                "p95_speedup": p95_speedup,
                "win_ge_5pct": speedup >= 1.05,
                "win_ge_10pct": speedup >= 1.10,
                "p95_not_worse": p95_speedup >= 1.0,
                "block16_ms": block16[metric] if block16 else "",
                "best_vs_block16_speedup": (block16[metric] / best[metric]) if block16 else "",
            }
        )
    return decisions


def normalized_regret(rows_by_block: dict[int, dict[str, Any]], chosen_block: int, metric: str) -> float:
    if chosen_block not in rows_by_block:
        return float("inf")
    best = best_row(rows_by_block, metric)
    return rows_by_block[chosen_block][metric] / best[metric]


def choose_block_for_group(workloads: list[dict[int, dict[str, Any]]], metric: str) -> int:
    scores: dict[int, list[float]] = defaultdict(list)
    for rows_by_block in workloads:
        for block_size in rows_by_block:
            scores[block_size].append(normalized_regret(rows_by_block, block_size, metric))
    return min(scores, key=lambda block_size: median(scores[block_size]))


def build_rule(groups: dict[tuple[Any, ...], dict[int, dict[str, Any]]], metric: str) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], int]]:
    grouped: dict[tuple[str, str, str], list[dict[int, dict[str, Any]]]] = defaultdict(list)
    for rows_by_block in groups.values():
        sample = next(iter(rows_by_block.values()))
        grouped[grouping_key(sample)].append(rows_by_block)

    rules: list[dict[str, Any]] = []
    rule_map: dict[tuple[str, str, str], int] = {}
    for key, workloads in sorted(grouped.items()):
        block = choose_block_for_group(workloads, metric)
        regrets = [normalized_regret(workload, block, metric) for workload in workloads]
        rules.append(
            {
                "phase": key[0],
                "shape": key[1],
                "seq_bucket": key[2],
                "chosen_block_size": block,
                "training_workloads": len(workloads),
                "median_regret_vs_oracle": median(regrets),
                "max_regret_vs_oracle": max(regrets),
            }
        )
        rule_map[key] = block
    return rules, rule_map


def fallback_rule_lookup(rule_map: dict[tuple[str, str, str], int], train_groups: dict[tuple[Any, ...], dict[int, dict[str, Any]]], sample: dict[str, Any], metric: str) -> int:
    key = grouping_key(sample)
    if key in rule_map:
        return rule_map[key]

    phase_shape = [
        workload
        for workload in train_groups.values()
        if next(iter(workload.values()))["phase"] == sample["phase"]
        and next(iter(workload.values()))["shape"] == sample["shape"]
    ]
    if phase_shape:
        return choose_block_for_group(phase_shape, metric)

    phase_only = [
        workload
        for workload in train_groups.values()
        if next(iter(workload.values()))["phase"] == sample["phase"]
    ]
    if phase_only:
        return choose_block_for_group(phase_only, metric)

    return choose_block_for_group(list(train_groups.values()), metric)


def leave_one_seq_len_out(groups: dict[tuple[Any, ...], dict[int, dict[str, Any]]], metric: str, l2_bytes: int) -> list[dict[str, Any]]:
    seq_lens = sorted({next(iter(workload.values()))["seq_len"] for workload in groups.values()})
    validations: list[dict[str, Any]] = []

    for held_seq in seq_lens:
        train = {
            key: workload
            for key, workload in groups.items()
            if next(iter(workload.values()))["seq_len"] != held_seq
        }
        test = {
            key: workload
            for key, workload in groups.items()
            if next(iter(workload.values()))["seq_len"] == held_seq
        }
        _, rule_map = build_rule(train, metric)

        for rows_by_block in test.values():
            sample = next(iter(rows_by_block.values()))
            baseline_block = shape_heuristic(sample, l2_bytes)
            if baseline_block not in rows_by_block:
                continue
            chosen_block = fallback_rule_lookup(rule_map, train, sample, metric)
            if chosen_block not in rows_by_block:
                continue
            best = best_row(rows_by_block, metric)
            baseline = rows_by_block[baseline_block]
            chosen = rows_by_block[chosen_block]
            validations.append(
                {
                    "heldout_seq_len": held_seq,
                    "phase": sample["phase"],
                    "shape": sample["shape"],
                    "batch_size": sample["batch_size"],
                    "exit_fraction": sample["exit_fraction"],
                    "baseline_block_size": baseline_block,
                    "chosen_block_size": chosen_block,
                    "oracle_block_size": best["block_size"],
                    "chosen_vs_baseline_speedup": baseline[metric] / chosen[metric],
                    "oracle_vs_baseline_speedup": baseline[metric] / best[metric],
                    "chosen_regret_vs_oracle": chosen[metric] / best[metric],
                }
            )
    return validations


def monotonic_larger_is_better(groups: dict[tuple[Any, ...], dict[int, dict[str, Any]]], metric: str) -> tuple[int, int]:
    total = 0
    monotonic = 0
    for rows_by_block in groups.values():
        if not all(block in rows_by_block for block in BLOCK_SIZES):
            continue
        values = [rows_by_block[block][metric] for block in BLOCK_SIZES]
        total += 1
        if all(values[i] >= values[i + 1] for i in range(len(values) - 1)):
            monotonic += 1
    return monotonic, total


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def summarize(args, rows: list[dict[str, Any]], groups: dict[tuple[Any, ...], dict[int, dict[str, Any]]], decisions: list[dict[str, Any]], rules: list[dict[str, Any]], validations: list[dict[str, Any]]) -> str:
    speedups = [row["speedup"] for row in decisions]
    p95_speedups = [row["p95_speedup"] for row in decisions]
    ge5 = sum(row["win_ge_5pct"] for row in decisions)
    ge10 = sum(row["win_ge_10pct"] for row in decisions)
    ge5_p95_ok = sum(row["win_ge_5pct"] and row["p95_not_worse"] for row in decisions)
    monotonic, monotonic_total = monotonic_larger_is_better(groups, args.metric)

    best_blocks = Counter(row["best_block_size"] for row in decisions)
    by_phase: dict[str, list[float]] = defaultdict(list)
    by_seq: dict[int, list[float]] = defaultdict(list)
    by_shape: dict[str, list[float]] = defaultdict(list)
    for row in decisions:
        by_phase[row["phase"]].append(row["speedup"])
        by_seq[row["seq_len"]].append(row["speedup"])
        by_shape[row["shape"]].append(row["speedup"])

    lines: list[str] = []
    lines.append("# SlabPool Block-Size Autotuning Summary")
    lines.append("")
    lines.append(f"- CSV rows: `{len(rows)}`")
    lines.append(f"- Workloads analyzed: `{len(decisions)}`")
    lines.append(f"- Layout held fixed: `{args.layout}`")
    lines.append(f"- Winner metric: `{args.metric}`")
    lines.append(f"- Assumed L2 bytes when missing from CSV: `{args.l2_bytes}`")
    lines.append("- Baseline: current PACE-style L2 heuristic block size for each shape")
    lines.append("")
    lines.append("## Main Result")
    lines.append("")
    lines.append(f"- Best empirical block size improves over the heuristic by >=5% in `{ge5}/{len(decisions)}` workloads ({pct(ge5, len(decisions))}).")
    lines.append(f"- Best empirical block size improves over the heuristic by >=10% in `{ge10}/{len(decisions)}` workloads ({pct(ge10, len(decisions))}).")
    lines.append(f"- >=5% wins with p95 not worse: `{ge5_p95_ok}/{len(decisions)}` workloads ({pct(ge5_p95_ok, len(decisions))}).")
    lines.append(f"- Median speedup over heuristic: `{fmt_speed(median(speedups))}`.")
    lines.append(f"- Max speedup over heuristic: `{fmt_speed(max(speedups))}`.")
    lines.append(f"- Median p95 speedup: `{fmt_speed(median(p95_speedups))}`.")
    lines.append(f"- Larger block size is monotonically better in only `{monotonic}/{monotonic_total}` workloads ({pct(monotonic, monotonic_total)}).")
    lines.append("")
    lines.append("## Best Block-Size Distribution")
    lines.append("")
    lines.append(markdown_table(["block_size", "workloads"], [[k, v] for k, v in best_blocks.most_common()]))
    lines.append("")
    lines.append("## Median Speedup By Phase")
    lines.append("")
    lines.append(markdown_table(["phase", "median_speedup"], [[k, fmt_speed(median(v))] for k, v in sorted(by_phase.items())]))
    lines.append("")
    lines.append("## Median Speedup By Sequence Length")
    lines.append("")
    lines.append(markdown_table(["seq_len", "median_speedup"], [[k, fmt_speed(median(v))] for k, v in sorted(by_seq.items())]))
    lines.append("")
    lines.append("## Median Speedup By Shape")
    lines.append("")
    lines.append(markdown_table(["shape", "median_speedup"], [[k, fmt_speed(median(v))] for k, v in sorted(by_shape.items())]))
    lines.append("")
    lines.append("## Learned Rule")
    lines.append("")
    lines.append("Rule features: `(phase, shape, sequence bucket)`, where buckets are `short <= 512`, `medium <= 2048`, and `long > 2048`.")
    lines.append("")
    lines.append(markdown_table(
        ["phase", "shape", "seq_bucket", "chosen_block_size", "training_workloads", "median_regret_vs_oracle"],
        [
            [
                row["phase"],
                row["shape"],
                row["seq_bucket"],
                row["chosen_block_size"],
                row["training_workloads"],
                f'{row["median_regret_vs_oracle"]:.3f}x',
            ]
            for row in rules
        ],
    ))
    lines.append("")

    if validations:
        val_speedups = [row["chosen_vs_baseline_speedup"] for row in validations]
        val_regrets = [row["chosen_regret_vs_oracle"] for row in validations]
        near_oracle = sum(row["chosen_regret_vs_oracle"] <= 1.05 for row in validations)
        lines.append("## Leave-One-Sequence-Length-Out Validation")
        lines.append("")
        lines.append(f"- Validation rows: `{len(validations)}`")
        lines.append(f"- Median rule speedup over heuristic: `{fmt_speed(median(val_speedups))}`")
        lines.append(f"- Rule within 5% of oracle best block: `{near_oracle}/{len(validations)}` workloads ({pct(near_oracle, len(validations))})")
        lines.append(f"- Median regret vs oracle: `{median(val_regrets):.3f}x`")
        lines.append("")

    lines.append("## Files")
    lines.append("")
    lines.append("- `decisions.csv`: per-workload baseline block, best block, speedup, and p95 comparison.")
    lines.append("- `blocksize_rule.csv`: learned lookup rule from `(phase, shape, seq_bucket)` to block size.")
    lines.append("- `validation.csv`: leave-one-sequence-length-out rule validation.")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", help="One or more block-size benchmark CSVs.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--layout", default="head_major")
    parser.add_argument("--metric", default="median_ms", choices=("median_ms", "mean_ms"))
    parser.add_argument("--l2-bytes", type=int, default=1024 * 1024)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_csvs(args.csv)
    groups = build_workloads(rows, args.layout)
    decisions = derive_decisions(groups, args.metric, args.l2_bytes)
    rules, _ = build_rule(groups, args.metric)
    validations = leave_one_seq_len_out(groups, args.metric, args.l2_bytes)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "decisions.csv", decisions)
    write_csv(out_dir / "blocksize_rule.csv", rules)
    write_csv(out_dir / "validation.csv", validations)

    summary = summarize(args, rows, groups, decisions, rules, validations)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
