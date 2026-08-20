#!/usr/bin/env python3
"""Inspect decode layout/traversal summary CSVs without external packages."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


BASELINE = "head_major_head_first"
CANDIDATE = "block_major_block_first"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--family")
    parser.add_argument("--payload", type=float)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--kv-heads", type=int)
    parser.add_argument("--decision", choices=("win", "tie", "baseline"))
    parser.add_argument(
        "--view",
        choices=("overview", "workloads", "launches"),
        default="overview",
    )
    parser.add_argument("--csv", action="store_true")
    return parser.parse_args()


def truthy(value: object) -> bool:
    return str(value).lower() == "true"


def final_decision(row: dict[str, str]) -> str:
    if truthy(row["co_designed_vs_current_repeatable_candidate"]):
        return "BM/BF"
    if truthy(row["co_designed_vs_current_repeatable_baseline"]):
        return "HM/HF"
    return "tie"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"summary CSV is empty: {path}")
    return rows


def filter_rows(
    rows: list[dict[str, str]], args: argparse.Namespace
) -> list[dict[str, str]]:
    result = []
    for row in rows:
        if args.family and row["case_family"] != args.family:
            continue
        if args.payload is not None and float(row["target_kv_mib"]) != args.payload:
            continue
        if args.batch is not None and int(row["batch_size"]) != args.batch:
            continue
        if args.kv_heads is not None and int(row["num_kv_heads"]) != args.kv_heads:
            continue
        decision = final_decision(row)
        requested = {"win": "BM/BF", "tie": "tie", "baseline": "HM/HF"}
        if args.decision and decision != requested[args.decision]:
            continue
        result.append(row)
    return result


def table(headers: list[str], rows: list[list[object]], csv_output: bool) -> None:
    text_rows = [[str(value) for value in row] for row in rows]
    if csv_output:
        writer = csv.writer(sys.stdout)
        writer.writerow(headers)
        writer.writerows(text_rows)
        return
    widths = [len(header) for header in headers]
    for row in text_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in text_rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def overview(rows: list[dict[str, str]], csv_output: bool) -> None:
    groups: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[float(row["target_kv_mib"])].append(row)
    output = []
    for payload, group in sorted(groups.items()):
        speedups = [float(row["co_designed_vs_current_speedup"]) for row in group]
        median_speedup = statistics.median(speedups)
        output.append(
            [
                f"{payload:g}",
                len(group),
                sum(final_decision(row) == "BM/BF" for row in group),
                f"{median_speedup:.3f}x",
                f"{(1 - 1 / median_speedup) * 100:.1f}%",
                f"{min(speedups):.3f}-{max(speedups):.3f}x",
            ]
        )
    table(
        [
            "payload_MiB",
            "workloads",
            "repeat_wins",
            "median_speedup",
            "latency_reduction",
            "speedup_range",
        ],
        output,
        csv_output,
    )
    if not csv_output:
        counts = Counter(final_decision(row) for row in rows)
        print(
            f"\nSelected workloads: {len(rows)}; repeatable BM/BF wins: "
            f"{counts['BM/BF']}; repeatable HM/HF wins: {counts['HM/HF']}; "
            f"ties: {counts['tie']}"
        )


def workloads(rows: list[dict[str, str]], csv_output: bool) -> None:
    output = []
    for row in sorted(
        rows,
        key=lambda item: (
            float(item["target_kv_mib"]),
            int(item["batch_size"]),
            int(item["num_kv_heads"]),
            int(item["seq_len"]),
        ),
    ):
        output.append(
            [
                row["case_family"],
                f"{float(row['target_kv_mib']):g}",
                row["batch_size"],
                row["num_kv_heads"],
                row["seq_len"],
                row["blocks_per_sequence"],
                f"{float(row[f'{BASELINE}_median_ms']):.6f}",
                f"{float(row[f'{CANDIDATE}_median_ms']):.6f}",
                f"{float(row['co_designed_vs_current_speedup']):.3f}x",
                f"{float(row['co_designed_vs_current_ci_low']):.3f}-"
                f"{float(row['co_designed_vs_current_ci_high']):.3f}",
                f"{float(row['co_designed_vs_current_win_rate']) * 100:.1f}%",
                final_decision(row),
            ]
        )
    table(
        [
            "family",
            "MiB",
            "B",
            "KVH",
            "seq",
            "blocks",
            "HM/HF_ms",
            "BM/BF_ms",
            "speedup",
            "95%_CI",
            "pair_wins",
            "decision",
        ],
        output,
        csv_output,
    )


def launches(rows: list[dict[str, str]], csv_output: bool) -> None:
    output = []
    for row in sorted(
        rows,
        key=lambda item: (
            float(item["target_kv_mib"]),
            int(item["batch_size"]),
            int(item["num_kv_heads"]),
        ),
    ):
        speedups = row["co_designed_vs_current_launch_speedups"].split(";")
        decisions = row["co_designed_vs_current_launch_decisions"].split(";")
        output.append(
            [
                f"{float(row['target_kv_mib']):g}",
                row["batch_size"],
                row["num_kv_heads"],
                row["seq_len"],
                " / ".join(f"{float(value):.3f}x" for value in speedups),
                " / ".join(decisions),
                final_decision(row),
            ]
        )
    table(
        ["MiB", "B", "KVH", "seq", "launch_speedups", "launch_decisions", "final"],
        output,
        csv_output,
    )


def main() -> None:
    args = arguments()
    rows = filter_rows(read_rows(args.summary), args)
    if not rows:
        raise RuntimeError("no rows match the requested filters")
    if args.view == "overview":
        overview(rows, args.csv)
    elif args.view == "workloads":
        workloads(rows, args.csv)
    else:
        launches(rows, args.csv)


if __name__ == "__main__":
    main()
