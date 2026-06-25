#!/usr/bin/env python3
# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Summarize SlabPool layout/block-size benchmark CSVs.

The benchmark emits one row per experimental case. This script groups rows by
workload, compares only layout/block_size choices within each workload, and
reports which choice had the lowest latency.

Examples:
    python benchmarks/slab/analyze_slab_results.py \
      benchmarks/slab/results/slab_sweep_6959.csv

    python benchmarks/slab/analyze_slab_results.py \
      benchmarks/slab/results/slab_sweep_6959.csv \
      --metric median_ms \
      --out-dir benchmarks/slab/results/analysis_6959
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


NUMERIC_FIELDS = {
    "block_size": int,
    "batch_size": int,
    "seq_len": int,
    "num_q_heads": int,
    "num_kv_heads": int,
    "head_dim": int,
    "warmups": int,
    "repeats": int,
    "tokens": int,
    "total_blocks": int,
    "exit_fraction": float,
    "mean_ms": float,
    "median_ms": float,
    "p95_ms": float,
    "min_ms": float,
    "max_ms": float,
    "tokens_per_second": float,
}

WORKLOAD_KEYS = ("phase", "batch_size", "seq_len", "shape", "exit_fraction")
CHOICE_KEYS = ("layout", "block_size")


def parse_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            converted: dict[str, object] = dict(row)
            for field, caster in NUMERIC_FIELDS.items():
                if field in converted:
                    converted[field] = caster(converted[field])
            rows.append(converted)
    return rows


def l2_heuristic_block_size(num_kv_heads: int, head_dim: int, l2_kib: int) -> int:
    """Mirror PACE's largest-fitting block-size heuristic."""
    target_bytes = (l2_kib * 1024) // 4
    bytes_per_token = 2 * num_kv_heads * head_dim * 2
    for block_size in (256, 128, 64, 32):
        if block_size * bytes_per_token <= target_bytes:
            return block_size
    return 32


def group_by(rows: Iterable[dict[str, object]], keys: tuple[str, ...]):
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    rendered = []
    rendered.append("| " + " | ".join(headers) + " |")
    rendered.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        rendered.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(rendered)


def summarize(args: argparse.Namespace) -> tuple[str, list[dict[str, object]]]:
    rows = parse_rows(Path(args.csv_path))
    groups = group_by(rows, WORKLOAD_KEYS)

    winners: list[dict[str, object]] = []
    for key, candidates in groups.items():
        ordered = sorted(candidates, key=lambda row: float(row[args.metric]))
        best = ordered[0]
        worst = ordered[-1]
        heuristic_bs = l2_heuristic_block_size(
            int(best["num_kv_heads"]), int(best["head_dim"]), args.l2_kib
        )
        winners.append(
            {
                "phase": key[0],
                "batch_size": key[1],
                "seq_len": key[2],
                "shape": key[3],
                "exit_fraction": key[4],
                "best_layout": best["layout"],
                "best_block_size": best["block_size"],
                f"best_{args.metric}": best[args.metric],
                "worst_layout": worst["layout"],
                "worst_block_size": worst["block_size"],
                f"worst_{args.metric}": worst[args.metric],
                "speedup_vs_worst": float(worst[args.metric]) / float(best[args.metric]),
                "heuristic_block_size": heuristic_bs,
                "heuristic_matched_best": int(heuristic_bs == int(best["block_size"])),
            }
        )

    choice_wins = Counter(
        (row["best_layout"], row["best_block_size"]) for row in winners
    )
    layout_wins = Counter(row["best_layout"] for row in winners)
    block_wins = Counter(row["best_block_size"] for row in winners)
    phase_wins = Counter(
        (row["phase"], row["best_layout"], row["best_block_size"])
        for row in winners
    )
    shape_wins = Counter(
        (row["shape"], row["best_layout"], row["best_block_size"])
        for row in winners
    )

    heuristic_matches = sum(int(row["heuristic_matched_best"]) for row in winners)
    speedups = [float(row["speedup_vs_worst"]) for row in winners]

    top_deltas = sorted(winners, key=lambda row: float(row["speedup_vs_worst"]), reverse=True)[
        : args.top
    ]

    report: list[str] = []
    report.append("# SlabPool Sweep Summary")
    report.append("")
    report.append(f"- CSV: `{args.csv_path}`")
    report.append(f"- Rows: `{len(rows)}`")
    report.append(f"- Workloads compared: `{len(winners)}`")
    report.append(f"- Metric: `{args.metric}`")
    report.append(f"- Assumed L2/core: `{args.l2_kib} KiB`")
    report.append(
        f"- L2 heuristic matched empirical best block size: "
        f"`{heuristic_matches}/{len(winners)}`"
    )
    report.append(
        f"- Best-vs-worst speedup across layout/block choices: "
        f"avg `{sum(speedups) / len(speedups):.2f}x`, "
        f"max `{max(speedups):.2f}x`"
    )
    report.append("")

    report.append("## Wins By Layout")
    report.append("")
    report.append(markdown_table(["layout", "wins"], layout_wins.most_common()))
    report.append("")

    report.append("## Wins By Block Size")
    report.append("")
    report.append(markdown_table(["block_size", "wins"], block_wins.most_common()))
    report.append("")

    report.append("## Wins By Layout And Block Size")
    report.append("")
    report.append(
        markdown_table(
            ["layout", "block_size", "wins"],
            [[layout, block_size, wins] for (layout, block_size), wins in choice_wins.most_common()],
        )
    )
    report.append("")

    report.append("## Wins By Phase")
    report.append("")
    report.append(
        markdown_table(
            ["phase", "layout", "block_size", "wins"],
            [
                [phase, layout, block_size, wins]
                for (phase, layout, block_size), wins in phase_wins.most_common()
            ],
        )
    )
    report.append("")

    report.append("## Wins By Shape")
    report.append("")
    report.append(
        markdown_table(
            ["shape", "layout", "block_size", "wins"],
            [
                [shape, layout, block_size, wins]
                for (shape, layout, block_size), wins in shape_wins.most_common()
            ],
        )
    )
    report.append("")

    report.append(f"## Top {args.top} Largest Differences")
    report.append("")
    report.append(
        markdown_table(
            [
                "phase",
                "shape",
                "batch",
                "seq_len",
                "exit",
                "best",
                f"best_{args.metric}",
                "worst",
                f"worst_{args.metric}",
                "speedup",
            ],
            [
                [
                    row["phase"],
                    row["shape"],
                    row["batch_size"],
                    row["seq_len"],
                    row["exit_fraction"],
                    f"{row['best_layout']}:{row['best_block_size']}",
                    f"{float(row[f'best_{args.metric}']):.6g}",
                    f"{row['worst_layout']}:{row['worst_block_size']}",
                    f"{float(row[f'worst_{args.metric}']):.6g}",
                    f"{float(row['speedup_vs_worst']):.2f}x",
                ]
                for row in top_deltas
            ],
        )
    )
    report.append("")

    return "\n".join(report), winners


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Benchmark CSV to summarize.")
    parser.add_argument(
        "--metric",
        default="median_ms",
        choices=("mean_ms", "median_ms", "p95_ms", "min_ms"),
        help="Latency metric used to decide winners.",
    )
    parser.add_argument(
        "--l2-kib",
        type=int,
        default=1024,
        help="Per-core L2 cache size in KiB for heuristic comparison.",
    )
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument(
        "--out-dir",
        default="",
        help="Optional directory for summary.md and winners.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, winners = summarize(args)
    print(report)

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.md").write_text(report + "\n")
        write_csv(out_dir / "winners.csv", winners)


if __name__ == "__main__":
    main()
