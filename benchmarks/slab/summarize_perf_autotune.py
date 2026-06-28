#!/usr/bin/env python3
# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Summarize hardware-counter sweeps for SlabPool autotuning evidence."""

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
    "query_len": int,
    "active_batch_size": int,
    "num_q_heads": int,
    "num_kv_heads": int,
    "head_dim": int,
    "exit_fraction": float,
    "median_ms": float,
    "p95_ms": float,
    "mean_ms": float,
    "tokens_per_second": float,
    "cache_miss_rate": float,
    "cache-misses": float,
    "cache-references": float,
    "cycles": float,
    "instructions": float,
    "ipc": float,
    "cycles_per_token": float,
    "instructions_per_token": float,
    "cache_misses_per_token": float,
}

WORKLOAD_KEYS = (
    "phase",
    "batch_size",
    "seq_len",
    "shape",
    "exit_fraction",
)


def parse_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            converted: dict[str, object] = dict(row)
            for field, caster in NUMERIC_FIELDS.items():
                if field in converted and converted[field] != "":
                    converted[field] = caster(converted[field])
            rows.append(converted)
    return rows


def group_by(rows: Iterable[dict[str, object]], keys: tuple[str, ...]):
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(row)
    return groups


def l2_heuristic_block_size(num_kv_heads: int, head_dim: int, l2_kib: int) -> int:
    target_bytes = (l2_kib * 1024) // 4
    bytes_per_token = 2 * num_kv_heads * head_dim * 2
    for block_size in (256, 128, 64, 32):
        if block_size * bytes_per_token <= target_bytes:
            return block_size
    return 32


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(args: argparse.Namespace) -> tuple[str, list[dict[str, object]]]:
    rows = parse_rows(Path(args.csv_path))
    groups = group_by(rows, WORKLOAD_KEYS)

    decisions: list[dict[str, object]] = []
    for key, candidates in groups.items():
        ordered = sorted(candidates, key=lambda row: float(row["median_ms"]))
        best = ordered[0]
        worst = ordered[-1]
        heuristic_bs = l2_heuristic_block_size(
            int(best["num_kv_heads"]), int(best["head_dim"]), args.l2_kib
        )
        default = None
        static16 = None
        for row in candidates:
            if row["layout"] == "head_major" and int(row["block_size"]) == heuristic_bs:
                default = row
            if row["layout"] == "head_major" and int(row["block_size"]) == 16:
                static16 = row
        baseline = default or static16
        if baseline is None:
            baseline = ordered[-1]

        best_misses = float(best.get("cache-misses", 0) or 0)
        base_misses = float(baseline.get("cache-misses", 0) or 0)
        miss_reduction = (
            1.0 - best_misses / base_misses if base_misses > 0 else ""
        )
        decisions.append(
            {
                "phase": key[0],
                "batch_size": key[1],
                "seq_len": key[2],
                "shape": key[3],
                "exit_fraction": key[4],
                "best_layout": best["layout"],
                "best_block_size": best["block_size"],
                "best_median_ms": best["median_ms"],
                "best_p95_ms": best.get("p95_ms", ""),
                "best_cache_miss_rate": best.get("cache_miss_rate", ""),
                "best_cache_misses": best.get("cache-misses", ""),
                "baseline_layout": baseline["layout"],
                "baseline_block_size": baseline["block_size"],
                "baseline_median_ms": baseline["median_ms"],
                "baseline_cache_miss_rate": baseline.get("cache_miss_rate", ""),
                "speedup_vs_baseline": float(baseline["median_ms"])
                / float(best["median_ms"]),
                "cache_miss_reduction_vs_baseline": miss_reduction,
                "worst_layout": worst["layout"],
                "worst_block_size": worst["block_size"],
                "speedup_vs_worst": float(worst["median_ms"]) / float(best["median_ms"]),
                "heuristic_block_size": heuristic_bs,
                "best_is_default": int(
                    best["layout"] == "head_major"
                    and int(best["block_size"]) == heuristic_bs
                ),
            }
        )

    best_choice = Counter(
        (row["best_layout"], row["best_block_size"]) for row in decisions
    )
    by_phase = Counter(
        (row["phase"], row["best_layout"], row["best_block_size"])
        for row in decisions
    )
    by_shape = Counter(
        (row["shape"], row["best_layout"], row["best_block_size"])
        for row in decisions
    )
    non_default = [row for row in decisions if not int(row["best_is_default"])]
    meaningful = [
        row for row in decisions if float(row["speedup_vs_baseline"]) >= args.min_speedup
    ]

    report: list[str] = []
    report.append("# SlabPool Perf Autotuning Summary")
    report.append("")
    report.append(f"- CSV: `{args.csv_path}`")
    report.append(f"- Rows: `{len(rows)}`")
    report.append(f"- Workloads compared: `{len(decisions)}`")
    report.append(f"- Baseline: `head_major` + PACE L2 heuristic block size when present")
    report.append(f"- Non-default best choices: `{len(non_default)}/{len(decisions)}`")
    report.append(
        f"- Best choice beat baseline by >= {args.min_speedup:.2f}x: "
        f"`{len(meaningful)}/{len(decisions)}`"
    )
    report.append("")
    report.append("## Best Choices")
    report.append("")
    report.append(
        markdown_table(
            ["layout", "block_size", "wins"],
            [[layout, block, wins] for (layout, block), wins in best_choice.most_common()],
        )
    )
    report.append("")
    report.append("## Best Choices By Phase")
    report.append("")
    report.append(
        markdown_table(
            ["phase", "layout", "block_size", "wins"],
            [
                [phase, layout, block, wins]
                for (phase, layout, block), wins in by_phase.most_common()
            ],
        )
    )
    report.append("")
    report.append("## Best Choices By Shape")
    report.append("")
    report.append(
        markdown_table(
            ["shape", "layout", "block_size", "wins"],
            [
                [shape, layout, block, wins]
                for (shape, layout, block), wins in by_shape.most_common()
            ],
        )
    )
    report.append("")
    report.append("## Largest Baseline Improvements")
    report.append("")
    top = sorted(
        decisions, key=lambda row: float(row["speedup_vs_baseline"]), reverse=True
    )[: args.top]
    report.append(
        markdown_table(
            [
                "phase",
                "shape",
                "batch",
                "seq",
                "exit",
                "best",
                "baseline",
                "speedup",
                "miss_reduction",
            ],
            [
                [
                    row["phase"],
                    row["shape"],
                    row["batch_size"],
                    row["seq_len"],
                    row["exit_fraction"],
                    f"{row['best_layout']}:{row['best_block_size']}",
                    f"{row['baseline_layout']}:{row['baseline_block_size']}",
                    f"{float(row['speedup_vs_baseline']):.2f}x",
                    (
                        f"{float(row['cache_miss_reduction_vs_baseline']) * 100:.1f}%"
                        if row["cache_miss_reduction_vs_baseline"] != ""
                        else ""
                    ),
                ]
                for row in top
            ],
        )
    )
    report.append("")
    return "\n".join(report), decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--l2-kib", type=int, default=1024)
    parser.add_argument("--min-speedup", type=float, default=1.05)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    report, decisions = summarize(args)
    print(report)
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "perf_autotune_summary.md").write_text(report + "\n")
        write_csv(out_dir / "perf_autotune_decisions.csv", decisions)


if __name__ == "__main__":
    main()
