#!/usr/bin/env python3
"""Select decision rows worth re-running under perf stat."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable


INT_FIELDS = {
    "batch_size",
    "seq_len",
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
    "baseline_block_size",
    "best_block_size",
}

FLOAT_FIELDS = {
    "exit_fraction",
    "baseline_ms",
    "best_ms",
    "speedup",
    "p95_speedup",
}

BOOL_FIELDS = {"win_ge_5pct", "win_ge_10pct", "p95_not_worse"}

OUTPUT_FIELDS = [
    "case_label",
    "phase",
    "shape",
    "batch_size",
    "seq_len",
    "seq_bucket",
    "exit_fraction",
    "baseline_block_size",
    "best_block_size",
    "speedup",
    "p95_speedup",
    "baseline_ms",
    "best_ms",
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
]


def read_decisions(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for field in INT_FIELDS:
                row[field] = int(float(row[field]))
            for field in FLOAT_FIELDS:
                row[field] = float(row[field])
            for field in BOOL_FIELDS:
                row[field] = row[field] == "True"
            rows.append(row)
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in OUTPUT_FIELDS})


def identity(row: dict) -> tuple:
    return (
        row["phase"],
        row["shape"],
        row["batch_size"],
        row["seq_len"],
        row["exit_fraction"],
    )


def add_best(
    selected: list[dict],
    candidates: list[dict],
    label: str,
    predicate: Callable[[dict], bool],
) -> None:
    matches = [row for row in candidates if predicate(row)]
    if not matches:
        return
    row = dict(max(matches, key=lambda item: item["speedup"]))
    if identity(row) in {identity(item) for item in selected}:
        return
    row["case_label"] = label
    selected.append(row)


def select_cases(rows: list[dict], max_cases: int) -> list[dict]:
    stable = [
        row
        for row in rows
        if row["win_ge_5pct"] and row["p95_not_worse"]
    ]
    selected: list[dict] = []

    add_best(
        selected,
        stable,
        "decode_short_llama",
        lambda row: row["phase"] == "decode"
        and row["shape"] == "llama_gqa"
        and row["seq_bucket"] == "short",
    )
    add_best(
        selected,
        stable,
        "decode_medium_mha",
        lambda row: row["phase"] == "decode"
        and row["shape"] == "mha"
        and row["seq_bucket"] == "medium",
    )
    add_best(
        selected,
        stable,
        "decode_long_llama",
        lambda row: row["phase"] == "decode"
        and row["shape"] == "llama_gqa"
        and row["seq_bucket"] == "long",
    )
    add_best(
        selected,
        stable,
        "mtd_short_llama",
        lambda row: row["phase"] == "mtd"
        and row["shape"] == "llama_gqa"
        and row["seq_bucket"] == "short",
    )
    add_best(
        selected,
        stable,
        "mtd_medium_mha",
        lambda row: row["phase"] == "mtd"
        and row["shape"] == "mha"
        and row["seq_bucket"] == "medium",
    )
    add_best(
        selected,
        stable,
        "mtd_long_slm",
        lambda row: row["phase"] == "mtd"
        and row["shape"] == "slm_gqa"
        and row["seq_bucket"] == "long",
    )
    add_best(
        selected,
        stable,
        "small_block_win",
        lambda row: row["best_block_size"] in (16, 32),
    )
    add_best(
        selected,
        stable,
        "large_block_win",
        lambda row: row["best_block_size"] == 256
        and row["baseline_block_size"] != 256,
    )

    for row in sorted(stable, key=lambda item: item["speedup"], reverse=True):
        if len(selected) >= max_cases:
            break
        if identity(row) in {identity(item) for item in selected}:
            continue
        copied = dict(row)
        copied["case_label"] = "top_stable_win"
        selected.append(copied)

    for phase in ("decode", "mtd"):
        controls = [
            row
            for row in rows
            if row["phase"] == phase and row["speedup"] < 1.01
        ]
        if controls:
            row = dict(
                max(
                    controls,
                    key=lambda item: (item["seq_len"], item["batch_size"]),
                )
            )
            if identity(row) not in {identity(item) for item in selected}:
                row["case_label"] = f"{phase}_heuristic_control"
                selected.append(row)

    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decisions_csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-cases", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_decisions(Path(args.decisions_csv))
    selected = select_cases(rows, args.max_cases)
    write_rows(Path(args.out), selected)
    print(f"Wrote {len(selected)} selected cases to {args.out}")


if __name__ == "__main__":
    main()
