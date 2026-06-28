#!/usr/bin/env python3
# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Combine parsed SlabPool perf CSV files into one table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PREFERRED_COLUMNS = [
    "job_id",
    "case_tag",
    "phase",
    "layout",
    "block_size",
    "batch_size",
    "seq_len",
    "query_len",
    "active_batch_size",
    "shape",
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
    "exit_fraction",
    "omp_threads",
    "slab_schedule",
    "warmups",
    "repeats",
    "mean_ms",
    "median_ms",
    "p95_ms",
    "tokens",
    "tokens_per_second",
    "total_blocks",
    "task-clock",
    "cycles",
    "instructions",
    "cache-references",
    "cache-misses",
    "LLC-loads",
    "LLC-load-misses",
    "ipc",
    "cache_miss_rate",
    "llc_load_miss_rate",
    "cycles_per_token",
    "instructions_per_token",
    "cache_misses_per_token",
    "llc_load_misses_per_token",
    "perf_events",
    "unavailable_events",
    "source_file",
]


def _sort_key(row: dict[str, str]) -> tuple:
    def as_float(field: str) -> float:
        try:
            return float(row.get(field, ""))
        except ValueError:
            return 0.0

    return (
        row.get("phase", ""),
        row.get("shape", ""),
        as_float("seq_len"),
        as_float("batch_size"),
        as_float("exit_fraction"),
        row.get("layout", ""),
        as_float("block_size"),
        row.get("job_id", ""),
    )


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["source_file"] = path.name
                rows.append(row)
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = set()
    for row in rows:
        columns.update(row)
    fieldnames = [column for column in PREFERRED_COLUMNS if column in columns]
    fieldnames.extend(sorted(columns - set(fieldnames)))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="benchmarks/slab/results/perf",
        help="Directory containing parsed *_perf.csv files.",
    )
    parser.add_argument("--glob", default="*_perf.csv")
    parser.add_argument(
        "--output",
        default="benchmarks/slab/results/perf_combined.csv",
        help="Combined CSV path.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    paths = sorted(
        p for p in input_dir.glob(args.glob) if not p.name.endswith("_perf_raw.csv")
    )
    if not paths:
        raise SystemExit(f"No parsed perf CSVs found in {input_dir}")
    rows = sorted(read_rows(paths), key=_sort_key)
    write_rows(Path(args.output), rows)
    print(f"Wrote {len(rows)} rows from {len(paths)} files to {args.output}")


if __name__ == "__main__":
    main()
