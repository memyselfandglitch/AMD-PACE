#!/usr/bin/env python3
# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Convert one `perf stat -x,` run into a compact CSV row.

The perf pass is intentionally separate from the latency sweep. The sweep tells
us which layout/block-size cases are interesting; this parser lets us attach
hardware counters to a small set of selected cases without complicating the main
benchmark harness.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


COUNTERS = (
    "task-clock",
    "cycles",
    "instructions",
    "cache-references",
    "cache-misses",
    "LLC-loads",
    "LLC-load-misses",
)


def _parse_value(raw: str) -> float | None:
    value = raw.strip().replace(",", "")
    if not value or value.startswith("<"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def _format(value: object) -> object:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6g}"
    if value is None:
        return ""
    return value


def _event_column(event: str) -> str:
    return "event_" + re.sub(r"[^A-Za-z0-9_]+", "_", event).strip("_")


def parse_perf_csv(path: Path) -> tuple[dict[str, float], list[str]]:
    counters: dict[str, float] = {}
    unavailable: list[str] = []

    with path.open(newline="") as f:
        reader = csv.reader(f)
        for fields in reader:
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) < 3:
                continue

            value = _parse_value(fields[0])
            event = fields[2].strip()
            if not event:
                continue

            if value is None:
                unavailable.append(event)
            else:
                counters[event] = value

    return counters, unavailable


def parse_benchmark_row(path: Path) -> dict[str, object]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise SystemExit(f"Expected exactly one benchmark row in {path}, got {len(rows)}")
    return rows[0]


def parse_metadata(items: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid metadata item {item!r}; expected key=value")
        key, value = item.split("=", 1)
        metadata[key] = value
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perf-raw", required=True, help="Raw perf stat CSV output.")
    parser.add_argument("--bench-csv", required=True, help="Single-row benchmark CSV.")
    parser.add_argument("--output", required=True, help="Parsed counter CSV path.")
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Extra key=value metadata to copy into the output row.",
    )
    args = parser.parse_args()

    counters, unavailable = parse_perf_csv(Path(args.perf_raw))
    bench = parse_benchmark_row(Path(args.bench_csv))
    row: dict[str, object] = {}
    row.update(parse_metadata(args.metadata))

    for field in (
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
    ):
        if field in bench:
            row[field] = bench[field]

    tokens = float(bench.get("tokens", 0) or 0)
    for counter in COUNTERS:
        row[counter] = counters.get(counter)

    for counter in sorted(set(counters) - set(COUNTERS)):
        row[_event_column(counter)] = counters[counter]

    row["ipc"] = _safe_div(counters.get("instructions"), counters.get("cycles"))
    row["cache_miss_rate"] = _safe_div(
        counters.get("cache-misses"), counters.get("cache-references")
    )
    row["llc_load_miss_rate"] = _safe_div(
        counters.get("LLC-load-misses"), counters.get("LLC-loads")
    )
    row["cycles_per_token"] = _safe_div(counters.get("cycles"), tokens)
    row["instructions_per_token"] = _safe_div(counters.get("instructions"), tokens)
    row["cache_misses_per_token"] = _safe_div(counters.get("cache-misses"), tokens)
    row["llc_load_misses_per_token"] = _safe_div(
        counters.get("LLC-load-misses"), tokens
    )
    row["unavailable_events"] = ";".join(sorted(set(unavailable)))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow({key: _format(value) for key, value in row.items()})


if __name__ == "__main__":
    main()
