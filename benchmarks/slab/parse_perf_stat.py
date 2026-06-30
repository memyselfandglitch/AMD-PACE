#!/usr/bin/env python3
"""Merge one perf-stat CSV output with one benchmark CSV row."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


EVENT_COLUMNS = [
    "task-clock",
    "cycles",
    "instructions",
    "cache-references",
    "cache-misses",
    "LLC-loads",
    "LLC-load-misses",
]


def parse_number(value: str) -> float | None:
    value = value.strip().replace(",", "")
    if not value or value.startswith("<not") or value.startswith("not"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_latency_row(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one latency row in {path}, got {len(rows)}")
    return rows[0]


def read_perf(path: Path) -> tuple[dict[str, float | None], list[str]]:
    events: dict[str, float | None] = {event: None for event in EVENT_COLUMNS}
    unavailable: list[str] = []

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for fields in reader:
            if len(fields) < 3:
                continue
            value = parse_number(fields[0])
            event = fields[2].strip()
            if event not in events:
                continue
            events[event] = value
            if value is None:
                unavailable.append(event)

    return events, unavailable


def as_float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def merge(latency_row: dict[str, Any], events: dict[str, float | None], unavailable: list[str]) -> dict[str, Any]:
    row = dict(latency_row)
    for event in EVENT_COLUMNS:
        row[event] = "" if events[event] is None else events[event]

    cycles = events.get("cycles")
    instructions = events.get("instructions")
    cache_refs = events.get("cache-references")
    cache_misses = events.get("cache-misses")
    llc_loads = events.get("LLC-loads")
    llc_misses = events.get("LLC-load-misses")
    tokens = as_float(row, "tokens")

    row["ipc"] = instructions / cycles if instructions and cycles else ""
    row["cache_miss_rate"] = cache_misses / cache_refs if cache_misses and cache_refs else ""
    row["llc_load_miss_rate"] = llc_misses / llc_loads if llc_misses and llc_loads else ""
    row["cycles_per_token"] = cycles / tokens if cycles and tokens and not math.isnan(tokens) else ""
    row["instructions_per_token"] = instructions / tokens if instructions and tokens and not math.isnan(tokens) else ""
    row["cache_misses_per_token"] = cache_misses / tokens if cache_misses and tokens and not math.isnan(tokens) else ""
    row["llc_load_misses_per_token"] = llc_misses / tokens if llc_misses and tokens and not math.isnan(tokens) else ""
    row["unavailable_events"] = ";".join(sorted(set(unavailable)))
    return row


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latency-csv", required=True)
    parser.add_argument("--perf-stat", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    latency_row = read_latency_row(Path(args.latency_csv))
    events, unavailable = read_perf(Path(args.perf_stat))
    append_row(Path(args.out), merge(latency_row, events, unavailable))


if __name__ == "__main__":
    main()
