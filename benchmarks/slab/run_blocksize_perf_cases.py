#!/usr/bin/env python3
"""Run selected block-size cases under perf stat.

Each selected case is run for the current heuristic block size and the empirical
best block size. The output is one combined CSV with latency and perf counters.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

from parse_perf_stat import read_latency_row, read_perf, merge


DEFAULT_EVENTS = (
    "task-clock,cycles,instructions,cache-references,cache-misses,"
    "LLC-loads,LLC-load-misses"
)


def sanitize(value: object) -> str:
    return str(value).replace(".", "p").replace("/", "_").replace(" ", "_")


def read_cases(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_one(args, case: dict, run_kind: str, block_size: str, case_index: int) -> None:
    tag = "_".join(
        [
            f"{case_index:03d}",
            sanitize(case["case_label"]),
            sanitize(run_kind),
            sanitize(case["phase"]),
            sanitize(case["shape"]),
            f"b{sanitize(case['batch_size'])}",
            f"s{sanitize(case['seq_len'])}",
            f"exit{sanitize(case['exit_fraction'])}",
            f"bs{sanitize(block_size)}",
        ]
    )

    latency_csv = args.raw_dir / f"{tag}.latency.csv"
    perf_csv = args.raw_dir / f"{tag}.perf.csv"

    cmd = [
        "perf",
        "stat",
        "-x,",
        "-e",
        args.perf_events,
        "-o",
        str(perf_csv),
        "--",
        args.python,
        "benchmarks/slab/bench_slab_blocksize.py",
        "--out",
        str(latency_csv),
        "--layouts",
        args.layout,
        "--block-sizes",
        str(block_size),
        "--phases",
        case["phase"],
        "--seq-lens",
        case["seq_len"],
        "--batch-sizes",
        case["batch_size"],
        "--shapes",
        case["shape"],
        "--exit-fractions",
        case["exit_fraction"],
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
    ]

    print("running", tag, flush=True)
    subprocess.run(cmd, check=True)

    latency_row = read_latency_row(latency_csv)
    events, unavailable = read_perf(perf_csv)
    merged = merge(latency_row, events, unavailable)

    metadata = {
        "case_label": case["case_label"],
        "run_kind": run_kind,
        "compared_heuristic_block_size": case["baseline_block_size"],
        "compared_best_block_size": case["best_block_size"],
        "decision_speedup": case["speedup"],
        "decision_p95_speedup": case["p95_speedup"],
    }
    row = {**metadata, **merged}
    append_row(args.out, row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--layout", default="head_major")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--perf-events", default=DEFAULT_EVENTS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    cases = read_cases(Path(args.cases))
    for index, case in enumerate(cases):
        block_sizes = [
            ("heuristic", case["baseline_block_size"]),
            ("best", case["best_block_size"]),
        ]
        seen: set[str] = set()
        for run_kind, block_size in block_sizes:
            if block_size in seen:
                continue
            seen.add(block_size)
            run_one(args, case, run_kind, block_size, index)

    print(f"Wrote perf comparison CSV to {args.out}")


if __name__ == "__main__":
    main()
