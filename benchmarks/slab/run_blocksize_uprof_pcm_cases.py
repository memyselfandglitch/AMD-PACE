#!/usr/bin/env python3
"""Run selected block-size cases under AMD uProf PCM.

The script intentionally does not try to interpret AMDuProfPcm's CSV format.
Different uProf versions and metric groups can emit different columns, so this
first pass preserves the raw PCM CSVs and writes a manifest that links each
heuristic/best run back to its latency result.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from pathlib import Path

from bench_slab_blocksize import FIELDNAMES as LATENCY_FIELDS
from parse_perf_stat import read_latency_row


DEFAULT_METRICS = "ipc,l2,l3"

METADATA_FIELDS = [
    "case_label",
    "run_kind",
    "compared_heuristic_block_size",
    "compared_best_block_size",
    "decision_speedup",
    "decision_p95_speedup",
    "uprof_metrics",
    "uprof_scope_args",
    "uprof_cumulative",
    "uprof_csv",
    "latency_csv",
    "stdout_path",
    "stderr_path",
    "returncode",
    "command",
]

OUTPUT_FIELDS = METADATA_FIELDS + [f"latency_{field}" for field in LATENCY_FIELDS]


def sanitize(value: object) -> str:
    return str(value).replace(".", "p").replace("/", "_").replace(" ", "_")


def read_cases(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in OUTPUT_FIELDS})


def latency_metadata(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    row = read_latency_row(path)
    return {f"latency_{field}": row.get(field, "") for field in LATENCY_FIELDS}


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
    uprof_csv = args.raw_dir / f"{tag}.uprof.csv"
    stdout_path = args.raw_dir / f"{tag}.uprof.stdout"
    stderr_path = args.raw_dir / f"{tag}.uprof.stderr"

    bench_cmd = [
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

    cmd = [
        args.amd_uprof_pcm,
        "-m",
        args.metrics,
        *shlex.split(args.scope_args),
        "-o",
        str(uprof_csv),
    ]
    if args.cumulative:
        cmd.append("-C")
    cmd.extend(["--", *bench_cmd])

    print("running", tag, flush=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        result = subprocess.run(cmd, stdout=stdout, stderr=stderr, check=False)

    row = {
        "case_label": case["case_label"],
        "run_kind": run_kind,
        "compared_heuristic_block_size": case["baseline_block_size"],
        "compared_best_block_size": case["best_block_size"],
        "decision_speedup": case["speedup"],
        "decision_p95_speedup": case["p95_speedup"],
        "uprof_metrics": args.metrics,
        "uprof_scope_args": args.scope_args,
        "uprof_cumulative": args.cumulative,
        "uprof_csv": str(uprof_csv),
        "latency_csv": str(latency_csv),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "returncode": result.returncode,
        "command": shlex.join(cmd),
        **latency_metadata(latency_csv),
    }
    append_row(args.out, row)

    if result.returncode != 0 and not args.continue_on_error:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--amd-uprof-pcm", required=True)
    parser.add_argument("--layout", default="head_major")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5000)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument("--scope-args", default="-a")
    parser.add_argument("--no-cumulative", dest="cumulative", action="store_false")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.set_defaults(cumulative=True)
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

    print(f"Wrote uProf PCM manifest to {args.out}")


if __name__ == "__main__":
    main()
