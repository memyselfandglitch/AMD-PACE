#!/usr/bin/env python3
"""Run selected P*V loop orders under AMD uProf PCM and consolidate metrics."""

from __future__ import annotations

import argparse
import csv
import random
import re
import subprocess
from pathlib import Path


CASES = (
    ("decode_control", 1, 8192, 64),
    ("kij_before_crossover", 512, 2048, 64),
    ("ikj_after_crossover", 512, 8192, 64),
    ("ikj_long_context", 512, 16384, 64),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--amd-uprof-pcm", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--minimum-seconds", type=float, default=5.0)
    parser.add_argument("--order-seed", type=int, default=20260809)
    parser.add_argument("--metrics", default="ipc,l2,l3,memory")
    parser.add_argument("--scope-args", default="-a")
    return parser.parse_args()


def parse_pcm(path: Path) -> list[tuple[str, float]]:
    metrics: list[tuple[str, float]] = []
    in_metric_table = False
    with path.open(newline="", errors="replace") as stream:
        for row in csv.reader(stream):
            if row and row[0].strip() == "Metric":
                in_metric_table = True
                continue
            if len(row) < 2:
                in_metric_table = False
                continue
            if not in_metric_table:
                continue
            name = row[0].strip()
            value = row[1].strip().replace(",", "")
            if not name:
                continue
            try:
                metrics.append((name, float(value)))
            except ValueError:
                continue
    return metrics


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def is_cumulative_metric(name: str) -> bool:
    normalized = name.lower()
    if (
        "%" in name
        or "latency" in normalized
        or "bw" in normalized
        or "per sec" in normalized
        or any(unit in normalized for unit in ("(pti)", "(ptc)", "(pto)"))
    ):
        return False
    return any(
        token in normalized
        for token in ("access", "miss", "instruction", "cycle", "request")
    ) and normalized not in {"ipc", "cpi"}


def main() -> None:
    args = arguments()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    runs = [(case, order) for case in CASES for order in ("ikj", "kij")]
    random.Random(args.order_seed).shuffle(runs)
    manifest_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for position, (case, order) in enumerate(runs):
        label, query_len, kv_len, head_dim = case
        stem = safe_name(
            f"{position:02d}_{label}_{order}_m{query_len}_n{kv_len}_d{head_dim}"
        )
        pcm_csv = raw_dir / f"{stem}.uprof.csv"
        stdout_path = raw_dir / f"{stem}.stdout"
        stderr_path = raw_dir / f"{stem}.stderr"
        benchmark_csv = raw_dir / f"{stem}.benchmark.csv"
        command = [
            str(args.amd_uprof_pcm),
            "--msr",
            "-m",
            args.metrics,
            *args.scope_args.split(),
            "-o",
            str(pcm_csv),
            "-C",
            "--",
            str(args.binary),
            "--order",
            order,
            "--query-len",
            str(query_len),
            "--kv-len",
            str(kv_len),
            "--head-dim",
            str(head_dim),
            "--minimum-seconds",
            str(args.minimum_seconds),
            "--result-csv",
            str(benchmark_csv),
        ]
        print(f"running {position + 1}/{len(runs)}: {label} {order}", flush=True)
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        stdout_path.write_text(result.stdout)
        stderr_path.write_text(result.stderr)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, command)
        if not benchmark_csv.exists():
            raise RuntimeError(f"benchmark did not write {benchmark_csv}")
        with benchmark_csv.open(newline="") as stream:
            benchmark = next(csv.DictReader(stream))
        iterations = int(benchmark["iterations"])
        manifest_rows.append(
            {
                "position": position,
                "case_label": label,
                "order": order,
                "query_len": query_len,
                "kv_len": kv_len,
                "head_dim": head_dim,
                "iterations": benchmark.get("iterations", ""),
                "kernel_elapsed_s": benchmark.get("elapsed_s", ""),
                "kernel_gflops": benchmark.get("gflops", ""),
                "checksum": benchmark.get("checksum", ""),
                "return_code": result.returncode,
                "pcm_csv": pcm_csv,
                "stdout": stdout_path,
                "stderr": stderr_path,
                "benchmark_csv": benchmark_csv,
            }
        )
        for metric, value in parse_pcm(pcm_csv):
            metric_rows.append(
                {
                    "case_label": label,
                    "order": order,
                    "query_len": query_len,
                    "kv_len": kv_len,
                    "head_dim": head_dim,
                    "metric": metric,
                    "value": value,
                    "iterations": iterations,
                    "value_per_iteration": (
                        value / iterations
                        if iterations and is_cumulative_metric(metric)
                        else ""
                    ),
                }
            )

    manifest = args.out_dir / "pv_uprof_manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)
    metrics = args.out_dir / "pv_uprof_metrics.csv"
    with metrics.open("w", newline="") as stream:
        if not metric_rows:
            raise RuntimeError("uProf completed but no numeric metrics were parsed")
        writer = csv.DictWriter(stream, fieldnames=metric_rows[0].keys())
        writer.writeheader()
        writer.writerows(metric_rows)
    values = {
        (str(row["case_label"]), str(row["order"]), str(row["metric"])): row
        for row in metric_rows
    }
    comparison_rows = []
    for label, query_len, kv_len, head_dim in CASES:
        metric_names = sorted(
            metric
            for case_label, order, metric in values
            if case_label == label and order == "ikj"
            and (label, "kij", metric) in values
        )
        for metric in metric_names:
            ikj_row = values[(label, "ikj", metric)]
            kij_row = values[(label, "kij", metric)]
            ikj_value = float(ikj_row["value"])
            kij_value = float(kij_row["value"])
            ikj_per_iteration = ikj_row["value_per_iteration"]
            kij_per_iteration = kij_row["value_per_iteration"]
            comparison_rows.append(
                {
                    "case_label": label,
                    "query_len": query_len,
                    "kv_len": kv_len,
                    "head_dim": head_dim,
                    "metric": metric,
                    "ikj_value": ikj_value,
                    "kij_value": kij_value,
                    "kij_over_ikj": kij_value / ikj_value if ikj_value else "",
                    "ikj_per_iteration": ikj_per_iteration,
                    "kij_per_iteration": kij_per_iteration,
                    "kij_over_ikj_per_iteration": (
                        float(kij_per_iteration) / float(ikj_per_iteration)
                        if ikj_per_iteration not in {"", 0}
                        and kij_per_iteration != ""
                        else ""
                    ),
                }
            )
    comparison = args.out_dir / "pv_uprof_comparison.csv"
    with comparison.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=comparison_rows[0].keys())
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"Manifest: {manifest}")
    print(f"Consolidated metrics: {metrics}")
    print(f"Paired comparison: {comparison}")


if __name__ == "__main__":
    main()
