#!/usr/bin/env python3
"""Build honest block-size latency summaries from a sweep result directory.

New benchmark results contain ``metrics.generation_times`` and therefore
support mean, minimum, and p95 calculations. Older results such as job 7055
retain only ``average_gen_time``; for those, the script reports the valid mean
but leaves minimum and p95 empty instead of inventing readings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Iterable, Optional


BLOCK_PATTERN = re.compile(r"_b(auto|\d+)_")
BLOCK_CHOICES = ("auto", "32", "64", "128", "256")
BLOCK_ORDER = {value: index for index, value in enumerate(BLOCK_CHOICES)}

# Geometry from the official model configs used by job 7055. Unknown models
# can still be analyzed; only their effective auto block size is left empty.
KNOWN_MODEL_GEOMETRY = {
    "Qwen/Qwen2.5-0.5B": (2, 64),
    "Qwen/Qwen2.5-7B-Instruct": (4, 128),
}

SUMMARY_FIELDS = (
    "model_class", "model_name", "input_tokens", "output_tokens", "batch_size",
    "requested_block_size", "effective_block_size", "configured_runs",
    "raw_readings_available", "mean_latency_seconds", "lowest_latency_seconds",
    "p95_latency_seconds", "reading_source", "result_file",
)

BEST_FIELDS = (
    "model_class", "model_name", "input_tokens", "output_tokens", "batch_size",
    "auto_effective_block_size", "auto_mean_latency_seconds",
    "best_requested_block_size", "best_effective_block_size",
    "best_mean_latency_seconds", "best_mean_improvement_vs_auto_pct",
)


def percentile(values: Iterable[float], probability: float) -> float:
    """Return an R-7/NumPy-linear percentile for a non-empty sample."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one reading")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def read_l2_size_bytes(explicit_kib: Optional[int]) -> Optional[int]:
    if explicit_kib is not None:
        return explicit_kib * 1024
    path = Path("/sys/devices/system/cpu/cpu0/cache/index2/size")
    try:
        raw = path.read_text(encoding="utf-8").strip()
        unit = raw[-1].upper()
        value = int(raw[:-1]) if unit in {"K", "M"} else int(raw)
        multiplier = 1024 if unit == "K" else 1024 * 1024 if unit == "M" else 1
        return value * multiplier
    except (OSError, ValueError, IndexError):
        return None


def autotune_block_size(model_name: str, l2_size_bytes: Optional[int]) -> Optional[int]:
    geometry = KNOWN_MODEL_GEOMETRY.get(model_name)
    if geometry is None or l2_size_bytes is None:
        return None
    num_kv_heads, head_dim = geometry
    bytes_per_token = 2 * num_kv_heads * head_dim * 2  # K + V, BF16
    target = l2_size_bytes // 4
    for block_size in (256, 128, 64, 32):
        if block_size * bytes_per_token <= target:
            return block_size
    return 32


def requested_block_size(path: Path) -> str:
    match = BLOCK_PATTERN.search(path.parent.name)
    if not match:
        raise ValueError(f"cannot determine requested block size from {path.parent.name!r}")
    return match.group(1)


def model_class(path: Path) -> str:
    prefix = path.parent.name.split("_", 1)[0]
    return prefix if prefix in {"SLM", "LLM"} else ""


def extract_readings(metrics: dict[str, Any]) -> tuple[list[float], str]:
    readings = metrics.get("generation_times")
    if readings is not None:
        values = [float(value) for value in readings]
        if not values:
            raise ValueError("generation_times is present but empty")
        return values, "raw_generation_times"
    return [float(metrics["average_gen_time"])], "aggregate_mean_only"


def load_rows(job_dir: Path, l2_size_bytes: Optional[int]) -> list[dict[str, Any]]:
    rows = []
    files = sorted((job_dir / "raw").glob("**/*_results.json"))
    if not files:
        raise ValueError(f"no *_results.json files found below {job_dir / 'raw'}")
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = payload["model_args"]["model_name"]
        generation = payload["generation_args"]
        metrics = payload["benchmark_results"][0]["metrics"]
        requested = requested_block_size(path)
        effective = (
            autotune_block_size(model, l2_size_bytes)
            if requested == "auto"
            else int(requested)
        )
        readings, source = extract_readings(metrics)
        has_raw = source == "raw_generation_times"
        if has_raw and len(readings) != int(payload["num_runs"]):
            raise ValueError(
                f"{path} contains {len(readings)} generation_times but num_runs is "
                f"{payload['num_runs']}"
            )
        rows.append(
            {
                "model_class": model_class(path),
                "model_name": model,
                "input_tokens": generation["input_tokens"],
                "output_tokens": generation["output_tokens"],
                "batch_size": generation["batch_size"],
                "requested_block_size": requested,
                "effective_block_size": effective if effective is not None else "",
                "configured_runs": payload["num_runs"],
                "raw_readings_available": len(readings) if has_raw else 0,
                "mean_latency_seconds": statistics.fmean(readings),
                "lowest_latency_seconds": min(readings) if has_raw else "",
                "p95_latency_seconds": percentile(readings, 0.95) if has_raw else "",
                "reading_source": source,
                "result_file": str(path),
            }
        )
    rows.sort(
        key=lambda row: (
            0 if row["model_class"] == "SLM" else 1,
            row["model_name"],
            int(row["input_tokens"]),
            int(row["output_tokens"]),
            int(row["batch_size"]),
            BLOCK_ORDER.get(str(row["requested_block_size"]), len(BLOCK_ORDER)),
        )
    )
    return rows


def workload_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        row[key]
        for key in ("model_class", "model_name", "input_tokens", "output_tokens", "batch_size")
    )


def best_rows(detail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in detail_rows:
        grouped.setdefault(workload_key(row), []).append(row)
    output = []
    key_fields = ("model_class", "model_name", "input_tokens", "output_tokens", "batch_size")
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (
            0 if item[0][0] == "SLM" else 1,
            item[0][1],
            int(item[0][2]),
            int(item[0][3]),
            int(item[0][4]),
        ),
    )
    for key, candidates in ordered_groups:
        requested = {row["requested_block_size"] for row in candidates}
        missing = set(BLOCK_CHOICES) - requested
        if missing:
            raise ValueError(f"workload {key} is missing block sizes: {sorted(missing)}")
        auto = next(row for row in candidates if row["requested_block_size"] == "auto")
        best = min(candidates, key=lambda row: float(row["mean_latency_seconds"]))
        auto_mean = float(auto["mean_latency_seconds"])
        best_mean = float(best["mean_latency_seconds"])
        output.append(
            dict(zip(key_fields, key))
            | {
                "auto_effective_block_size": auto["effective_block_size"],
                "auto_mean_latency_seconds": auto_mean,
                "best_requested_block_size": best["requested_block_size"],
                "best_effective_block_size": best["effective_block_size"],
                "best_mean_latency_seconds": best_mean,
                "best_mean_improvement_vs_auto_pct": (auto_mean - best_mean) / auto_mean * 100,
            }
        )
    return output


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path, help="job directory containing raw/")
    parser.add_argument("--output-dir", type=Path, help="default: JOB_DIR/analysis")
    parser.add_argument(
        "--l2-cache-kib", type=int,
        help="L2 KiB per core; auto-detected from sysfs when omitted",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.job_dir / "analysis"
    rows = load_rows(args.job_dir, read_l2_size_bytes(args.l2_cache_kib))
    best = best_rows(rows)
    write_csv(output_dir / "latency_summary.csv", SUMMARY_FIELDS, rows)
    write_csv(output_dir / "best_latency_by_workload.csv", BEST_FIELDS, best)

    raw_count = sum(int(row["raw_readings_available"]) for row in rows)
    unavailable = sum(row["reading_source"] == "aggregate_mean_only" for row in rows)
    print(f"Analyzed {len(rows)} block configurations across {len(best)} workloads.")
    print(f"Wrote {output_dir / 'latency_summary.csv'}")
    print(f"Wrote {output_dir / 'best_latency_by_workload.csv'}")
    if unavailable:
        print(
            f"WARNING: {unavailable} configurations contain only an aggregate mean; "
            "minimum and p95 are unavailable for those rows.",
            file=sys.stderr,
        )
    else:
        print(f"Computed distributions from {raw_count} raw timed readings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
