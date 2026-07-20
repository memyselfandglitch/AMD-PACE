#!/usr/bin/env python3
"""Run and summarize decode-oriented SlabPool block-size sweeps.

Each point runs benchmark_llm_offline.py in a new process because the slab block
size is resolved while the model/cache is initialized.  The script uses only
the Python standard library so it can also be used to inspect results on a
machine without the benchmark dependencies installed.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import shlex
import statistics
import subprocess
import sys
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE / "benchmark_llm_offline.py"
FIELDS = (
    "model_class", "model_name", "framework", "phase", "requested_block_size",
    "effective_block_size", "input_tokens", "output_tokens", "batch_size",
    "kv_cache_type", "warmup_runs", "measured_runs", "mean_generation_latency_ms",
    "median_generation_latency_ms", "p95_generation_latency_ms",
    "minimum_generation_latency_ms", "maximum_generation_latency_ms", "output_tps",
    "average_latency_per_token_ms", "recommended_block_size", "is_recommended",
    "rank_by_median", "margin_over_runner_up_pct", "recommended_p95_not_worse",
    "recommendation_stable", "result_file",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def points(spec: dict[str, Any]):
    axes = spec["axes"]
    workloads = itertools.product(
        spec["models"], spec.get("frameworks", ["pace"]),
        axes["input_tokens"], axes["output_tokens"], axes["batch_sizes"],
    )
    for workload_index, (model, framework, input_tokens, output_tokens, batch_size) in enumerate(workloads):
        block_sizes = list(axes["block_sizes"])
        if spec.get("randomize_block_order", True):
            random.Random(spec.get("order_seed", 0) + workload_index).shuffle(block_sizes)
        for block_size in block_sizes:
            yield model, framework, block_size, input_tokens, output_tokens, batch_size


def block_size_env(framework: str) -> str:
    if framework == "pace":
        return "SLAB_BLOCK_SIZE"
    if framework == "vllm_zentorch":
        return "PACE_VLLM_SLAB_BLOCK_SIZE"
    raise ValueError(
        f"Block-size sweeps support pace and vllm_zentorch, not {framework!r}"
    )


def make_benchmark_config(spec, model, framework, input_tokens, output_tokens, batch_size, point_dir):
    return {
        "frameworks": [framework],
        "model_args": {
            "model_name": model["name"],
            "dtype": model.get("dtype", "bf16"),
            "llm_operators": model.get("llm_operators", spec.get("llm_operators")),
            "spec_config": None,
        },
        "use_real_data": spec.get("use_real_data", False),
        "generation_args": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "batch_size": batch_size,
            "kv_cache_type": "SLAB_POOL",
            "do_sample": False,
            "manual_seed": spec.get("manual_seed", 0),
        },
        "warmup_runs": spec.get("warmup_runs", 2),
        "num_runs": spec.get("num_runs", 5),
        "verbose": False,
        "output_dir": str(point_dir),
        "token_metrics": {"time_to_first_token": False, "time_per_tokens": False},
        "system_metrics": False,
    }


def percentile(values: list[float], probability: float) -> float:
    """Return the linearly interpolated percentile used by NumPy's default."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def read_l2_size_bytes() -> int | None:
    try:
        raw = Path("/sys/devices/system/cpu/cpu0/cache/index2/size").read_text(
            encoding="utf-8"
        ).strip()
        unit = raw[-1].upper()
        value = int(raw[:-1]) if unit in {"K", "M"} else int(raw)
        return value * (1024 if unit == "K" else 1024 * 1024 if unit == "M" else 1)
    except (OSError, ValueError, IndexError):
        return None


def effective_block_size(model: dict[str, Any], requested: int | None, l2_bytes: int | None):
    if requested is not None:
        return requested
    num_kv_heads = model.get("num_kv_heads")
    head_dim = model.get("head_dim")
    if num_kv_heads is None or head_dim is None or l2_bytes is None:
        return ""
    bytes_per_token = 2 * num_kv_heads * head_dim * 2  # K + V, BF16
    target = l2_bytes // 4
    for candidate in (256, 128, 64, 32):
        if candidate * bytes_per_token <= target:
            return candidate
    return 32


def row_from_result(path, model, framework, block_size, l2_bytes):
    result = load_json(path)
    generation = result["generation_args"]
    metrics = result["benchmark_results"][0]["metrics"]
    readings = metrics.get("generation_times")
    if not readings:
        raise ValueError(
            f"{path} has no metrics.generation_times; rebuild with the raw-timing "
            "Metrics changes before running this sweep"
        )
    readings_ms = [float(value) * 1000.0 for value in readings]
    if len(readings_ms) != int(result["num_runs"]):
        raise ValueError(
            f"{path} contains {len(readings_ms)} timings for {result['num_runs']} runs"
        )
    return {
        "model_class": model["class"],
        "model_name": model["name"],
        "framework": framework,
        "phase": "decode_generation",
        "requested_block_size": "auto" if block_size is None else block_size,
        "effective_block_size": effective_block_size(model, block_size, l2_bytes),
        "input_tokens": generation["input_tokens"],
        "output_tokens": generation["output_tokens"],
        "batch_size": generation["batch_size"],
        "kv_cache_type": generation["kv_cache_type"],
        "warmup_runs": result["warmup_runs"],
        "measured_runs": result["num_runs"],
        "mean_generation_latency_ms": statistics.fmean(readings_ms),
        "median_generation_latency_ms": statistics.median(readings_ms),
        "p95_generation_latency_ms": percentile(readings_ms, 0.95),
        "minimum_generation_latency_ms": min(readings_ms),
        "maximum_generation_latency_ms": max(readings_ms),
        "output_tps": metrics["output_tps"],
        "average_latency_per_token_ms": float(metrics["average_latency_per_token"]) * 1000.0,
        "recommended_block_size": "",
        "is_recommended": "",
        "rank_by_median": "",
        "margin_over_runner_up_pct": "",
        "recommended_p95_not_worse": "",
        "recommendation_stable": "",
        "result_file": str(path),
    }


def workload_key(row):
    return tuple(
        row[key]
        for key in (
            "model_class", "model_name", "framework", "phase", "input_tokens",
            "output_tokens", "batch_size", "kv_cache_type",
        )
    )


def annotate_recommendations(rows, expected_fixed_sizes, min_margin_pct):
    """Annotate complete workloads; incomplete groups remain visibly undecided."""
    for row in rows:
        for field in (
            "recommended_block_size", "is_recommended", "rank_by_median",
            "margin_over_runner_up_pct", "recommended_p95_not_worse",
            "recommendation_stable",
        ):
            row[field] = ""

    groups = {}
    for row in rows:
        groups.setdefault(workload_key(row), []).append(row)

    for candidates in groups.values():
        fixed = [row for row in candidates if row["requested_block_size"] != "auto"]
        present = {int(row["requested_block_size"]) for row in fixed}
        if not expected_fixed_sizes.issubset(present):
            continue
        ranked = sorted(fixed, key=lambda row: float(row["median_generation_latency_ms"]))
        best = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else ranked[0]
        best_median = float(best["median_generation_latency_ms"])
        runner_median = float(runner_up["median_generation_latency_ms"])
        margin = (runner_median / best_median - 1.0) * 100.0 if best_median else 0.0
        p95_not_worse = (
            float(best["p95_generation_latency_ms"])
            <= float(runner_up["p95_generation_latency_ms"])
        )
        recommended = int(best["effective_block_size"])
        rank_by_size = {
            int(row["effective_block_size"]): rank
            for rank, row in enumerate(ranked, 1)
        }
        for row in candidates:
            row["recommended_block_size"] = recommended
            row["is_recommended"] = (
                row["requested_block_size"] != "auto"
                and int(row["effective_block_size"]) == recommended
            )
            effective = row["effective_block_size"]
            row["rank_by_median"] = rank_by_size.get(int(effective), "") if effective != "" else ""
            row["margin_over_runner_up_pct"] = round(margin, 3)
            row["recommended_p95_not_worse"] = p95_not_worse
            row["recommendation_stable"] = margin >= min_margin_pct and p95_not_worse
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run(args):
    spec = load_json(args.spec)
    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_points = list(points(spec))
    rows = []
    failures = []
    l2_bytes = read_l2_size_bytes()
    expected_fixed_sizes = {
        int(value) for value in spec["axes"]["block_sizes"] if value is not None
    }

    metadata = {
        "spec": str(args.spec.resolve()), "platform": platform.platform(),
        "processor": platform.processor(), "python": sys.version,
        "command": " ".join(shlex.quote(x) for x in sys.argv),
        "warmup_runs": spec.get("warmup_runs", 2),
        "measured_runs": spec.get("num_runs", 10),
        "selection_metric": "median_generation_latency_ms",
        "l2_size_bytes": l2_bytes,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    for index, (model, framework, block_size, input_tokens, output_tokens, batch_size) in enumerate(all_points, 1):
        tag = (f"{model['class']}_b{'auto' if block_size is None else block_size}"
               f"_i{input_tokens}_o{output_tokens}_n{batch_size}_{framework}")
        point_dir = raw_dir / tag
        prior = sorted(point_dir.glob("*_results.json"))
        print(f"[{index}/{len(all_points)}] {tag}", flush=True)
        if prior and not args.rerun:
            rows.append(row_from_result(prior[-1], model, framework, block_size, l2_bytes))
            annotate_recommendations(rows, expected_fixed_sizes, args.min_margin_pct)
            write_csv(output_dir / "results.csv", rows)
            continue

        config = make_benchmark_config(
            spec, model, framework, input_tokens, output_tokens, batch_size, point_dir
        )
        if args.dry_run:
            env_name = block_size_env(framework)
            print(f"  {env_name}={'<unset>' if block_size is None else block_size} "
                  f"{sys.executable} {BENCHMARK} --config <generated-config>")
            continue

        point_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env_name = block_size_env(framework)
        if block_size is None:
            env.pop(env_name, None)
        else:
            env[env_name] = str(block_size)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as config_file:
            json.dump(config, config_file, indent=2)
            config_path = config_file.name
        try:
            completed = subprocess.run(
                [sys.executable, str(BENCHMARK), "--config", config_path],
                cwd=HERE, env=env, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            (point_dir / "benchmark.log").write_text(completed.stdout, encoding="utf-8")
            result_files = sorted(point_dir.glob("*_results.json"))
            if completed.returncode or not result_files:
                failures.append({"point": tag, "returncode": completed.returncode})
                print(f"  FAILED (see {point_dir / 'benchmark.log'})", file=sys.stderr)
                if not args.keep_going:
                    break
            else:
                rows.append(row_from_result(result_files[-1], model, framework, block_size, l2_bytes))
                annotate_recommendations(rows, expected_fixed_sizes, args.min_margin_pct)
                write_csv(output_dir / "results.csv", rows)
        finally:
            Path(config_path).unlink(missing_ok=True)

    annotate_recommendations(rows, expected_fixed_sizes, args.min_margin_pct)
    write_csv(output_dir / "results.csv", rows)
    (output_dir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    print(f"Wrote {len(rows)} results to {output_dir / 'results.csv'}")
    return 1 if failures else 0


def summarize(args):
    with args.results.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    recommendations = [row for row in rows if row["is_recommended"].lower() == "true"]

    output = args.output or args.results.with_name("recommendations.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(recommendations)
    print(f"Wrote {len(recommendations)} workload recommendations to {output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="execute a sweep")
    run_parser.add_argument("--spec", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--rerun", action="store_true")
    run_parser.add_argument("--keep-going", action="store_true")
    run_parser.add_argument("--min-margin-pct", type=float, default=5.0)
    run_parser.set_defaults(func=run)
    summary_parser = subparsers.add_parser("summarize", help="select per-workload winners")
    summary_parser.add_argument("--results", type=Path, required=True)
    summary_parser.add_argument("--output", type=Path)
    summary_parser.add_argument("--min-margin-pct", type=float, default=3.0)
    summary_parser.set_defaults(func=summarize)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(parsed.func(parsed) or 0)
