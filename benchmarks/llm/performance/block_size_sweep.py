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
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE / "benchmark_llm_offline.py"
FIELDS = (
    "model_class", "model_name", "framework", "block_size", "input_tokens",
    "output_tokens", "batch_size", "kv_cache_type", "output_tps",
    "average_latency_per_token", "average_gen_time", "result_file",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def points(spec: dict[str, Any]):
    axes = spec["axes"]
    for model, framework, block_size, input_tokens, output_tokens, batch_size in itertools.product(
        spec["models"], spec.get("frameworks", ["pace"]), axes["block_sizes"],
        axes["input_tokens"], axes["output_tokens"], axes["batch_sizes"],
    ):
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


def row_from_result(path, model_class, model_name, framework, block_size):
    result = load_json(path)
    generation = result["generation_args"]
    metrics = result["benchmark_results"][0]["metrics"]
    return {
        "model_class": model_class,
        "model_name": model_name,
        "framework": framework,
        "block_size": "auto" if block_size is None else block_size,
        "input_tokens": generation["input_tokens"],
        "output_tokens": generation["output_tokens"],
        "batch_size": generation["batch_size"],
        "kv_cache_type": generation["kv_cache_type"],
        "output_tps": metrics["output_tps"],
        "average_latency_per_token": metrics["average_latency_per_token"],
        "average_gen_time": metrics["average_gen_time"],
        "result_file": str(path),
    }


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    spec = load_json(args.spec)
    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_points = list(points(spec))
    rows = []
    failures = []

    metadata = {
        "spec": str(args.spec.resolve()), "platform": platform.platform(),
        "processor": platform.processor(), "python": sys.version,
        "command": " ".join(shlex.quote(x) for x in sys.argv),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    for index, (model, framework, block_size, input_tokens, output_tokens, batch_size) in enumerate(all_points, 1):
        tag = (f"{model['class']}_b{'auto' if block_size is None else block_size}"
               f"_i{input_tokens}_o{output_tokens}_n{batch_size}_{framework}")
        point_dir = raw_dir / tag
        prior = sorted(point_dir.glob("*_results.json"))
        print(f"[{index}/{len(all_points)}] {tag}", flush=True)
        if prior and not args.rerun:
            rows.append(row_from_result(prior[-1], model["class"], model["name"], framework, block_size))
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
                rows.append(row_from_result(result_files[-1], model["class"], model["name"], framework, block_size))
                write_csv(output_dir / "results.csv", rows)
        finally:
            Path(config_path).unlink(missing_ok=True)

    write_csv(output_dir / "results.csv", rows)
    (output_dir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    print(f"Wrote {len(rows)} results to {output_dir / 'results.csv'}")
    return 1 if failures else 0


def summarize(args):
    with args.results.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups = {}
    keys = ("model_class", "model_name", "framework", "input_tokens", "output_tokens", "batch_size")
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)

    recommendations = []
    for key, candidates in sorted(groups.items()):
        candidates.sort(key=lambda row: float(row["output_tps"]), reverse=True)
        best = candidates[0]
        runner_up = candidates[1] if len(candidates) > 1 else best
        margin = (float(best["output_tps"]) / float(runner_up["output_tps"]) - 1) * 100
        recommendations.append(dict(zip(keys, key)) | {
            "recommended_block_size": best["block_size"],
            "output_tps": best["output_tps"],
            "margin_over_runner_up_pct": round(margin, 2),
            "stable_winner": margin >= args.min_margin_pct,
        })

    output = args.output or args.results.with_name("recommendations.csv")
    fields = (*keys, "recommended_block_size", "output_tps", "margin_over_runner_up_pct", "stable_winner")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
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
