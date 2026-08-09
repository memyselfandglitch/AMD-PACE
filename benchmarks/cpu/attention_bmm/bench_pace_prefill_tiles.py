#!/usr/bin/env python3
"""Benchmark real SlabPool prefill while varying its BRGeMM query tile."""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
import pace  # noqa: F401 - loads the PACE shared library


SHAPES = {
    "slm": (14, 2, 64),
    "llm": (28, 4, 128),
}


def integer_list(text: str) -> list[int]:
    values = [int(value) for value in text.split(",")]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shapes", default="slm,llm")
    parser.add_argument("--query-lens", type=integer_list, default=integer_list("128,512"))
    parser.add_argument(
        "--kv-lens", type=integer_list, default=integer_list("2048,8192,16384")
    )
    parser.add_argument("--batch-sizes", type=integer_list, default=integer_list("1,4"))
    parser.add_argument("--tiles", type=integer_list, default=integer_list("16,32,64,128"))
    parser.add_argument("--data-seeds", type=integer_list, default=integer_list("11,29,47"))
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--order-seed", type=int, default=20260809)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--profile-one-call", action="store_true")
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    generator = random.Random(seed)
    medians = []
    for _ in range(samples):
        sample = [generator.choice(values) for _ in values]
        medians.append(statistics.median(sample))
    return percentile(medians, 0.025), percentile(medians, 0.975)


def make_pool(
    batch_size: int,
    kv_len: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    seed: int,
):
    torch.manual_seed(seed)
    blocks_per_sequence = (kv_len + block_size - 1) // block_size
    pool = torch.classes.pace.SlabPool(
        batch_size * blocks_per_sequence + 64,
        num_kv_heads,
        head_dim,
        block_size,
    )
    sequence_ids = list(range(batch_size))
    for sequence_id in sequence_ids:
        pool.create_sequence(sequence_id, kv_len + block_size)
    keys = torch.randn(
        batch_size, kv_len, num_kv_heads, head_dim, dtype=torch.bfloat16
    )
    values = torch.randn_like(keys)
    pool.cache_update(sequence_ids, keys, values, [])
    return pool, sequence_ids


def attention(pool, sequence_ids, query, query_len: int, head_dim: int):
    return pool.attention(
        sequence_ids,
        query,
        [query_len] * len(sequence_ids),
        [],
        1.0 / math.sqrt(head_dim),
        0,
        torch.tensor([]),
    )


def summarize(args: argparse.Namespace, trials: list[dict[str, object]]):
    workload_fields = ("shape", "batch_size", "query_len", "kv_len")
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in trials:
        groups[tuple(row[field] for field in workload_fields)].append(row)

    summaries: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    for workload_index, (key, rows) in enumerate(sorted(groups.items())):
        by_tile: dict[int, list[dict[str, object]]] = defaultdict(list)
        by_pair: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
        for row in rows:
            tile = int(row["query_tile"])
            by_tile[tile].append(row)
            by_pair[(int(row["data_seed"]), int(row["round"]))][tile] = float(
                row["elapsed_ms"]
            )
        baseline_rows = by_tile[64]
        baseline_median = statistics.median(
            float(row["elapsed_ms"]) for row in baseline_rows
        )
        baseline_p95 = percentile(
            [float(row["elapsed_ms"]) for row in baseline_rows], 0.95
        )
        tile_medians = {
            tile: statistics.median(float(row["elapsed_ms"]) for row in tile_rows)
            for tile, tile_rows in by_tile.items()
        }
        best_tile = min(tile_medians, key=tile_medians.get)

        tile_summaries: dict[int, dict[str, object]] = {}
        for tile, tile_rows in sorted(by_tile.items()):
            times = [float(row["elapsed_ms"]) for row in tile_rows]
            ratios = [pair[64] / pair[tile] for pair in by_pair.values()]
            ci_low, ci_high = bootstrap_ci(
                ratios, args.bootstrap_samples, args.order_seed + workload_index * 17 + tile
            )
            speedup = statistics.median(ratios)
            p95 = percentile(times, 0.95)
            win_rate = sum(ratio > 1.0 for ratio in ratios) / len(ratios)
            strict_win = (
                tile != 64
                and speedup >= 1.05
                and win_rate >= 0.80
                and ci_low > 1.0
                and p95 <= baseline_p95
            )
            summary = {
                **dict(zip(workload_fields, key)),
                "query_tile": tile,
                "pairs": len(times),
                "median_ms": statistics.median(times),
                "p95_ms": p95,
                "speedup_vs_64": speedup,
                "speedup_ci_low": ci_low,
                "speedup_ci_high": ci_high,
                "win_rate_vs_64": win_rate,
                "all_correct": all(bool(row["correct"]) for row in tile_rows),
                "strict_win_vs_64": strict_win,
                "empirical_best": tile == best_tile,
            }
            summaries.append(summary)
            tile_summaries[tile] = summary

        best = tile_summaries[best_tile]
        recommended_tile = best_tile if bool(best["strict_win_vs_64"]) else 64
        decisions.append(
            {
                **dict(zip(workload_fields, key)),
                "baseline_tile": 64,
                "empirical_best_tile": best_tile,
                "recommended_tile": recommended_tile,
                "baseline_median_ms": baseline_median,
                "best_median_ms": best["median_ms"],
                "best_speedup_vs_64": best["speedup_vs_64"],
                "best_p95_ms": best["p95_ms"],
                "baseline_p95_ms": baseline_p95,
                "strict_improvement": recommended_tile != 64,
            }
        )
    return summaries, decisions


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, args: argparse.Namespace, decisions: list[dict[str, object]]):
    wins = [row for row in decisions if bool(row["strict_improvement"])]
    with path.open("w") as stream:
        stream.write("# Real PACE SlabPool Prefill Tile Summary\n\n")
        stream.write(f"- Workloads: `{len(decisions)}`\n")
        stream.write("- Current/default tile: `64`\n")
        stream.write(
            f"- Strictly improved workloads: `{len(wins)}/{len(decisions)}`\n"
        )
        stream.write(
            "- Strict win: >=5% paired median improvement, >=80% pair wins, "
            "95% bootstrap CI above 1, and p95 not worse.\n\n"
        )
        stream.write(
            "| shape | batch | query | KV | best | recommended | speedup | "
            "p95 baseline->best |\n"
        )
        stream.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        for row in decisions:
            stream.write(
                f"| {row['shape']} | {row['batch_size']} | {row['query_len']} | "
                f"{row['kv_len']} | {row['empirical_best_tile']} | "
                f"{row['recommended_tile']} | {float(row['best_speedup_vs_64']):.3f}x | "
                f"{float(row['baseline_p95_ms']):.3f}->{float(row['best_p95_ms']):.3f} ms |\n"
            )
        stream.write("\n## Guardrail\n\n")
        stream.write(
            "This measures the real BF16 SlabPool prefill path, including packing, "
            "online softmax, oneDNN BRGeMM, OpenMP dispatch, and output normalization. "
            "It is not an end-to-end model-generation benchmark.\n"
        )


def main() -> None:
    args = arguments()
    shape_names = args.shapes.split(",")
    if any(shape not in SHAPES for shape in shape_names):
        raise ValueError(f"unknown shape in {shape_names}")
    if 64 not in args.tiles or any(tile not in {16, 32, 64, 128} for tile in args.tiles):
        raise ValueError("tiles must include 64 and contain only 16,32,64,128")
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("invalid warmup/repeat count")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trials: list[dict[str, object]] = []

    for shape in shape_names:
        num_q_heads, num_kv_heads, head_dim = SHAPES[shape]
        for batch_size in args.batch_sizes:
            for query_len in args.query_lens:
                for kv_len in args.kv_lens:
                    if query_len > kv_len:
                        continue
                    for data_seed in args.data_seeds:
                        if args.profile_one_call:
                            os.environ["PACE_LOG_LEVEL"] = "profile"
                        pool, sequence_ids = make_pool(
                            batch_size,
                            kv_len,
                            num_kv_heads,
                            head_dim,
                            64,
                            data_seed,
                        )
                        os.environ["PACE_LOG_LEVEL"] = "none"
                        torch.manual_seed(data_seed + 1)
                        query = torch.randn(
                            batch_size,
                            query_len,
                            num_q_heads,
                            head_dim,
                            dtype=torch.bfloat16,
                        )
                        os.environ["PACE_SLAB_PREFILL_Q_TILE"] = "64"
                        reference = attention(
                            pool, sequence_ids, query, query_len, head_dim
                        )
                        tile_order = list(args.tiles)
                        random.Random(
                            args.order_seed
                            + data_seed
                            + batch_size * 101
                            + query_len * 1009
                            + kv_len * 10007
                            + head_dim * 100003
                        ).shuffle(tile_order)
                        for position, tile in enumerate(tile_order):
                            os.environ["PACE_SLAB_PREFILL_Q_TILE"] = str(tile)
                            for _ in range(args.warmups):
                                attention(pool, sequence_ids, query, query_len, head_dim)
                            if args.profile_one_call:
                                os.environ["PACE_LOG_LEVEL"] = "profile"
                                attention(pool, sequence_ids, query, query_len, head_dim)
                                os.environ["PACE_LOG_LEVEL"] = "none"
                            candidate = None
                            for repeat in range(args.repeats):
                                begin = time.perf_counter_ns()
                                candidate = attention(
                                    pool, sequence_ids, query, query_len, head_dim
                                )
                                elapsed_ms = (time.perf_counter_ns() - begin) / 1.0e6
                                error = float(
                                    (candidate.float() - reference.float()).abs().max()
                                )
                                trials.append(
                                    {
                                        "shape": shape,
                                        "num_q_heads": num_q_heads,
                                        "num_kv_heads": num_kv_heads,
                                        "head_dim": head_dim,
                                        "batch_size": batch_size,
                                        "query_len": query_len,
                                        "kv_len": kv_len,
                                        "query_tile": tile,
                                        "data_seed": data_seed,
                                        "round": repeat,
                                        "tile_order_position": position,
                                        "elapsed_ms": elapsed_ms,
                                        "max_abs_error": error,
                                        "correct": bool(
                                            torch.allclose(
                                                candidate.float(),
                                                reference.float(),
                                                atol=0.02,
                                                rtol=0.02,
                                            )
                                        ),
                                    }
                                )

    summaries, decisions = summarize(args, trials)
    write_csv(args.out_dir / "pace_prefill_tile_trials.csv", trials)
    write_csv(args.out_dir / "pace_prefill_tile_summary.csv", summaries)
    write_csv(args.out_dir / "pace_prefill_tile_decisions.csv", decisions)
    write_report(args.out_dir / "pace_prefill_tile_report.md", args, decisions)
    if not all(bool(row["correct"]) for row in trials):
        raise RuntimeError("at least one tile failed correctness validation")
    print(f"Wrote {len(trials)} trials and {len(decisions)} decisions to {args.out_dir}")


if __name__ == "__main__":
    main()
