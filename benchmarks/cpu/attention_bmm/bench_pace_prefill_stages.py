#!/usr/bin/env python3
"""Measure real SlabPool prefill stages and the ceiling from packing reuse."""

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
QUERY_TILE = 64
BLOCK_SIZE = 64
EMPTY_SINKS = torch.tensor([])
PROFILE_FIELDS = (
    "timer_pair_cost_ns",
    "timer_empty_interval_ns",
    "dispatch_wall_ns",
    "stage_sum_ns",
    "cache_init_ns",
    "q_prepare_ns",
    "k_pack_ns",
    "qk_ns",
    "softmax_ns",
    "v_pack_ns",
    "pv_ns",
    "normalize_ns",
    "cache_init_pairs",
    "q_prepare_pairs",
    "k_pack_pairs",
    "qk_pairs",
    "softmax_pairs",
    "v_pack_pairs",
    "pv_pairs",
    "normalize_pairs",
    "timer_pairs",
    "prefill_work_items",
    "kv_blocks",
    "active_threads",
    "total_work_items",
    "omp_threads",
)
STAGES = (
    "cache_init",
    "q_prepare",
    "k_pack",
    "qk",
    "softmax",
    "v_pack",
    "pv",
    "normalize",
)


def integer_list(text: str) -> list[int]:
    values = [int(value) for value in text.split(",")]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shapes", default="slm,llm")
    parser.add_argument(
        "--query-lens", type=integer_list, default=integer_list("128,256,512,1024")
    )
    parser.add_argument(
        "--kv-lens", type=integer_list, default=integer_list("2048,8192")
    )
    parser.add_argument("--batch-sizes", type=integer_list, default=integer_list("1,4"))
    parser.add_argument("--data-seeds", type=integer_list, default=integer_list("11,29,47"))
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--order-seed", type=int, default=20260809)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--max-profile-overhead", type=float, default=0.02)
    parser.add_argument("--max-timer-bias", type=float, default=0.01)
    parser.add_argument("--minimum-ceiling-speedup", type=float, default=1.05)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_median_ci(
    values: list[float], samples: int, seed: int
) -> tuple[float, float]:
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
    seed: int,
):
    torch.manual_seed(seed)
    blocks_per_sequence = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    pool = torch.classes.pace.SlabPool(
        batch_size * blocks_per_sequence + 64,
        num_kv_heads,
        head_dim,
        BLOCK_SIZE,
    )
    sequence_ids = list(range(batch_size))
    for sequence_id in sequence_ids:
        pool.create_sequence(sequence_id, kv_len + BLOCK_SIZE)
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
        EMPTY_SINKS,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def corrected_stage(profile: dict[str, float], stage: str) -> float:
    measured = profile[f"{stage}_ns"]
    estimated_bias = (
        profile[f"{stage}_pairs"] * profile["timer_empty_interval_ns"]
    )
    return max(0.0, measured - estimated_bias)


def ideal_reuse_speedup(packing_fraction: float, query_tiles: int) -> float:
    return 1.0 / (1.0 - packing_fraction + packing_fraction / query_tiles)


def summarize(
    args: argparse.Namespace,
    latency_rows: list[dict[str, object]],
    profile_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    workload_fields = ("shape", "batch_size", "query_len", "kv_len")
    profiles: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    latency_pairs: dict[
        tuple[object, ...], dict[tuple[int, int], dict[str, float]]
    ] = defaultdict(lambda: defaultdict(dict))
    for row in profile_rows:
        profiles[tuple(row[field] for field in workload_fields)].append(row)
    for row in latency_rows:
        key = tuple(row[field] for field in workload_fields)
        pair = (int(row["data_seed"]), int(row["round"]))
        mode = "profiled" if bool(row["profile_enabled"]) else "unprofiled"
        latency_pairs[key][pair][mode] = float(row["elapsed_ms"])

    summaries = []
    for index, key in enumerate(sorted(profiles)):
        rows = profiles[key]
        query_tiles = int(rows[0]["query_tiles"])
        packing_fractions = [float(row["packing_fraction_corrected"]) for row in rows]
        ideal_speedups = [float(row["ideal_reuse_speedup"]) for row in rows]
        timer_biases = [float(row["estimated_timer_bias_fraction"]) for row in rows]
        complete_pairs = [
            pair for pair in latency_pairs[key].values() if len(pair) == 2
        ]
        overhead_ratios = [pair["profiled"] / pair["unprofiled"] for pair in complete_pairs]
        pack_ci = bootstrap_median_ci(
            packing_fractions, args.bootstrap_samples, args.order_seed + index * 31
        )
        ceiling_ci = bootstrap_median_ci(
            ideal_speedups, args.bootstrap_samples, args.order_seed + index * 31 + 1
        )
        overhead_ci = bootstrap_median_ci(
            overhead_ratios, args.bootstrap_samples, args.order_seed + index * 31 + 2
        )
        timer_bias_ci = bootstrap_median_ci(
            timer_biases, args.bootstrap_samples, args.order_seed + index * 31 + 3
        )
        overhead_median = statistics.median(overhead_ratios)
        timer_bias_median = statistics.median(timer_biases)
        instrumentation_valid = (
            overhead_ci[1] <= 1.0 + args.max_profile_overhead
            and timer_bias_ci[1] <= args.max_timer_bias
        )
        worth_synthetic = (
            instrumentation_valid and ceiling_ci[0] >= args.minimum_ceiling_speedup
        )
        stage_fractions = {
            f"{stage}_fraction": statistics.median(
                float(row[f"{stage}_fraction_corrected"]) for row in rows
            )
            for stage in STAGES
        }
        summaries.append(
            {
                **dict(zip(workload_fields, key)),
                "query_tiles": query_tiles,
                "pairs": len(complete_pairs),
                "packing_fraction_median": statistics.median(packing_fractions),
                "packing_fraction_ci_low": pack_ci[0],
                "packing_fraction_ci_high": pack_ci[1],
                "ideal_reuse_speedup_median": statistics.median(ideal_speedups),
                "ideal_reuse_speedup_ci_low": ceiling_ci[0],
                "ideal_reuse_speedup_ci_high": ceiling_ci[1],
                "profile_overhead_ratio_median": overhead_median,
                "profile_overhead_ci_low": overhead_ci[0],
                "profile_overhead_ci_high": overhead_ci[1],
                "estimated_timer_bias_fraction": timer_bias_median,
                "estimated_timer_bias_ci_low": timer_bias_ci[0],
                "estimated_timer_bias_ci_high": timer_bias_ci[1],
                "instrumentation_valid": instrumentation_valid,
                "worth_synthetic_reuse_test": worth_synthetic,
                **stage_fractions,
            }
        )
    return summaries


def write_report(path: Path, args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    valid = [row for row in rows if bool(row["instrumentation_valid"])]
    proceed = [row for row in rows if bool(row["worth_synthetic_reuse_test"])]
    with path.open("w") as stream:
        stream.write("# Real PACE Prefill Stage Profile\n\n")
        stream.write(f"- Workloads: `{len(rows)}`\n")
        stream.write("- Fixed PACE layout: `head_major`\n")
        stream.write("- Fixed KV block size / query tile: `64 / 64`\n")
        stream.write(f"- Instrumentation valid: `{len(valid)}/{len(rows)}`\n")
        stream.write(
            f"- Packing-reuse ceiling >= {args.minimum_ceiling_speedup:.2f}x: "
            f"`{len(proceed)}/{len(rows)}` valid workloads\n"
        )
        stream.write(
            "- Packing fraction uses calibrated timer-bias correction; raw stage "
            "times remain in the profile CSV.\n\n"
        )
        stream.write("## Decision Table\n\n")
        stream.write(
            "| shape | batch | query | tiles | KV | packing % (95% CI) | "
            "ideal ceiling (95% CI) | profiler overhead (95% CI) | "
            "valid | proceed |\n"
        )
        stream.write(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n"
        )
        for row in rows:
            stream.write(
                f"| {row['shape']} | {row['batch_size']} | {row['query_len']} | "
                f"{row['query_tiles']} | {row['kv_len']} | "
                f"{float(row['packing_fraction_median']):.1%} "
                f"({float(row['packing_fraction_ci_low']):.1%}-"
                f"{float(row['packing_fraction_ci_high']):.1%}) | "
                f"{float(row['ideal_reuse_speedup_median']):.3f}x "
                f"({float(row['ideal_reuse_speedup_ci_low']):.3f}-"
                f"{float(row['ideal_reuse_speedup_ci_high']):.3f}) | "
                f"{float(row['profile_overhead_ratio_median']):.3f}x "
                f"({float(row['profile_overhead_ci_low']):.3f}-"
                f"{float(row['profile_overhead_ci_high']):.3f}) | "
                f"{row['instrumentation_valid']} | "
                f"{row['worth_synthetic_reuse_test']} |\n"
            )
        stream.write("\n## Guardrail\n\n")
        stream.write(
            "Stage durations are CPU time summed across OpenMP work items. The ideal "
            "reuse speedup is an optimistic Amdahl ceiling, not a measured kernel "
            "speedup. A workload proceeds only when instrumentation overhead is within "
            "the configured limits and the ceiling's lower 95% bound reaches the "
            "pre-registered threshold.\n"
        )


def main() -> None:
    args = arguments()
    shape_names = args.shapes.split(",")
    if any(shape not in SHAPES for shape in shape_names):
        raise ValueError(f"unknown shape in {shape_names}")
    if any(query_len <= 64 for query_len in args.query_lens):
        raise ValueError("query lengths must exceed 64 to exercise PACE prefill")
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("invalid warmup/repeat count")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PACE_SLAB_PREFILL_Q_TILE"] = str(QUERY_TILE)
    os.environ["PACE_LOG_LEVEL"] = "none"
    latency_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []

    for shape in shape_names:
        num_q_heads, num_kv_heads, head_dim = SHAPES[shape]
        for batch_size in args.batch_sizes:
            for query_len in args.query_lens:
                query_tiles = (query_len + QUERY_TILE - 1) // QUERY_TILE
                for kv_len in args.kv_lens:
                    if query_len > kv_len:
                        continue
                    for data_seed in args.data_seeds:
                        pool, sequence_ids = make_pool(
                            batch_size, kv_len, num_kv_heads, head_dim, data_seed
                        )
                        torch.manual_seed(data_seed + 1)
                        query = torch.randn(
                            batch_size,
                            query_len,
                            num_q_heads,
                            head_dim,
                            dtype=torch.bfloat16,
                        )
                        pool.set_stage_profile(False)
                        reference = attention(
                            pool, sequence_ids, query, query_len, head_dim
                        )
                        for _ in range(args.warmups):
                            attention(pool, sequence_ids, query, query_len, head_dim)

                        # Warm the one-time timer calibration and profiled code path.
                        pool.set_stage_profile(True)
                        attention(pool, sequence_ids, query, query_len, head_dim)
                        if len(pool.get_stage_profile()) != len(PROFILE_FIELDS):
                            raise RuntimeError("PACE returned an unexpected profile schema")
                        pool.set_stage_profile(False)

                        for repeat in range(args.repeats):
                            modes = [False, True]
                            random.Random(
                                args.order_seed
                                + data_seed * 101
                                + batch_size * 1009
                                + query_len * 10007
                                + kv_len * 100003
                                + repeat * 1000003
                            ).shuffle(modes)
                            for position, enabled in enumerate(modes):
                                pool.set_stage_profile(enabled)
                                begin = time.perf_counter_ns()
                                output = attention(
                                    pool, sequence_ids, query, query_len, head_dim
                                )
                                elapsed_ms = (time.perf_counter_ns() - begin) / 1.0e6
                                correct = bool(
                                    torch.allclose(
                                        output.float(),
                                        reference.float(),
                                        atol=0.02,
                                        rtol=0.02,
                                    )
                                )
                                common = {
                                    "shape": shape,
                                    "num_q_heads": num_q_heads,
                                    "num_kv_heads": num_kv_heads,
                                    "head_dim": head_dim,
                                    "batch_size": batch_size,
                                    "query_len": query_len,
                                    "query_tiles": query_tiles,
                                    "kv_len": kv_len,
                                    "block_size": BLOCK_SIZE,
                                    "data_seed": data_seed,
                                    "round": repeat,
                                }
                                latency_rows.append(
                                    {
                                        **common,
                                        "profile_enabled": enabled,
                                        "order_position": position,
                                        "elapsed_ms": elapsed_ms,
                                        "correct": correct,
                                    }
                                )
                                if enabled:
                                    values = [float(value) for value in pool.get_stage_profile()]
                                    profile = dict(zip(PROFILE_FIELDS, values))
                                    corrected = {
                                        stage: corrected_stage(profile, stage)
                                        for stage in STAGES
                                    }
                                    corrected_sum = sum(corrected.values())
                                    packing_fraction = (
                                        (corrected["k_pack"] + corrected["v_pack"])
                                        / corrected_sum
                                        if corrected_sum
                                        else 0.0
                                    )
                                    timer_bias = (
                                        profile["timer_pairs"]
                                        * profile["timer_empty_interval_ns"]
                                    )
                                    estimated_instrumentation_cost = (
                                        profile["timer_pairs"]
                                        * profile["timer_pair_cost_ns"]
                                    )
                                    profile_rows.append(
                                        {
                                            **common,
                                            **profile,
                                            **{
                                                f"{stage}_ns_corrected": corrected[stage]
                                                for stage in STAGES
                                            },
                                            **{
                                                f"{stage}_fraction_corrected": (
                                                    corrected[stage] / corrected_sum
                                                    if corrected_sum
                                                    else 0.0
                                                )
                                                for stage in STAGES
                                            },
                                            "corrected_stage_sum_ns": corrected_sum,
                                            "packing_fraction_corrected": packing_fraction,
                                            "ideal_reuse_speedup": ideal_reuse_speedup(
                                                packing_fraction, query_tiles
                                            ),
                                            "estimated_timer_bias_fraction": (
                                                timer_bias / profile["stage_sum_ns"]
                                                if profile["stage_sum_ns"]
                                                else 0.0
                                            ),
                                            "estimated_instrumentation_cost_ns": (
                                                estimated_instrumentation_cost
                                            ),
                                            "profile_order_position": position,
                                            "profiled_elapsed_ms": elapsed_ms,
                                            "correct": correct,
                                        }
                                    )

    if not profile_rows:
        raise RuntimeError("no valid prefill workloads were selected")
    summaries = summarize(args, latency_rows, profile_rows)
    write_csv(args.out_dir / "pace_prefill_stage_latency_trials.csv", latency_rows)
    write_csv(args.out_dir / "pace_prefill_stage_profiles.csv", profile_rows)
    write_csv(args.out_dir / "pace_prefill_stage_summary.csv", summaries)
    write_report(args.out_dir / "pace_prefill_stage_report.md", args, summaries)
    if not all(bool(row["correct"]) for row in latency_rows):
        raise RuntimeError("at least one profiled call failed correctness validation")
    print(
        f"Wrote {len(latency_rows)} latency trials and {len(profile_rows)} profiles "
        f"to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
