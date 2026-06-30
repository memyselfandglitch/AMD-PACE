#!/usr/bin/env python3
"""Focused SlabPool block-size benchmark.

This benchmark intentionally keeps the SlabPool layout fixed by default and
sweeps only block size across phase, sequence length, batch size, and attention
shape. The goal is to characterize whether PACE's cache-derived block-size
heuristic should become workload-aware.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Shape:
    name: str
    num_q_heads: int
    num_kv_heads: int
    head_dim: int


DEFAULT_SHAPES = (
    Shape("slm_gqa", num_q_heads=8, num_kv_heads=4, head_dim=64),
    Shape("llama_gqa", num_q_heads=32, num_kv_heads=8, head_dim=128),
    Shape("mha", num_q_heads=8, num_kv_heads=8, head_dim=64),
)

SHAPES_BY_NAME = {shape.name: shape for shape in DEFAULT_SHAPES}

FIELDNAMES = [
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
    "l2_bytes",
    "heuristic_block_size",
    "bytes_per_token",
    "block_bytes",
    "kv_history_bytes",
    "warmups",
    "repeats",
    "mean_ms",
    "median_ms",
    "p95_ms",
    "min_ms",
    "max_ms",
    "tokens",
    "tokens_per_second",
    "total_blocks",
    "omp_threads",
    "slab_schedule",
]


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_shapes(value: str) -> list[Shape]:
    if value.strip() == "default":
        return list(DEFAULT_SHAPES)

    shapes: list[Shape] = []
    for item in parse_str_list(value):
        if item in SHAPES_BY_NAME:
            shapes.append(SHAPES_BY_NAME[item])
            continue

        # Custom shape syntax: name:q_heads:kv_heads:head_dim
        parts = item.split(":")
        if len(parts) != 4:
            known = ", ".join(sorted(SHAPES_BY_NAME))
            raise ValueError(
                f"Unknown shape '{item}'. Use one of {{{known}}} or "
                "custom syntax name:q_heads:kv_heads:head_dim."
            )
        name, q_heads, kv_heads, head_dim = parts
        shapes.append(
            Shape(
                name=name,
                num_q_heads=int(q_heads),
                num_kv_heads=int(kv_heads),
                head_dim=int(head_dim),
            )
        )
    return shapes


def read_l2_bytes() -> int:
    path = Path("/sys/devices/system/cpu/cpu0/cache/index2/size")
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return 0

    if not text:
        return 0
    suffix = text[-1]
    number = int(text.rstrip("KkMm"))
    if suffix in "Kk":
        return number * 1024
    if suffix in "Mm":
        return number * 1024 * 1024
    return number


def heuristic_block_size(num_kv_heads: int, head_dim: int, l2_bytes: int) -> int:
    if l2_bytes <= 0:
        return 64

    # K + V, BF16. This matches pace.llm.attention.slab.cache.autotune_block_size.
    bytes_per_token = 2 * num_kv_heads * head_dim * 2
    target = l2_bytes // 4
    for block_size in (256, 128, 64, 32):
        if block_size * bytes_per_token <= target:
            return block_size
    return 32


def phase_query_len(phase: str, seq_len: int, mtd_query_len: int) -> int:
    if phase == "prefill":
        return seq_len
    if phase == "decode":
        return 1
    if phase == "mtd":
        return mtd_query_len
    raise ValueError(f"unknown phase: {phase}")


def active_batch_size(batch_size: int, exit_fraction: float) -> int:
    if exit_fraction <= 0:
        return batch_size
    return max(1, int(math.ceil(batch_size * (1.0 - exit_fraction))))


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    index = int(math.ceil((pct / 100.0) * len(sorted_values))) - 1
    index = max(0, min(index, len(sorted_values) - 1))
    return sorted_values[index]


def make_bf16(shape: Iterable[int]):
    import torch

    shape = tuple(int(dim) for dim in shape)
    try:
        return torch.empty(shape, dtype=torch.bfloat16).normal_()
    except RuntimeError:
        return torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)


def run_case(args, shape: Shape, phase: str, layout: str, block_size: int, batch_size: int, seq_len: int, exit_fraction: float, case_index: int) -> dict:
    import torch
    import pace  # noqa: F401  # Loads torch.classes.pace.SlabPool.
    from pace.llm.attention.slab.cache import create_slab_pool

    os.environ["SLAB_LAYOUT"] = layout
    torch.manual_seed(args.seed + case_index)

    query_len = phase_query_len(phase, seq_len, args.mtd_query_len)
    active_batch = active_batch_size(batch_size, exit_fraction)
    seq_ids = list(range(active_batch))
    total_blocks = active_batch * math.ceil(seq_len / block_size) + args.extra_blocks

    pool = create_slab_pool(
        total_blocks=total_blocks,
        num_kv_heads=shape.num_kv_heads,
        head_dim=shape.head_dim,
        block_size=block_size,
    )
    for seq_id in seq_ids:
        pool.create_sequence(seq_id, seq_len + query_len)

    keys = make_bf16((active_batch, seq_len, shape.num_kv_heads, shape.head_dim))
    values = make_bf16((active_batch, seq_len, shape.num_kv_heads, shape.head_dim))
    pool.cache_update(seq_ids, keys, values, [])
    del keys, values

    query = make_bf16((active_batch, query_len, shape.num_q_heads, shape.head_dim))
    sinks = torch.empty(0, dtype=torch.float32)
    scale = 1.0 / math.sqrt(shape.head_dim)

    def attention_once():
        return pool.attention(seq_ids, query, [], [], scale, 0, sinks)

    for _ in range(args.warmups):
        attention_once()

    timings_ms: list[float] = []
    for _ in range(args.repeats):
        start_ns = time.perf_counter_ns()
        attention_once()
        end_ns = time.perf_counter_ns()
        timings_ms.append((end_ns - start_ns) / 1_000_000.0)

    timings_sorted = sorted(timings_ms)
    tokens = active_batch * query_len
    median_ms = statistics.median(timings_ms)
    mean_ms = statistics.mean(timings_ms)

    bytes_per_token = 2 * shape.num_kv_heads * shape.head_dim * 2
    block_bytes = block_size * bytes_per_token
    kv_history_bytes = active_batch * seq_len * bytes_per_token
    l2_bytes = args.l2_bytes or read_l2_bytes()

    result = {
        "phase": phase,
        "layout": layout,
        "block_size": block_size,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "query_len": query_len,
        "active_batch_size": active_batch,
        "shape": shape.name,
        "num_q_heads": shape.num_q_heads,
        "num_kv_heads": shape.num_kv_heads,
        "head_dim": shape.head_dim,
        "exit_fraction": exit_fraction,
        "l2_bytes": l2_bytes,
        "heuristic_block_size": heuristic_block_size(
            shape.num_kv_heads, shape.head_dim, l2_bytes
        ),
        "bytes_per_token": bytes_per_token,
        "block_bytes": block_bytes,
        "kv_history_bytes": kv_history_bytes,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": percentile(timings_sorted, 95),
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
        "tokens": tokens,
        "tokens_per_second": tokens / (median_ms / 1000.0),
        "total_blocks": total_blocks,
        "omp_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "slab_schedule": os.environ.get("SLAB_SCHEDULE", "auto"),
    }

    del pool, query, sinks
    gc.collect()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--phases", default="prefill,decode,mtd")
    parser.add_argument("--layouts", default="head_major")
    parser.add_argument("--block-sizes", default="16,32,64,128,256")
    parser.add_argument("--batch-sizes", default="1,4,16")
    parser.add_argument("--seq-lens", default="128,512,2048,8192")
    parser.add_argument("--shapes", default="default")
    parser.add_argument("--exit-fractions", default="0.0,0.25,0.5,0.75")
    parser.add_argument("--mtd-query-len", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--extra-blocks", type=int, default=64)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--l2-bytes", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.torch_threads > 0:
        import torch

        torch.set_num_threads(args.torch_threads)

    phases = parse_str_list(args.phases)
    layouts = parse_str_list(args.layouts)
    block_sizes = parse_int_list(args.block_sizes)
    batch_sizes = parse_int_list(args.batch_sizes)
    seq_lens = parse_int_list(args.seq_lens)
    shapes = parse_shapes(args.shapes)
    exit_fractions = parse_float_list(args.exit_fractions)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    case_index = 0
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for phase in phases:
            phase_exits = [0.0] if phase == "prefill" else exit_fractions
            for layout in layouts:
                for block_size in block_sizes:
                    for batch_size in batch_sizes:
                        for seq_len in seq_lens:
                            for shape in shapes:
                                for exit_fraction in phase_exits:
                                    if args.verbose:
                                        print(
                                            "running",
                                            phase,
                                            layout,
                                            block_size,
                                            "batch",
                                            batch_size,
                                            "seq",
                                            seq_len,
                                            shape.name,
                                            "exit",
                                            exit_fraction,
                                            flush=True,
                                        )
                                    row = run_case(
                                        args,
                                        shape,
                                        phase,
                                        layout,
                                        block_size,
                                        batch_size,
                                        seq_len,
                                        exit_fraction,
                                        case_index,
                                    )
                                    writer.writerow(row)
                                    f.flush()
                                    case_index += 1

    print(f"Wrote {case_index} rows to {output_path}")


if __name__ == "__main__":
    main()
