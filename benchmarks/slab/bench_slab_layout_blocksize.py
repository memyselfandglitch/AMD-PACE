# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this file consist of AI-generated content
# ******************************************************************************

"""Microbenchmark SlabPool layout and block-size sensitivity.

This benchmark intentionally calls the C++ SlabPool binding directly instead of
running full model generation. That keeps the first experiment focused on the KV
cache layout, block size, and attention phase behavior Arun asked about.

Examples:
    python benchmarks/slab/bench_slab_layout_blocksize.py --quick

    SLAB_LAYOUT=head_major python benchmarks/slab/bench_slab_layout_blocksize.py \
        --layouts head_major --block-sizes 64 --seq-lens 2048 --phases decode

    perf stat -e LLC-loads,LLC-load-misses \
        python benchmarks/slab/bench_slab_layout_blocksize.py --quick
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pace  # noqa: F401 - registers torch.classes.pace.SlabPool
except (ModuleNotFoundError, OSError) as exc:
    raise SystemExit(
        "Could not import PACE SlabPool bindings. Build or install PACE first so "
        "`pace/lib/libpace_cpp.so` exists, then rerun this benchmark."
    ) from exc


@dataclass(frozen=True)
class Shape:
    name: str
    num_q_heads: int
    num_kv_heads: int
    head_dim: int


@dataclass(frozen=True)
class Case:
    phase: str
    layout: str
    block_size: int
    batch_size: int
    seq_len: int
    shape: Shape
    exit_fraction: float = 0.0


DEFAULT_SHAPES = (
    Shape("slm_gqa", num_q_heads=8, num_kv_heads=4, head_dim=64),
    Shape("llama_gqa", num_q_heads=32, num_kv_heads=8, head_dim=128),
    Shape("mha", num_q_heads=8, num_kv_heads=8, head_dim=64),
)


def _parse_ints(raw: str) -> list[int]:
    return [int(item) for item in raw.split(",") if item.strip()]


def _parse_floats(raw: str) -> list[float]:
    return [float(item) for item in raw.split(",") if item.strip()]


def _parse_shapes(raw: str) -> list[Shape]:
    """Parse shape specs like name:q_heads:kv_heads:head_dim."""
    if raw == "default":
        return list(DEFAULT_SHAPES)

    shapes: list[Shape] = []
    for item in raw.split(","):
        if not item.strip():
            continue
        parts = item.split(":")
        if len(parts) != 4:
            raise ValueError(
                f"Invalid shape '{item}'. Expected name:q_heads:kv_heads:head_dim."
            )
        name, q_heads, kv_heads, head_dim = parts
        shapes.append(Shape(name, int(q_heads), int(kv_heads), int(head_dim)))
    return shapes


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * pct)
    return ordered[idx]


def _synchronize() -> None:
    # CPU tensors do not need a device sync, but keeping this hook makes it
    # harder to accidentally under-measure if GPU cases are added later.
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _make_pool(case: Case, total_blocks: int):
    os.environ["SLAB_LAYOUT"] = case.layout
    return torch.classes.pace.SlabPool(
        total_blocks,
        case.shape.num_kv_heads,
        case.shape.head_dim,
        case.block_size,
    )


def _create_filled_pool(case: Case, prefill_len: int):
    """Create a SlabPool, register sequences, and prefill KV cache."""
    blocks_per_seq = math.ceil((prefill_len + 512) / case.block_size)
    total_blocks = case.batch_size * blocks_per_seq + 64
    pool = _make_pool(case, total_blocks)
    seq_ids = list(range(case.batch_size))
    max_seq_len = prefill_len + 512

    for seq_id in seq_ids:
        pool.create_sequence(seq_id, max_seq_len)

    keys = torch.randn(
        case.batch_size,
        prefill_len,
        case.shape.num_kv_heads,
        case.shape.head_dim,
        dtype=torch.bfloat16,
    )
    values = torch.randn_like(keys)
    pool.cache_update(seq_ids, keys, values, [])
    return pool, seq_ids


def _apply_premature_exit(pool, seq_ids: list[int], exit_fraction: float) -> list[int]:
    if exit_fraction <= 0.0:
        return seq_ids

    exit_count = int(len(seq_ids) * exit_fraction)
    if exit_count <= 0:
        return seq_ids

    exiting = seq_ids[-exit_count:]
    remaining = seq_ids[:-exit_count]
    for seq_id in exiting:
        pool.remove_sequence(seq_id)
    return remaining


def _run_prefill(case: Case, warmups: int, repeats: int) -> tuple[list[float], int, int]:
    blocks_per_seq = math.ceil((case.seq_len + 512) / case.block_size)
    total_blocks = case.batch_size * blocks_per_seq + 64
    seq_ids = list(range(case.batch_size))
    scale = 1.0 / math.sqrt(case.shape.head_dim)
    pool = _make_pool(case, total_blocks)
    for seq_id in seq_ids:
        pool.create_sequence(seq_id, case.seq_len + 512)

    keys = torch.randn(
        case.batch_size,
        case.seq_len,
        case.shape.num_kv_heads,
        case.shape.head_dim,
        dtype=torch.bfloat16,
    )
    values = torch.randn_like(keys)
    query = torch.randn(
        case.batch_size,
        case.seq_len,
        case.shape.num_q_heads,
        case.shape.head_dim,
        dtype=torch.bfloat16,
    )
    query_lens = [case.seq_len] * case.batch_size

    def reset() -> None:
        for seq_id in seq_ids:
            pool.truncate_sequence(seq_id, case.seq_len)

    def once() -> None:
        pool.cache_update(seq_ids, keys, values, [])
        pool.attention(
            seq_ids, query, query_lens, [], scale, 0, torch.tensor([])
        )

    for _ in range(warmups):
        once()
        reset()
    gc.collect()
    timings: list[float] = []
    for _ in range(repeats):
        _synchronize()
        start = time.perf_counter_ns()
        once()
        _synchronize()
        timings.append((time.perf_counter_ns() - start) / 1_000_000.0)
        reset()
    return timings, case.batch_size * case.seq_len, total_blocks


def _run_decode_like(
    case: Case,
    warmups: int,
    repeats: int,
    query_len: int,
) -> tuple[list[float], int, int]:
    pool, seq_ids = _create_filled_pool(case, prefill_len=case.seq_len)
    seq_ids = _apply_premature_exit(pool, seq_ids, case.exit_fraction)
    scale = 1.0 / math.sqrt(case.shape.head_dim)

    keys = torch.randn(
        len(seq_ids),
        query_len,
        case.shape.num_kv_heads,
        case.shape.head_dim,
        dtype=torch.bfloat16,
    )
    values = torch.randn_like(keys)
    query = torch.randn(
        len(seq_ids),
        query_len,
        case.shape.num_q_heads,
        case.shape.head_dim,
        dtype=torch.bfloat16,
    )
    query_lens = [query_len] * len(seq_ids)

    def once() -> None:
        pool.cache_update(seq_ids, keys, values, [])
        pool.attention(seq_ids, query, query_lens, [], scale, 0, torch.tensor([]))

    def reset() -> None:
        pool.truncate_sequence(seq_ids[0], query_len)
        for seq_id in seq_ids[1:]:
            pool.truncate_sequence(seq_id, query_len)

    for _ in range(warmups):
        once()
        reset()
    gc.collect()
    timings: list[float] = []
    for _ in range(repeats):
        _synchronize()
        start = time.perf_counter_ns()
        once()
        _synchronize()
        timings.append((time.perf_counter_ns() - start) / 1_000_000.0)
        reset()
    processed_tokens = len(seq_ids) * query_len
    total_blocks = pool.get_free_block_count() + sum(
        math.ceil(pool.get_sequence_length(seq_id) / case.block_size)
        for seq_id in seq_ids
    )
    return timings, processed_tokens, total_blocks


def run_case(case: Case, warmups: int, repeats: int) -> dict[str, object]:
    if case.phase == "prefill":
        timings, tokens, total_blocks = _run_prefill(case, warmups, repeats)
    elif case.phase == "decode":
        timings, tokens, total_blocks = _run_decode_like(case, warmups, repeats, 1)
    elif case.phase == "mtd":
        timings, tokens, total_blocks = _run_decode_like(case, warmups, repeats, 8)
    else:
        raise ValueError(f"Unsupported phase: {case.phase}")

    mean_ms = statistics.fmean(timings)
    return {
        "phase": case.phase,
        "layout": case.layout,
        "block_size": case.block_size,
        "batch_size": case.batch_size,
        "seq_len": case.seq_len,
        "shape": case.shape.name,
        "num_q_heads": case.shape.num_q_heads,
        "num_kv_heads": case.shape.num_kv_heads,
        "head_dim": case.shape.head_dim,
        "exit_fraction": case.exit_fraction,
        "warmups": warmups,
        "repeats": repeats,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(timings),
        "p95_ms": _percentile(timings, 0.95),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "tokens": tokens,
        "tokens_per_second": (tokens * 1000.0 / mean_ms) if mean_ms > 0 else 0.0,
        "total_blocks": total_blocks,
    }


def iter_cases(args: argparse.Namespace) -> Iterable[Case]:
    for phase in args.phases:
        for layout in args.layouts:
            for block_size in args.block_sizes:
                for batch_size in args.batch_sizes:
                    for seq_len in args.seq_lens:
                        for shape in args.shapes:
                            exit_fractions = (
                                args.exit_fractions
                                if phase in {"decode", "mtd"}
                                else [0.0]
                            )
                            for exit_fraction in exit_fractions:
                                yield Case(
                                    phase=phase,
                                    layout=layout,
                                    block_size=block_size,
                                    batch_size=batch_size,
                                    seq_len=seq_len,
                                    shape=shape,
                                    exit_fraction=exit_fraction,
                                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SlabPool layout and block-size sensitivity."
    )
    parser.add_argument(
        "--layouts",
        default="head_major,block_major",
        help="Comma-separated layouts: head_major,block_major.",
    )
    parser.add_argument(
        "--block-sizes",
        default="16,32,64,128,256",
        type=_parse_ints,
        help="Comma-separated SlabPool block sizes.",
    )
    parser.add_argument(
        "--seq-lens",
        default="128,512,2048,8192",
        type=_parse_ints,
        help="Comma-separated sequence lengths.",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,4,16",
        type=_parse_ints,
        help="Comma-separated request/batch counts.",
    )
    parser.add_argument(
        "--phases",
        default="prefill,decode,mtd",
        help="Comma-separated phases: prefill,decode,mtd.",
    )
    parser.add_argument(
        "--shapes",
        default="default",
        type=_parse_shapes,
        help="default or comma-separated name:q_heads:kv_heads:head_dim specs.",
    )
    parser.add_argument(
        "--exit-fractions",
        default="0.0,0.25,0.5",
        type=_parse_floats,
        help="Premature-exit fractions for decode/mtd phases.",
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--output",
        default="benchmarks/slab/results/slab_layout_blocksize.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small smoke benchmark suitable for local validation.",
    )
    args = parser.parse_args()

    args.layouts = [item.strip() for item in args.layouts.split(",") if item.strip()]
    args.phases = [item.strip() for item in args.phases.split(",") if item.strip()]

    if args.quick:
        args.block_sizes = [16, 64]
        args.seq_lens = [128, 512]
        args.batch_sizes = [1, 4]
        args.shapes = [DEFAULT_SHAPES[0], DEFAULT_SHAPES[1]]
        args.exit_fractions = [0.0, 0.5]
        args.warmups = 1
        args.repeats = 3

    return args


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "phase",
        "layout",
        "block_size",
        "batch_size",
        "seq_len",
        "shape",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "exit_fraction",
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
    ]

    cases = list(iter_cases(args))
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index, case in enumerate(cases, start=1):
            print(
                f"[{index}/{len(cases)}] {case.phase} layout={case.layout} "
                f"bs={case.block_size} batch={case.batch_size} seq={case.seq_len} "
                f"shape={case.shape.name} exit={case.exit_fraction}"
            )
            row = run_case(case, args.warmups, args.repeats)
            writer.writerow(row)
            f.flush()
            print(
                f"  mean={row['mean_ms']:.3f} ms "
                f"p95={row['p95_ms']:.3f} ms "
                f"tokens/s={row['tokens_per_second']:.1f}"
            )

    print(f"Wrote {len(cases)} rows to {output_path}")


if __name__ == "__main__":
    main()
