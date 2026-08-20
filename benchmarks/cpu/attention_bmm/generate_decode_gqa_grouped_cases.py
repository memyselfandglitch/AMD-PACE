#!/usr/bin/env python3
"""Generate fixed-geometry GQA decode cases for grouped-query evaluation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


BYTES_PER_BF16_KV_ELEMENT = 4  # One BF16 key plus one BF16 value.


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", default="32,16,8,4,2,1")
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--seq-lens", default="512,2048,8192,32768,65536")
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    return parser.parse_args()


def integers(text: str) -> list[int]:
    values = [int(value) for value in text.split(",")]
    if not values or any(value <= 0 for value in values):
        raise ValueError("lists must contain positive integers")
    return values


def make_cases(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.num_q_heads <= 0:
        raise ValueError("num_q_heads must be positive")
    if args.head_dim not in {64, 128, 256}:
        raise ValueError("head_dim must be 64, 128, or 256")
    if args.block_size <= 0 or args.block_size > 256 or args.block_size % 16:
        raise ValueError("block_size must be a multiple of 16 no larger than 256")

    rows: list[dict[str, object]] = []
    for seq_len in integers(args.seq_lens):
        for batch_size in integers(args.batch_sizes):
            for num_kv_heads in integers(args.kv_heads):
                if args.num_q_heads % num_kv_heads:
                    raise ValueError(
                        "num_q_heads must be divisible by every KV-head count"
                    )
                target_bytes = (
                    batch_size
                    * seq_len
                    * num_kv_heads
                    * args.head_dim
                    * BYTES_PER_BF16_KV_ELEMENT
                )
                target_mib = target_bytes / (1 << 20)
                ratio = args.num_q_heads // num_kv_heads
                rows.append(
                    {
                        "case_family": "gqa_grouped",
                        "case_name": (
                            f"b{batch_size}_q{args.num_q_heads}_kv{num_kv_heads}_"
                            f"r{ratio}_s{seq_len}"
                        ),
                        "shape": f"gqa_r{ratio}_q{args.num_q_heads}_kv{num_kv_heads}",
                        "num_q_heads": args.num_q_heads,
                        "num_kv_heads": num_kv_heads,
                        "head_dim": args.head_dim,
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "block_size": args.block_size,
                        "target_kv_mib": target_mib,
                    }
                )
    return rows


def main() -> None:
    args = arguments()
    rows = make_cases(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    ratios = sorted({args.num_q_heads // int(row["num_kv_heads"]) for row in rows})
    print(
        f"Wrote {len(rows)} GQA cases to {args.out}; "
        f"GQA ratios={','.join(map(str, ratios))}"
    )


if __name__ == "__main__":
    main()
