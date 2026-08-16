#!/usr/bin/env python3
"""Generate byte-matched decode cases plus the MHA-2 transition control lane."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


MIB = 1 << 20
BYTES_PER_BF16_KV_ELEMENT = 4  # One BF16 key and one BF16 value.


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--target-mib", default="8,16,24,32,48,64,96,128"
    )
    parser.add_argument("--kv-heads", default="2,4,8,16")
    parser.add_argument("--batch-sizes", default="1,4")
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument(
        "--transition-seq-lens", default="6144,8192,10240,12288,14336,16384"
    )
    return parser.parse_args()


def integers(text: str) -> list[int]:
    values = [int(value) for value in text.split(",")]
    if not values or any(value <= 0 for value in values):
        raise ValueError("lists must contain positive integers")
    return values


def make_cases(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    targets = integers(args.target_mib)
    kv_heads_values = integers(args.kv_heads)
    batches = integers(args.batch_sizes)

    for target_mib in targets:
        target_bytes = target_mib * MIB
        for batch_size in batches:
            for num_kv_heads in kv_heads_values:
                bytes_per_token = (
                    batch_size
                    * num_kv_heads
                    * args.head_dim
                    * BYTES_PER_BF16_KV_ELEMENT
                )
                if target_bytes % bytes_per_token:
                    raise ValueError(
                        f"{target_mib} MiB cannot be represented exactly by "
                        f"batch={batch_size}, KV heads={num_kv_heads}"
                    )
                seq_len = target_bytes // bytes_per_token
                rows.append(
                    {
                        "case_family": "kv_byte_target",
                        "case_name": (
                            f"target{target_mib}mib_b{batch_size}_"
                            f"h{num_kv_heads}_s{seq_len}"
                        ),
                        "shape": f"mha{num_kv_heads}",
                        "num_q_heads": num_kv_heads,
                        "num_kv_heads": num_kv_heads,
                        "head_dim": args.head_dim,
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "block_size": args.block_size,
                        "target_kv_mib": target_mib,
                    }
                )

    for seq_len in integers(args.transition_seq_lens):
        batch_size = 4
        num_kv_heads = 2
        target_bytes = (
            batch_size
            * seq_len
            * num_kv_heads
            * args.head_dim
            * BYTES_PER_BF16_KV_ELEMENT
        )
        rows.append(
            {
                "case_family": "mha2_transition",
                "case_name": f"mha2_b4_s{seq_len}",
                "shape": "mha2",
                "num_q_heads": num_kv_heads,
                "num_kv_heads": num_kv_heads,
                "head_dim": args.head_dim,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "block_size": args.block_size,
                "target_kv_mib": target_bytes / MIB,
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
    byte_cases = sum(row["case_family"] == "kv_byte_target" for row in rows)
    transition_cases = len(rows) - byte_cases
    print(
        f"Wrote {len(rows)} cases to {args.out} "
        f"({byte_cases} byte-target, {transition_cases} transition)"
    )


if __name__ == "__main__":
    main()
