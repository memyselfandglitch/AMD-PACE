#!/usr/bin/env python3
"""Analyze paired KV-layout and decode-traversal trials."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


CANDIDATES = (
    "head_major_head_first",
    "block_major_head_first",
    "head_major_block_first",
    "block_major_block_first",
)

COMPARISONS = (
    ("layout_only_head_first", CANDIDATES[0], CANDIDATES[1]),
    ("traversal_only_head_major", CANDIDATES[0], CANDIDATES[2]),
    ("co_designed_vs_current", CANDIDATES[0], CANDIDATES[3]),
    ("traversal_under_block_major", CANDIDATES[1], CANDIDATES[3]),
    ("layout_under_block_first", CANDIDATES[2], CANDIDATES[3]),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trials", type=Path, nargs="+")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260816)
    parser.add_argument("--minimum-effect", type=float, default=0.05)
    parser.add_argument("--minimum-win-rate", type=float, default=0.80)
    parser.add_argument("--minimum-launch-wins", type=int, default=2)
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
        medians.append(
            statistics.median(generator.choice(values) for _ in values)
        )
    return percentile(medians, 0.025), percentile(medians, 0.975)


def compare(
    pairs: list[dict[str, float]],
    baseline: str,
    candidate: str,
    p95: dict[str, float],
    args: argparse.Namespace,
    seed: int,
) -> dict[str, float | str]:
    speedups = [pair[baseline] / pair[candidate] for pair in pairs]
    speedup = statistics.median(speedups)
    ci_low, ci_high = bootstrap_median_ci(
        speedups, args.bootstrap_samples, seed
    )
    win_rate = sum(value > 1.0 for value in speedups) / len(speedups)
    decision = "tie"
    if (
        speedup >= 1.0 + args.minimum_effect
        and win_rate >= args.minimum_win_rate
        and ci_low > 1.0
        and p95[candidate] <= p95[baseline]
    ):
        decision = candidate
    elif (
        speedup <= 1.0 / (1.0 + args.minimum_effect)
        and win_rate <= 1.0 - args.minimum_win_rate
        and ci_high < 1.0
        and p95[baseline] <= p95[candidate]
    ):
        decision = baseline
    return {
        "speedup": speedup,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "win_rate": win_rate,
        "decision": decision,
    }


def normalized_trial(
    row: dict[str, str], source_index: int
) -> dict[str, str]:
    result = dict(row)
    result.setdefault("shape_family", "default")
    result.setdefault("case_family", result["shape_family"])
    result.setdefault(
        "case_name",
        f"{result['shape']}_b{result['batch_size']}_s{result['seq_len']}_"
        f"bs{result['block_size']}",
    )
    batch = int(result["batch_size"])
    sequence = int(result["seq_len"])
    kv_heads = int(result["num_kv_heads"])
    head_dim = int(result["head_dim"])
    block_size = int(result["block_size"])
    blocks = math.ceil(sequence / block_size)
    kv_bytes_per_sequence = sequence * kv_heads * head_dim * 4
    allocated_per_sequence = blocks * block_size * kv_heads * head_dim * 4
    result.setdefault("target_kv_mib", str(kv_bytes_per_sequence * batch / (1 << 20)))
    result.setdefault("batch_semantics", "sequential_outer_loop")
    result.setdefault("kv_bytes_per_sequence", str(kv_bytes_per_sequence))
    result.setdefault("kv_bytes_per_call", str(kv_bytes_per_sequence * batch))
    result.setdefault(
        "allocated_kv_bytes_per_sequence", str(allocated_per_sequence)
    )
    result.setdefault(
        "allocated_kv_bytes_per_call", str(allocated_per_sequence * batch)
    )
    result.setdefault("blocks_per_sequence", str(blocks))
    result.setdefault("blocks_per_call", str(blocks * batch))
    result.setdefault("alignment_bytes", "unspecified")
    result.setdefault("order_policy", "independent_random_shuffle")
    result.setdefault("process_launch", f"source{source_index}")
    return result


def analyze(args: argparse.Namespace) -> list[dict[str, object]]:
    trials: list[dict[str, str]] = []
    for source_index, path in enumerate(args.trials):
        with path.open() as stream:
            trials.extend(
                normalized_trial(row, source_index)
                for row in csv.DictReader(stream)
            )
    if not trials:
        raise RuntimeError("trial CSV is empty")
    if not all(row["correct"] == "true" for row in trials):
        raise RuntimeError("at least one trial failed correctness")

    workload_fields = (
        "case_family",
        "case_name",
        "shape_family",
        "shape",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "batch_size",
        "seq_len",
        "block_size",
    )
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in trials:
        groups[
            tuple(
                row[field]
                for field in workload_fields
            )
        ].append(row)

    summaries: list[dict[str, object]] = []
    expected = set(CANDIDATES)
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            item[0][2],
            item[0][3],
            *(int(value) for value in item[0][4:]),
        ),
    )
    for group_index, (key, rows) in enumerate(ordered_groups):
        by_candidate: dict[str, list[float]] = defaultdict(list)
        paired: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
        launch_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            elapsed = float(row["elapsed_ms"])
            by_candidate[row["candidate"]].append(elapsed)
            paired[
                (row["process_launch"], row["data_seed"], row["round"])
            ][row["candidate"]] = elapsed
            launch_rows[row["process_launch"]].append(row)
        if any(set(pair) != expected for pair in paired.values()):
            raise RuntimeError(f"incomplete randomized quadruple for {key}")

        pair_rows = list(paired.values())
        medians = {
            name: statistics.median(by_candidate[name]) for name in CANDIDATES
        }
        p95 = {
            name: percentile(by_candidate[name], 0.95) for name in CANDIDATES
        }
        results = {}
        launch_results: dict[str, list[dict[str, float | str]]] = defaultdict(list)
        for comparison_index, (label, baseline, candidate) in enumerate(COMPARISONS):
            results[label] = compare(
                pair_rows,
                baseline,
                candidate,
                p95,
                args,
                args.bootstrap_seed + group_index * len(COMPARISONS) + comparison_index,
            )
            for launch_index, launch in enumerate(sorted(launch_rows)):
                rows_for_launch = launch_rows[launch]
                launch_pairs: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
                launch_candidates: dict[str, list[float]] = defaultdict(list)
                for launch_row in rows_for_launch:
                    elapsed = float(launch_row["elapsed_ms"])
                    launch_pairs[(launch_row["data_seed"], launch_row["round"])][
                        launch_row["candidate"]
                    ] = elapsed
                    launch_candidates[launch_row["candidate"]].append(elapsed)
                if any(set(pair) != expected for pair in launch_pairs.values()):
                    raise RuntimeError(
                        f"incomplete randomized quadruple for {key}, launch {launch}"
                    )
                launch_p95 = {
                    name: percentile(launch_candidates[name], 0.95)
                    for name in CANDIDATES
                }
                launch_results[label].append(
                    compare(
                        list(launch_pairs.values()),
                        baseline,
                        candidate,
                        launch_p95,
                        args,
                        args.bootstrap_seed
                        + 1_000_000
                        + group_index * len(COMPARISONS) * 10
                        + comparison_index * 10
                        + launch_index,
                    )
                )

        row: dict[str, object] = dict(zip(workload_fields, key))
        first = rows[0]
        metadata_fields = (
            "target_kv_mib",
            "batch_semantics",
            "kv_bytes_per_sequence",
            "kv_bytes_per_call",
            "allocated_kv_bytes_per_sequence",
            "allocated_kv_bytes_per_call",
            "blocks_per_sequence",
            "blocks_per_call",
            "alignment_bytes",
            "order_policy",
        )
        for field in metadata_fields:
            row[field] = first[field]
        launches = sorted(launch_rows)
        row["process_launches"] = ";".join(launches)
        row["independent_launches"] = len(launches)
        row["gqa_ratio"] = int(row["num_q_heads"]) // int(row["num_kv_heads"])
        row["paired_quadruples"] = len(pair_rows)
        for name in CANDIDATES:
            row[f"{name}_median_ms"] = medians[name]
            row[f"{name}_p95_ms"] = p95[name]
        for label, _, _ in COMPARISONS:
            result = results[label]
            row[f"{label}_speedup"] = result["speedup"]
            row[f"{label}_ci_low"] = result["ci_low"]
            row[f"{label}_ci_high"] = result["ci_high"]
            row[f"{label}_win_rate"] = result["win_rate"]
            row[f"{label}_decision"] = result["decision"]
            per_launch = launch_results[label]
            candidate_name = next(
                candidate for item_label, _, candidate in COMPARISONS
                if item_label == label
            )
            baseline_name = next(
                baseline for item_label, baseline, _ in COMPARISONS
                if item_label == label
            )
            candidate_wins = sum(
                launch["decision"] == candidate_name for launch in per_launch
            )
            baseline_wins = sum(
                launch["decision"] == baseline_name for launch in per_launch
            )
            has_repeatability = len(per_launch) >= args.minimum_launch_wins
            row[f"{label}_launch_speedups"] = ";".join(
                f"{float(launch['speedup']):.6f}" for launch in per_launch
            )
            row[f"{label}_launch_decisions"] = ";".join(
                str(launch["decision"]) for launch in per_launch
            )
            row[f"{label}_launch_candidate_wins"] = candidate_wins
            row[f"{label}_launch_baseline_wins"] = baseline_wins
            row[f"{label}_repeatable_candidate"] = (
                has_repeatability
                and result["decision"] == candidate_name
                and candidate_wins >= args.minimum_launch_wins
            )
            row[f"{label}_repeatable_baseline"] = (
                has_repeatability
                and result["decision"] == baseline_name
                and baseline_wins >= args.minimum_launch_wins
            )

        strict_candidates = []
        baseline_comparisons = {
            CANDIDATES[1]: "layout_only_head_first",
            CANDIDATES[2]: "traversal_only_head_major",
            CANDIDATES[3]: "co_designed_vs_current",
        }
        for candidate, label in baseline_comparisons.items():
            if (
                row[f"{label}_repeatable_candidate"]
                or (
                    len(launches) < args.minimum_launch_wins
                    and row[f"{label}_decision"] == candidate
                )
            ):
                strict_candidates.append(candidate)
        recommendation = CANDIDATES[0]
        if strict_candidates:
            recommendation = min(strict_candidates, key=medians.get)
        row["recommendation"] = recommendation
        summaries.append(row)
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path, args: argparse.Namespace, rows: list[dict[str, object]]
) -> None:
    order_policies = sorted({str(row["order_policy"]) for row in rows})
    batch_semantics = sorted({str(row["batch_semantics"]) for row in rows})
    co_design_wins = sum(
        row["co_designed_vs_current_decision"] == "block_major_block_first"
        for row in rows
    )
    current_wins = sum(
        row["co_designed_vs_current_decision"] == "head_major_head_first"
        for row in rows
    )
    repeatable_co_design_wins = sum(
        bool(row["co_designed_vs_current_repeatable_candidate"])
        for row in rows
    )
    repeatable_current_wins = sum(
        bool(row["co_designed_vs_current_repeatable_baseline"])
        for row in rows
    )
    recommendations = defaultdict(int)
    for row in rows:
        recommendations[str(row["recommendation"])] += 1

    family_shapes: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    family_regions: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        family_shapes[(str(row["shape_family"]), str(row["shape"]))].append(row)
        family_regions[
            (
                str(row["shape_family"]),
                str(row["seq_len"]),
                str(row["batch_size"]),
            )
        ].append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        stream.write("# Decode KV Layout and Traversal Summary\n\n")
        stream.write("## Experiment\n\n")
        stream.write("- Current baseline: head-major storage plus head-first traversal.\n")
        stream.write("- Layout-only control: block-major storage plus head-first traversal.\n")
        stream.write("- Traversal-only control: head-major storage plus block-first traversal.\n")
        stream.write("- Co-designed candidate: block-major storage plus block-first traversal.\n")
        stream.write(
            "- Each candidate performs the same fused BF16 decode: QK dot products, "
            "blockwise online softmax, and weighted V accumulation.\n"
        )
        if order_policies == ["latin_square_cycle4"]:
            stream.write(
                "- Candidate order uses randomized four-round Latin-square cycles, "
                "so every candidate occupies every timing position equally often. "
                "All outputs must pass correctness before timing.\n"
            )
        else:
            stream.write(
                "- Recorded candidate-order policies: `"
                + "`, `".join(order_policies)
                + "`. All outputs must pass correctness before timing.\n"
            )
        stream.write(
            "- Recorded batch semantics: `"
            + "`, `".join(batch_semantics)
            + "`. In `sequential_outer_loop`, batch elements contribute to total "
            "call traffic but are not interleaved as one simultaneous cache working "
            "set.\n"
        )
        stream.write(
            f"- Strict winner: >={args.minimum_effect:.0%} paired median effect, "
            f">={args.minimum_win_rate:.0%} pair wins, 95% CI excluding one, "
            "and non-regressing p95. A repeatable winner must also be strict in "
            f">={args.minimum_launch_wins} independent process launches when that "
            "many launches are available.\n\n"
        )
        stream.write("## Main Result\n\n")
        stream.write(f"- Workloads: `{len(rows)}`\n")
        stream.write(
            "- Strict block-major/block-first wins over current baseline: "
            f"`{co_design_wins}/{len(rows)}`\n"
        )
        stream.write(
            "- Strict current-baseline wins over block-major/block-first: "
            f"`{current_wins}/{len(rows)}`\n"
        )
        stream.write(
            "- Repeatable block-major/block-first wins: "
            f"`{repeatable_co_design_wins}/{len(rows)}`\n"
        )
        stream.write(
            "- Repeatable current-baseline wins: "
            f"`{repeatable_current_wins}/{len(rows)}`\n"
        )
        for candidate in CANDIDATES:
            stream.write(
                f"- Recommended `{candidate}`: `{recommendations[candidate]}` workloads\n"
            )

        byte_target_rows = [
            row for row in rows if row["case_family"] == "kv_byte_target"
        ]
        if byte_target_rows:
            stream.write("\n## KV-Byte Target Matrix\n\n")
            stream.write(
                "`call MiB` is total logical K+V traffic across the sequential batch "
                "loop. `sequence MiB` is the cache-relevant footprint of one sequence. "
                "Block count is reported separately because bytes and traversal "
                "iterations co-vary at fixed block size.\n\n"
            )
            stream.write(
                "| call MiB | geometries | sequence MiB range | blocks/call range | "
                "median co-design speedup | repeatable wins |\n"
            )
            stream.write("| ---: | ---: | ---: | ---: | ---: | ---: |\n")
            by_target: dict[float, list[dict[str, object]]] = defaultdict(list)
            for row in byte_target_rows:
                by_target[float(row["target_kv_mib"])].append(row)
            for target, group in sorted(by_target.items()):
                sequence_mib = [
                    int(row["kv_bytes_per_sequence"]) / (1 << 20)
                    for row in group
                ]
                blocks = [int(row["blocks_per_call"]) for row in group]
                speedups = [
                    float(row["co_designed_vs_current_speedup"]) for row in group
                ]
                wins = sum(
                    bool(row["co_designed_vs_current_repeatable_candidate"])
                    for row in group
                )
                stream.write(
                    f"| {target:g} | {len(group)} | "
                    f"{min(sequence_mib):g}-{max(sequence_mib):g} | "
                    f"{min(blocks)}-{max(blocks)} | "
                    f"{statistics.median(speedups):.3f}x | {wins}/{len(group)} |\n"
                )

        transition_rows = [
            row for row in rows if row["case_family"] == "mha2_transition"
        ]
        if transition_rows:
            stream.write("\n## MHA-2 Transition Control\n\n")
            stream.write(
                "This lane keeps all four configurations and directly retests the "
                "isolated head-major/block-first recommendation around sequence 12288.\n\n"
            )
            stream.write(
                "| seq | call MiB | blocks/sequence | HM/BF speedup | HM/BF launches | "
                "BM/BF speedup | BM/BF launches | recommendation |\n"
            )
            stream.write(
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
            )
            for row in sorted(transition_rows, key=lambda item: int(item["seq_len"])):
                stream.write(
                    f"| {row['seq_len']} | {float(row['target_kv_mib']):g} | "
                    f"{row['blocks_per_sequence']} | "
                    f"{float(row['traversal_only_head_major_speedup']):.3f}x | "
                    f"{row['traversal_only_head_major_launch_candidate_wins']}/"
                    f"{row['independent_launches']} | "
                    f"{float(row['co_designed_vs_current_speedup']):.3f}x | "
                    f"{row['co_designed_vs_current_launch_candidate_wins']}/"
                    f"{row['independent_launches']} | {row['recommendation']} |\n"
                )
        stream.write("\n## Controlled Shape Families\n\n")
        stream.write(
            "| family | shape | Q/KV | D | GQA ratio | workloads | "
            "median co-design speedup | strict wins |\n"
        )
        stream.write(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        )
        for (family, shape), group in sorted(family_shapes.items()):
            speedups = [
                float(row["co_designed_vs_current_speedup"]) for row in group
            ]
            wins = sum(
                row["co_designed_vs_current_decision"]
                == "block_major_block_first"
                for row in group
            )
            first = group[0]
            stream.write(
                f"| {family} | {shape} | {first['num_q_heads']}/{first['num_kv_heads']} | "
                f"{first['head_dim']} | {first['gqa_ratio']} | {len(group)} | "
                f"{statistics.median(speedups):.3f}x | {wins}/{len(group)} |\n"
            )

        stream.write("\n## Sequence And Batch Regions\n\n")
        stream.write(
            "| family | sequence | batch | shapes | median co-design speedup | "
            "strict wins |\n"
        )
        stream.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        for (family, sequence, batch), group in sorted(
            family_regions.items(),
            key=lambda item: (item[0][0], int(item[0][1]), int(item[0][2])),
        ):
            speedups = [
                float(row["co_designed_vs_current_speedup"]) for row in group
            ]
            wins = sum(
                row["co_designed_vs_current_decision"]
                == "block_major_block_first"
                for row in group
            )
            stream.write(
                f"| {family} | {sequence} | {batch} | {len(group)} | "
                f"{statistics.median(speedups):.3f}x | {wins}/{len(group)} |\n"
            )

        stream.write("\n## Co-design Win/Loss Maps\n\n")
        stream.write(
            "Cells show current-baseline latency divided by block-major/block-first "
            "latency. `+` is a strict co-design win, `-` is a strict current-baseline "
            "win, and `~` is a tie under the preregistered criterion.\n\n"
        )
        families = sorted({str(row["shape_family"]) for row in rows})
        for family in families:
            family_rows = [row for row in rows if row["shape_family"] == family]
            for block_size in sorted({int(row["block_size"]) for row in family_rows}):
                block_rows = [
                    row
                    for row in family_rows
                    if int(row["block_size"]) == block_size
                ]
                columns = sorted(
                    {
                        (int(row["seq_len"]), int(row["batch_size"]))
                        for row in block_rows
                    }
                )
                stream.write(f"### {family}, block size {block_size}\n\n")
                stream.write("| shape | " + " | ".join(
                    f"S{sequence}/B{batch}" for sequence, batch in columns
                ) + " |\n")
                stream.write("| --- | " + " | ".join("---:" for _ in columns) + " |\n")
                by_cell = {
                    (
                        str(row["shape"]),
                        int(row["seq_len"]),
                        int(row["batch_size"]),
                    ): row
                    for row in block_rows
                }
                for shape in sorted({str(row["shape"]) for row in block_rows}):
                    cells = []
                    for sequence, batch in columns:
                        row = by_cell[(shape, sequence, batch)]
                        decision = row["co_designed_vs_current_decision"]
                        marker = "~"
                        if decision == "block_major_block_first":
                            marker = "+"
                        elif decision == "head_major_head_first":
                            marker = "-"
                        cells.append(
                            f"{float(row['co_designed_vs_current_speedup']):.3f}x {marker}"
                        )
                    stream.write(f"| {shape} | " + " | ".join(cells) + " |\n")
                stream.write("\n")

        stream.write("\n## Individual Workloads\n\n")
        stream.write(
            "| family | shape | batch | seq | block | current ms | layout-only | traversal-only | "
            "co-designed | recommendation |\n"
        )
        stream.write(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n"
        )
        for row in rows:
            stream.write(
                f"| {row['shape_family']} | {row['shape']} | {row['batch_size']} | "
                f"{row['seq_len']} | {row['block_size']} | "
                f"{float(row['head_major_head_first_median_ms']):.4f} | "
                f"{float(row['layout_only_head_first_speedup']):.3f}x "
                f"[{row['layout_only_head_first_decision']}] | "
                f"{float(row['traversal_only_head_major_speedup']):.3f}x "
                f"[{row['traversal_only_head_major_decision']}] | "
                f"{float(row['co_designed_vs_current_speedup']):.3f}x "
                f"[{row['co_designed_vs_current_decision']}] | "
                f"{row['recommendation']} |\n"
            )
        stream.write("\n## Scope Guardrail\n\n")
        stream.write(
            "This is a single-threaded decode-mechanism benchmark. It mirrors PACE's "
            "GQA blockwise online-softmax dataflow and AVX-512 BF16 arithmetic, but it "
            "does not yet modify SlabPool, exercise its allocator's non-contiguous "
            "physical block mapping, use Split-K, or include OpenMP scheduling. A "
            "production prototype is justified only if a repeatable co-design signal "
            "survives this controlled test.\n"
        )


def main() -> None:
    args = arguments()
    rows = analyze(args)
    write_csv(args.summary, rows)
    write_report(args.report, args, rows)
    print(f"Wrote {len(rows)} workload summaries to {args.summary}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
