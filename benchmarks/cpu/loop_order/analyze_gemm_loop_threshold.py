#!/usr/bin/env python3
"""Infer whether ikj and kij have a reproducible square-GEMM crossover."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--minimum-effect", type=float, default=0.05)
    parser.add_argument("--minimum-win-rate", type=float, default=0.80)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260809)
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


def threshold_statement(rows: list[dict[str, object]]) -> str:
    strict = [
        (int(row["matrix_size"]), str(row["decision"]))
        for row in rows
        if row["decision"] in {"ikj", "kij"}
    ]
    winners = {decision for _, decision in strict}
    if len(winners) < 2:
        winner = next(iter(winners), "neither order")
        return (
            "No two-sided crossover was established: "
            f"the strict results only identify {winner}."
        )

    transitions = [
        index
        for index in range(1, len(strict))
        if strict[index - 1][1] != strict[index][1]
    ]
    if len(transitions) != 1:
        return (
            "No single size threshold was established because the strict winner "
            "changes direction more than once."
        )

    index = transitions[0]
    lower_size, lower_winner = strict[index - 1]
    upper_size, upper_winner = strict[index]
    return (
        f"Among strict decisions, a candidate crossover lies between N={lower_size} "
        f"and N={upper_size}: {lower_winner} wins below the interval and "
        f"{upper_winner} wins above it. "
        "This statement applies only to the sampled square FP32 GEMM sizes."
    )


def main() -> None:
    args = arguments()
    if not 0.0 <= args.minimum_effect < 1.0:
        raise ValueError("minimum effect must be in [0, 1)")
    if not 0.5 <= args.minimum_win_rate <= 1.0:
        raise ValueError("minimum win rate must be in [0.5, 1]")

    with args.trials.open(newline="") as stream:
        trials = list(csv.DictReader(stream))
    with args.summary.open(newline="") as stream:
        summaries = list(csv.DictReader(stream))

    paired: dict[int, dict[int, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in trials:
        order = row["loop_order"]
        if order in {"ikj", "kij"}:
            paired[int(row["matrix_size"])][int(row["round"])][order] = float(
                row["elapsed_ms"]
            )

    summary_by_key = {
        (int(row["matrix_size"]), row["loop_order"]): row for row in summaries
    }
    rows: list[dict[str, object]] = []
    for position, size in enumerate(sorted(paired)):
        pairs = [
            times
            for _, times in sorted(paired[size].items())
            if {"ikj", "kij"} <= times.keys()
        ]
        if not pairs:
            raise RuntimeError(f"no complete ikj/kij pairs for N={size}")
        ratios = [times["kij"] / times["ikj"] for times in pairs]
        median_speedup = statistics.median(ratios)
        ci_low, ci_high = bootstrap_median_ci(
            ratios, args.bootstrap_samples, args.seed + position
        )
        ikj_win_rate = sum(ratio > 1.0 for ratio in ratios) / len(ratios)
        ikj = summary_by_key[(size, "ikj")]
        kij = summary_by_key[(size, "kij")]
        ikj_p95 = float(ikj["p95_ms"])
        kij_p95 = float(kij["p95_ms"])

        if (
            median_speedup >= 1.0 + args.minimum_effect
            and ikj_win_rate >= args.minimum_win_rate
            and ci_low > 1.0
            and ikj_p95 <= kij_p95
        ):
            decision = "ikj"
        elif (
            median_speedup <= 1.0 / (1.0 + args.minimum_effect)
            and ikj_win_rate <= 1.0 - args.minimum_win_rate
            and ci_high < 1.0
            and kij_p95 <= ikj_p95
        ):
            decision = "kij"
        elif 1.0 / (1.0 + args.minimum_effect) < median_speedup < 1.0 + args.minimum_effect:
            decision = "tie"
        else:
            decision = "inconclusive"

        all_six_best = next(
            row["loop_order"]
            for row in summaries
            if int(row["matrix_size"]) == size and row["is_best"] == "true"
        )
        rows.append(
            {
                "matrix_size": size,
                "working_set_kib": 3 * size * size * 4 / 1024,
                "pairs": len(pairs),
                "all_six_best": all_six_best,
                "ikj_median_ms": float(ikj["median_ms"]),
                "kij_median_ms": float(kij["median_ms"]),
                "ikj_p95_ms": ikj_p95,
                "kij_p95_ms": kij_p95,
                "ikj_speedup": median_speedup,
                "speedup_ci_low": ci_low,
                "speedup_ci_high": ci_high,
                "ikj_win_rate": ikj_win_rate,
                "decision": decision,
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    threshold = threshold_statement(rows)
    counts = Counter(str(row["decision"]) for row in rows)
    report = [
        "# Square GEMM Loop-Order Threshold",
        "",
        f"- Sizes tested: `{len(rows)}`",
        f"- Paired rounds per size: `{rows[0]['pairs']}`",
        f"- Strict effect threshold: `{args.minimum_effect * 100:.1f}%`",
        f"- Minimum pair-win rate: `{args.minimum_win_rate * 100:.1f}%`",
        (
            "- Decisions: "
            + ", ".join(f"`{name}={count}`" for name, count in sorted(counts.items()))
        ),
        "",
        "## Threshold Result",
        "",
        threshold,
        "",
        "## Paired Results",
        "",
        "| N | A+B+C KiB | all-six fastest | ikj ms | kij ms | ikj speedup | 95% CI | ikj win rate | decision |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        report.append(
            f"| {row['matrix_size']} | {row['working_set_kib']:.1f} | "
            f"{row['all_six_best']} | "
            f"{row['ikj_median_ms']:.6f} | {row['kij_median_ms']:.6f} | "
            f"{row['ikj_speedup']:.3f}x | "
            f"{row['speedup_ci_low']:.3f}-{row['speedup_ci_high']:.3f} | "
            f"{row['ikj_win_rate']:.1%} | {row['decision']} |"
        )
    report.extend(
        [
            "",
            "## Guardrail",
            "",
            "An `ikj` speedup is paired `kij latency / ikj latency`; values above 1 favor ikj.",
            "Strict winners require the minimum median effect, pair-win rate, a bootstrap CI excluding 1, and non-regressing p95.",
            "This is unblocked, single-threaded, square FP32 GEMM; it does not establish an attention-kernel threshold.",
            "",
        ]
    )
    args.out_md.write_text("\n".join(report))
    print(f"Wrote threshold decisions to {args.out_csv}")
    print(f"Wrote threshold report to {args.out_md}")
    print(threshold)


if __name__ == "__main__":
    main()
