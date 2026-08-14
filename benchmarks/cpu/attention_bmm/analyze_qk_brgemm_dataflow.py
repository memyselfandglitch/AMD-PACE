#!/usr/bin/env python3
"""Analyze randomized QK^T full-K, IKJ, and KIJ BRGeMM trials."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trials", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    parser.add_argument("--minimum-effect", type=float, default=0.05)
    parser.add_argument("--minimum-win-rate", type=float, default=0.80)
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


def comparison(
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


def analyze(args: argparse.Namespace) -> list[dict[str, object]]:
    with args.trials.open() as stream:
        trials = list(csv.DictReader(stream))
    if not trials:
        raise RuntimeError("trial CSV is empty")
    if not all(row["correct"] == "true" for row in trials):
        raise RuntimeError("at least one trial failed correctness")

    workload_fields = ("head_dim", "query_len", "kv_len", "k_chunk")
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in trials:
        groups[tuple(row[field] for field in workload_fields)].append(row)

    ordered_groups = sorted(
        groups.items(), key=lambda item: tuple(map(int, item[0]))
    )
    summaries: list[dict[str, object]] = []
    expected = {"pace_fullk", "ikj", "kij"}
    for index, (key, rows) in enumerate(ordered_groups):
        by_candidate: dict[str, list[float]] = defaultdict(list)
        paired: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
        for row in rows:
            elapsed = float(row["elapsed_ms"])
            by_candidate[row["dataflow"]].append(elapsed)
            paired[(row["data_seed"], row["round"])][row["dataflow"]] = elapsed
        if any(set(pair) != expected for pair in paired.values()):
            raise RuntimeError(f"incomplete randomized triple for workload {key}")

        pair_rows = list(paired.values())
        medians = {
            name: statistics.median(by_candidate[name]) for name in expected
        }
        p95 = {name: percentile(by_candidate[name], 0.95) for name in expected}
        pace_ikj = comparison(
            pair_rows,
            "pace_fullk",
            "ikj",
            p95,
            args,
            args.bootstrap_seed + index * 3,
        )
        pace_kij = comparison(
            pair_rows,
            "pace_fullk",
            "kij",
            p95,
            args,
            args.bootstrap_seed + index * 3 + 1,
        )
        ikj_kij = comparison(
            pair_rows,
            "ikj",
            "kij",
            p95,
            args,
            args.bootstrap_seed + index * 3 + 2,
        )

        recommendation = "pace_fullk"
        strict_improvements = [
            name
            for name, result in (("ikj", pace_ikj), ("kij", pace_kij))
            if result["decision"] == name
        ]
        if strict_improvements:
            recommendation = min(strict_improvements, key=medians.get)

        head_dim, query_len, kv_len, k_chunk = map(int, key)
        summaries.append(
            {
                "head_dim": head_dim,
                "query_len": query_len,
                "query_tiles": query_len // 64,
                "kv_len": kv_len,
                "kv_tiles": kv_len // 64,
                "k_chunk": k_chunk,
                "k_chunks": head_dim // k_chunk,
                "triples": len(pair_rows),
                "pace_median_ms": medians["pace_fullk"],
                "pace_p95_ms": p95["pace_fullk"],
                "ikj_median_ms": medians["ikj"],
                "ikj_p95_ms": p95["ikj"],
                "kij_median_ms": medians["kij"],
                "kij_p95_ms": p95["kij"],
                "pace_to_ikj_speedup": pace_ikj["speedup"],
                "pace_to_ikj_ci_low": pace_ikj["ci_low"],
                "pace_to_ikj_ci_high": pace_ikj["ci_high"],
                "ikj_win_rate": pace_ikj["win_rate"],
                "pace_vs_ikj": pace_ikj["decision"],
                "pace_to_kij_speedup": pace_kij["speedup"],
                "pace_to_kij_ci_low": pace_kij["ci_low"],
                "pace_to_kij_ci_high": pace_kij["ci_high"],
                "kij_win_rate": pace_kij["win_rate"],
                "pace_vs_kij": pace_kij["decision"],
                "ikj_to_kij_speedup": ikj_kij["speedup"],
                "ikj_to_kij_ci_low": ikj_kij["ci_low"],
                "ikj_to_kij_ci_high": ikj_kij["ci_high"],
                "kij_over_ikj_win_rate": ikj_kij["win_rate"],
                "ikj_vs_kij": ikj_kij["decision"],
                "recommendation": recommendation,
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path, args: argparse.Namespace, rows: list[dict[str, object]]
) -> None:
    ikj_wins = sum(row["pace_vs_ikj"] == "ikj" for row in rows)
    kij_wins = sum(row["pace_vs_kij"] == "kij" for row in rows)
    with path.open("w") as stream:
        stream.write("# QK^T BF16 BRGeMM Dataflow Comparison\n\n")
        stream.write("## Experiment\n\n")
        stream.write(
            "- `pace_fullk`: query tile -> KV tile, reducing the complete "
            "head dimension in one BRGeMM, as PACE does today.\n"
        )
        stream.write(
            "- `ikj`: query tile -> K chunk -> KV tile.\n"
        )
        stream.write(
            "- `kij`: K chunk -> query tile -> KV tile.\n"
        )
        stream.write(
            "- All candidates use identical BF16 Q and prepacked K inputs, "
            "64x64 output tiles, and the same oneDNN ukernel API.\n"
        )
        stream.write(
            "- Packing is outside timing; execution is single-threaded and "
            "candidate order is randomized.\n"
        )
        stream.write(
            f"- Strict winner: >={args.minimum_effect:.0%} paired median effect, "
            f">={args.minimum_win_rate:.0%} pair wins, 95% CI excluding one, "
            "and non-regressing p95.\n\n"
        )
        stream.write("## Result\n\n")
        stream.write(f"- Workloads: `{len(rows)}`\n")
        stream.write(f"- Strict IKJ improvements over PACE: `{ikj_wins}`\n")
        stream.write(f"- Strict KIJ improvements over PACE: `{kij_wins}`\n")
        stream.write(
            "- Proceed to a PACE prototype only if IKJ or KIJ strictly beats "
            "the full-K baseline in at least two workloads.\n\n"
        )
        stream.write(
            "| N | M | K | K chunk | PACE ms | IKJ ms | KIJ ms | "
            "PACE/IKJ | PACE/KIJ | IKJ/KIJ | recommendation |\n"
        )
        stream.write(
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | --- |\n"
        )
        for row in rows:
            stream.write(
                f"| {row['kv_len']} | {row['query_len']} | {row['head_dim']} | "
                f"{row['k_chunk']} | {float(row['pace_median_ms']):.4f} | "
                f"{float(row['ikj_median_ms']):.4f} | "
                f"{float(row['kij_median_ms']):.4f} | "
                f"{float(row['pace_to_ikj_speedup']):.3f}x "
                f"[{row['pace_vs_ikj']}] | "
                f"{float(row['pace_to_kij_speedup']):.3f}x "
                f"[{row['pace_vs_kij']}] | "
                f"{float(row['ikj_to_kij_speedup']):.3f}x "
                f"[{row['ikj_vs_kij']}] | "
                f"{row['recommendation']} |\n"
            )
        stream.write("\n## Scope Guardrail\n\n")
        stream.write(
            "The split-K candidates are faithful tiled translations of scalar "
            "IKJ/KIJ, but splitting K adds BRGeMM invocations relative to PACE. "
            "That overhead is intentionally included because a useful ordering "
            "must improve the existing full-K implementation, not only beat "
            "another split-K candidate. This benchmark excludes softmax, P*V, "
            "physical SlabPool layout, GQA batching, and OpenMP scheduling.\n"
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
