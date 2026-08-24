"""Add paired bootstrap confidence intervals to a deadkv_ab.py summary."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Callable


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_mean(
    values: list[float], *, samples: int, rng: random.Random
) -> dict[str, object]:
    draws = [
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(samples)
    ]
    return {
        "paired_values": values,
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "bootstrap_mean_95ci": [quantile(draws, 0.025), quantile(draws, 0.975)],
    }


def percent_change(before: float, after: float) -> float:
    return 100.0 * (before - after) / before


def available_values(
    pairs: list[dict], extractor: Callable[[dict], float]
) -> list[float]:
    values: list[float] = []
    for pair in pairs:
        try:
            value = float(extractor(pair))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    pairs = summary["aggregate"]["pairs"]
    if not pairs:
        raise SystemExit("summary has no paired trials")

    extractors: dict[str, Callable[[dict], float]] = {
        "dead_kv_reduction_pct": lambda pair: 100.0
        * pair["dead_kv_reduction_fraction"],
        "live_ttft_improvement_pct": lambda pair: 100.0
        * pair["anchor_ttft_improvement_fraction"],
        "all_ttft_p50_improvement_pct": lambda pair: 100.0
        * pair["all_request_ttft_p50_improvement_fraction"],
        "inference_throughput_change_pct": lambda pair: 100.0
        * pair["inference_throughput_change_fraction"],
        "pipeline_throughput_change_pct": lambda pair: 100.0
        * pair["pipeline_throughput_change_fraction"],
        "hbm_dead_auc_reduction_pct": lambda pair: percent_change(
            pair["baseline"]["dead_byte_seconds_auc"]["HBM"],
            pair["ours"]["dead_byte_seconds_auc"]["HBM"],
        ),
        "dram_dead_auc_reduction_pct": lambda pair: percent_change(
            pair["baseline"]["dead_byte_seconds_auc"]["DRAM"],
            pair["ours"]["dead_byte_seconds_auc"]["DRAM"],
        ),
        "peak_hbm_reduction_pct": lambda pair: percent_change(
            pair["baseline"]["peak_pool_used_bytes"]["HBM"],
            pair["ours"]["peak_pool_used_bytes"]["HBM"],
        ),
    }
    rng = random.Random(args.seed)
    analyses: dict[str, object] = {}
    skipped_metrics: list[str] = []
    for name, extractor in extractors.items():
        values = available_values(pairs, extractor)
        if values:
            analyses[name] = bootstrap_mean(values, samples=args.samples, rng=rng)
        else:
            skipped_metrics.append(name)

    prompt_tokens = summary["configuration"]["total_input_tokens"]
    cache_delta = available_values(
        pairs, lambda pair: pair["anchor_cached_tokens_delta"]
    )
    cache_metric = "live_cache_hit_gain_percentage_points"
    if isinstance(prompt_tokens, (int, float)) and prompt_tokens > 0 and cache_delta:
        analyses[cache_metric] = bootstrap_mean(
            [100.0 * value / prompt_tokens for value in cache_delta],
            samples=args.samples,
            rng=rng,
        )
    else:
        skipped_metrics.append(cache_metric)

    output = {
        "source_summary": str(args.summary),
        "pair_count": len(pairs),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "metrics": analyses,
        "skipped_metrics": skipped_metrics,
        "note": (
            "Intervals are non-parametric bootstrap intervals for the paired mean. "
            f"This analysis contains {len(pairs)} paired comparison(s); a result from "
            "one synthetic workload is not a production-wide performance claim."
        ),
    }
    destination = args.output or args.summary.with_name("paired_bootstrap.json")
    destination.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
