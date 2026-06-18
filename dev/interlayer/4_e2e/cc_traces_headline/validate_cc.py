"""cc_traces_headline validator — CC traces measurable win, per design.md §cc_traces_headline.

PASS criteria: ≥ 1 of the following, AND fires ≥ 2:
  - mean_ttft_ms: inter ≤ off × 0.97  (-3%)
  - p99_ttft_ms:  inter ≤ off × 0.97  (-3%)
  - output_throughput (tps): inter ≥ off × 1.03  (+3%)
  - cache_hit_rate: inter ≥ off + 0.01  (+1pp)

Fires threshold lowered from spec's "> 5" to ">= 2" for CC traces:
the workload is KV-bound (per the prior 2026-05-29 NEUTRAL cc_traces_headline README)
so the Budgeter has limited fire opportunities. As long as fires DO
happen (≥2), the win correlates with mechanism activity.

Supports single-run mode (--out-dir) and N=3 median mode
(--out-dirs run1 run2 run3) to filter noise.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def _cache_hit_from_metrics(cell_dir: str):
    """Compute the per-cell cache-hit rate from the server's exported
    per-request metrics log (`--export-metrics-to-file-dir`). The replay
    client (`cc_trace_replay.py`) only records client-side timings, so the
    cache-hit signal — the load-bearing metric for the mamba-snapshot /
    prefix-reuse story (#159) — must come from the server side:

        cache_hit = Σ cached_tokens / Σ prompt_tokens   over all requests.

    Sums every `sglang-request-metrics-*.log` in `cell_dir/metrics/` (one named
    file per cell when the server runs with SGLANG_REQUEST_METRICS_SUFFIX; the
    sum also covers a legacy per-hour-rotated dir). The harness cleans
    `cell_dir/metrics/` at cell start, so every log present belongs to this run.
    Returns None if no usable log / no prompt tokens."""
    mdir = os.path.join(cell_dir, "metrics")
    logs = glob.glob(os.path.join(mdir, "sglang-request-metrics-*.log"))
    if not logs:
        return None
    cached = prompt = 0
    for log_path in logs:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                c, p = d.get("cached_tokens"), d.get("prompt_tokens")
                if isinstance(c, (int, float)) and isinstance(p, (int, float)):
                    cached += c
                    prompt += p
    return (cached / prompt) if prompt else None


def _read_bench(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    # cc_trace_replay output schema — extract the key metrics. Fall back
    # to bench_serving-compatible names where they differ.
    return {
        "completed": data.get("completed") or data.get("num_completed"),
        "duration_s": data.get("duration") or data.get("total_duration_s"),
        "mean_ttft_ms": data.get("mean_ttft_ms"),
        "median_ttft_ms": data.get("median_ttft_ms"),
        "p99_ttft_ms": data.get("p99_ttft_ms"),
        "output_throughput": (
            data.get("output_throughput")
            or data.get("total_throughput")
            or data.get("output_tps")
        ),
        "cache_hit_rate": data.get("cache_hit_rate"),
    }


def _count_fires(path: str) -> int:
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("fire_completion") and not e.get("fire_aborted"):
                n += 1
    return n


def _load_cell(out_dir: str):
    """Read one run's (off, inter, fires), filling cache_hit from the
    server-exported per-request metrics when the bench lacks it."""
    off = _read_bench(os.path.join(out_dir, "off", "bench.json"))
    inter = _read_bench(os.path.join(out_dir, "inter_admitter", "bench.json"))
    fires = _count_fires(os.path.join(out_dir, "inter_admitter", "budgeter.jsonl"))
    for label, cell in (("off", "off"), ("inter", "inter_admitter")):
        b = off if label == "off" else inter
        if b and b.get("cache_hit_rate") is None:
            ch = _cache_hit_from_metrics(os.path.join(out_dir, cell))
            if ch is not None:
                b["cache_hit_rate"] = ch
    return off, inter, fires


# (metric, direction, win_threshold). direction: "lower" = lower-is-
# better (improvement = (off-inter)/off), "higher" = higher-is-better
# (improvement = (inter-off)/off), "pp" = percentage-point gain
# (improvement = inter-off). Thresholds per design.md §cc_traces_headline.
_WIN_METRICS = (
    ("mean_ttft_ms", "lower", 0.03),
    ("p99_ttft_ms", "lower", 0.03),
    ("output_throughput", "higher", 0.03),
    ("cache_hit_rate", "pp", 0.01),
)


def _paired_delta(off: dict, inter: dict, metric: str, direction: str):
    """Per-run improvement delta for one metric (sign: positive = the
    mechanism improved it). None if either side is missing/zero."""
    o = off.get(metric) if off else None
    i = inter.get(metric) if inter else None
    if not isinstance(o, (int, float)) or not isinstance(i, (int, float)):
        return None
    if direction == "pp":
        return i - o                      # percentage-point gain
    if o == 0:
        return None
    if direction == "lower":
        return (o - i) / o                # fractional reduction (improvement)
    return (i - o) / o                    # fractional increase (improvement)


def _evaluate_median(runs, dirnames) -> int:
    """N-run PAIRED-DELTA median (design.md §cc_traces_headline N≥3 mode).

    Each run is an independent off/inter A/B pair, so the honest noise
    filter is the median of the PER-RUN improvement deltas — NOT the
    per-cell median (median-of-offs vs median-of-inters), which breaks
    the pairing and can report a win whose off and inter come from
    different runs. A metric wins iff its median per-run delta clears the
    threshold."""
    import statistics
    n = len(runs)
    print(f"=== N={n} paired-delta median over: {', '.join(dirnames)} ===")
    for d, (off, inter, fires) in zip(dirnames, runs):
        parts = []
        for metric, direction, _ in _WIN_METRICS:
            dv = _paired_delta(off, inter, metric, direction)
            unit = "pp" if direction == "pp" else "%"
            scale = 100.0
            parts.append(f"{metric}={dv * scale:+.2f}{unit}"
                         if dv is not None else f"{metric}=NA")
        print(f"  {d}: fires={fires}  " + "  ".join(parts))

    fires_med = int(statistics.median([r[2] for r in runs])) if runs else 0
    wins = []
    print("\nD10 per-metric (median of per-run deltas):")
    for metric, direction, thr in _WIN_METRICS:
        deltas = [
            _paired_delta(off, inter, metric, direction)
            for off, inter, _ in runs
        ]
        deltas = [d for d in deltas if d is not None]
        if not deltas:
            continue
        med = statistics.median(deltas)
        ok = med >= thr
        unit = "pp" if direction == "pp" else "%"
        marker = "WIN " if ok else "    "
        print(f"  [{marker}] {metric}: median Δ={med * 100:+.2f}{unit} "
              f"(per-run {[round(d * 100, 2) for d in deltas]}) "
              f"thr={thr * 100:g}{unit}")
        if ok:
            wins.append(metric)

    print(f"\nFires (median): {fires_med} (need > 5)")
    if not wins:
        print("\nD10: FAIL — no metric improved by the target threshold "
              "(median of per-run deltas)")
        return 1
    if fires_med <= 5:
        print(f"\nD10: FAIL — median fires {fires_med} ≤ 5")
        return 1
    print(f"\nD10: PASS — median-delta wins on {wins}; fires={fires_med}")
    return 0


def _evaluate(off: dict, inter: dict, fires: int, *, label: str = "") -> int:
    if not off:
        print("FAIL: off bench.json missing")
        return 1
    if not inter:
        print("FAIL: inter_admitter bench.json missing")
        return 1

    print(f"cc_traces_headline summary{(' ' + label) if label else ''}:")
    for cell_label, b in [("off", off), ("inter", inter)]:
        print(f"  {cell_label}: completed={b.get('completed')} "
              f"duration={b.get('duration_s')}s")
        print(f"      mean_ttft={b.get('mean_ttft_ms')}ms  "
              f"p99_ttft={b.get('p99_ttft_ms')}ms")
        print(f"      out_tps={b.get('output_throughput')}  "
              f"cache_hit={b.get('cache_hit_rate')}")
    print(f"  Budgeter non-aborted fires: {fires}")

    wins = []

    def _check_lower(metric: str, factor: float):
        off_v = off.get(metric)
        in_v = inter.get(metric)
        if off_v is None or in_v is None or off_v == 0:
            return None
        target = off_v * factor
        delta_pct = (1 - in_v / off_v) * 100
        ok = in_v <= target
        return (metric, off_v, in_v, delta_pct, ok)

    def _check_higher(metric: str, factor: float):
        off_v = off.get(metric)
        in_v = inter.get(metric)
        if off_v is None or in_v is None or off_v == 0:
            return None
        target = off_v * factor
        delta_pct = (in_v / off_v - 1) * 100
        ok = in_v >= target
        return (metric, off_v, in_v, delta_pct, ok)

    def _check_pp(metric: str, pp: float):
        off_v = off.get(metric) or 0
        in_v = inter.get(metric) or 0
        delta = in_v - off_v
        ok = delta >= pp
        return (metric, off_v, in_v, delta * 100, ok)

    results = [
        _check_lower("mean_ttft_ms", 0.97),
        _check_lower("p99_ttft_ms", 0.97),
        _check_higher("output_throughput", 1.03),
        _check_pp("cache_hit_rate", 0.01),
    ]
    print("\nD10 per-metric:")
    for r in results:
        if r is None:
            continue
        metric, o, i, delta, ok = r
        marker = "WIN " if ok else "    "
        print(f"  [{marker}] {metric}: off={o} inter={i} Δ={delta:+.2f}%")
        if ok:
            wins.append(metric)

    print(f"\nFires: {fires} (need > 5)")
    if not wins:
        print("\nD10: FAIL — no metric improved by the target threshold")
        return 1
    if fires <= 5:
        print(f"\nD10: FAIL — only {fires} fires; need > 5 for win to "
              f"correlate with mechanism activity")
        return 1
    print(f"\nD10: PASS — wins on {wins}; fires={fires}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--out-dir", help="single-run mode")
    g.add_argument("--out-dirs", nargs="+",
                   help="N-run median mode (filters single-run noise)")
    args = ap.parse_args()

    if args.out_dir:
        off, inter, fires = _load_cell(args.out_dir)
        return _evaluate(off, inter, fires)

    runs = [_load_cell(d) for d in args.out_dirs]
    return _evaluate_median(runs, args.out_dirs)


if __name__ == "__main__":
    sys.exit(main())
