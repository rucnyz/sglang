"""Re-aggregate sweep results into results.csv from raw per-knob artifacts.

Handles both bench_serving (top-level keys) and bench_multiturn (nested under
'summary') output JSONL formats. Run after a sweep completes if the original
driver missed fields.

Usage:
  python dev/0/aggregate_results.py <sweep_dir>

Looks at <sweep_dir>/<knob>.{bench.json,bench.jsonl,metrics.txt,metrics_samples.jsonl}
and writes <sweep_dir>/results.csv (overwrites).
"""
import argparse, csv, json, os, re, statistics, sys
from pathlib import Path


def load_bench(path):
    """Return a flat dict of bench metrics from either format."""
    if not os.path.exists(path):
        return {}
    flat = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict):
            continue
        # bench_multiturn: {"timestamp": ..., "tag": ..., "summary": {...}}
        if 'summary' in d and isinstance(d['summary'], dict):
            flat.update(d['summary'])
        # bench_serving: top-level keys
        flat.update({k: v for k, v in d.items() if k != 'summary'})
    return flat


def get_throughput(b):
    return (b.get('input_throughput')
            or b.get('total_input_throughput')
            or b.get('total_throughput_input')
            or b.get('input_token_throughput')
            or '')


def get_output_throughput(b):
    return (b.get('output_throughput')
            or b.get('total_output_throughput')
            or b.get('total_throughput_output')
            or b.get('output_token_throughput')
            or '')


def get_ttft_mean_ms(b):
    """Return mean TTFT in milliseconds. bench_serving exports *_ttft_ms; bench_multiturn exports *_ttft in seconds."""
    if 'mean_ttft_ms' in b:
        return b['mean_ttft_ms']
    if 'avg_ttft_ms' in b:
        return b['avg_ttft_ms']
    if 'average_ttft' in b:
        return float(b['average_ttft']) * 1000.0
    return ''


def get_ttft_p99_ms(b):
    if 'p99_ttft_ms' in b:
        return b['p99_ttft_ms']
    if 'p99_ttft' in b:
        return float(b['p99_ttft']) * 1000.0
    return ''


def gauge(metrics_path, name):
    if not os.path.exists(metrics_path):
        return float('nan')
    pat = re.compile(rf'^{re.escape(name)}\b[^\s]*\s+([0-9.eE+-]+)$', re.M)
    m = pat.search(open(metrics_path).read())
    return float(m.group(1)) if m else float('nan')


def agg_samples(path, name, fn=max):
    if not os.path.exists(path):
        return float('nan')
    vals = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if name in d:
            vals.append(d[name])
    return fn(vals) if vals else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir")
    args = ap.parse_args()

    sd = Path(args.sweep_dir)
    if not sd.is_dir():
        sys.exit(f"not a directory: {sd}")

    # Find all knob tags from .server_info.json files (one per knob value)
    tags = sorted({p.stem.replace('.server_info', '') for p in sd.glob('*.server_info.json')})
    if not tags:
        # fall back to .metrics.txt
        tags = sorted({p.stem.replace('.metrics', '') for p in sd.glob('*.metrics.txt')})

    # Try numeric sort
    try:
        tags = sorted(tags, key=lambda x: float(x))
    except ValueError:
        pass

    out_csv = sd / 'results.csv'
    rows = []
    for tag in tags:
        bench_path = None
        for ext in ('jsonl', 'json'):
            p = sd / f"{tag}.bench.{ext}"
            if p.exists():
                bench_path = p
                break
        metrics_path = sd / f"{tag}.metrics.txt"
        samples_path = sd / f"{tag}.metrics_samples.jsonl"

        b = load_bench(str(bench_path)) if bench_path else {}

        # Duration: prefer bench's `duration`, then derive from total_requests / throughput
        duration = b.get('duration', b.get('total_duration', ''))
        if duration == '' and 'total_requests' in b and 'throughput' in b:
            try:
                duration = float(b['total_requests']) / float(b['throughput'])
            except (TypeError, ZeroDivisionError):
                pass

        # Cache hit rate: prefer the bench-reported value (workload-level, unambiguous);
        # fall back to the prometheus gauge.
        chr = b.get('cache_hit_rate')
        if chr is None:
            chr = gauge(str(metrics_path), 'sglang:cache_hit_rate')

        row = [
            tag,
            get_throughput(b),
            get_output_throughput(b),
            get_ttft_mean_ms(b),
            get_ttft_p99_ms(b),
            agg_samples(str(samples_path), 'token_usage', max),
            (statistics.mean([json.loads(l)['token_usage']
                              for l in open(samples_path)
                              if l.strip() and 'token_usage' in json.loads(l)])
             if samples_path.exists() else float('nan')),
            chr,
            agg_samples(str(samples_path), 'full_token_usage', max),
            agg_samples(str(samples_path), 'swa_token_usage', max),
            agg_samples(str(samples_path), 'mamba_usage', max),
            duration,
        ]
        rows.append(row)

    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['knob_value','throughput_input_tps','throughput_output_tps',
                    'mean_ttft_ms','p99_ttft_ms',
                    'token_usage_peak','token_usage_mean','cache_hit_rate',
                    'full_token_usage_peak','swa_token_usage_peak','mamba_usage_peak',
                    'duration_s'])
        for r in rows:
            w.writerow(r)

    print(f"wrote {out_csv} with {len(rows)} rows")
    for r in rows:
        print('  ', r)


if __name__ == '__main__':
    main()
