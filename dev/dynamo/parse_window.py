#!/usr/bin/env python3
"""Sum re-prefill (#new-token) / cache-hit (#cached-token) from the sglang backend
log over a [start_epoch, end_epoch] wall-clock window. Run inside the container.

Usage: python parse_window.py <logpath> <start_epoch> <end_epoch>
Prints one JSON line: {"new":..,"cached":..,"total":..,"cache_hit_pct":..,"peak_util":..}
"""
import re, sys, json
from datetime import datetime

def epoch(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()

logpath, start, end = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
rx = re.compile(
    r"(\d{4}-\d{2}-\d{2}T[\d:.]+Z).*#new-token: (\d+), #cached-token: (\d+), token usage: ([\d.]+)")
new = cached = 0
peak = 0.0
for line in open(logpath, errors="ignore"):
    line = re.sub(r"\x1b\[[0-9;]*m", "", line)
    m = rx.search(line)
    if not m:
        continue
    t = epoch(m.group(1))
    if start <= t <= end:
        new += int(m.group(2)); cached += int(m.group(3))
        peak = max(peak, float(m.group(4)))
tot = new + cached
print(json.dumps({"new": new, "cached": cached, "total": tot,
                  "cache_hit_pct": round(cached / tot * 100, 2) if tot else 0.0,
                  "peak_util": peak}))
