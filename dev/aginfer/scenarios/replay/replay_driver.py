"""Deterministic trace-replay driver (#231).

Replays a captured request trace (see daemon/trace_capture.py) against a
running stack — ours or baseline — through the daemon proxy, FORCING the
generated length per request (``max_tokens=output_len`` + ``ignore_eos``)
and ``temperature=0``.  Because the work is pinned, the two arms process
byte-identical requests, so any difference in serving latency is
attributable to the daemon alone.

Metrics (per request, then aggregated): TTFT, TPOT (inter-token), end-to-end
latency, throughput.  Two replay modes:

  * ``arrival`` (default) — open-loop: dispatch each request at its recorded
    arrival offset × ``--slowdown``.  Reproduces the real offered load and
    thus the KV-pressure profile the daemon reacts to.  This is the
    do-no-harm / characterization mode.
  * ``closed`` — ignore timestamps, keep ``--max-concurrency`` in flight.
    A saturating stress probe.

Usage:
    python replay_driver.py --trace trace.jsonl \
        --base-url http://127.0.0.1:9100/v1 --out metrics.json \
        [--mode arrival|closed] [--slowdown 1.0] [--max-concurrency 64] \
        [--limit N] [--label ours]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx


# ----------------------------------------------------------------- pure core


def build_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    """Turn a captured trace record into a deterministic replay request.

    Forces: temperature 0, exact length (max_tokens == output_len with
    ignore_eos), streaming (so we can time first-token + inter-token).
    program_id is injected top-level where the proxy reads it.  output_len
    is clamped to >=1 (max_tokens=0 is rejected; a 0-length capture replays
    as a single token, negligible and flagged in the row).
    """
    body = record.get("body") or {}
    out_len = int(record.get("output_len") or 0)
    payload: Dict[str, Any] = {
        "model": body.get("model", "default"),
        "messages": body.get("messages") or [],
        "temperature": 0.0,
        "max_tokens": max(1, out_len),
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": False},
    }
    pid = record.get("program_id")
    if pid is not None:
        payload["program_id"] = pid
    return payload


def _pct(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _summ(vals: List[float]) -> Dict[str, float]:
    s = sorted(v for v in vals if v == v)  # drop NaN
    if not s:
        return {"n": 0}
    return {
        "n": len(s),
        "mean": sum(s) / len(s),
        "p50": _pct(s, 0.50),
        "p90": _pct(s, 0.90),
        "p99": _pct(s, 0.99),
        "max": s[-1],
    }


def aggregate(rows: List[Dict[str, Any]], wall_s: float) -> Dict[str, Any]:
    """Pure aggregation of per-request rows into summary metrics.

    Each row: {ok, ttft_ms, e2e_ms, tpot_ms, n_out, want_out}.  TPOT is
    only defined for rows with n_out >= 2.  Throughput is total generated
    tokens over the measured wall-clock.
    """
    ok = [r for r in rows if r.get("ok")]
    errs = [r for r in rows if not r.get("ok")]
    ttft = [r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None]
    e2e = [r["e2e_ms"] for r in ok if r.get("e2e_ms") is not None]
    tpot = [r["tpot_ms"] for r in ok if r.get("tpot_ms") is not None]
    total_out = sum(int(r.get("n_out") or 0) for r in ok)
    len_match = sum(
        1 for r in ok if r.get("n_out") is not None and r.get("want_out") is not None
        and abs(int(r["n_out"]) - int(r["want_out"])) <= 1
    )
    return {
        "n_requests": len(rows),
        "n_ok": len(ok),
        "n_error": len(errs),
        "wall_s": round(wall_s, 3),
        "throughput_tok_s": round(total_out / wall_s, 1) if wall_s > 0 else 0.0,
        "total_out_tokens": total_out,
        "len_match_rate": round(len_match / len(ok), 4) if ok else 0.0,
        "ttft_ms": _summ(ttft),
        "tpot_ms": _summ(tpot),
        "e2e_ms": _summ(e2e),
    }


# -------------------------------------------------------------- replay engine


async def _one_request(
    cli: httpx.AsyncClient, url: str, record: Dict[str, Any]
) -> Dict[str, Any]:
    payload = build_payload(record)
    want_out = int(record.get("output_len") or 0)
    t0 = time.perf_counter()
    ttft_ms: Optional[float] = None
    last_tok_t: Optional[float] = None
    n_out = 0
    carry = b""
    try:
        async with cli.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                await resp.aread()
                return {"ok": False, "status": resp.status_code, "want_out": want_out}
            async for chunk in resp.aiter_bytes():
                now = time.perf_counter()
                buf = carry + chunk
                lines = buf.split(b"\n")
                carry = lines[-1]
                for line in lines[:-1]:
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    p = line[5:].strip()
                    if p == b"[DONE]" or not p:
                        continue
                    try:
                        obj = json.loads(p)
                    except Exception:
                        continue
                    for ch in obj.get("choices") or ():
                        if (ch.get("delta") or {}).get("content"):
                            if ttft_ms is None:
                                ttft_ms = (now - t0) * 1000.0
                            n_out += 1
                            last_tok_t = now
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "want_out": want_out}

    e2e_ms = (time.perf_counter() - t0) * 1000.0
    tpot_ms = None
    if ttft_ms is not None and last_tok_t is not None and n_out >= 2:
        decode_ms = (last_tok_t - t0) * 1000.0 - ttft_ms
        tpot_ms = decode_ms / (n_out - 1)
    return {
        "ok": True,
        "ttft_ms": ttft_ms,
        "e2e_ms": e2e_ms,
        "tpot_ms": tpot_ms,
        "n_out": n_out,
        "want_out": want_out,
    }


async def replay(
    records: List[Dict[str, Any]],
    *,
    base_url: str,
    mode: str,
    slowdown: float,
    max_concurrency: int,
) -> Tuple[List[Dict[str, Any]], float]:
    url = base_url.rstrip("/") + "/chat/completions"
    rows: List[Dict[str, Any]] = [None] * len(records)  # type: ignore
    sem = asyncio.Semaphore(max_concurrency)
    limits = httpx.Limits(max_connections=max_concurrency + 8)
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0),
        limits=limits,
    ) as cli:
        wall0 = time.perf_counter()

        async def run(i: int, rec: Dict[str, Any]) -> None:
            async with sem:
                rows[i] = await _one_request(cli, url, rec)

        tasks: List[Any] = []
        if mode == "arrival":
            base_t = records[0].get("t", 0.0) if records else 0.0
            for i, rec in enumerate(records):
                delay = (float(rec.get("t", 0.0)) - base_t) * slowdown
                wait = wall0 + delay - time.perf_counter()
                if wait > 0:
                    await asyncio.sleep(wait)
                tasks.append(asyncio.ensure_future(run(i, rec)))
            await asyncio.gather(*tasks)
        else:  # closed
            for i, rec in enumerate(records):
                tasks.append(asyncio.ensure_future(run(i, rec)))
            await asyncio.gather(*tasks)
        wall_s = time.perf_counter() - wall0
    return rows, wall_s


def load_trace(path: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
            if limit and len(out) >= limit:
                break
    out.sort(key=lambda r: float(r.get("t", 0.0)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:9100/v1")
    ap.add_argument("--mode", choices=["arrival", "closed"], default="arrival")
    ap.add_argument("--slowdown", type=float, default=1.0)
    ap.add_argument("--max-concurrency", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    records = load_trace(args.trace, args.limit or None)
    if not records:
        print("no records in trace")
        return 1
    rows, wall_s = asyncio.run(
        replay(
            records,
            base_url=args.base_url,
            mode=args.mode,
            slowdown=args.slowdown,
            max_concurrency=args.max_concurrency,
        )
    )
    metrics = aggregate(rows, wall_s)
    metrics["label"] = args.label
    metrics["mode"] = args.mode
    print(json.dumps(metrics, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"metrics": metrics, "rows": rows}, fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
