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

    FAIRNESS NOTE (audit C1): keeping program_id means the daemon's FULL
    machinery runs on the a3 arm — including admission, which may PAUSE a
    replayed request (its TTFT then includes gate-park time that a3_kvoff,
    admission-off, never sees).  This is deliberate: the do-no-harm question
    is the daemon's *whole* latency footprint.  In open-loop `arrival` mode
    it is CONSERVATIVE for a3 — admission's cost is counted but its
    open-loop-invisible benefit (back-pressure → the next arrival backing
    off) is not; that benefit shows up in closed-loop `session` mode.  So
    arrival-mode is a pessimistic do-no-harm bound for a3, session-mode the
    realistic one.  (To isolate pure serving latency, run an admission-off
    variant; see README.)
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


def build_sessions(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group a trace into closed-loop sessions (pure).

    Records are grouped by ``program_id`` (program_id=None → each its own
    singleton session), ordered by arrival ``t``.  For each step we derive
    the **tool-think gap** before the NEXT request from the captured timing:

        gap_after_N = max(0, t_{N+1} - (t_N + ref_e2e_N))

    i.e. the real wall time the agent spent between request N's response
    landing and issuing request N+1 — the tool/bash/reasoning the daemon
    cannot speed up.  Replaying steps as ``dispatch N → await → sleep
    gap_N → dispatch N+1`` reproduces the closed loop: faster serving makes
    the next request arrive sooner, so the session finishes sooner.

    Each session: {program_id, start_t, steps:[{record, gap_after}]}.
    """
    by_pid: Dict[Any, List[Dict[str, Any]]] = {}
    singletons: List[List[Dict[str, Any]]] = []
    for r in records:
        pid = r.get("program_id")
        if pid is None:
            singletons.append([r])
        else:
            by_pid.setdefault(pid, []).append(r)

    groups: List[List[Dict[str, Any]]] = [
        sorted(recs, key=lambda r: float(r.get("t", 0.0))) for recs in by_pid.values()
    ] + singletons

    out: List[Dict[str, Any]] = []
    for recs in groups:
        steps: List[Dict[str, Any]] = []
        for i, r in enumerate(recs):
            gap = 0.0
            if i + 1 < len(recs):
                t_n = float(r.get("t", 0.0))
                e2e_n = float(r.get("ref_e2e_ms", 0.0)) / 1000.0
                t_next = float(recs[i + 1].get("t", 0.0))
                gap = max(0.0, t_next - (t_n + e2e_n))
            steps.append({"record": r, "gap_after": gap})
        out.append(
            {
                "program_id": recs[0].get("program_id"),
                "start_t": float(recs[0].get("t", 0.0)),
                "steps": steps,
            }
        )
    out.sort(key=lambda s: s["start_t"])
    return out


def aggregate_sessions(
    session_rows: List[Dict[str, Any]], makespan_s: float
) -> Dict[str, Any]:
    """Pure: per-session completion times → end-to-end summary.

    makespan_s (total wall to drain all sessions) is THE closed-loop
    headline — it captures the feedback open-loop replay cannot.
    """
    e2e = [r["session_e2e_s"] for r in session_rows if r.get("session_e2e_s") is not None]
    return {
        "n_sessions": len(session_rows),
        "makespan_s": round(makespan_s, 3),
        "session_e2e_s": _summ(e2e),
        "total_steps": sum(int(r.get("n_steps") or 0) for r in session_rows),
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
            # m3: flush a final un-terminated data line (stream ended without
            # a trailing newline) so the last token isn't silently dropped.
            tail = carry.strip()
            if tail.startswith(b"data:"):
                p = tail[5:].strip()
                if p and p != b"[DONE]":
                    try:
                        obj = json.loads(p)
                        for ch in obj.get("choices") or ():
                            if (ch.get("delta") or {}).get("content"):
                                if ttft_ms is None:
                                    ttft_ms = (time.perf_counter() - t0) * 1000.0
                                n_out += 1
                                last_tok_t = time.perf_counter()
                    except Exception:
                        pass
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


class _InflightGauge:
    """Tracks concurrent in-flight requests + whether the concurrency cap
    was ever hit (M1: a saturated cap silently reshapes the offered-load /
    KV-pressure profile the replay is meant to reproduce)."""

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.cur = 0
        self.peak = 0
        self.saturated = False

    def enter(self) -> None:
        self.cur += 1
        if self.cur > self.peak:
            self.peak = self.cur
        if self.cur >= self.cap:
            self.saturated = True

    def exit(self) -> None:
        self.cur -= 1


async def _request_with_deadline(
    cli: httpx.AsyncClient, url: str, rec: Dict[str, Any], deadline_s: float
) -> Dict[str, Any]:
    """_one_request with a wall deadline (audit C-1).

    The a3 arm runs admission ON, so a replayed request can PARK in the
    proxy gate and — if a resume is ever lost (knapsack can't fit, dropped
    event, apply-failed) — never return.  With ``read=None`` on the client
    that would hang asyncio.gather forever and wedge the whole multi-trial
    run.  A generous per-request deadline converts that hang into a counted
    error that the sanity gate then catches.
    """
    try:
        return await asyncio.wait_for(_one_request(cli, url, rec), timeout=deadline_s)
    except asyncio.TimeoutError:
        return {"ok": False, "error": "request-deadline",
                "want_out": int(rec.get("output_len") or 0)}


async def replay(
    records: List[Dict[str, Any]],
    *,
    base_url: str,
    mode: str,
    slowdown: float,
    max_concurrency: int,
    request_deadline_s: float = 300.0,
) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
    url = base_url.rstrip("/") + "/chat/completions"
    rows: List[Dict[str, Any]] = [None] * len(records)  # type: ignore
    sem = asyncio.Semaphore(max_concurrency)
    gauge = _InflightGauge(max_concurrency)
    limits = httpx.Limits(max_connections=max_concurrency + 8)
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0),
        limits=limits,
    ) as cli:
        wall0 = time.perf_counter()

        async def run(i: int, rec: Dict[str, Any]) -> None:
            async with sem:
                gauge.enter()
                try:
                    rows[i] = await _request_with_deadline(
                        cli, url, rec, request_deadline_s
                    )
                finally:
                    gauge.exit()

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
    return rows, wall_s, {"peak_inflight": gauge.peak, "cap_saturated": gauge.saturated}


async def replay_sessions(
    sessions: List[Dict[str, Any]],
    *,
    base_url: str,
    slowdown: float,
    max_concurrency: int,
    zero_tool_time: bool = False,
    request_deadline_s: float = 300.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float, Dict[str, Any]]:
    """Closed-loop replay: each session is a dependent request chain.

    Sessions start at their recorded first-arrival offset (preserving
    inter-session concurrency); within a session, request N+1 is dispatched
    only after N completes + its tool-think gap.  ``zero_tool_time`` drops
    the gaps (benefit upper bound).  Returns (per-request rows, per-session
    rows, makespan_s, gauge).

    M3: ``max_concurrency`` MUST be >= len(sessions), else sessions queue at
    the semaphore and the makespan is throttled (equally for both arms, so
    the comparison still holds, but the absolute number is an artifact).
    The caller is warned via gauge.cap_saturated.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    # M2: rows is append-as-completed (NOT trace-ordered like open-loop's
    # indexed list).  aggregate() is order-independent so this is fine;
    # do NOT join these rows back to the trace by index.
    rows: List[Dict[str, Any]] = []
    sess_rows: List[Dict[str, Any]] = []
    sem = asyncio.Semaphore(max_concurrency)
    gauge = _InflightGauge(max_concurrency)
    limits = httpx.Limits(max_connections=max_concurrency + 8)
    base_t = min((s["start_t"] for s in sessions), default=0.0)
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0),
        limits=limits,
    ) as cli:
        wall0 = time.perf_counter()

        async def run_session(s: Dict[str, Any]) -> None:
            delay = (s["start_t"] - base_t) * slowdown
            wait = wall0 + delay - time.perf_counter()
            if wait > 0:
                await asyncio.sleep(wait)
            sess_start = time.perf_counter()
            last_done = sess_start
            for step in s["steps"]:
                async with sem:
                    gauge.enter()
                    try:
                        row = await _request_with_deadline(
                            cli, url, step["record"], request_deadline_s
                        )
                    finally:
                        gauge.exit()
                rows.append(row)
                last_done = time.perf_counter()
                gap = 0.0 if zero_tool_time else step["gap_after"]
                if gap > 0:
                    await asyncio.sleep(gap * slowdown)
            sess_rows.append(
                {
                    "program_id": s["program_id"],
                    "session_e2e_s": last_done - sess_start,
                    "n_steps": len(s["steps"]),
                }
            )

        await asyncio.gather(*[run_session(s) for s in sessions])
        makespan_s = time.perf_counter() - wall0
    ginfo = {
        "peak_inflight": gauge.peak,
        "cap_saturated": gauge.saturated,
        "n_sessions": len(sessions),
    }
    return rows, sess_rows, makespan_s, ginfo


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
    ap.add_argument(
        "--mode", choices=["arrival", "closed", "session"], default="arrival",
        help="arrival/closed = open-loop (per-request latency, do-no-harm); "
             "session = closed-loop dependent chains (end-to-end makespan)",
    )
    ap.add_argument("--slowdown", type=float, default=1.0)
    # M1/M3: default high so the cap does not throttle the captured offered
    # load (32-way + runaway) / serialize sessions.  Surfaced via
    # cap_saturated if it is ever hit.
    ap.add_argument("--max-concurrency", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="")
    ap.add_argument("--zero-tool-time", action="store_true",
                    help="session mode: drop tool-think gaps (benefit upper bound)")
    ap.add_argument("--request-deadline", type=float, default=300.0,
                    help="per-request wall deadline (s); a parked/hung request "
                         "becomes a counted error instead of wedging the run")
    args = ap.parse_args()

    records = load_trace(args.trace, args.limit or None)
    if not records:
        print("no records in trace")
        return 1

    # M4: a trace missing ref_e2e_ms makes closed-loop gaps wrong (over-long).
    n_missing_e2e = sum(1 for r in records if "ref_e2e_ms" not in r)
    if args.mode == "session" and n_missing_e2e:
        print(f"WARNING: {n_missing_e2e}/{len(records)} records lack ref_e2e_ms "
              f"— closed-loop tool-think gaps will be OVER-estimated", flush=True)

    if args.mode == "session":
        sessions = build_sessions(records)
        rows, sess_rows, makespan, ginfo = asyncio.run(
            replay_sessions(
                sessions,
                base_url=args.base_url,
                slowdown=args.slowdown,
                max_concurrency=args.max_concurrency,
                zero_tool_time=args.zero_tool_time,
                request_deadline_s=args.request_deadline,
            )
        )
        metrics = aggregate(rows, makespan)
        metrics["sessions"] = aggregate_sessions(sess_rows, makespan)
        extra_rows: Dict[str, Any] = {"session_rows": sess_rows}
    else:
        rows, wall_s, ginfo = asyncio.run(
            replay(
                records,
                base_url=args.base_url,
                mode=args.mode,
                slowdown=args.slowdown,
                max_concurrency=args.max_concurrency,
                request_deadline_s=args.request_deadline,
            )
        )
        metrics = aggregate(rows, wall_s)
        extra_rows = {}

    metrics["peak_inflight"] = ginfo.get("peak_inflight")
    metrics["cap_saturated"] = ginfo.get("cap_saturated")
    if ginfo.get("cap_saturated"):
        print(f"WARNING: concurrency cap {args.max_concurrency} SATURATED "
              f"(peak_inflight={ginfo.get('peak_inflight')}) — offered-load / "
              f"pressure profile may be throttled; raise --max-concurrency",
              flush=True)
    metrics["label"] = args.label
    metrics["mode"] = args.mode
    print(json.dumps(metrics, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"metrics": metrics, "rows": rows, **extra_rows}, fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
