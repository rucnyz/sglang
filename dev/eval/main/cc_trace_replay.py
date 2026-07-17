#!/usr/bin/env python3
"""
cc_trace_replay.py — replay real Claude-Code agent traces against an
OpenAI-compatible serving endpoint (vLLM or SGLang).

Each input session is a multi-turn conversation extracted from cc-share-hf
JSONL or archit11/complete-conversations-dataset.  The client drives the bench
by feeding the original user/tool-result messages turn-by-turn; the model under
test generates fresh assistant responses (we do NOT replay the trace's
original assistant content).  This produces realistic long-horizon agent
workload pressure while staying engine-agnostic.

Output: bench.json with the same schema as genai-bench / sglang.bench_serving:
    wall_s, num_concurrency, num_requests_total, num_requests_valid,
    num_errors, mean_ttft_ms, p50_ttft_ms, p99_ttft_ms, mean_e2e_ms,
    p50_e2e_ms, p99_e2e_ms, output_tps, input_tps, requests_per_second,
    error_rate

Usage:
    python cc_trace_replay.py \
        --api-base http://127.0.0.1:34000 \
        --model moonshotai/Kimi-Linear-48B-A3B-Instruct \
        --traces /path/to/traces.jsonl \
        --num-concurrency 14 \
        --max-time-min 10 \
        --max-tokens 1024 \
        --output-file /path/to/bench.json

The --traces argument may be:
  - a single jsonl file with one trace per line (archit11 schema), OR
  - a directory of cc-share-hf JSONL session files (one session per file)
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp


def extract_conversation(events: list[dict]) -> list[dict]:
    """Pull just the user-role messages (text or tool_result) from a list of
    cc-share-hf events. Returns a list of {"role": "user", "content": str}.
    Assistant outputs from the trace are discarded — the model under test
    generates them fresh on replay."""
    out = []
    for e in events:
        if e.get("type") != "user":
            continue
        m = e.get("message", {})
        if not isinstance(m, dict):
            continue
        content = m.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict):
                    t = p.get("text") or p.get("content") or ""
                    if isinstance(t, (dict, list)):
                        t = json.dumps(t)
                    parts.append(str(t))
                else:
                    parts.append(str(p))
            text = "\n".join(parts)
        else:
            text = str(content)
        if text.strip():
            out.append({"role": "user", "content": text})
    return out


def load_traces(path: str) -> list[list[dict]]:
    """Return a list of traces, where each trace is a list of {role, content}
    user messages ready to be replayed turn-by-turn.

    Supports:
      - directory of cc-share-hf JSONL session files (one file = one session)
      - single JSONL file with one trace per line (archit11 schema:
        each line has 'conversations' or 'messages' field)
    """
    traces: list[list[dict]] = []
    p = Path(path)
    if p.is_dir():
        for f in sorted(p.glob("*.jsonl")):
            events = []
            with open(f) as fh:
                for line in fh:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            conv = extract_conversation(events)
            if conv:
                traces.append(conv)
    else:
        with open(p) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Try a few schemas: cc-share-hf (events list), archit11 (conversations / messages)
                if "conversations" in rec:
                    conv = []
                    for m in rec["conversations"]:
                        if m.get("from") in ("human", "user"):
                            v = m.get("value", "")
                            if v.strip():
                                conv.append({"role": "user", "content": v})
                elif "messages" in rec:
                    conv = []
                    for m in rec["messages"]:
                        if m.get("role") == "user":
                            c = m.get("content", "")
                            if isinstance(c, list):
                                c = "\n".join(p.get("text", str(p)) for p in c if isinstance(p, dict))
                            if isinstance(c, str) and c.strip():
                                conv.append({"role": "user", "content": c})
                else:
                    # Treat as a flat list of events
                    events = rec if isinstance(rec, list) else [rec]
                    conv = extract_conversation(events)
                if conv:
                    traces.append(conv)
    return traces


def filter_traces(traces: list[list[dict]], min_turns: int, min_chars: int) -> list[list[dict]]:
    """Keep traces with at least `min_turns` user turns AND `min_chars` total
    user-content characters (a coarse stand-in for tokens)."""
    out = []
    for t in traces:
        if len(t) < min_turns:
            continue
        total = sum(len(m["content"]) for m in t)
        if total < min_chars:
            continue
        out.append(t)
    return out


class Stats:
    """Per-cell aggregator. Worker tasks append latency samples after each
    completed request; the main task computes summary stats at the end."""

    def __init__(self) -> None:
        self.ttft_ms: list[float] = []
        self.e2e_ms: list[float] = []
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.requests_total: int = 0
        self.requests_valid: int = 0
        self.errors: int = 0
        self.start: float = time.monotonic()


async def run_one_session(
    session: aiohttp.ClientSession,
    trace: list[dict],
    api_base: str,
    model: str,
    max_tokens: int,
    deadline: float,
    stats: Stats,
) -> None:
    """Replay one trace as a single multi-turn conversation. We accumulate
    history as we go, send each user turn via /v1/chat/completions, and stop
    when the trace runs out OR the cell-wide deadline passes."""
    history: list[dict] = []
    for user_msg in trace:
        if time.monotonic() >= deadline:
            return
        history.append(user_msg)
        body = {
            "model": model,
            "messages": history,
            "max_tokens": max_tokens,
            "stream": True,
        }
        url = f"{api_base.rstrip('/')}/v1/chat/completions"
        t_start = time.monotonic()
        ttft = None
        n_out = 0
        full_text = ""
        try:
            async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=600)) as r:
                async for chunk in r.content:
                    if not chunk:
                        continue
                    s = chunk.decode("utf-8", errors="ignore")
                    for line in s.splitlines():
                        line = line.strip()
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            continue
                        try:
                            d = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if not d.get("choices"):
                            continue
                        delta = d["choices"][0].get("delta", {})
                        # Qwen3 reasoning models split output between `content` and
                        # `reasoning_content` (the latter holds chain-of-thought
                        # tokens when --reasoning-parser is enabled). Both count
                        # as model output for our serving-side metrics.
                        token_text = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                        if token_text:
                            if ttft is None:
                                ttft = (time.monotonic() - t_start) * 1000.0
                            full_text += token_text
                            n_out += len(token_text) // 3 + 1  # rough token count
            stats.requests_total += 1
            if ttft is not None and full_text:
                stats.ttft_ms.append(ttft)
                stats.e2e_ms.append((time.monotonic() - t_start) * 1000.0)
                stats.input_tokens += sum(len(m["content"]) for m in history) // 4  # rough
                stats.output_tokens += n_out
                stats.requests_valid += 1
                history.append({"role": "assistant", "content": full_text})
            else:
                stats.errors += 1
                return
        except Exception:
            stats.requests_total += 1
            stats.errors += 1
            return


async def run_bench(args: argparse.Namespace) -> dict:
    traces = load_traces(args.traces)
    traces = filter_traces(traces, args.min_turns, args.min_chars)
    if not traces:
        raise RuntimeError(f"no traces matched filters in {args.traces}")
    print(f"[cc_trace_replay] loaded {len(traces)} traces (filtered ≥{args.min_turns} turns, ≥{args.min_chars} chars)", flush=True)
    print(f"[cc_trace_replay] turns/trace: min={min(len(t) for t in traces)}  max={max(len(t) for t in traces)}  median={sorted(len(t) for t in traces)[len(traces)//2]}", flush=True)

    stats = Stats()
    deadline = time.monotonic() + args.max_time_min * 60.0

    conn = aiohttp.TCPConnector(limit=args.num_concurrency * 2)
    async with aiohttp.ClientSession(connector=conn) as session:
        # Each worker pulls traces from the shared list, looping back to the
        # start if it runs out before the deadline.
        idx = 0
        lock = asyncio.Lock()

        async def worker(slot: int) -> None:
            nonlocal idx
            while time.monotonic() < deadline:
                async with lock:
                    # Request-bounded mode (--max-sessions > 0): stop after
                    # N sessions so two cells process the IDENTICAL session
                    # set (idx increments deterministically over the same
                    # trace order). Makes tail metrics (p99 TTFT) comparable
                    # across cells — the default time-bounded mode lets the
                    # faster cell process a larger/different request set,
                    # which confounds the tail.
                    if args.max_sessions and idx >= args.max_sessions:
                        return
                    if idx >= len(traces) and not args.repeat:
                        return
                    trace = traces[idx % len(traces)]
                    idx += 1
                await run_one_session(session, trace, args.api_base, args.model, args.max_tokens, deadline, stats)

        await asyncio.gather(*[worker(i) for i in range(args.num_concurrency)])

    wall_s = time.monotonic() - stats.start
    out = {
        "wall_s": wall_s,
        "num_concurrency": args.num_concurrency,
        "num_requests_total": stats.requests_total,
        "num_requests_valid": stats.requests_valid,
        "num_errors": stats.errors,
        "mean_ttft_ms": statistics.mean(stats.ttft_ms) if stats.ttft_ms else 0,
        "p50_ttft_ms": statistics.median(stats.ttft_ms) if stats.ttft_ms else 0,
        "p99_ttft_ms": sorted(stats.ttft_ms)[int(0.99 * len(stats.ttft_ms))] if stats.ttft_ms else 0,
        "mean_e2e_ms": statistics.mean(stats.e2e_ms) if stats.e2e_ms else 0,
        "p50_e2e_ms": statistics.median(stats.e2e_ms) if stats.e2e_ms else 0,
        "p99_e2e_ms": sorted(stats.e2e_ms)[int(0.99 * len(stats.e2e_ms))] if stats.e2e_ms else 0,
        "output_tps": stats.output_tokens / wall_s if wall_s > 0 else 0,
        "input_tps": stats.input_tokens / wall_s if wall_s > 0 else 0,
        "requests_per_second": stats.requests_valid / wall_s if wall_s > 0 else 0,
        "error_rate": stats.errors / max(1, stats.requests_total),
    }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--api-base", required=True, help="e.g. http://127.0.0.1:34000")
    p.add_argument("--model", required=True)
    p.add_argument("--traces", required=True, help="directory of cc-share-hf JSONL OR single JSONL with archit11 schema")
    p.add_argument("--num-concurrency", type=int, default=14)
    p.add_argument("--max-time-min", type=float, default=10.0)
    p.add_argument("--max-sessions", type=int, default=0,
                   help="request-bounded mode: stop after N sessions so "
                        "two cells process the identical set (0 = time-bound)")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--min-turns", type=int, default=15, help="filter: keep traces with at least this many user turns")
    p.add_argument("--min-chars", type=int, default=30000, help="filter: keep traces with at least this many user-content chars (user-only; full context grows further with assistant generations)")
    p.add_argument("--output-file", required=True)
    p.add_argument("--repeat", action="store_true", default=True, help="loop back to first trace when list exhausts (default: True)")
    args = p.parse_args()

    out = asyncio.run(run_bench(args))
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[cc_trace_replay] DONE  wall_s={out['wall_s']:.0f}  reqs={out['num_requests_valid']}/{out['num_requests_total']}  out_tps={out['output_tps']:.0f}  p99_ttft={out['p99_ttft_ms']:.0f}ms  errors={out['num_errors']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
