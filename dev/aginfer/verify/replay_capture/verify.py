"""#231 verify: request-trace capture (daemon proxy → replay trace).

Covers the capture half of the deterministic replay benchmark:

  A. Pure helpers (deterministic, no server)
    A0 count_sse_content_tokens counts content deltas, even when a JSON
       event is split across two byte chunks (httpx aiter_bytes boundary)
    A1 count_sse_content_tokens ignores [DONE], empty deltas, role-only
       deltas, and malformed lines
    A2 usage_completion_tokens reads the non-stream usage block; None on junk

  B. TraceRecorder
    B0 note_arrival rebases t0: first offset == 0, later offsets monotonic >0
    B1 write emits valid JSONL with t / program_id / slim body / output_len;
       body keeps messages + sampling keys, drops bulk/extra fields
    B2 write never raises on a non-dict body (capture must not break serving)

  C. Integration through the REAL proxy create_app + stub sglang
    C0 streaming request → one trace line, output_len == #content chunks,
       program_id + messages captured, arrival offset present
    C1 non-streaming request → output_len from usage.completion_tokens
    C2 trace_recorder=None (capture off) → request still served, no file

Usage:  python dev/aginfer/verify/replay_capture/verify.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

_AGINFER_ROOT = Path(__file__).resolve().parents[2]
if str(_AGINFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGINFER_ROOT))

from daemon.proxy import create_app  # noqa: E402
from daemon.trace_capture import (  # noqa: E402
    TraceRecorder,
    count_sse_content_tokens,
    usage_completion_tokens,
)

_FAILS: List[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        _FAILS.append(msg)


def _sse(content: str) -> bytes:
    return b"data: " + json.dumps(
        {"choices": [{"delta": {"content": content}}]}
    ).encode() + b"\n\n"


# ----------------------------------------------------------------- A. helpers


def test_helpers() -> None:
    print("A. pure helpers")

    # A0 — split a content event across a chunk boundary.
    full = _sse("hello") + _sse("world")
    carry: Dict[str, bytes] = {}
    cut = len(full) // 2
    n = count_sse_content_tokens(full[:cut], carry)
    n += count_sse_content_tokens(full[cut:], carry)
    check(n == 2, f"A0 split-chunk content count == 2 (got {n})")

    # A1 — ignore [DONE], empty/role-only deltas, malformed lines.
    carry = {}
    junk = (
        _sse("tok")
        + b"data: [DONE]\n\n"
        + b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        + b'data: {"choices":[{"delta":{"content":""}}]}\n\n'
        + b"data: not-json\n\n"
        + b": keep-alive comment\n\n"
    )
    n = count_sse_content_tokens(junk, carry)
    check(n == 1, f"A1 only the one real content delta counts (got {n})")

    # A1b — CRLF line endings + multiple events in one chunk.
    carry = {}
    crlf = (b"data: " + json.dumps({"choices": [{"delta": {"content": "x"}}]}).encode()
            + b"\r\n\r\n"
            + b"data: " + json.dumps({"choices": [{"delta": {"content": "y"}}]}).encode()
            + b"\r\n\r\n")
    check(count_sse_content_tokens(crlf, carry) == 2, "A1b CRLF + multi-event-per-chunk == 2")

    # A1c — reasoning_content deltas count as tokens (reasoning models split
    # the chain-of-thought into reasoning_content; they are real decode
    # tokens occupying KV).  Mix of reasoning + content + a [DONE].
    carry = {}
    rc = (b'data: {"choices":[{"delta":{"reasoning_content":"think1"}}]}\n\n'
          + b'data: {"choices":[{"delta":{"reasoning_content":"think2"}}]}\n\n'
          + _sse("answer")
          + b"data: [DONE]\n\n")
    check(count_sse_content_tokens(rc, carry) == 3,
          "A1c reasoning_content + content both counted (2+1=3)")

    # A2 — usage parse.
    body = json.dumps({"usage": {"completion_tokens": 42}}).encode()
    check(usage_completion_tokens(body) == 42, "A2 usage_completion_tokens == 42")
    check(usage_completion_tokens(b"garbage") is None, "A2 junk -> None")
    check(
        usage_completion_tokens(json.dumps({"usage": {}}).encode()) is None,
        "A2 missing field -> None",
    )


# --------------------------------------------------------------- B. recorder


def test_recorder() -> None:
    print("B. TraceRecorder")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "trace.jsonl")
        rec = TraceRecorder(p)

        # B0 — t0 rebase.
        a0 = rec.note_arrival()
        # tiny busy gap so monotonic advances
        for _ in range(100000):
            pass
        a1 = rec.note_arrival()
        check(a0 == 0.0, f"B0 first arrival offset == 0 (got {a0})")
        check(a1 > 0.0, f"B0 second arrival offset > 0 (got {a1})")

        # B1 — write slims body + emits valid JSONL.
        rec.write(
            arrival_offset=a1,
            program_id="prog-7",
            body={
                "messages": [{"role": "user", "content": "hi"}],
                "model": "m",
                "temperature": 0.0,
                "stream": True,            # dropped (not a sampling key)
                "extra_body": {"x": 1},    # dropped
            },
            output_len=5,
            ref_e2e_ms=123.4,
        )
        # B2 — non-dict body must not raise / must not write.
        rec.write(arrival_offset=0.0, program_id=None, body="not-a-dict", output_len=0)
        rec.close()

        lines = [json.loads(x) for x in open(p) if x.strip()]
        check(len(lines) == 1, f"B2 non-dict body produced no extra line (n={len(lines)})")
        r = lines[0]
        check(r["program_id"] == "prog-7", "B1 program_id captured")
        check(r["output_len"] == 5, "B1 output_len captured")
        check(abs(r["t"] - round(a1, 6)) < 1e-6, "B1 arrival offset captured")
        check(r["body"].get("messages") == [{"role": "user", "content": "hi"}],
              "B1 messages captured verbatim")
        check("temperature" in r["body"] and r["body"]["temperature"] == 0.0,
              "B1 sampling key kept")
        check("stream" not in r["body"] and "extra_body" not in r["body"],
              "B1 bulk/non-sampling keys dropped")
        check(abs(r.get("ref_e2e_ms", 0) - 123.4) < 1e-6, "B1 ref_e2e_ms captured")


# ------------------------------------------------------------ C. integration

def make_stub_sglang() -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(raw: Request) -> Any:
        body = await raw.json()
        if body.get("stream") is True:
            async def gen():
                for tok in ["a", "b", "c", "d"]:
                    yield _sse(tok)
                yield b"data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        return JSONResponse(
            {
                "choices": [{"message": {"role": "assistant", "content": "abc"}}],
                "usage": {"completion_tokens": 3},
            }
        )

    return app


class _Server:
    def __init__(self, app: FastAPI, port: int) -> None:
        cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        self._srv = uvicorn.Server(cfg)
        self._task: Any = None

    async def __aenter__(self) -> "_Server":
        self._task = asyncio.ensure_future(self._srv.serve())
        while not self._srv.started:
            await asyncio.sleep(0.02)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._srv.should_exit = True
        await self._task


async def test_integration() -> None:
    print("C. integration through real proxy + stub sglang")
    import contextlib
    import socket

    def free_port() -> int:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
        s.close()
        return p

    sg_port, dm_port = free_port(), free_port()
    with tempfile.TemporaryDirectory() as d:
        trace = os.path.join(d, "trace.jsonl")
        rec = TraceRecorder(trace)
        stub = make_stub_sglang()
        app = create_app(
            sglang_base_url=f"http://127.0.0.1:{sg_port}",
            enable_event_router=False,
            trace_recorder=rec,
        )
        async with _Server(stub, sg_port), _Server(app, dm_port):
            async with httpx.AsyncClient(timeout=10.0) as cli:
                # C0 — streaming
                async with cli.stream(
                    "POST",
                    f"http://127.0.0.1:{dm_port}/v1/chat/completions",
                    json={
                        "model": "m",
                        "program_id": "sess-A",
                        "stream": True,
                        "messages": [{"role": "user", "content": "go"}],
                    },
                ) as resp:
                    async for _ in resp.aiter_bytes():
                        pass
                # C1 — non-streaming
                r = await cli.post(
                    f"http://127.0.0.1:{dm_port}/v1/chat/completions",
                    json={
                        "model": "m",
                        "program_id": "sess-B",
                        "messages": [{"role": "user", "content": "go2"}],
                    },
                )
                check(r.status_code == 200, "C1 non-stream forwarded ok")
        rec.close()

        recs = [json.loads(x) for x in open(trace) if x.strip()]
        by_pid = {r["program_id"]: r for r in recs}
        check(len(recs) == 2, f"C captured 2 requests (got {len(recs)})")
        check(
            by_pid.get("sess-A", {}).get("output_len") == 4,
            f"C0 streaming output_len == 4 content chunks (got {by_pid.get('sess-A',{}).get('output_len')})",
        )
        check(
            by_pid.get("sess-B", {}).get("output_len") == 3,
            f"C1 non-stream output_len == usage 3 (got {by_pid.get('sess-B',{}).get('output_len')})",
        )
        check(
            by_pid.get("sess-A", {}).get("body", {}).get("messages")
            == [{"role": "user", "content": "go"}],
            "C0 messages captured",
        )
        check(
            all(r.get("ref_e2e_ms", 0) > 0 for r in recs),
            "C ref_e2e_ms measured (>0) for both real requests",
        )

    # C2 — capture OFF: request still served, recorder is None.
    sg_port2, dm_port2 = free_port(), free_port()
    stub2 = make_stub_sglang()
    app2 = create_app(
        sglang_base_url=f"http://127.0.0.1:{sg_port2}",
        enable_event_router=False,
        trace_recorder=None,
    )
    async with _Server(stub2, sg_port2), _Server(app2, dm_port2):
        async with httpx.AsyncClient(timeout=10.0) as cli:
            r = await cli.post(
                f"http://127.0.0.1:{dm_port2}/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "x"}]},
            )
            check(r.status_code == 200, "C2 capture-off request served ok")
            check(getattr(app2.state, "trace_recorder", "x") is None,
                  "C2 trace_recorder is None")


def main() -> int:
    test_helpers()
    test_recorder()
    asyncio.run(test_integration())
    print()
    if _FAILS:
        print(f"FAILED ({len(_FAILS)}):")
        for f in _FAILS:
            print("  - " + f)
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
