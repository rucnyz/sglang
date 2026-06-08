"""#231 verify: deterministic replay driver (pure metrics + live replay).

  A. Pure functions
    A0 build_payload forces temp=0, max_tokens=max(1,output_len), ignore_eos,
       stream; injects program_id; keeps messages+model
    A1 _pct / _summ percentile + summary correctness on a known list
    A2 aggregate: rows -> n_ok/n_error, throughput, len_match_rate, ttft summary

  B. Live replay against a stub that HONORS max_tokens
    B0 arrival mode: every request replayed, n_out == forced max_tokens,
       ttft + tpot measured, len_match_rate == 1.0
    B1 program_id reaches the server (proxy contract: body.program_id)
    B2 a server 502 is counted as an error, not a crash

Usage:  python dev/aginfer/verify/replay_driver/verify.py
"""
from __future__ import annotations

import asyncio
import json
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

_REPLAY_DIR = Path(__file__).resolve().parents[2] / "scenarios" / "replay"
sys.path.insert(0, str(_REPLAY_DIR))

from replay_driver import (  # noqa: E402
    aggregate,
    build_payload,
    replay,
    _pct,
    _summ,
)

_FAILS: List[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        _FAILS.append(msg)


def test_pure() -> None:
    print("A. pure functions")
    # A0 build_payload
    p = build_payload(
        {"body": {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
         "program_id": "P1", "output_len": 7}
    )
    check(p["temperature"] == 0.0, "A0 temperature forced 0")
    check(p["max_tokens"] == 7, "A0 max_tokens == output_len")
    check(p["ignore_eos"] is True, "A0 ignore_eos True")
    check(p["stream"] is True, "A0 stream True")
    check(p["program_id"] == "P1", "A0 program_id injected")
    check(p["messages"] == [{"role": "user", "content": "hi"}], "A0 messages kept")
    p0 = build_payload({"body": {}, "output_len": 0})
    check(p0["max_tokens"] == 1, "A0 output_len 0 -> max_tokens 1")
    check("program_id" not in p0, "A0 no program_id when absent")

    # A1 percentile
    vals = [float(i) for i in range(1, 101)]  # 1..100
    check(abs(_pct(vals, 0.50) - 50.5) < 1e-9, "A1 p50 of 1..100 == 50.5")
    check(abs(_pct(vals, 0.99) - 99.01) < 1e-9, f"A1 p99 ~ 99.01 (got {_pct(vals,0.99)})")
    s = _summ([10.0, 20.0, 30.0])
    check(s["mean"] == 20.0 and s["max"] == 30.0 and s["n"] == 3, "A1 _summ basic")
    check(_summ([])["n"] == 0, "A1 _summ empty")

    # A2 aggregate
    rows = [
        {"ok": True, "ttft_ms": 10.0, "e2e_ms": 100.0, "tpot_ms": 5.0, "n_out": 8, "want_out": 8},
        {"ok": True, "ttft_ms": 20.0, "e2e_ms": 200.0, "tpot_ms": 6.0, "n_out": 9, "want_out": 8},
        {"ok": False, "want_out": 4},
    ]
    agg = aggregate(rows, wall_s=2.0)
    check(agg["n_requests"] == 3 and agg["n_ok"] == 2 and agg["n_error"] == 1,
          "A2 counts")
    check(agg["total_out_tokens"] == 17, "A2 total out tokens 8+9")
    check(abs(agg["throughput_tok_s"] - 8.5) < 1e-6, "A2 throughput 17/2.0")
    check(agg["len_match_rate"] == 1.0, "A2 len_match within ±1 -> 1.0")
    check(agg["ttft_ms"]["mean"] == 15.0, "A2 ttft mean 15")


# ----------------------------------------------------------- live stub server

def make_stub(per_tok_delay_s: float = 0.002, fail: bool = False) -> FastAPI:
    app = FastAPI()
    app.state.seen_pids: List[Any] = []

    @app.post("/v1/chat/completions")
    async def chat(raw: Request) -> Any:
        body = await raw.json()
        app.state.seen_pids.append(body.get("program_id"))
        if fail:
            return JSONResponse({"error": "boom"}, status_code=502)
        n = int(body.get("max_tokens") or 1)  # HONOR forced length

        async def gen():
            for _ in range(n):
                await asyncio.sleep(per_tok_delay_s)
                yield b"data: " + json.dumps(
                    {"choices": [{"delta": {"content": "x"}}]}
                ).encode() + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class _Server:
    def __init__(self, app: FastAPI, port: int) -> None:
        self._srv = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        )
        self._task: Any = None

    async def __aenter__(self) -> "_Server":
        self._task = asyncio.ensure_future(self._srv.serve())
        while not self._srv.started:
            await asyncio.sleep(0.02)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._srv.should_exit = True
        await self._task


async def test_live() -> None:
    print("B. live replay against max_tokens-honoring stub")
    records = [
        {"t": 0.0, "program_id": "A", "body": {"model": "m", "messages": [{"role": "user", "content": "1"}]}, "output_len": 5},
        {"t": 0.05, "program_id": "B", "body": {"model": "m", "messages": [{"role": "user", "content": "2"}]}, "output_len": 8},
        {"t": 0.10, "program_id": "C", "body": {"model": "m", "messages": [{"role": "user", "content": "3"}]}, "output_len": 3},
    ]

    # B0/B1 — success path
    port = _free_port()
    stub = make_stub()
    async with _Server(stub, port):
        rows, wall = await replay(
            records, base_url=f"http://127.0.0.1:{port}/v1",
            mode="arrival", slowdown=1.0, max_concurrency=16,
        )
    agg = aggregate(rows, wall)
    check(agg["n_ok"] == 3 and agg["n_error"] == 0, "B0 all 3 replayed ok")
    n_outs = sorted(r["n_out"] for r in rows)
    check(n_outs == [3, 5, 8], f"B0 forced lengths honored {n_outs}")
    check(agg["len_match_rate"] == 1.0, "B0 len_match_rate 1.0")
    check(all(r["ttft_ms"] is not None for r in rows), "B0 ttft measured")
    check(all(r["tpot_ms"] is not None for r in rows if r["n_out"] >= 2), "B0 tpot measured")
    check(set(stub.state.seen_pids) == {"A", "B", "C"}, "B1 program_id reached server")

    # B2 — error path
    port2 = _free_port()
    stub2 = make_stub(fail=True)
    async with _Server(stub2, port2):
        rows2, wall2 = await replay(
            records, base_url=f"http://127.0.0.1:{port2}/v1",
            mode="closed", slowdown=1.0, max_concurrency=16,
        )
    agg2 = aggregate(rows2, wall2)
    check(agg2["n_error"] == 3 and agg2["n_ok"] == 0, "B2 502s counted as errors")


def main() -> int:
    test_pure()
    asyncio.run(test_live())
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
