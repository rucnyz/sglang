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
    aggregate_sessions,
    build_payload,
    build_sessions,
    replay,
    replay_sessions,
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
        rows, wall, ginfo = await replay(
            records, base_url=f"http://127.0.0.1:{port}/v1",
            mode="arrival", slowdown=1.0, max_concurrency=16,
        )
    check(ginfo["peak_inflight"] >= 1 and not ginfo["cap_saturated"],
          "B0 gauge: peak tracked, cap not saturated at 16")
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
        rows2, wall2, _g2 = await replay(
            records, base_url=f"http://127.0.0.1:{port2}/v1",
            mode="closed", slowdown=1.0, max_concurrency=16,
        )
    agg2 = aggregate(rows2, wall2)
    check(agg2["n_error"] == 3 and agg2["n_ok"] == 0, "B2 502s counted as errors")

    # B3 (audit C-1): a request that HANGS past the deadline becomes a
    # counted error, not an indefinite wedge.
    port3 = _free_port()
    stub3 = make_stub(per_tok_delay_s=5.0)  # 5s/token >> deadline
    async with _Server(stub3, port3):
        t0 = asyncio.get_event_loop().time()
        rows3, wall3, _g3 = await replay(
            [{"t": 0.0, "program_id": "H", "output_len": 10,
              "body": {"model": "m", "messages": [{"role": "user", "content": "h"}]}}],
            base_url=f"http://127.0.0.1:{port3}/v1",
            mode="arrival", slowdown=1.0, max_concurrency=4,
            request_deadline_s=0.5,
        )
        elapsed = asyncio.get_event_loop().time() - t0
    check(rows3[0]["ok"] is False and rows3[0].get("error") == "request-deadline",
          "B3 hung request -> counted deadline error")
    check(elapsed < 4.0, f"B3 deadline fired fast (not the 50s hang) ({elapsed:.1f}s)")


def test_sessions_pure() -> None:
    print("C. closed-loop session reconstruction (pure)")
    # program A: 2 turns; gap before turn2 = t1 - (t0 + ref_e2e0/1000)
    #   t0=0.0 ref_e2e0=200ms -> done at 0.2; t1=1.0 -> gap 0.8
    # program B: 1 turn; one None-singleton
    recs = [
        {"t": 0.0, "program_id": "A", "ref_e2e_ms": 200.0, "output_len": 3, "body": {}},
        {"t": 1.0, "program_id": "A", "ref_e2e_ms": 100.0, "output_len": 4, "body": {}},
        {"t": 0.5, "program_id": "B", "ref_e2e_ms": 50.0, "output_len": 2, "body": {}},
        {"t": 0.7, "program_id": None, "ref_e2e_ms": 10.0, "output_len": 1, "body": {}},
    ]
    sessions = build_sessions(recs)
    check(len(sessions) == 3, f"C n_sessions == 3 (got {len(sessions)})")
    a = next(s for s in sessions if s["program_id"] == "A")
    check(len(a["steps"]) == 2, "C session A has 2 steps")
    check(abs(a["steps"][0]["gap_after"] - 0.8) < 1e-9,
          f"C A gap = t1-(t0+e2e0) = 0.8 (got {a['steps'][0]['gap_after']})")
    check(a["steps"][1]["gap_after"] == 0.0, "C last step gap 0")
    check(sessions[0]["start_t"] <= sessions[-1]["start_t"], "C sessions sorted by start_t")

    # M4: missing ref_e2e_ms -> gap overestimated to full t_next-t_n (=1.0),
    # not the true 0.8.  (Driver warns; here we pin the documented behavior.)
    recs_noe2e = [
        {"t": 0.0, "program_id": "A", "output_len": 3, "body": {}},
        {"t": 1.0, "program_id": "A", "output_len": 4, "body": {}},
    ]
    a2 = build_sessions(recs_noe2e)[0]
    check(abs(a2["steps"][0]["gap_after"] - 1.0) < 1e-9,
          f"M4 missing ref_e2e_ms -> gap = full interval 1.0 (got {a2['steps'][0]['gap_after']})")

    agg = aggregate_sessions(
        [{"program_id": "A", "session_e2e_s": 1.2, "n_steps": 2},
         {"program_id": "B", "session_e2e_s": 0.3, "n_steps": 1}],
        makespan_s=2.5,
    )
    check(agg["n_sessions"] == 2 and agg["total_steps"] == 3, "C aggregate_sessions counts")
    check(agg["makespan_s"] == 2.5, "C makespan carried")
    check(abs(agg["session_e2e_s"]["mean"] - 0.75) < 1e-9, "C session e2e mean")


async def test_session_live() -> None:
    print("D. live closed-loop replay (gaps honored, makespan measured)")
    # one session, 2 turns, a 0.3s tool gap between them.
    recs = [
        {"t": 0.0, "program_id": "S", "ref_e2e_ms": 0.0, "output_len": 3,
         "body": {"model": "m", "messages": [{"role": "user", "content": "1"}]}},
        {"t": 0.3, "program_id": "S", "ref_e2e_ms": 0.0, "output_len": 3,
         "body": {"model": "m", "messages": [{"role": "user", "content": "2"}]}},
    ]
    sessions = build_sessions(recs)
    check(abs(sessions[0]["steps"][0]["gap_after"] - 0.3) < 1e-9, "D gap 0.3 derived")
    port = _free_port()
    stub = make_stub(per_tok_delay_s=0.001)
    async with _Server(stub, port):
        rows, sess_rows, makespan, _g = await replay_sessions(
            sessions, base_url=f"http://127.0.0.1:{port}/v1",
            slowdown=1.0, max_concurrency=8,
        )
        # zero-tool-time variant must be faster (no 0.3s gap)
        rows0, sess0, makespan0, _g0 = await replay_sessions(
            sessions, base_url=f"http://127.0.0.1:{port}/v1",
            slowdown=1.0, max_concurrency=8, zero_tool_time=True,
        )
    check(len(rows) == 2 and all(r["ok"] for r in rows), "D both turns served")
    check(len(sess_rows) == 1, "D one session row")
    check(sess_rows[0]["session_e2e_s"] >= 0.3, "D session e2e includes the 0.3s gap")
    check(makespan0 < makespan, "D zero-tool-time makespan < real-gap makespan")


def main() -> int:
    test_pure()
    asyncio.run(test_live())
    test_sessions_pure()
    asyncio.run(test_session_live())
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
