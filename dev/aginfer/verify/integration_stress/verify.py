"""integration_stress (PLAN §2 / #147): six cross-component stress
flavors A–F, run against a real sglang + daemon stack.

Why this verify exists
----------------------
The per-task verifies (T7/T8/T36/T42/etc.) exercise components in
isolation against stubs.  This file is the **only** verify that
launches the full sglang + daemon process pair and pushes real
traffic through.  Each flavor stresses a different cross-component
interaction; the pass criteria are paper §9 deployment thresholds
or the per-task ceilings from sibling verifies.

Flavors
-------
* **A** — daemon proxy hot-path latency under load
  (proxy + event_router + sglang)
* **B** — `/aginfer/state` cost under sustained traffic
  (sglang state-dump + daemon polling; PLAN F3-revisit threshold)
* **C** — event-router fan-in throughput
  (event_router + program_tracker + kv_scheduler.handle)
* **D** — migrate dispatch under traffic
  (outbound queue + sglang migrate endpoint + APPLY_FAILED webhook)
* **E** — threshold PUT atomicity under traffic
  (daemon thresholds endpoint + kv_scheduler + admission_controller)
* **F** — dead-sglang resilience under low traffic
  (event_router + state-fetch failure path + #164's low-traffic
  no-escalate contract)

Run
---
::
    python dev/aginfer/verify/integration_stress/verify.py

About 5–10 minutes wall clock on GPUs 5–6 (sglang startup +
~30–60 s per flavor).  Sequential to avoid cross-flavor interference.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from statistics import median
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx


_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from harness import (  # noqa: E402
    DAEMON_HOST,
    DAEMON_PORT,
    GPUS,
    MODEL,
    SGLANG_HOST,
    SGLANG_PORT,
    StackHandles,
    _launch_daemon,
    _launch_sglang,
    stack,
    wait_daemon,
    wait_sglang,
)


def _green(s: str) -> str: return f"\033[32m{s}\033[0m"
def _red(s: str) -> str:   return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


def _quantile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = int(q * (len(s) - 1))
    return s[k]


# ---- shared helpers -----------------------------------------------


_PAD = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed "
    "do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
) * 6  # ~700 tokens


def _chat_body(worker_idx: int, n: int) -> Dict[str, Any]:
    """Unique-prompt body that anchors its own radix chain (defeats
    sharing) — same pattern as t14/stress_probe.py + T15 driver."""
    return {
        "model": MODEL,
        "messages": [
            {"role": "system",
             "content": f"{_PAD} (tag-{worker_idx}-{uuid.uuid4().hex[:8]})"},
            {"role": "user",
             "content": (
                 f"reply with the literal token 'ack' (req {n} "
                 f"worker {worker_idx} nonce {uuid.uuid4().hex[:12]})"
             )},
        ],
        "max_tokens": 8,
        "temperature": 0.0,
    }


async def _post_chat(
    cli: httpx.AsyncClient, base: str, body: Dict[str, Any],
) -> Tuple[bool, float]:
    """Returns (success, wall_ms)."""
    t0 = time.perf_counter()
    try:
        r = await cli.post(
            f"http://{base}/v1/chat/completions", json=body, timeout=60.0,
        )
        ok = r.status_code == 200
    except (httpx.RequestError, asyncio.TimeoutError):
        ok = False
    return ok, (time.perf_counter() - t0) * 1000.0


# ============================================================ A. proxy latency


async def _flavor_a_proxy_overhead(stack_h: StackHandles) -> Dict[str, Any]:
    """Drive N=24 concurrent chats through the daemon proxy for 60s.
    Pass criteria:
      * 95% of requests succeed (i.e. proxy doesn't drop on its own)
      * end-to-end p99 < 5s on Qwen3-0.6B (sane for small model)
      * the daemon process is still alive at end
    """
    DAEMON_BASE = f"{DAEMON_HOST}:{DAEMON_PORT}"
    DURATION = 60.0
    WORKERS = 24

    deadline = time.time() + DURATION
    samples_ms: List[float] = []
    succeeded = 0
    failed = 0

    async def worker(idx: int) -> None:
        nonlocal succeeded, failed
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5, read=60, write=10, pool=5),
        ) as cli:
            n = 0
            while time.time() < deadline:
                ok, ms = await _post_chat(cli, DAEMON_BASE, _chat_body(idx, n))
                samples_ms.append(ms)
                if ok:
                    succeeded += 1
                else:
                    failed += 1
                n += 1

    await asyncio.gather(*(worker(i) for i in range(WORKERS)))
    total = succeeded + failed
    p50 = _quantile(samples_ms, 0.5)
    p99 = _quantile(samples_ms, 0.99)
    success_rate = (succeeded / total) if total else 0.0
    return {
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "success_rate": success_rate,
        "p50_ms": p50,
        "p99_ms": p99,
        "daemon_alive": (stack_h.daemon_proc.poll() is None),
    }


def stage_a(stack_h: StackHandles) -> None:
    res = asyncio.run(_flavor_a_proxy_overhead(stack_h))
    print(f"  [A] {res['total']} req | succ {res['succeeded']} "
          f"({res['success_rate']:.1%}) | "
          f"p50 {res['p50_ms']:.1f} ms | p99 {res['p99_ms']:.1f} ms")
    if not res["daemon_alive"]:
        raise StageFail("daemon died mid-flavor-A")
    if res["success_rate"] < 0.95:
        raise StageFail(
            f"proxy success rate {res['success_rate']:.1%} below 95%"
        )
    if res["p99_ms"] > 5000.0:
        raise StageFail(
            f"proxy p99 {res['p99_ms']:.0f} ms above 5000 ms ceiling"
        )


# ============================================================ B. state-dump under traffic


async def _flavor_b_state_dump_under_traffic(
    stack_h: StackHandles,
) -> Dict[str, Any]:
    """Sustain 16-worker chat traffic via daemon for 90s; poll
    /aginfer/state at 2 Hz; aggregate state_dump_metrics across
    samples.  Pass: aggregate p99 < 200 ms (loose ceiling — Qwen3-
    0.6B on TP=1 dumps a small tree).  PLAN F3-revisit hard
    threshold is 50 ms, but at this fixture size we expect well
    under that."""
    DAEMON_BASE = f"{DAEMON_HOST}:{DAEMON_PORT}"
    SGLANG_BASE = f"{SGLANG_HOST}:{SGLANG_PORT}"
    DURATION = 90.0
    WORKERS = 16

    deadline = time.time() + DURATION
    p50_samples: List[float] = []
    p99_samples: List[float] = []
    unit_counts: List[int] = []

    async def worker(idx: int) -> None:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5, read=60, write=10, pool=5),
        ) as cli:
            n = 0
            while time.time() < deadline:
                await _post_chat(cli, DAEMON_BASE, _chat_body(idx, n))
                n += 1

    async def poller() -> None:
        async with httpx.AsyncClient(timeout=10.0) as cli:
            while time.time() < deadline:
                try:
                    r = await cli.get(
                        f"http://{SGLANG_BASE}/aginfer/state"
                    )
                    if r.status_code == 200:
                        body = r.json()
                        # Single-rank dump (TP=1, no per_rank wrap).
                        m = body.get("state_dump_metrics", {})
                        if "p50_ms" in m:
                            p50_samples.append(float(m["p50_ms"]))
                        if "p99_ms" in m:
                            p99_samples.append(float(m["p99_ms"]))
                        unit_counts.append(len(body.get("units", [])))
                except Exception:
                    pass
                await asyncio.sleep(0.5)

    await asyncio.gather(
        *(worker(i) for i in range(WORKERS)), poller(),
    )
    return {
        "p99_samples": p99_samples,
        "p50_median": median(p50_samples) if p50_samples else 0.0,
        "p99_max": max(p99_samples) if p99_samples else 0.0,
        "units_peak": max(unit_counts) if unit_counts else 0,
        "samples": len(p99_samples),
        "daemon_alive": (stack_h.daemon_proc.poll() is None),
    }


def stage_b(stack_h: StackHandles) -> None:
    res = asyncio.run(_flavor_b_state_dump_under_traffic(stack_h))
    print(f"  [B] samples={res['samples']} | p50_median {res['p50_median']:.2f} ms"
          f" | p99_max {res['p99_max']:.2f} ms | units peak {res['units_peak']}")
    if not res["daemon_alive"]:
        raise StageFail("daemon died mid-flavor-B")
    if res["samples"] < 30:
        raise StageFail(
            f"too few state samples ({res['samples']}); sglang state "
            f"endpoint may be flaky"
        )
    if res["p99_max"] > 200.0:
        raise StageFail(
            f"state-dump p99 max {res['p99_max']:.1f} ms > 200 ms "
            f"loose ceiling (PLAN F3-revisit hard threshold = 50 ms)"
        )


# ============================================================ C. event-router fan-in


async def _flavor_c_event_router_fanin(
    stack_h: StackHandles,
) -> Dict[str, Any]:
    """Synthesize webhook events directly via POST /aginfer/event at
    high rate.  Daemon's event_router consumes them serially; we
    measure that none are lost (sum of accepted == total fired) and
    the daemon process stays alive."""
    DAEMON_BASE = f"{DAEMON_HOST}:{DAEMON_PORT}"
    TARGET = 200  # webhook events to fire
    DURATION_LIMIT = 60.0

    fired = 0
    accepted = 0
    transport_errors = 0
    start = time.time()

    async def fire_one(cli: httpx.AsyncClient, kind: str, sid: str) -> None:
        nonlocal accepted, transport_errors
        body = {"kind": kind, "session": sid}
        try:
            r = await cli.post(
                f"http://{DAEMON_BASE}/aginfer/event", json=body, timeout=10.0,
            )
            if r.status_code == 200:
                accepted += 1
        except (httpx.RequestError, asyncio.TimeoutError):
            transport_errors += 1

    # Mix of paper §4 event kinds.  SESSION_ARRIVAL first per pid
    # so program_tracker sees a valid state machine.
    kinds_seq = [
        "session_arrival", "llm_prefill", "tool_call_start",
        "tool_call_end", "llm_prefill",
    ]
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5, read=10, write=5, pool=5),
    ) as cli:
        tasks: List[asyncio.Task] = []
        for i in range(TARGET):
            if time.time() - start > DURATION_LIMIT:
                break
            kind = kinds_seq[i % len(kinds_seq)]
            sid = f"intc-{i // len(kinds_seq):04d}"
            tasks.append(asyncio.create_task(fire_one(cli, kind, sid)))
            fired += 1
            if len(tasks) >= 32:
                await asyncio.gather(*tasks)
                tasks = []
        if tasks:
            await asyncio.gather(*tasks)
    return {
        "fired": fired, "accepted": accepted,
        "transport_errors": transport_errors,
        "daemon_alive": (stack_h.daemon_proc.poll() is None),
    }


def stage_c(stack_h: StackHandles) -> None:
    res = asyncio.run(_flavor_c_event_router_fanin(stack_h))
    print(f"  [C] fired {res['fired']} | accepted {res['accepted']} | "
          f"transport_err {res['transport_errors']}")
    if not res["daemon_alive"]:
        raise StageFail("daemon died mid-flavor-C")
    if res["fired"] == 0:
        raise StageFail("no events fired (driver bug)")
    # Daemon accepts everything; only network errors should reduce.
    accept_rate = res["accepted"] / res["fired"]
    if accept_rate < 0.99:
        raise StageFail(
            f"accept rate {accept_rate:.2%} below 99% — daemon may have "
            f"dropped events"
        )


# ============================================================ D. migrate under traffic


async def _flavor_d_migrate_under_traffic(
    stack_h: StackHandles,
) -> Dict[str, Any]:
    """While sustained chat traffic flows, post 200 migrate batches
    directly to sglang's /aginfer/migrate (mimicking what the
    daemon's outbound worker would emit).  Pass: every batch returns
    200 OK with applied + skipped accounting; sglang stays alive.

    We POST a synthetic batch with `add_tiers=[]` `remove_tiers=[]`
    so the migrate is a structural no-op — we're testing
    accept/idempotence/observability under traffic, not the actual
    eviction logic (which the per-action tests cover)."""
    SGLANG_BASE = f"{SGLANG_HOST}:{SGLANG_PORT}"
    DAEMON_BASE = f"{DAEMON_HOST}:{DAEMON_PORT}"
    DURATION = 45.0
    WORKERS = 12
    TARGET_BATCHES = 200

    deadline = time.time() + DURATION
    batches_fired = 0
    batches_ok = 0

    async def chat_worker(idx: int) -> None:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5, read=60, write=10, pool=5),
        ) as cli:
            n = 0
            while time.time() < deadline:
                await _post_chat(cli, DAEMON_BASE, _chat_body(idx, n))
                n += 1

    async def migrate_dispatcher() -> None:
        nonlocal batches_fired, batches_ok
        async with httpx.AsyncClient(timeout=10.0) as cli:
            while batches_fired < TARGET_BATCHES and time.time() < deadline:
                body = {
                    "actions": [
                        {
                            "hash": f"no-such-hash-{batches_fired}",
                            "add_tiers": [],
                            "remove_tiers": [],
                            "action_id": uuid.uuid4().hex,
                        }
                    ],
                    "batch_id": uuid.uuid4().hex,
                }
                try:
                    r = await cli.post(
                        f"http://{SGLANG_BASE}/aginfer/migrate",
                        json=body, timeout=10.0,
                    )
                    if r.status_code == 200:
                        batches_ok += 1
                except (httpx.RequestError, asyncio.TimeoutError):
                    pass
                batches_fired += 1
                await asyncio.sleep(0.05)

    await asyncio.gather(
        *(chat_worker(i) for i in range(WORKERS)),
        migrate_dispatcher(),
    )
    return {
        "batches_fired": batches_fired,
        "batches_ok": batches_ok,
        "sglang_alive": (stack_h.sglang_proc.poll() is None),
        "daemon_alive": (stack_h.daemon_proc.poll() is None),
    }


def stage_d(stack_h: StackHandles) -> None:
    res = asyncio.run(_flavor_d_migrate_under_traffic(stack_h))
    print(f"  [D] batches fired {res['batches_fired']} | ok {res['batches_ok']}")
    if not res["sglang_alive"]:
        raise StageFail("sglang died mid-flavor-D")
    if not res["daemon_alive"]:
        raise StageFail("daemon died mid-flavor-D")
    if res["batches_fired"] == 0:
        raise StageFail("no migrate batches dispatched")
    ok_rate = res["batches_ok"] / res["batches_fired"]
    if ok_rate < 0.95:
        raise StageFail(
            f"sglang migrate accept rate {ok_rate:.2%} below 95%"
        )


# ============================================================ E. threshold PUT atomicity


async def _flavor_e_threshold_put_under_traffic(
    stack_h: StackHandles,
) -> Dict[str, Any]:
    """While chat traffic flows, PUT /aginfer/thresholds (sglang
    side) repeatedly with alternating valid profiles.  Pass:
      * every PUT returns 200 (sglang accepts the apply)
      * sglang stays alive
      * daemon's view via GET /aginfer/thresholds (daemon side)
        always returns a sane (lo < hi < crit) tuple
        (any torn read here = atomic-swap contract broken)

    Sglang has PUT only; daemon has GET only.  The GET reads from
    the daemon's local cache (not synced from sglang under T22 v1).
    The daemon's GET should NEVER show inconsistent ordering — those
    fields are individual float reads in CPython, but a future
    refactor that bundles them into a tuple+lock could regress.
    """
    DAEMON_BASE = f"{DAEMON_HOST}:{DAEMON_PORT}"
    SGLANG_BASE = f"{SGLANG_HOST}:{SGLANG_PORT}"
    DURATION = 30.0
    WORKERS = 8

    deadline = time.time() + DURATION
    puts_fired = 0
    puts_ok = 0
    torn_reads = 0
    daemon_gets = 0

    # Two valid threshold profiles; alternate.  Sglang's PUT
    # requires all 4 fields including `heartbeat_s` — missing it
    # returns 400.
    profiles = [
        {"theta_lo": 0.65, "theta_hi": 0.80, "theta_crit": 0.90,
         "heartbeat_s": 5.0},
        {"theta_lo": 0.70, "theta_hi": 0.85, "theta_crit": 0.92,
         "heartbeat_s": 4.0},
    ]

    async def chat_worker(idx: int) -> None:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5, read=60, write=10, pool=5),
        ) as cli:
            n = 0
            while time.time() < deadline:
                await _post_chat(cli, DAEMON_BASE, _chat_body(idx, n))
                n += 1

    first_400_body: List[str] = []  # capture first failure body for diag

    async def threshold_writer() -> None:
        nonlocal puts_fired, puts_ok
        i = 0
        async with httpx.AsyncClient(timeout=10.0) as cli:
            while time.time() < deadline:
                body = profiles[i % len(profiles)]
                try:
                    r = await cli.put(
                        f"http://{SGLANG_BASE}/aginfer/thresholds",
                        json=body, timeout=10.0,
                    )
                    if r.status_code == 200:
                        puts_ok += 1
                    elif not first_400_body:
                        first_400_body.append(
                            f"status={r.status_code} body={r.text[:300]}"
                        )
                except (httpx.RequestError, asyncio.TimeoutError):
                    pass
                puts_fired += 1
                i += 1
                await asyncio.sleep(0.1)

    async def daemon_threshold_reader() -> None:
        nonlocal torn_reads, daemon_gets
        async with httpx.AsyncClient(timeout=10.0) as cli:
            while time.time() < deadline:
                try:
                    r = await cli.get(
                        f"http://{DAEMON_BASE}/aginfer/thresholds",
                        timeout=10.0,
                    )
                    if r.status_code == 200:
                        body = r.json()
                        lo = float(body.get("theta_lo", 0.0))
                        hi = float(body.get("theta_hi", 0.0))
                        crit = float(body.get("theta_crit", 0.0))
                        daemon_gets += 1
                        if not (lo < hi < crit):
                            torn_reads += 1
                except (httpx.RequestError, asyncio.TimeoutError):
                    pass
                await asyncio.sleep(0.05)

    await asyncio.gather(
        *(chat_worker(i) for i in range(WORKERS)),
        threshold_writer(),
        daemon_threshold_reader(),
    )
    return {
        "puts_fired": puts_fired, "puts_ok": puts_ok,
        "torn_reads": torn_reads, "daemon_gets": daemon_gets,
        "first_400_body": first_400_body,
        "sglang_alive": (stack_h.sglang_proc.poll() is None),
        "daemon_alive": (stack_h.daemon_proc.poll() is None),
    }


def stage_e(stack_h: StackHandles) -> None:
    res = asyncio.run(_flavor_e_threshold_put_under_traffic(stack_h))
    print(f"  [E] PUT fired {res['puts_fired']} | ok {res['puts_ok']} | "
          f"daemon_gets {res['daemon_gets']} | torn {res['torn_reads']}")
    if res["first_400_body"]:
        print(f"  [E] first 400: {res['first_400_body'][0]}")
    if not res["sglang_alive"]:
        raise StageFail("sglang died mid-flavor-E")
    if not res["daemon_alive"]:
        raise StageFail("daemon died mid-flavor-E")
    if res["puts_fired"] == 0:
        raise StageFail("no PUTs fired")
    ok_rate = res["puts_ok"] / res["puts_fired"]
    if ok_rate < 0.95:
        raise StageFail(
            f"sglang threshold PUT accept rate {ok_rate:.2%} below 95%"
        )
    if res["daemon_gets"] < 10:
        raise StageFail(
            f"daemon /aginfer/thresholds returned <10 successful "
            f"reads ({res['daemon_gets']}); endpoint may be dead"
        )
    if res["torn_reads"] > 0:
        raise StageFail(
            f"observed {res['torn_reads']} torn reads of daemon "
            f"thresholds (lo < hi < crit violated) under concurrent "
            f"traffic"
        )


# ============================================================ G. SESSION_END migrate e2e


async def _flavor_g_session_end_migrate(stack_h: StackHandles) -> Dict[str, Any]:
    """#191 (closes #187 audit G3): the LIVE SESSION_END seam end-to-
    end — webhook `POST /aginfer/event {session_end}` → the daemon's
    composed handler (end → kv_scheduler.handle migrate D_t → PUT
    {ENDED}) → sglang's `per_program_usage[pid].state` reflects ENDED.

    Deterministic assertion: the ENDED state PROPAGATES (the F5 PUT
    round-trip).  The actual demote/drop of the session-scoped units is
    policy + cost-model dependent (cold-start `h_max≈0` makes V_u(keep)
    positive, so the policy may decline) — so demotion is OBSERVED and
    logged, not hard-gated.  Both terminal shapes count as success:
    (a) ENDED echoed with units still resident (policy declined), or
    (b) the units fully dropped + the entry GC'd (#186) — the stronger
    "migrate dropped everything" outcome."""
    SGLANG_BASE = f"{SGLANG_HOST}:{SGLANG_PORT}"
    DAEMON_BASE = f"{DAEMON_HOST}:{DAEMON_PORT}"
    pid = f"sessend-{uuid.uuid4().hex[:8]}"

    async def get_state(cli: httpx.AsyncClient) -> Dict[str, Any]:
        r = await cli.get(f"http://{SGLANG_BASE}/aginfer/state", timeout=30.0)
        r.raise_for_status()
        return r.json()

    def _units_of(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [u for u in state.get("units", []) if pid in u.get("session_ids", [])]

    def _hbm(units: List[Dict[str, Any]]) -> int:
        return sum(1 for u in units if "HBM" in u.get("residence", []))

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5, read=60, write=10, pool=5),
    ) as cli:
        # 1. Tag EXCLUSIVE units with pid — unique prompts, direct to
        #    sglang (the session_id_passthrough path), so the units are
        #    p's alone (holders == {pid}) → they ARE its session_scoped
        #    set at SESSION_END.
        for n in range(6):
            body = {
                "model": MODEL,
                "messages": [
                    {"role": "system",
                     "content": f"{_PAD} (sessend {pid} {n} {uuid.uuid4().hex[:8]})"},
                    {"role": "user",
                     "content": f"reply 'ack' ({pid} {n} {uuid.uuid4().hex[:12]})"},
                ],
                "max_tokens": 8, "temperature": 0.0,
                "program_id": pid,
            }
            r = await cli.post(
                f"http://{SGLANG_BASE}/v1/chat/completions", json=body, timeout=60.0,
            )
            r.raise_for_status()

        st = await get_state(cli)
        units_before = _units_of(st)
        ppu_before = st.get("per_program_usage", {}).get(pid)
        hbm_before = _hbm(units_before)

        # #199 audit: assert the REAL sglang dump emits decode_bytes_per_
        # token on every HBM subpool (the daemon's forecast_inflight_demand
        # input).  Without this, a silent regression in sglang's dump would
        # leave the daemon defaulting to 0 forever and every other test
        # would still pass.
        hbm_subpools = st["pool_usage"]["HBM"]["subpools"]
        decode_bpt_ok = bool(hbm_subpools) and all(
            isinstance(e.get("decode_bytes_per_token"), int)
            and e["decode_bytes_per_token"] >= 0
            for e in hbm_subpools.values()
        )
        decode_bpt_sample = {
            sp: e.get("decode_bytes_per_token")
            for sp, e in hbm_subpools.items()
        }

        # 2. Fire SESSION_END at the DAEMON.
        ev = await cli.post(
            f"http://{DAEMON_BASE}/aginfer/event",
            json={"kind": "session_end", "session": pid}, timeout=10.0,
        )
        event_accepted = ev.status_code == 200

        # 3. Poll sglang for the terminal SESSION_END shape (~10s).
        ended = False
        gc_dropped = False
        units_after = units_before
        hbm_after = hbm_before
        for _ in range(50):
            await asyncio.sleep(0.2)
            st = await get_state(cli)
            ppu = st.get("per_program_usage", {}).get(pid)
            units_after = _units_of(st)
            hbm_after = _hbm(units_after)
            if ppu and ppu.get("state") == "ENDED":
                ended = True
                break
            if ppu is None and not units_after and ppu_before is not None:
                # fully ended → units dropped → entry GC'd (#186)
                ended = True
                gc_dropped = True
                break

    return {
        "pid": pid,
        "units_before": len(units_before),
        "ppu_before_state": (ppu_before or {}).get("state"),
        "event_accepted": event_accepted,
        "ended": ended,
        "gc_dropped": gc_dropped,
        "hbm_before": hbm_before,
        "hbm_after": hbm_after,
        "units_after": len(units_after),
        "demoted": (hbm_after < hbm_before) or (len(units_after) < len(units_before)),
        "decode_bpt_ok": decode_bpt_ok,
        "decode_bpt_sample": decode_bpt_sample,
        "sglang_alive": stack_h.sglang_proc.poll() is None,
        "daemon_alive": stack_h.daemon_proc.poll() is None,
    }


def stage_g(stack_h: StackHandles) -> None:
    res = asyncio.run(_flavor_g_session_end_migrate(stack_h))
    demote_note = (
        "GC'd (units dropped)" if res["gc_dropped"]
        else (f"demoted {res['hbm_before']}→{res['hbm_after']} HBM units"
              if res["demoted"] else "policy declined (cold-start V_u; units kept)")
    )
    print(f"  [G] pid={res['pid']} units_before={res['units_before']} "
          f"ppu_before={res['ppu_before_state']} → ENDED={res['ended']} "
          f"| migrate: {demote_note} | decode_bpt={res['decode_bpt_sample']}")
    if not res["sglang_alive"]:
        raise StageFail("sglang died mid-flavor-G")
    if not res["daemon_alive"]:
        raise StageFail("daemon died mid-flavor-G")
    # #199 audit: the real sglang dump must emit decode_bytes_per_token on
    # every HBM subpool (the §8 forecast trajectory input).
    if not res["decode_bpt_ok"]:
        raise StageFail(
            "sglang /aginfer/state HBM subpools missing decode_bytes_per_"
            f"token (#199); got {res['decode_bpt_sample']}")
    if res["units_before"] == 0:
        raise StageFail(
            "no units tagged with the SESSION_END pid — program_id "
            "tagging or launch config broken (driver precondition)"
        )
    if not res["event_accepted"]:
        raise StageFail("daemon rejected the SESSION_END webhook")
    if not res["ended"]:
        raise StageFail(
            "SESSION_END did not propagate to sglang: "
            "per_program_usage[pid].state never became ENDED (and the "
            "units were not dropped+GC'd) — the webhook→daemon→PUT→"
            "sglang seam is broken"
        )


# ============================================================ T26 measurement


async def _flavor_t26_measurement(stack_h: StackHandles) -> Dict[str, Any]:
    """T26 (#200): under program-tagged decode load, the REAL sglang
    /aginfer/state must now report measured throughput / in-flight (was
    0.0/{} placeholders): ``throughput_ema.prefill_bps > 0``,
    ``decode_per_program[pid] > 0`` for actively-decoding programs, and
    ``per_program_usage[pid].hbm.inflight`` populated.

    Fires a few PROGRAM-TAGGED chat requests with longer outputs (so
    they decode for a while) straight at sglang, and polls /aginfer/state
    concurrently, capturing the peak observed values."""
    SGLANG_BASE = f"{SGLANG_HOST}:{SGLANG_PORT}"
    N_PROGS = 4
    DURATION = 25.0
    prog_ids = [f"t26-{i}-{uuid.uuid4().hex[:6]}" for i in range(N_PROGS)]
    deadline = time.time() + DURATION

    tagged = set(prog_ids)
    seen = {
        "prefill_bps_max": 0.0,
        "decode_pos_tagged": set(),   # TAGGED pids seen with decode rate > 0
        "inflight_tagged": set(),     # TAGGED pids seen with inflight > 0
        "inflight_polls": 0,
        "polls": 0,
    }

    async def worker(pid: str) -> None:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5, read=120, write=10, pool=5),
        ) as cli:
            n = 0
            while time.time() < deadline:
                body = {
                    "model": MODEL,
                    "messages": [
                        {"role": "system",
                         "content": f"{_PAD} ({pid} {n} {uuid.uuid4().hex[:8]})"},
                        {"role": "user",
                         "content": (f"Count slowly from 1 to 40, one number "
                                     f"per line. ({pid} {n} "
                                     f"{uuid.uuid4().hex[:12]})")},
                    ],
                    # Longer decode so the request is observably in-flight.
                    "max_tokens": 96, "temperature": 0.0,
                    "program_id": pid,
                }
                try:
                    await cli.post(
                        f"http://{SGLANG_BASE}/v1/chat/completions",
                        json=body, timeout=120.0)
                except Exception:
                    pass
                n += 1

    async def poller() -> None:
        async with httpx.AsyncClient(timeout=10.0) as cli:
            while time.time() < deadline:
                try:
                    r = await cli.get(f"http://{SGLANG_BASE}/aginfer/state")
                    if r.status_code == 200:
                        body = r.json()
                        seen["polls"] += 1
                        te = body.get("throughput_ema", {})
                        seen["prefill_bps_max"] = max(
                            seen["prefill_bps_max"],
                            float(te.get("prefill_bps", 0.0)))
                        for pid, rate in (te.get("decode_per_program", {}) or {}).items():
                            if pid in tagged and float(rate) > 0.0:
                                seen["decode_pos_tagged"].add(pid)
                        ppu = body.get("per_program_usage", {}) or {}
                        any_inflight = False
                        for pid, e in ppu.items():
                            infl = e.get("hbm", {}).get("inflight") or {}
                            if pid in tagged and any(v > 0 for v in infl.values()):
                                seen["inflight_tagged"].add(pid)
                                any_inflight = True
                        if any_inflight:
                            seen["inflight_polls"] += 1
                except Exception:
                    pass
                await asyncio.sleep(0.25)

    await asyncio.gather(*(worker(p) for p in prog_ids), poller())
    return {
        "prefill_bps_max": seen["prefill_bps_max"],
        "decode_pos_tagged": len(seen["decode_pos_tagged"]),
        "inflight_tagged": len(seen["inflight_tagged"]),
        "inflight_polls": seen["inflight_polls"],
        "n_tagged": len(tagged),
        "polls": seen["polls"],
        "sglang_alive": stack_h.sglang_proc.poll() is None,
        "daemon_alive": stack_h.daemon_proc.poll() is None,
    }


def stage_t26(stack_h: StackHandles) -> None:
    res = asyncio.run(_flavor_t26_measurement(stack_h))
    n = res["n_tagged"]
    print(f"  [T26] polls={res['polls']} prefill_bps_max={res['prefill_bps_max']:.3g} "
          f"decode_pos_tagged={res['decode_pos_tagged']}/{n} "
          f"inflight_tagged={res['inflight_tagged']}/{n} "
          f"inflight_polls={res['inflight_polls']}")
    if not res["sglang_alive"]:
        raise StageFail("sglang died mid-flavor-T26")
    if res["polls"] == 0:
        raise StageFail("no /aginfer/state polls succeeded")
    if res["prefill_bps_max"] <= 0.0:
        raise StageFail("T26: prefill_bps never became > 0 under prefill load "
                        "(measurement not wired)")
    # EVERY tagged program must be measured (not just one) — a single stuck
    # program or a misattributed pid would otherwise pass (#200 audit).
    if res["decode_pos_tagged"] < n:
        raise StageFail(f"T26: only {res['decode_pos_tagged']}/{n} tagged "
                        f"programs got a positive decode rate (per-program "
                        f"decode measurement incomplete)")
    if res["inflight_tagged"] < n:
        raise StageFail(f"T26: only {res['inflight_tagged']}/{n} tagged "
                        f"programs had inflight populated")


# ============================================================ F. escalate-to-fatal


async def _flavor_f_dead_sglang_resilience(
    results_dir: Path,
) -> Dict[str, Any]:
    """F: launch a SEPARATE daemon pointed at a dead sglang URL,
    fire low-volume events, assert the daemon STAYS ALIVE.

    DESIGN §10 contract (T164 stage A2): low-traffic dead-sglang
    MUST NOT escalate to fatal — every event fails fast at the
    state-fetch step, the outbound queue never accumulates, the
    sustained-escalate counters stay below threshold.  The daemon
    is expected to keep accepting events on /aginfer/event and to
    increment its state_fetch_failed observability counter.

    The POSITIVE escalate-to-fatal case (true high-volume
    sustained-fail → daemon self-kills) lives in T164 stage B0 (in-
    process subprocess) and C0 (uvicorn-hosted subprocess) — those
    populate the outbound queue directly via the OutboundQueue API
    rather than through the event-router, because event-driven
    state-fetch-failure path returns early before the queue gets
    populated.  Integrating the queue-populating path into this
    flavor would require a fake-sglang stub that responds OK on
    /aginfer/state but fails on /aginfer/migrate — out of scope
    here; T164 owns that integration.
    """
    DEAD_PORT = 30099  # nothing listening
    test_daemon_port = 9131

    ts = time.strftime("%Y%m%d_%H%M%S")
    daemon_log = results_dir / f"{ts}_flavor_f_daemon.log"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(_HERE.parent.parent)
    env["AGINFER_DATA_DIR"] = str(results_dir / f"flavor_f_data_{ts}")
    args = [
        sys.executable, "-m", "daemon.main",
        "--sglang-base-url", f"http://127.0.0.1:{DEAD_PORT}",
        "--host", "127.0.0.1", "--port", str(test_daemon_port),
        "--kv-scheduler", "enabled",
        "--admission-controller", "enabled",
        # Tight thresholds: if the daemon WERE to populate the
        # outbound queue, these would fire fast.  Since it doesn't
        # (low-traffic path), they stay below threshold.
        "--sustained-escalate-fails", "5",
        "--sustained-escalate-age-s", "5.0",
    ]
    f = open(daemon_log, "w")
    proc = subprocess.Popen(
        args, env=env, stdout=f, stderr=subprocess.STDOUT,
        cwd=str(_HERE.parent.parent),
    )
    events_accepted = 0
    try:
        async with httpx.AsyncClient(timeout=2.0) as cli:
            deadline = time.time() + 30.0
            while time.time() < deadline:
                try:
                    r = await cli.get(
                        f"http://127.0.0.1:{test_daemon_port}/health"
                    )
                    if r.status_code == 200:
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.5)
            else:
                raise RuntimeError("flavor-F daemon never came up")

        # Fire a low-volume burst (8 events / 8 s).  Each one
        # triggers state-fetch → fails on dead port → early return.
        # No migrate enqueued → outbound queue stays empty → no
        # escalate-to-fatal trip even with the tight thresholds.
        async with httpx.AsyncClient(timeout=10.0) as cli:
            for i in range(8):
                body = {
                    "kind": "memory_pressure",
                    "session": f"flavor-f-{i}",
                }
                try:
                    r = await cli.post(
                        f"http://127.0.0.1:{test_daemon_port}/aginfer/event",
                        json=body, timeout=5.0,
                    )
                    if r.status_code == 200:
                        events_accepted += 1
                except Exception:
                    pass
                await asyncio.sleep(1.0)

        # Wait 8 more seconds — total elapsed > 5 s escalate-age
        # threshold.  If the daemon were going to fatal, it would
        # have by now.  Verify it's still alive AND /health still
        # answers.
        await asyncio.sleep(8.0)
        daemon_still_alive = (proc.poll() is None)
        health_ok = False
        if daemon_still_alive:
            try:
                async with httpx.AsyncClient(timeout=2.0) as cli:
                    r = await cli.get(
                        f"http://127.0.0.1:{test_daemon_port}/health"
                    )
                    health_ok = (r.status_code == 200)
            except Exception:
                health_ok = False
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    rc = proc.returncode

    forensic_dir = Path(env["AGINFER_DATA_DIR"]) / "forensic"
    forensic_files = (
        list(forensic_dir.glob("sglang_sustained_unreachable_*.json"))
        if forensic_dir.exists() else []
    )

    return {
        "daemon_returncode": rc,
        "daemon_still_alive_during_test": daemon_still_alive,
        "health_ok_during_test": health_ok,
        "events_accepted": events_accepted,
        "forensic_count": len(forensic_files),
        "forensic_dir": str(forensic_dir),
        "daemon_log": str(daemon_log),
    }


def stage_f() -> None:
    """Stage F: dead-sglang resilience.  Verify the daemon SURVIVES
    low-volume traffic against a dead sglang (no false-positive
    escalate-to-fatal).  T164 stages B0/C0 cover the positive case
    (true high-volume sustained-fail → fatal)."""
    results_dir = _HERE / "results"
    results_dir.mkdir(exist_ok=True)
    res = asyncio.run(_flavor_f_dead_sglang_resilience(results_dir))
    print(f"  [F] daemon alive during test: {res['daemon_still_alive_during_test']} | "
          f"health ok: {res['health_ok_during_test']} | "
          f"events accepted: {res['events_accepted']} | "
          f"forensic: {res['forensic_count']}")
    if not res["daemon_still_alive_during_test"]:
        raise StageFail(
            f"daemon died (rc={res['daemon_returncode']}) under "
            f"low-volume dead-sglang traffic — false-positive "
            f"escalate-to-fatal trip; log={res['daemon_log']}"
        )
    if not res["health_ok_during_test"]:
        raise StageFail(
            "daemon process alive but /health is not responding — "
            "event loop may be stuck"
        )
    if res["events_accepted"] < 6:
        raise StageFail(
            f"only {res['events_accepted']}/8 events accepted — "
            f"daemon may have stopped consuming the event endpoint"
        )
    if res["forensic_count"] > 0:
        raise StageFail(
            f"daemon dropped {res['forensic_count']} forensic "
            f"file(s) under low-volume dead-sglang — false-positive "
            f"fatal trip"
        )


# ============================================================ run


def main() -> int:
    results_dir = _HERE / "results"
    results_dir.mkdir(exist_ok=True)
    failures: List[str] = []

    # Skip F at first — it brings up its own daemon against a dead
    # sglang port, so the shared stack would interfere if we ran
    # F first.  Use the shared stack for A–E, run F last (after the
    # shared stack is torn down so port reuse is clean).
    print("[integration_stress] launching shared sglang+daemon stack…")
    t_start = time.time()
    daemon_log_path = None
    try:
        with stack(results_dir) as stack_h:
            daemon_log_path = stack_h.daemon_log  # for the post-teardown check
            t_ready = time.time() - t_start
            print(f"[integration_stress] stack ready in {t_ready:.0f}s")
            for label, fn in [
                ("A proxy hot-path under load",     stage_a),
                ("B state-dump under sustained traffic", stage_b),
                ("C event-router fan-in throughput", stage_c),
                ("D migrate under traffic",         stage_d),
                ("E threshold PUT atomicity",       stage_e),
                ("G SESSION_END migrate e2e",       stage_g),
                ("T26 throughput/inflight measurement", stage_t26),
            ]:
                try:
                    print(f"[stage {label}] starting…")
                    t0 = time.time()
                    fn(stack_h)
                    dt = time.time() - t0
                    print(f"  {_green('PASS')} stage {label} "
                          f"({dt:.1f}s)")
                except StageFail as exc:
                    failures.append(label)
                    print(f"  {_red('FAIL')} stage {label}: {exc}")
                except Exception as exc:  # noqa: BLE001
                    failures.append(label)
                    print(
                        f"  {_red('FAIL')} stage {label}: "
                        f"unexpected {type(exc).__name__}: {exc}"
                    )
    except Exception as exc:
        print(f"[integration_stress] shared stack failed: {exc}")
        failures.append("shared-stack-init")

    # Stage H (post-teardown): the daemon's SIGTERM shutdown handler must
    # run CLEAN — emit the cycle_summary metric + the final observability
    # summary, with no traceback.  Regression guard for the #194 bug where
    # _emit_cycle_summary referenced a deleted `admission` var → NameError
    # aborted the whole shutdown handler and dropped the telemetry.
    try:
        print("[stage H] shutdown-handler clean…")
        if daemon_log_path is None:
            raise StageFail("no daemon log captured (stack never came up)")
        txt = Path(daemon_log_path).read_text(errors="replace")
        if "event=cycle_summary" not in txt:
            raise StageFail(
                "daemon never emitted event=cycle_summary on shutdown "
                f"(handler crashed?); log={daemon_log_path}")
        for bad in ("Application shutdown failed", "NameError",
                    "Traceback (most recent call last)"):
            # Only flag tracebacks in the shutdown region (after the
            # cycle_summary attempt) — startup-phase noise is unrelated.
            tail = txt.split("event=cycle_summary", 1)[-1]
            if bad in tail:
                raise StageFail(
                    f"daemon shutdown handler error ({bad!r}); "
                    f"log={daemon_log_path}")
        print(f"  {_green('PASS')} stage H shutdown-handler clean")
    except StageFail as exc:
        failures.append("H")
        print(f"  {_red('FAIL')} stage H: {exc}")

    # Stage F runs after the shared stack is down.
    try:
        print(f"[stage F] starting…")
        t0 = time.time()
        stage_f()
        print(f"  {_green('PASS')} stage F escalate-to-fatal "
              f"({time.time() - t0:.1f}s)")
    except StageFail as exc:
        failures.append("F")
        print(f"  {_red('FAIL')} stage F: {exc}")
    except Exception as exc:  # noqa: BLE001
        failures.append("F")
        print(f"  {_red('FAIL')} stage F: "
              f"unexpected {type(exc).__name__}: {exc}")

    n_total = 7
    if failures:
        print(_red(
            f"\nintegration_stress FAILED ({len(failures)}/{n_total}): "
            f"{failures}"
        ))
        return 1
    print(_green(
        f"\nintegration_stress PASS — all {n_total} flavors green"
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
