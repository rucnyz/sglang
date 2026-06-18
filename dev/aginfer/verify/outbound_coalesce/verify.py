"""#228 / #227 — outbound coalescing + freshness bound + LATENCY PROOF.

Root cause (TP=4 A3): the daemon pushes a `hints` PUT every event (T40), so
hints are ~99% of outbound traffic (≈11.7k vs ≈130 migrate per cycle).  The
outbound channel is single-flight at sglang (one communicator, serialised
apply ≈ one scheduler iteration per POST), so every time-sensitive `migrate`
waits behind that idempotent flood — ageing ~1.8 s until the radix tree
diverges and sglang rejects it (`remove_*_not_leaf`).

Fix: the worker drains the queued burst and coalesces it per endpoint —
hints merge to ONE PUT (overwrite-by-stamp, latest per hash); migrates ALSO
merge to ONE POST (latest decision per hash) after the #227 freshness drop
— the live a3 cycle-2 evidence showed a migrate BURST dispatched
individually re-clogs the single-flight channel (oldest-age 700 ms, rejects
back); program_paused coalesces by pid.

Stages:

  A. _partition_and_coalesce — pure correctness (deterministic, injected clock)
    A0 N hints batches  → ONE hints PUT, every hash present, latest-stamp wins
    A1 migrate burst → ONE POST, every unit, latest-decision-per-hash
    A2 #227 freshness: a stale migrate action is dropped before the merge
    A3 program_paused coalesced by pid (latest state wins), never dropped
    A4 dispatch ORDER: program_paused → migrate → hints (time-sensitive never
       waits behind the idempotent flood); unknown endpoint passes through

  B. LATENCY PROOF — real OutboundQueue worker, recording HTTP client
    B0 with N=200 hints queued AHEAD of a migrate, the migrate POST is among
       the FIRST HTTP calls (O(1)), NOT the 201st — and the 200 hints
       collapse to ONE PUT carrying all 200.  (Old FIFO: migrate waits behind
       200 → 201st call.)
    B1 wall-clock: migrate dispatch latency under the flood is bounded by a
       few apply-times, not N×apply-time (N≥3 trials, report mean).
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

_HERE = Path(__file__).resolve().parent
_AGINFER_ROOT = _HERE.parent.parent
if str(_AGINFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGINFER_ROOT))

from daemon.outbound import (  # noqa: E402
    OutboundBatch, OutboundQueue, _partition_and_coalesce,
)


def _green(s: str) -> str: return f"\033[32m{s}\033[0m"
def _red(s: str) -> str:   return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


# ---- fixtures -------------------------------------------------------------


def _hint_batch(ts: float, hints: List[Dict[str, Any]]) -> OutboundBatch:
    return OutboundBatch(batch_id=f"h{ts}", endpoint="hints",
                         body={"hints": hints, "batch_id": f"h{ts}"},
                         enqueue_ts=ts, method="PUT")


def _migrate_batch(ts: float, actions: List[Dict[str, Any]]) -> OutboundBatch:
    return OutboundBatch(batch_id=f"m{ts}", endpoint="migrate",
                         body={"actions": actions, "batch_id": f"m{ts}"},
                         enqueue_ts=ts, method="POST")


def _paused_batch(ts: float, pid: str, state: str) -> OutboundBatch:
    return OutboundBatch(batch_id=f"p{ts}", endpoint="program_paused",
                         body={"pid": pid, "state": state,
                               "pre_pause_state": None, "batch_id": f"p{ts}"},
                         enqueue_ts=ts, method="PUT")


# ============================================================ A. pure


def stage_a0_hints_coalesce_latest_wins() -> None:
    now = 1000.0
    batches = [
        _hint_batch(now - 0.3, [{"hash": "u1", "p_hat": 0.1, "stamp": 1},
                                {"hash": "u2", "p_hat": 0.2, "stamp": 1}]),
        _hint_batch(now - 0.2, [{"hash": "u2", "p_hat": 0.9, "stamp": 2}]),
        _hint_batch(now - 0.1, [{"hash": "u3", "p_hat": 0.3, "stamp": 3}]),
    ]
    out, stats = _partition_and_coalesce(
        batches, now_ts=now, migrate_freshness_ms=30000)
    hint_posts = [b for b in out if b.endpoint == "hints"]
    if len(hint_posts) != 1:
        raise StageFail(f"A0: 3 hint batches must coalesce to 1 PUT; "
                        f"got {len(hint_posts)}")
    hints = {h["hash"]: h for h in hint_posts[0].body["hints"]}
    if set(hints) != {"u1", "u2", "u3"}:
        raise StageFail(f"A0: coalesced PUT must carry every hash; got {set(hints)}")
    if hints["u2"]["p_hat"] != 0.9 or hints["u2"]["stamp"] != 2:
        raise StageFail(f"A0: latest stamp must win for u2; got {hints['u2']}")
    if stats["hints_in"] != 3 or stats["hints_out"] != 1:
        raise StageFail(f"A0: stats wrong: {stats}")
    # audit #4: HIGHEST stamp wins even when enqueued OUT OF ORDER (a higher
    # stamp enqueued EARLIER must not be overwritten by a later-enqueued
    # lower stamp) — robust to burst reordering, matches sglang's §10
    # overwrite-by-stamp.
    inv = [
        _hint_batch(now - 0.3, [{"hash": "w", "p_hat": 0.9, "stamp": 5}]),
        _hint_batch(now - 0.1, [{"hash": "w", "p_hat": 0.1, "stamp": 2}]),
    ]
    out2, _ = _partition_and_coalesce(inv, now_ts=now, migrate_freshness_ms=30000)
    hw = {h["hash"]: h for h in
          next(b for b in out2 if b.endpoint == "hints").body["hints"]}["w"]
    if hw["stamp"] != 5:
        raise StageFail(f"A0: highest stamp (5) must win regardless of enqueue "
                        f"order; got stamp={hw['stamp']}")
    print(_green("  [A0] hints coalesce → 1 PUT, all hashes, max-stamp-wins "
                 "(order-independent) OK"))


def stage_a1_migrates_coalesce_latest_wins() -> None:
    """#228-complete (cycle-2 evidence): a migrate BURST coalesces to ONE
    POST, latest decision per unit hash — individual dispatch of a burst
    re-clogged the single-flight channel (oldest-age 700 ms, rejects back).
    The single-flight ceiling remains, but a burst is one round-trip, not N."""
    now = 1000.0
    batches = [
        _migrate_batch(now - 0.3, [{"hash": "u1", "add_tiers": [],
                                    "remove_tiers": ["HBM"], "action_id": "a1"}]),
        _migrate_batch(now - 0.2, [{"hash": "u2", "add_tiers": [],
                                    "remove_tiers": ["HBM"], "action_id": "a2"}]),
        # later decision for u1 SUPERSEDES the earlier one (latest-per-hash):
        _migrate_batch(now - 0.1, [{"hash": "u1", "add_tiers": ["DRAM"],
                                    "remove_tiers": [], "action_id": "a3"}]),
    ]
    out, stats = _partition_and_coalesce(
        batches, now_ts=now, migrate_freshness_ms=30000)
    migs = [b for b in out if b.endpoint == "migrate"]
    if len(migs) != 1:
        raise StageFail(f"A1: a migrate burst must coalesce to 1 POST; "
                        f"got {len(migs)}")
    acts = {a["hash"]: a for a in migs[0].body["actions"]}
    if set(acts) != {"u1", "u2"}:
        raise StageFail(f"A1: coalesced POST must carry every unit; got {set(acts)}")
    if acts["u1"]["action_id"] != "a3":
        raise StageFail(f"A1: latest decision per hash must win — u1 should be "
                        f"a3, got {acts['u1']['action_id']}")
    if stats["migrate_in"] != 3 or stats["migrate_out"] != 1:
        raise StageFail(f"A1: stats wrong: {stats}")
    print(_green("  [A1] migrate burst → 1 POST, every unit, latest-per-hash OK"))


def stage_a2_migrate_freshness_drop() -> None:
    now = 1000.0
    batches = [
        _migrate_batch(now - 5.0, [{"hash": "stale", "add_tiers": [],
                                    "remove_tiers": ["HBM"], "action_id": "s"}]),
        _migrate_batch(now - 0.1, [{"hash": "fresh", "add_tiers": [],
                                    "remove_tiers": ["HBM"], "action_id": "f"}]),
    ]
    # freshness=1000ms: the 5 s-old batch is stale-dropped BEFORE the merge;
    # only the fresh action survives into the single coalesced POST.
    out, stats = _partition_and_coalesce(
        batches, now_ts=now, migrate_freshness_ms=1000)
    migs = [b for b in out if b.endpoint == "migrate"]
    acts = {a["hash"] for b in migs for a in b.body["actions"]}
    if acts != {"fresh"}:
        raise StageFail(f"A2: stale migrate must be dropped, only fresh kept; "
                        f"got {acts}")
    if stats["migrate_dropped_stale"] != 1 or stats["migrate_out"] != 1:
        raise StageFail(f"A2: stats wrong: {stats}")
    # freshness=0 disables the bound → BOTH actions survive (in 1 coalesced POST).
    out0, _ = _partition_and_coalesce(batches, now_ts=now, migrate_freshness_ms=0)
    acts0 = {a["hash"] for b in out0 if b.endpoint == "migrate"
             for a in b.body["actions"]}
    if acts0 != {"stale", "fresh"}:
        raise StageFail(f"A2: freshness=0 must keep both actions; got {acts0}")
    print(_green("  [A2] #227 freshness: stale action dropped, fresh kept, "
                 "0=disabled (both actions) OK"))


def stage_a3_paused_coalesce_by_pid() -> None:
    now = 1000.0
    batches = [
        _paused_batch(now - 0.3, "p1", "PAUSED"),
        _paused_batch(now - 0.2, "p1", "REASONING"),   # newer for p1
        _paused_batch(now - 0.1, "p2", "ENDED"),
    ]
    out, stats = _partition_and_coalesce(
        batches, now_ts=now, migrate_freshness_ms=30000)
    paused = [b for b in out if b.endpoint == "program_paused"]
    by_pid = {b.body["pid"]: b.body["state"] for b in paused}
    if by_pid != {"p1": "REASONING", "p2": "ENDED"}:
        raise StageFail(f"A3: paused coalesce-by-pid latest-wins wrong: {by_pid}")
    if stats["paused_out"] != 2:
        raise StageFail(f"A3: paused_out wrong: {stats}")
    print(_green("  [A3] program_paused coalesced by pid, latest-wins OK"))


def stage_a4_dispatch_order() -> None:
    now = 1000.0
    batches = [
        _hint_batch(now - 0.4, [{"hash": "u1", "p_hat": 0.1, "stamp": 1}]),
        _migrate_batch(now - 0.3, [{"hash": "u2", "add_tiers": [],
                                    "remove_tiers": ["HBM"], "action_id": "a"}]),
        _paused_batch(now - 0.2, "p1", "PAUSED"),
    ]
    out, _ = _partition_and_coalesce(
        batches, now_ts=now, migrate_freshness_ms=30000)
    order = [b.endpoint for b in out]
    if order != ["program_paused", "migrate", "hints"]:
        raise StageFail(f"A4: order must be paused→migrate→hints (time-sensitive "
                        f"never behind the flood); got {order}")
    print(_green("  [A4] dispatch order paused→migrate→hints OK"))


# ============================================================ B. latency


class _RecordingClient:
    """Records (verb, endpoint, monotonic_ts) per call; simulates the
    single-flight sglang apply with a fixed per-POST sleep."""
    def __init__(self, apply_s: float):
        self.apply_s = apply_s
        self.calls: List[Tuple[str, str, float]] = []
        self._t0 = time.monotonic()

    def _ep(self, url: str) -> str:
        return url.rstrip("/").rsplit("/", 1)[-1]

    async def post(self, url, json=None):
        self.calls.append(("POST", self._ep(url), time.monotonic() - self._t0))
        await asyncio.sleep(self.apply_s)
        return _Resp()

    async def request(self, method, url, json=None):
        self.calls.append((method, self._ep(url), time.monotonic() - self._t0))
        await asyncio.sleep(self.apply_s)
        return _Resp()

    async def aclose(self):
        return None


class _Resp:
    status_code = 200
    text = ""
    def json(self): return {}


def stage_b0_migrate_not_behind_hint_flood() -> None:
    """The decisive structural proof: a migrate queued BEHIND 200 hint PUTs
    must dispatch as an O(1) call, not the 201st — and the 200 hints collapse
    to ONE PUT carrying all 200 hashes."""
    async def _go():
        client = _RecordingClient(apply_s=0.005)
        ob = OutboundQueue(sglang_base_url="http://x", http_client=client)
        N = 200
        # Enqueue 200 hint PUTs, THEN one migrate (worst case: at the back).
        for i in range(N):
            ob.enqueue_hints([{"hash": f"u{i}", "p_hat": 0.1,
                               "lambda": 0.01, "stamp": i}])
        ob.enqueue_migrate([{"hash": "evict-me", "add_tiers": [],
                             "remove_tiers": ["HBM"], "action_id": "a0"}])
        await ob.start()
        await ob.queue.join()
        await ob.stop()
        return client.calls

    calls = asyncio.run(_go())
    posts = [c for c in calls if c[1] == "migrate"]
    puts = [c for c in calls if c[1] == "hints"]
    if len(posts) != 1:
        raise StageFail(f"B0: expected 1 migrate POST; got {len(posts)}")
    if len(puts) != 1:
        raise StageFail(f"B0: 200 hint PUTs must coalesce to 1; got {len(puts)}")
    # The migrate must be among the FIRST calls (order = paused→migrate→hints),
    # NOT the 201st — i.e. it did NOT wait behind the flood.
    migrate_idx = calls.index(posts[0])
    if migrate_idx > 1:
        raise StageFail(f"B0: migrate dispatched as call #{migrate_idx} — it "
                        f"waited behind the hint flood (should be O(1))")
    print(_green(f"  [B0] migrate is call #{migrate_idx} (not #{200}); 200 hints "
                 f"→ 1 PUT — migrate does NOT wait behind the flood OK"))


def stage_b1_latency_bounded_under_flood() -> None:
    """Wall-clock: migrate enqueue→dispatch latency under a 200-hint flood is
    bounded by a few apply-times, not N×apply-time.  N≥3 trials, mean."""
    APPLY = 0.005
    N = 200

    async def _trial() -> float:
        client = _RecordingClient(apply_s=APPLY)
        ob = OutboundQueue(sglang_base_url="http://x", http_client=client)
        for i in range(N):
            ob.enqueue_hints([{"hash": f"u{i}", "p_hat": 0.1,
                               "lambda": 0.01, "stamp": i}])
        t_enq = time.monotonic()
        ob.enqueue_migrate([{"hash": "evict-me", "add_tiers": [],
                             "remove_tiers": ["HBM"], "action_id": "a0"}])
        await ob.start()
        await ob.queue.join()
        await ob.stop()
        # absolute monotonic of the migrate call:
        mt = next(c[2] for c in client.calls if c[1] == "migrate")
        return (client._t0 + mt) - t_enq

    lats = [asyncio.run(_trial()) for _ in range(3)]
    mean = sum(lats) / len(lats)
    serial_floor = N * APPLY  # what FIFO-behind-the-flood would cost
    # Coalesced: migrate fires near-first, so latency << serial_floor.
    if mean >= serial_floor * 0.25:
        raise StageFail(
            f"B1: migrate latency {mean*1000:.1f}ms not << serial floor "
            f"{serial_floor*1000:.0f}ms (N×apply) — flood not un-clogged. "
            f"trials={[f'{x*1000:.1f}ms' for x in lats]}")
    print(_green(f"  [B1] migrate latency mean={mean*1000:.1f}ms over 3 trials "
                 f"<< serial-floor {serial_floor*1000:.0f}ms (N={N}×{APPLY*1000:.0f}ms) OK"))


_STAGES = [
    ("A0 hints coalesce latest-wins", stage_a0_hints_coalesce_latest_wins),
    ("A1 migrates coalesce latest-wins", stage_a1_migrates_coalesce_latest_wins),
    ("A2 migrate freshness drop", stage_a2_migrate_freshness_drop),
    ("A3 paused coalesce by pid", stage_a3_paused_coalesce_by_pid),
    ("A4 dispatch order", stage_a4_dispatch_order),
    ("B0 migrate not behind hint flood", stage_b0_migrate_not_behind_hint_flood),
    ("B1 latency bounded under flood", stage_b1_latency_bounded_under_flood),
]


def main() -> int:
    failures = []
    for name, fn in _STAGES:
        try:
            fn()
        except StageFail as e:
            print(_red(f"  FAIL {name}: {e}"))
            failures.append(name)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(_red(f"  FAIL {name}: unexpected {e!r}"))
            failures.append(name)
    print("=" * 60)
    if failures:
        print(_red(f"outbound_coalesce FAILED ({len(failures)}): {failures}"))
        return 1
    print(_green(f"outbound_coalesce PASS — all {len(_STAGES)} stages green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
