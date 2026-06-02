"""T40 — F2 hint emitter (#184, DESIGN §6 `PUT /aginfer/hints` + §10).

Per event the daemon re-scores the units in D_t and pushes ALL of
them to sglang via a fire-and-forget `PUT /aginfer/hints`.  Two hard
invariants from DESIGN §10:

  * **No daemon-side hint cache**: the daemon keeps NO shadow
    `{hash: last_pushed_value}` map.  Every D_t unit is pushed
    unconditionally every event; sglang's hint table dedupes on its
    side (overwrite-by-stamp).
  * **Hint table covers every live unit**, overwrite-by-stamp: a
    PUT whose `stamp` is newer wins; a stale (older) stamp is
    rejected; an equal stamp is an idempotent no-op (DESIGN §10 R2).

This task is the FULL round-trip: daemon emitter + sglang receiving
side (`PUT /aginfer/hints` → `set_aginfer_hints` overwrite-by-stamp
storage on the radix cache).  The inline-scorer CONSUMPTION of the
table (eviction order), unit-birth seeding, and eviction-time hint
clear are deferred (T27 / T28 / #177 / follow-ons).

Stages:

  A. daemon outbound — enqueue_hints
    A0 enqueue_hints builds {hints:[...], batch_id} body,
       endpoint=hints, method=PUT

  B. daemon kv_scheduler — the emitter (drives handle())
    B0 non-empty D_t → one hints PUT, one hint per D_t unit, each
       carrying the EXACT p_hat / lambda the scorer computed +
       stamp == time_counter
    B1 push is UNCONDITIONAL: policy declines to migrate → hints
       STILL pushed (independent of the migrate decision)
    B2 empty D_t (LLM_PREFILL) → NO hints pushed
    B3 NO shadow cache: two consecutive events re-push the SAME
       unit (no suppression of "unchanged" values), second stamp
       strictly newer

  C. sglang validator — _validate_hints_body
    C0 well-formed body accepted, normalized list returned
    C1 malformed bodies rejected (not dict / missing hints / hint
       missing hash / bad numeric types / negative stamp)

  D. sglang storage — set_aginfer_hints overwrite-by-stamp
    D0 first push applies (applied == n), values readable
    D1 idempotent re-push (same stamp) → applied 0
    D2 newer stamp overwrites → applied counts, value updated
    D3 stale (older) stamp rejected → applied 0, value unchanged

  E. wire round-trip
    E0 the EXACT body the daemon emits passes sglang's
       _validate_hints_body AND set_aginfer_hints (catches a
       "lambda" vs "lambda_rate" field-name mismatch)

  F. e2e (env-gated AGINFER_VERIFY_BASE): PUT /aginfer/hints against
     a live sglang, read it back via /aginfer/state n_aginfer_hints
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


_HERE = Path(__file__).resolve().parent
_AGINFER_ROOT = _HERE.parent.parent
if str(_AGINFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGINFER_ROOT))
_SGLANG_PY = "/scratch/yuzhou/projects/sglang/python"
if _SGLANG_PY not in sys.path:
    sys.path.insert(0, _SGLANG_PY)

import asyncio  # noqa: E402

from baselines.base import Action, Tier  # noqa: E402
from daemon import kv_scheduler as kvs  # noqa: E402
from daemon.events import Event, EventKind  # noqa: E402
from daemon.outbound import OutboundBatch, OutboundQueue  # noqa: E402
from daemon.program_tracker import ProgramTracker  # noqa: E402


def _green(s: str) -> str: return f"\033[32m{s}\033[0m"
def _red(s: str) -> str:   return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


# ============================================================ stubs / fixtures


class _Resp:
    status_code = 200
    text = ""
    def json(self): return {}


class _DummyHttp:
    async def post(self, *a, **k): return _Resp()
    async def request(self, *a, **k): return _Resp()
    async def aclose(self): return None


class _RecordingHttp:
    """Records every (verb, url, body) so a stage can assert the
    outbound worker routed a batch to the right endpoint + HTTP verb."""
    def __init__(self):
        self.calls: List[Tuple[str, str, Any]] = []

    async def post(self, url, json=None):
        self.calls.append(("POST", url, json))
        return _Resp()

    async def request(self, method, url, json=None):
        self.calls.append((method, url, json))
        return _Resp()

    async def aclose(self):
        return None


def _new_outbound() -> OutboundQueue:
    return OutboundQueue(
        sglang_base_url="http://unused", http_client=_DummyHttp(),
    )


def _unit(
    *,
    uhash: str,
    residence: List[str],
    holders: List[str],
    n_tokens: int = 1000,
    last_access_time: int = 0,
    hit_count: int = 1,
    subpool: str = "kv",
) -> Dict[str, Any]:
    n_bytes = {t: {subpool: n_tokens * 2048} for t in residence}
    return {
        "hash": uhash,
        "residence": list(residence),
        "n_tokens": n_tokens,
        "n_bytes": n_bytes,
        "last_access_time": last_access_time,
        "hit_count": hit_count,
        "session_ids": list(holders),
    }


def _state_json(
    *,
    units: List[Dict[str, Any]],
    time_counter: int = 100,
    subpool: str = "kv",
) -> Dict[str, Any]:
    GB = 1024 * 1024 * 1024

    def _pool(used: int, cap: int) -> Dict[str, Any]:
        return {"subpools": {subpool: {
            "used_bytes": used, "cap_bytes": cap,
            "available_bytes": max(0, cap - used),
            "evictable_bytes": used, "page_bytes": 64 * 1024,
        }}}
    return {
        "time_counter": time_counter,
        "throughput_ema": {"prefill_bps": 0.0, "decode_per_program": {}},
        "pool_usage": {
            "HBM": _pool(1 * GB, 10 * GB),
            "DRAM": _pool(1 * GB, 40 * GB),
            "DISK": _pool(0, 200 * GB),
        },
        "per_program_usage": {},
        "units": units,
        "link_stats": {link: {
            "peak_bw_bps": 64 * GB, "recent_throughput_bps": 0.0,
            "time_since_last_sample_s": 5.0,
        } for link in ("HBM->DRAM", "DRAM->HBM", "DRAM->DISK", "DISK->DRAM")},
        "tier_holding_cost": {tier: {subpool: {"h_max_per_byte_sec": 0.0}}
                              for tier in ("HBM", "DRAM", "DISK")},
    }


class _FakeRouter:
    """Minimal EventRouter stand-in: handle() only needs fetch_state()
    + observability."""
    def __init__(self, state_json: Dict[str, Any]):
        self._sj = state_json
        self.observability = None

    async def fetch_state(self) -> Dict[str, Any]:
        return self._sj


class _DeclinePolicy:
    """Returns no migrate assignments (Vt non-positive everywhere)."""
    def decide(self, state) -> Action:  # noqa: ANN001
        return Action(assignments=[])


class _MigratePolicy:
    """Demotes the first D_t unit HBM→DRAM, so handle() also enqueues
    a migrate POST alongside the hints PUT."""
    def decide(self, state) -> Action:  # noqa: ANN001
        for uid in state.decision_set:
            u = state.units.get(uid)
            if u is not None and Tier.HBM in u.residence:
                return Action(assignments=[(uid, [Tier.DRAM], [Tier.HBM])])
        return Action(assignments=[])


def _sched(tracker: ProgramTracker, ob: OutboundQueue, policy) -> "kvs.KvScheduler":
    return kvs.KvScheduler(
        tracker=tracker, sglang_base_url="http://unused",
        policy=policy, outbound=ob,
    )


def _drain(ob: OutboundQueue) -> List[OutboundBatch]:
    out: List[OutboundBatch] = []
    while ob.queue.qsize():
        out.append(ob.queue.get_nowait())
    return out


def _hints_batch(batches: List[OutboundBatch]) -> Optional[OutboundBatch]:
    for b in batches:
        if b.endpoint == "hints":
            return b
    return None


# ============================================================ A. enqueue_hints


def stage_a0_enqueue_hints_shape() -> None:
    ob = _new_outbound()
    hints = [
        {"hash": "u0", "p_hat": 1.0, "lambda": 0.5, "stamp": 100},
        {"hash": "u1", "p_hat": 0.3, "lambda": 0.01, "stamp": 100},
    ]
    batch_id = ob.enqueue_hints(hints)
    if ob.queue.qsize() != 1:
        raise StageFail(f"queue size: {ob.queue.qsize()}")
    batch = ob.queue.get_nowait()
    if batch.endpoint != "hints":
        raise StageFail(f"endpoint: {batch.endpoint}")
    if batch.method != "PUT":
        raise StageFail(f"method should be PUT; got {batch.method}")
    if batch.body.get("hints") != hints:
        raise StageFail(f"body hints: {batch.body!r}")
    if batch.body.get("batch_id") != batch_id:
        raise StageFail(f"batch_id mismatch: {batch.body!r} vs {batch_id}")


# ============================================================ B. emitter


def _expected_hints(sched_state) -> Dict[str, Dict[str, Any]]:  # noqa: ANN001
    exp = {}
    for uid in sched_state.decision_set:
        u = sched_state.units[uid]
        exp[uid] = {
            "hash": uid,
            "p_hat": u.p_hat,
            "lambda": u.lambda_rate,
            "stamp": int(sched_state.t),
        }
    return exp


def stage_b0_emit_one_hint_per_dt_unit() -> None:
    """handle() pushes exactly one hint per D_t unit, each carrying the
    EXACT p_hat / lambda the scorer computed and stamp == time_counter."""
    async def _go():
        tracker = ProgramTracker()
        tracker.observe_arrival("p0")  # REASONING, alive
        sj = _state_json(
            units=[_unit(uhash="u0", residence=["HBM"], holders=["p0"])],
            time_counter=100,
        )
        ev = Event(EventKind.TOOL_CALL_END, session="p0")
        sched_state = kvs.build_paper_state(
            sj, event=ev, tracker=tracker, unknown_tier_log=set(),
        )
        ob = _new_outbound()
        sched = _sched(tracker, ob, _DeclinePolicy())
        await sched.handle(ev, _FakeRouter(sj))
        return sched_state, _drain(ob)
    sched_state, batches = asyncio.run(_go())
    if not sched_state.decision_set:
        raise StageFail("fixture bug: D_t should be non-empty for TOOL_CALL_END")
    hb = _hints_batch(batches)
    if hb is None:
        raise StageFail(f"no hints PUT enqueued; batches={[b.endpoint for b in batches]}")
    got = {h["hash"]: h for h in hb.body["hints"]}
    exp = _expected_hints(sched_state)
    if set(got) != set(exp):
        raise StageFail(f"hint hashes: got {set(got)} exp {set(exp)}")
    # Literal anchor (audit B0-tautology): the fixture is deterministic
    # — an alive REASONING holder gives p_hat == 1.0 exactly.  Pin it
    # so a regression that zeroes every hint can't pass by also
    # zeroing the `exp` derived from the same build_paper_state.
    if abs(float(got["u0"]["p_hat"]) - 1.0) > 1e-9:
        raise StageFail(f"u0 p_hat must be exactly 1.0 (alive holder); got {got['u0']}")
    for uid, e in exp.items():
        g = got[uid]
        if g.get("stamp") != e["stamp"]:
            raise StageFail(f"{uid} stamp: got {g.get('stamp')} exp {e['stamp']} (must == time_counter)")
        if abs(float(g.get("p_hat")) - e["p_hat"]) > 1e-9:
            raise StageFail(f"{uid} p_hat: got {g.get('p_hat')} exp {e['p_hat']}")
        if abs(float(g.get("lambda")) - e["lambda"]) > 1e-9:
            raise StageFail(f"{uid} lambda: got {g.get('lambda')} exp {e['lambda']}")


def stage_b1_push_unconditional_when_policy_declines() -> None:
    """The hint push is independent of the migrate decision: even when
    the policy declines to migrate (assignments empty), every D_t unit's
    hint is still pushed."""
    async def _go():
        tracker = ProgramTracker()
        tracker.observe_arrival("p0")
        sj = _state_json(
            units=[_unit(uhash="u0", residence=["HBM"], holders=["p0"])],
            time_counter=100,
        )
        ev = Event(EventKind.TOOL_CALL_END, session="p0")
        ob = _new_outbound()
        sched = _sched(tracker, ob, _DeclinePolicy())
        await sched.handle(ev, _FakeRouter(sj))
        return _drain(ob)
    batches = asyncio.run(_go())
    if any(b.endpoint == "migrate" for b in batches):
        raise StageFail("decline policy must not enqueue a migrate")
    hb = _hints_batch(batches)
    if hb is None or not hb.body["hints"]:
        raise StageFail("hints must be pushed even when the policy declines migrate")


def stage_b1b_push_alongside_migrate() -> None:
    """When the policy DOES migrate, the hints PUT is enqueued alongside
    the migrate POST (both, independently)."""
    async def _go():
        tracker = ProgramTracker()
        tracker.observe_arrival("p0")
        sj = _state_json(
            units=[_unit(uhash="u0", residence=["HBM"], holders=["p0"])],
            time_counter=100,
        )
        ev = Event(EventKind.TOOL_CALL_END, session="p0")
        ob = _new_outbound()
        sched = _sched(tracker, ob, _MigratePolicy())
        await sched.handle(ev, _FakeRouter(sj))
        return _drain(ob)
    batches = asyncio.run(_go())
    eps = sorted(b.endpoint for b in batches)
    if eps != ["hints", "migrate"]:
        raise StageFail(f"expected both hints+migrate enqueued; got {eps}")


def stage_b2_empty_dt_no_hints() -> None:
    """LLM_PREFILL → D_t is empty → no hints (and no migrate)."""
    async def _go():
        tracker = ProgramTracker()
        tracker.observe_arrival("p0")
        sj = _state_json(
            units=[_unit(uhash="u0", residence=["HBM"], holders=["p0"])],
            time_counter=100,
        )
        ev = Event(EventKind.LLM_PREFILL, session="p0")
        ob = _new_outbound()
        sched = _sched(tracker, ob, _DeclinePolicy())
        await sched.handle(ev, _FakeRouter(sj))
        return _drain(ob)
    batches = asyncio.run(_go())
    if batches:
        raise StageFail(f"empty D_t must push nothing; got {[b.endpoint for b in batches]}")


def stage_b3_no_shadow_cache_repush() -> None:
    """No daemon-side shadow map: the SAME unit is re-pushed on a
    second event (unchanged value is NOT suppressed), with a strictly
    newer stamp from the advanced time_counter."""
    async def _go():
        tracker = ProgramTracker()
        tracker.observe_arrival("p0")
        ev = Event(EventKind.TOOL_CALL_END, session="p0")
        ob = _new_outbound()
        sched = _sched(tracker, ob, _DeclinePolicy())
        u = _unit(uhash="u0", residence=["HBM"], holders=["p0"])
        await sched.handle(ev, _FakeRouter(_state_json(units=[u], time_counter=100)))
        await sched.handle(ev, _FakeRouter(_state_json(units=[u], time_counter=200)))
        return _drain(ob)
    batches = asyncio.run(_go())
    hint_puts = [b for b in batches if b.endpoint == "hints"]
    if len(hint_puts) != 2:
        raise StageFail(f"expected 2 hint PUTs (re-pushed, no suppression); got {len(hint_puts)}")
    stamps = []
    for b in hint_puts:
        hs = {h["hash"]: h for h in b.body["hints"]}
        if "u0" not in hs:
            raise StageFail(f"u0 missing from a re-push: {b.body!r}")
        stamps.append(hs["u0"]["stamp"])
    if not (stamps[0] == 100 and stamps[1] == 200):
        raise StageFail(f"stamps should track time_counter [100,200]; got {stamps}")


# ============================================================ C. sglang validator


def _validator():
    from sglang.srt.entrypoints.http_server import _validate_hints_body
    return _validate_hints_body


def stage_c0_validator_accepts() -> None:
    v = _validator()
    body = {"hints": [
        {"hash": "u0", "p_hat": 1.0, "lambda": 0.5, "stamp": 100},
        {"hash": "u1", "p_hat": 0.25, "lambda": 0.0, "stamp": 100},
    ], "batch_id": "abc"}
    hints = v(body)
    if len(hints) != 2:
        raise StageFail(f"validator should return 2 hints; got {hints!r}")
    h0 = hints[0]
    for k in ("hash", "p_hat", "lambda", "stamp"):
        if k not in h0:
            raise StageFail(f"normalized hint missing {k!r}: {h0!r}")


def stage_c1_validator_rejects() -> None:
    v = _validator()
    bad_cases = [
        ("not a dict", []),
        ("missing hints", {"batch_id": "x"}),
        ("hints not a list", {"hints": {}}),
        ("hint not a dict", {"hints": [42]}),
        ("hint missing hash", {"hints": [{"p_hat": 1.0, "lambda": 0.0, "stamp": 1}]}),
        ("empty hash", {"hints": [{"hash": "", "p_hat": 1.0, "lambda": 0.0, "stamp": 1}]}),
        ("p_hat not numeric", {"hints": [{"hash": "u", "p_hat": "x", "lambda": 0.0, "stamp": 1}]}),
        ("lambda not numeric", {"hints": [{"hash": "u", "p_hat": 1.0, "lambda": None, "stamp": 1}]}),
        ("stamp not int", {"hints": [{"hash": "u", "p_hat": 1.0, "lambda": 0.0, "stamp": 1.5}]}),
        ("stamp negative", {"hints": [{"hash": "u", "p_hat": 1.0, "lambda": 0.0, "stamp": -1}]}),
        ("p_hat out of range", {"hints": [{"hash": "u", "p_hat": 2.0, "lambda": 0.0, "stamp": 1}]}),
        ("lambda negative", {"hints": [{"hash": "u", "p_hat": 1.0, "lambda": -0.1, "stamp": 1}]}),
        # audit A4: non-finite must be rejected at the door (the
        # validator is the safety boundary for the inline scorer).
        ("p_hat nan", {"hints": [{"hash": "u", "p_hat": float("nan"), "lambda": 0.0, "stamp": 1}]}),
        ("p_hat inf", {"hints": [{"hash": "u", "p_hat": float("inf"), "lambda": 0.0, "stamp": 1}]}),
        ("lambda inf", {"hints": [{"hash": "u", "p_hat": 1.0, "lambda": float("inf"), "stamp": 1}]}),
        ("lambda nan", {"hints": [{"hash": "u", "p_hat": 1.0, "lambda": float("nan"), "stamp": 1}]}),
    ]
    for label, body in bad_cases:
        try:
            v(body)
        except ValueError:
            continue
        raise StageFail(f"validator should reject: {label} ({body!r})")


# ============================================================ D. storage


def _fresh_cache():
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache._aginfer_hints = {}
    return cache


def stage_d0_set_hints_applies() -> None:
    cache = _fresh_cache()
    hints = [
        {"hash": "u0", "p_hat": 1.0, "lambda": 0.5, "stamp": 100},
        {"hash": "u1", "p_hat": 0.25, "lambda": 0.0, "stamp": 100},
    ]
    ok, reason, applied = cache.set_aginfer_hints(hints)
    if not ok:
        raise StageFail(f"set_aginfer_hints failed: {reason!r}")
    if applied != 2:
        raise StageFail(f"applied should be 2; got {applied}")
    got = cache.get_aginfer_hint("u0")
    if got is None or abs(got["p_hat"] - 1.0) > 1e-9 or got["stamp"] != 100:
        raise StageFail(f"stored u0 wrong: {got!r}")


def stage_d1_idempotent_same_stamp() -> None:
    cache = _fresh_cache()
    hints = [{"hash": "u0", "p_hat": 1.0, "lambda": 0.5, "stamp": 100}]
    cache.set_aginfer_hints(hints)
    ok, reason, applied = cache.set_aginfer_hints(hints)  # re-apply
    if not ok:
        raise StageFail(f"re-apply failed: {reason!r}")
    if applied != 0:
        raise StageFail(f"idempotent re-apply (same stamp) → applied 0; got {applied}")


def stage_d2_newer_stamp_overwrites() -> None:
    cache = _fresh_cache()
    cache.set_aginfer_hints([{"hash": "u0", "p_hat": 0.2, "lambda": 0.1, "stamp": 100}])
    ok, reason, applied = cache.set_aginfer_hints(
        [{"hash": "u0", "p_hat": 0.9, "lambda": 0.7, "stamp": 150}]
    )
    if applied != 1:
        raise StageFail(f"newer stamp should apply; got applied={applied}")
    got = cache.get_aginfer_hint("u0")
    if abs(got["p_hat"] - 0.9) > 1e-9 or got["stamp"] != 150:
        raise StageFail(f"newer value should win: {got!r}")


def stage_d3_stale_stamp_rejected() -> None:
    cache = _fresh_cache()
    cache.set_aginfer_hints([{"hash": "u0", "p_hat": 0.9, "lambda": 0.7, "stamp": 150}])
    ok, reason, applied = cache.set_aginfer_hints(
        [{"hash": "u0", "p_hat": 0.2, "lambda": 0.1, "stamp": 100}]  # stale
    )
    if applied != 0:
        raise StageFail(f"stale stamp must be rejected; got applied={applied}")
    got = cache.get_aginfer_hint("u0")
    if abs(got["p_hat"] - 0.9) > 1e-9 or got["stamp"] != 150:
        raise StageFail(f"stale push must not clobber newer value: {got!r}")


def stage_d4_mixed_batch_applied_count() -> None:
    """audit A5: the realistic daemon case — a re-push of D_t where
    only SOME units advanced.  `applied` must count ONLY the hashes
    whose stamp strictly advanced (not len(batch), not 1)."""
    cache = _fresh_cache()
    cache.set_aginfer_hints([
        {"hash": "u0", "p_hat": 0.2, "lambda": 0.1, "stamp": 100},
        {"hash": "u1", "p_hat": 0.3, "lambda": 0.2, "stamp": 100},
    ])
    ok, reason, applied = cache.set_aginfer_hints([
        {"hash": "u0", "p_hat": 0.9, "lambda": 0.7, "stamp": 150},  # newer → apply
        {"hash": "u1", "p_hat": 0.9, "lambda": 0.7, "stamp": 100},  # equal → skip
        {"hash": "u2", "p_hat": 0.5, "lambda": 0.5, "stamp": 100},  # new   → apply
    ])
    if not ok or applied != 2:
        raise StageFail(f"mixed batch should apply exactly 2 (u0 newer + u2 new); got applied={applied} reason={reason!r}")
    if abs(cache.get_aginfer_hint("u0")["p_hat"] - 0.9) > 1e-9:
        raise StageFail("u0 should have advanced")
    if abs(cache.get_aginfer_hint("u1")["p_hat"] - 0.3) > 1e-9:
        raise StageFail("u1 (equal stamp) must NOT have changed")
    if cache.get_aginfer_hint("u2") is None:
        raise StageFail("u2 (new) should have been stored")


def stage_d5_clear_aginfer_hint() -> None:
    """audit A11: clear_aginfer_hint primitive — present → True +
    entry gone; not-present → False."""
    cache = _fresh_cache()
    cache.set_aginfer_hints([{"hash": "u0", "p_hat": 1.0, "lambda": 0.5, "stamp": 100}])
    if cache.clear_aginfer_hint("u0") is not True:
        raise StageFail("clearing a present hint should return True")
    if cache.get_aginfer_hint("u0") is not None:
        raise StageFail("entry should be gone after clear")
    if cache.clear_aginfer_hint("u0") is not False:
        raise StageFail("clearing an absent hint should return False")


# ============================================================ G. dump-path parity


def stage_g0_dump_paths_both_echo_count() -> None:
    """audit #1: guard the dump-path-divergence bug class (#181).  The
    live /aginfer/state uses the BYTES path (get_aginfer_state →
    dump_aginfer_state_bytes when available) — F0 exercises it end-to-
    end — but the DICT path (in-process callers) must echo the same
    key, and a typo/omission in either path's `n_aginfer_hints` write
    would otherwise ship green.  Both must reference the SAME source
    (`self._aginfer_hints`) and emit the key."""
    import inspect
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    for meth in ("_dump_aginfer_state_dict", "_dump_aginfer_state_bytes"):
        src = inspect.getsource(getattr(UnifiedRadixCache, meth))
        if "n_aginfer_hints" not in src:
            raise StageFail(f"{meth} does not emit n_aginfer_hints (dump-path divergence)")
        if "_aginfer_hints" not in src:
            raise StageFail(f"{meth} does not read self._aginfer_hints")


# ============================================================ H. outbound routing


def stage_h0_worker_routes_hints_to_put() -> None:
    """audit #2: the outbound worker's PUT dispatch for endpoint=hints.
    A0 only checks the enqueued body shape; this drives the batch
    through `_post_one` and asserts it issues a PUT to /aginfer/hints
    (NOT a POST — the migrate hot path stays on .post()).  Closes the
    one dispatch branch t36/t41 don't cover for hints."""
    async def _go():
        rec = _RecordingHttp()
        ob = OutboundQueue(sglang_base_url="http://sg", http_client=rec)
        ob.enqueue_hints([{"hash": "u0", "p_hat": 1.0, "lambda": 0.5, "stamp": 1}])
        batch = ob.queue.get_nowait()
        await ob._post_one(batch)
        return rec.calls
    calls = asyncio.run(_go())
    if len(calls) != 1:
        raise StageFail(f"expected exactly one HTTP call; got {calls!r}")
    verb, url, body = calls[0]
    if verb != "PUT":
        raise StageFail(f"hints must dispatch via PUT (not {verb}); migrate stays on .post()")
    if not url.endswith("/aginfer/hints"):
        raise StageFail(f"wrong URL: {url!r}")
    if "hints" not in (body or {}):
        raise StageFail(f"PUT body missing 'hints': {body!r}")


# ============================================================ E. wire round-trip


def stage_e0_wire_round_trip() -> None:
    """The EXACT body the daemon enqueues must pass sglang's validator
    AND its storage setter — catches a wire field-name mismatch
    (e.g. the daemon emitting 'lambda_rate' while sglang reads 'lambda',
    or 'stamp' vs 'seq')."""
    async def _go():
        tracker = ProgramTracker()
        tracker.observe_arrival("p0")
        sj = _state_json(
            units=[_unit(uhash="u0", residence=["HBM"], holders=["p0"])],
            time_counter=123,
        )
        ev = Event(EventKind.TOOL_CALL_END, session="p0")
        ob = _new_outbound()
        sched = _sched(tracker, ob, _DeclinePolicy())
        await sched.handle(ev, _FakeRouter(sj))
        return _hints_batch(_drain(ob))
    hb = asyncio.run(_go())
    if hb is None:
        raise StageFail("no hints PUT to round-trip")
    body = hb.body  # literal wire body the worker PUTs

    v = _validator()
    hints = v(body)  # must accept
    if not hints:
        raise StageFail("validator returned no hints from the daemon's body")

    cache = _fresh_cache()
    ok, reason, applied = cache.set_aginfer_hints(hints)
    if not (ok and applied == len(hints)):
        raise StageFail(f"cache rejected the daemon's wire body: ok={ok} reason={reason!r} applied={applied}")
    # the stamp the scorer used (time_counter) must be what landed
    got = cache.get_aginfer_hint("u0")
    if got is None or got["stamp"] != 123:
        raise StageFail(f"round-trip stamp lost: {got!r}")


# ============================================================ F. e2e (env-gated)


def stage_f0_e2e_live_put_readback() -> None:
    base = os.environ.get("AGINFER_VERIFY_BASE")
    if not base:
        raise _Skip("set AGINFER_VERIFY_BASE=http://127.0.0.1:PORT to run e2e")
    import json
    import time
    import urllib.request

    def _req(method: str, path: str, body: Optional[dict] = None):
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(
            f"{base}{path}", data=data, method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode())

    # Per-run unique stamp + hashes so repeated runs against a
    # persistent server stay independent (the table is overwrite-by-
    # stamp and accumulates across runs).  Wall-clock is fine here —
    # this is the verify layer, NOT the daemon policy path.
    run = int(time.time())
    h0, h1 = f"e2e-u0-{run}", f"e2e-u1-{run}"
    hints = [
        {"hash": h0, "p_hat": 1.0, "lambda": 0.5, "stamp": run},
        {"hash": h1, "p_hat": 0.4, "lambda": 0.0, "stamp": run},
    ]
    status, out = _req("PUT", "/aginfer/hints", {"hints": hints, "batch_id": "e2e"})
    if status != 200 or not out.get("ok"):
        raise StageFail(f"live PUT /aginfer/hints failed: {status} {out!r}")
    if out.get("applied", 0) < 2:
        raise StageFail(f"live PUT applied < 2: {out!r}")
    # idempotent re-apply (same stamp) → applied 0
    _, out2 = _req("PUT", "/aginfer/hints", {"hints": hints, "batch_id": "e2e2"})
    if out2.get("applied", -1) != 0:
        raise StageFail(f"live idempotent re-apply should be applied=0: {out2!r}")
    # newer stamp overwrites → applied advances for both hashes.
    # NOTE: the HTTP layer SUMS applied across ranks, so a 2-hint PUT
    # returns applied == 2 * n_ranks (==2 at TP=1).  Assert >= 2 so
    # this stays correct at TP>1 (audit A7).
    newer = [dict(h, stamp=run + 1, p_hat=0.1) for h in hints]
    _, out3 = _req("PUT", "/aginfer/hints", {"hints": newer, "batch_id": "e2e3"})
    if out3.get("applied", -1) < 2:
        raise StageFail(f"live newer-stamp re-apply should advance both (>=2): {out3!r}")
    # read back the count via /aginfer/state
    _, state = _req("GET", "/aginfer/state")
    n = state.get("n_aginfer_hints")
    if n is None or n < 2:
        raise StageFail(f"/aginfer/state n_aginfer_hints should be >=2; got {n!r}")


class _Skip(Exception):
    pass


# ============================================================ run


_STAGES: List[Tuple[str, Callable[[], None]]] = [
    ("A0 enqueue_hints PUT shape",                  stage_a0_enqueue_hints_shape),
    ("B0 one hint per D_t unit (exact p_hat/lambda/stamp)", stage_b0_emit_one_hint_per_dt_unit),
    ("B1 push unconditional when policy declines",  stage_b1_push_unconditional_when_policy_declines),
    ("B1b hints pushed alongside a migrate",        stage_b1b_push_alongside_migrate),
    ("B2 empty D_t → no hints",                     stage_b2_empty_dt_no_hints),
    ("B3 no shadow cache: re-push same unit, newer stamp", stage_b3_no_shadow_cache_repush),
    ("C0 _validate_hints_body accepts well-formed", stage_c0_validator_accepts),
    ("C1 _validate_hints_body rejects malformed",   stage_c1_validator_rejects),
    ("D0 set_aginfer_hints applies",                stage_d0_set_hints_applies),
    ("D1 idempotent re-apply (same stamp) → 0",     stage_d1_idempotent_same_stamp),
    ("D2 newer stamp overwrites",                   stage_d2_newer_stamp_overwrites),
    ("D3 stale stamp rejected",                     stage_d3_stale_stamp_rejected),
    ("D4 mixed batch → applied counts only advanced", stage_d4_mixed_batch_applied_count),
    ("D5 clear_aginfer_hint present/absent",        stage_d5_clear_aginfer_hint),
    ("E0 daemon wire body round-trips sglang validator+setter", stage_e0_wire_round_trip),
    ("G0 both dump paths echo n_aginfer_hints (no divergence)", stage_g0_dump_paths_both_echo_count),
    ("H0 outbound worker routes hints → PUT /aginfer/hints", stage_h0_worker_routes_hints_to_put),
    ("F0 e2e live PUT /aginfer/hints + state readback", stage_f0_e2e_live_put_readback),
]


def main() -> int:
    failures: List[str] = []
    skipped: List[str] = []
    for label, fn in _STAGES:
        try:
            fn()
            print(f"  {_green('PASS')}  Stage {label}")
        except _Skip as exc:
            skipped.append(label)
            print(f"  \033[33mSKIP\033[0m  Stage {label}: {exc}")
        except StageFail as exc:
            failures.append(label)
            print(f"  {_red('FAIL')}  Stage {label}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(label)
            print(f"  {_red('FAIL')}  Stage {label}: "
                  f"unexpected {type(exc).__name__}: {exc}")
    if failures:
        print(_red(f"\nT40 FAILED ({len(failures)}): {failures}"))
        return 1
    n_ran = len(_STAGES) - len(skipped)
    suffix = f" ({len(skipped)} skipped)" if skipped else ""
    print(_green(f"\nT40 PASS — {n_ran} stages green{suffix}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
