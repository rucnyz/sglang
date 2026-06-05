"""T26 (#200) — pure measurement helpers for the sglang throughput /
in-flight instrumentation.

The hot-path hooks (scheduler) and the cold-path dump (radix cache) are
exercised end-to-end on the real stack by verify/integration_stress; this
file pins the PURE helpers (no GPU): the EMA, the per-program in-flight
byte computation, the decode-token counting, and the running-program
projection.

Usage:
    python dev/aginfer/verify/t26/verify.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

# sglang is importable in the agsched env.
from sglang.srt.mem_cache.aginfer_metrics import (  # noqa: E402
    AGINFER_THROUGHPUT_EMA_ALPHA,
    decode_tokens_by_program,
    ema_update,
    inflight_bytes_by_program,
    running_program_view,
)


def _green(s): return f"\033[32m{s}\033[0m"
def _red(s): return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


class _Req:
    """Minimal stand-in for a sglang Req (the helpers only read these)."""
    def __init__(self, program_id, allocated=1, committed=1, extend_input_len=0):
        self.program_id = program_id
        self.kv_allocated_len = allocated
        self.kv_committed_len = committed
        self.extend_input_len = extend_input_len


# ============================================================ stages


def stage_ema() -> None:
    a = 0.5
    # first sample seeds.
    if ema_update(None, 10.0, a) != 10.0:
        raise StageFail("ema: first sample must seed the average")
    # blend.
    if ema_update(10.0, 20.0, a) != 15.0:
        raise StageFail("ema: 0.5*20 + 0.5*10 must be 15")
    # alpha applied correctly.
    v = ema_update(100.0, 0.0, 0.3)
    if abs(v - 70.0) > 1e-9:
        raise StageFail(f"ema: 0.3*0 + 0.7*100 = 70; got {v}")
    # malformed sample ignored (keeps prev).
    for bad in (float("nan"), float("inf"), -5.0):
        if ema_update(42.0, bad, a) != 42.0:
            raise StageFail(f"ema: malformed sample {bad!r} must keep prev")
    # malformed sample with no prev → 0.
    if ema_update(None, float("nan"), a) != 0.0:
        raise StageFail("ema: malformed first sample → 0.0")
    # poisoned prev recovers to the sample.
    if ema_update(float("nan"), 7.0, a) != 7.0:
        raise StageFail("ema: non-finite prev must recover to sample")
    # prev=0.0 is a VALID prior (a real zero EMA), NOT "unset" → blend,
    # don't re-seed (#200 audit).
    if ema_update(0.0, 10.0, 0.5) != 5.0:
        raise StageFail("ema: prev=0.0 must blend (0.5*10+0.5*0), not re-seed")
    if not (0.0 < AGINFER_THROUGHPUT_EMA_ALPHA < 1.0):
        raise StageFail("ema: default alpha must be in (0,1)")
    print(_green("  [ema] seed / blend / alpha / prev=0 / malformed-guard OK"))


def stage_inflight() -> None:
    BPT = 2048
    SP = "full"
    # inflight = current KV = kv_allocated_len × bpt (committed is ignored;
    # in sglang it tracks allocated, so allocated−committed≡0).
    reqs = [
        _Req("A", allocated=100, committed=100),   # 100 tok current KV
        _Req("B", allocated=50, committed=50),     # 50 tok
        _Req(None, allocated=99, committed=99),    # untagged → skipped
        _Req("C", allocated=0, committed=0),       # no KV yet → skipped
    ]
    got = inflight_bytes_by_program(reqs, BPT, SP)
    want = {"A": {SP: 100 * BPT}, "B": {SP: 50 * BPT}}
    if got != want:
        raise StageFail(f"inflight: expected {want}, got {got}")
    # multiple running reqs of one program accumulate.
    got2 = inflight_bytes_by_program(
        [_Req("A", 100, 100), _Req("A", 30, 30)], BPT, SP)
    if got2 != {"A": {SP: (100 + 30) * BPT}}:
        raise StageFail(f"inflight: same-program reqs must sum; got {got2}")
    # committed value is irrelevant (allocated is the current-KV signal).
    if inflight_bytes_by_program([_Req("A", 80, 0)], BPT, SP) != {"A": {SP: 80 * BPT}}:
        raise StageFail("inflight: committed must be ignored; allocated is KV")
    # bpt<=0 → empty (cache couldn't report bytes/token).
    if inflight_bytes_by_program(reqs, 0, SP) != {}:
        raise StageFail("inflight: bpt<=0 must yield {}")
    # no running reqs → empty.
    if inflight_bytes_by_program([], BPT, SP) != {}:
        raise StageFail("inflight: no reqs → {}")
    print(_green("  [inflight] current-KV per program, skip-untagged/empty, "
                 "sum, bpt-guard OK"))


def stage_decode_counts() -> None:
    reqs = [_Req("A", 1, 0), _Req("A", 1, 0), _Req("B", 1, 0), _Req(None, 1, 0)]
    c = decode_tokens_by_program(reqs)
    if c != {"A": 2, "B": 1}:
        raise StageFail(f"decode-counts: 1 tok/req, skip untagged; got {c}")
    # accumulation across calls (spec-decode multi-token).
    c2 = decode_tokens_by_program([_Req("A", 1, 0)], counts={"A": 5, "B": 1})
    if c2 != {"A": 6, "B": 1}:
        raise StageFail(f"decode-counts: must accumulate into counts; got {c2}")
    # per_req_tokens (#206 spec-decode accept_lens): index-aligned with reqs,
    # untagged reqs still consume an index so alignment holds.
    reqs2 = [_Req("A"), _Req(None), _Req("B"), _Req("A")]
    c3 = decode_tokens_by_program(reqs2, per_req_tokens=[3, 9, 1, 2])
    if c3 != {"A": 5, "B": 1}:  # A: 3+2, B: 1, untagged 9 dropped
        raise StageFail(f"decode-counts: per_req_tokens must align past "
                        f"untagged reqs; got {c3}")
    # a per-req count ≤ 0 contributes nothing (all-rejected guard).
    c4 = decode_tokens_by_program([_Req("A"), _Req("B")], per_req_tokens=[0, 4])
    if c4 != {"B": 4}:
        raise StageFail(f"decode-counts: per_req ≤0 contributes nothing; got {c4}")
    # short per_req_tokens → reqs past the end default to 1.
    c5 = decode_tokens_by_program([_Req("A"), _Req("B")], per_req_tokens=[7])
    if c5 != {"A": 7, "B": 1}:
        raise StageFail(f"decode-counts: missing per_req entry defaults to 1; "
                        f"got {c5}")
    print(_green("  [decode-counts] 1 tok/req + accumulation + per_req_tokens "
                 "(spec accept_lens) OK"))


def stage_running_view() -> None:
    ema = {"A": 100.0, "B": 50.0, "C": 7.0}
    # only A and C are still running → B's stale EMA is dropped.
    v = running_program_view(ema, ["A", "C", "Z"])
    if v != {"A": 100.0, "C": 7.0}:
        raise StageFail(f"running-view: must project onto live pids; got {v}")
    if running_program_view(ema, []) != {}:
        raise StageFail("running-view: no running pids → {}")
    print(_green("  [running-view] projects EMA onto live programs OK"))


import time as _time
import types as _types


def _Cache():
    class _C:
        def _aginfer_bytes_per_token(self):
            return 2048
    return _C()


def _spec(on):
    return _types.SimpleNamespace(is_none=lambda: not on)


def _sched_self(spec_on=False):
    """A SimpleNamespace standing in for a Scheduler, with the real
    instrumentation methods bound onto it so the unbound-method calls
    inside ``_aginfer_record_throughput_inner`` resolve."""
    from sglang.srt.managers.scheduler import Scheduler
    s = _types.SimpleNamespace(
        tree_cache=_Cache(),
        spec_algorithm=_spec(spec_on),
        _aginfer_last_decode_t=None, _aginfer_last_prefill_t=None,
        _aginfer_decode_ema={}, _aginfer_prefill_bps_ema=None,
        _aginfer_throughput_warned=False)
    for name in ("_aginfer_record_throughput_inner", "_aginfer_update_decode",
                 "_aginfer_update_prefill", "_aginfer_record_spec_decode"):
        setattr(s, name, _types.MethodType(getattr(Scheduler, name), s))
    # staticmethod: attach the plain function (no self injection).
    s._aginfer_extend_token_count = Scheduler._aginfer_extend_token_count
    return s


def _batch(mode, reqs=None, ent=None, decoding_reqs=None, is_spec_v2=False):
    return _types.SimpleNamespace(
        forward_mode=mode, reqs=reqs or [], extend_num_tokens=ent,
        extend_lens=None, input_ids=None, decoding_reqs=decoding_reqs,
        is_spec_v2=is_spec_v2)


def stage_scheduler_routing() -> None:
    """#200/#206: pin the scheduler pre-forward hook's mode routing — pure
    DECODE→decode (unless spec, then post-forward), pure EXTEND→prefill,
    MIXED→split (prefill+decode), and the per-forward wrapper never raises."""
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    # pure DECODE (no spec) → decode EMA populated for the batch's programs.
    s = _sched_self(); s._aginfer_last_decode_t = _time.perf_counter() - 0.1
    Scheduler._aginfer_record_throughput_inner(
        s, _batch(ForwardMode.DECODE, reqs=[_Req("A"), _Req("B")]))
    if set(s._aginfer_decode_ema) != {"A", "B"} or not all(
            v > 0 for v in s._aginfer_decode_ema.values()):
        raise StageFail(f"routing: pure DECODE must populate decode EMA; "
                        f"got {s._aginfer_decode_ema}")
    if s._aginfer_prefill_bps_ema is not None:
        raise StageFail("routing: DECODE must not touch prefill_bps")

    # spec-v2 DECODE (overlap-on) → pre-forward skips (decode counted
    # post-forward via accept_lens, not 1/req here) — #206.
    s = _sched_self(spec_on=True)
    s._aginfer_last_decode_t = _time.perf_counter() - 0.1
    Scheduler._aginfer_record_throughput_inner(
        s, _batch(ForwardMode.DECODE, reqs=[_Req("A"), _Req("B")],
                  is_spec_v2=True))
    if s._aginfer_decode_ema:
        raise StageFail(f"routing: spec-v2 DECODE must skip pre-forward decode "
                        f"(counted post-forward); got {s._aginfer_decode_ema}")

    # spec-v1 DECODE (overlap OFF, e.g. ngram) → no accept_lens post-forward,
    # so it KEEPS the conservative 1/req here (never empty, no regression).
    s = _sched_self(spec_on=True)
    s._aginfer_last_decode_t = _time.perf_counter() - 0.1
    Scheduler._aginfer_record_throughput_inner(
        s, _batch(ForwardMode.DECODE, reqs=[_Req("A"), _Req("B")],
                  is_spec_v2=False))
    if set(s._aginfer_decode_ema) != {"A", "B"}:
        raise StageFail(f"routing: spec-v1 DECODE must keep 1/req pre-forward "
                        f"(no regression); got {s._aginfer_decode_ema}")

    # pure EXTEND → prefill_bps populated, decode untouched.
    s = _sched_self(); s._aginfer_last_prefill_t = _time.perf_counter() - 0.1
    Scheduler._aginfer_record_throughput_inner(
        s, _batch(ForwardMode.EXTEND, ent=100))
    if not (s._aginfer_prefill_bps_ema and s._aginfer_prefill_bps_ema > 0):
        raise StageFail("routing: pure EXTEND must populate prefill_bps")
    if s._aginfer_decode_ema:
        raise StageFail("routing: EXTEND must not touch decode EMA")

    # raise-safety: the WRAPPER must swallow an inner error (reqs not
    # iterable → TypeError inside) — never propagate into the forward loop.
    s = _sched_self(); s._aginfer_last_decode_t = _time.perf_counter() - 0.1
    bad = _batch(ForwardMode.DECODE); bad.reqs = 5  # not iterable
    try:
        Scheduler._aginfer_record_throughput(s, bad)
    except Exception as e:  # noqa: BLE001
        raise StageFail(f"routing: per-forward hook must NOT raise; got {e!r}")
    if not s._aginfer_throughput_warned:
        raise StageFail("routing: a suppressed error must set the warned flag")
    print(_green("  [routing] DECODE→decode, spec-v2→skip, spec-v1→1/req, "
                 "EXTEND→prefill, raise-safe OK"))


def stage_mixed() -> None:
    """#206: a MIXED batch (chunked prefill + running decode) must split via
    batch.decoding_reqs — prefill reqs' extend_input_len → prefill_bps, decode
    reqs → decode EMA.  extend_num_tokens (whole-batch) must NOT pollute."""
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    # prefill req X (256-token chunk) + two decode reqs A, B.
    pre = _Req("X", extend_input_len=256)
    da, db = _Req("A"), _Req("B")
    s = _sched_self()
    s._aginfer_last_prefill_t = _time.perf_counter() - 0.1
    s._aginfer_last_decode_t = _time.perf_counter() - 0.1
    # extend_num_tokens=258 (256 prefill + 1×2 decode) — the trap value the
    # split must IGNORE in favour of the prefill reqs' extend_input_len.
    Scheduler._aginfer_record_throughput_inner(
        s, _batch(ForwardMode.MIXED, reqs=[pre, da, db], ent=258,
                  decoding_reqs=[da, db]))
    if not (s._aginfer_prefill_bps_ema and s._aginfer_prefill_bps_ema > 0):
        raise StageFail("mixed: prefill portion must feed prefill_bps")
    # decode reqs A,B present in EMA; prefill req X absent from decode EMA.
    if set(s._aginfer_decode_ema) != {"A", "B"}:
        raise StageFail(f"mixed: decode reqs (not prefill X) must feed decode "
                        f"EMA; got {set(s._aginfer_decode_ema)}")

    # Deterministic proof that extend_num_tokens is NOT used: an all-decode
    # MIXED batch has zero prefill reqs → ntok=0 → prefill_bps stays unset,
    # even though extend_num_tokens=258 is nonzero (the old trap).
    s0 = _sched_self()
    s0._aginfer_last_prefill_t = _time.perf_counter() - 0.1
    s0._aginfer_last_decode_t = _time.perf_counter() - 0.1
    Scheduler._aginfer_record_throughput_inner(
        s0, _batch(ForwardMode.MIXED, reqs=[da, db], ent=258,
                   decoding_reqs=[da, db]))
    if s0._aginfer_prefill_bps_ema is not None:
        raise StageFail(f"mixed: all-decode batch (no prefill reqs) must NOT "
                        f"set prefill_bps from extend_num_tokens; "
                        f"got {s0._aginfer_prefill_bps_ema}")
    if set(s0._aginfer_decode_ema) != {"A", "B"}:
        raise StageFail("mixed: all-decode MIXED must still feed decode EMA")

    # spec-v2 MIXED → decode portion is STILL counted here (1/req), NOT
    # deferred: the post-forward accept_lens hook only fires for pure
    # is_decode() batches, so deferring a MIXED batch would drop its decode
    # tokens entirely (#206 audit F2).  prefill still counts too.
    s2 = _sched_self(spec_on=True)
    s2._aginfer_last_prefill_t = _time.perf_counter() - 0.1
    s2._aginfer_last_decode_t = _time.perf_counter() - 0.1
    Scheduler._aginfer_record_throughput_inner(
        s2, _batch(ForwardMode.MIXED, reqs=[pre, da, db], ent=258,
                   decoding_reqs=[da, db], is_spec_v2=True))
    if not (s2._aginfer_prefill_bps_ema and s2._aginfer_prefill_bps_ema > 0):
        raise StageFail("mixed(spec-v2): prefill portion must still feed prefill_bps")
    if set(s2._aginfer_decode_ema) != {"A", "B"}:
        raise StageFail(f"mixed(spec-v2): decode portion must be counted 1/req "
                        f"here (post-forward hook can't reach MIXED), not "
                        f"dropped; got {s2._aginfer_decode_ema}")
    print(_green("  [mixed] split prefill→bps / decode→EMA via decoding_reqs; "
                 "extend_num_tokens not used; spec-v2 MIXED still counts decode OK"))


def stage_bpt_cache() -> None:
    """#209 regression guard: UnifiedRadixCache._aginfer_bytes_per_token must
    NOT cache a transient 0 (kvcache not yet wired when an early #200/#206
    hot-path caller asks).  Caching 0 poisoned every pool_usage cap_bytes
    → occ_hbm≡0 → the daemon never saw HBM pressure.  Only a positive bpt
    may be cached; a 0 must recompute next call."""
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    class _KV:
        def __init__(self, bpt):
            self._bpt = bpt
        def get_bytes_per_token(self):
            return self._bpt

    class _Alloc:
        def __init__(self, kv):
            self._kvcache = kv

    kv = _KV(0)
    s = _types.SimpleNamespace(token_to_kv_pool_allocator=_Alloc(kv))
    fn = UnifiedRadixCache._aginfer_bytes_per_token

    # 1st call: kvcache reports 0 (early/transient) → returns 0, NOT cached.
    if fn(s) != 0:
        raise StageFail("bpt-cache: transient 0 must return 0")
    if getattr(s, "_aginfer_bpt_cache", 0):
        raise StageFail("bpt-cache: a 0 must NOT be cached (the #209 poison)")
    # 2nd call: kvcache now reports the real value → returns it AND caches it.
    kv._bpt = 2048
    if fn(s) != 2048:
        raise StageFail("bpt-cache: must recompute the real bpt after a 0")
    if getattr(s, "_aginfer_bpt_cache", 0) != 2048:
        raise StageFail("bpt-cache: a positive bpt must be cached")
    # 3rd call: even if the pool transiently reports 0 again, the cached
    # positive value stands (layout never changes once known).
    kv._bpt = 0
    if fn(s) != 2048:
        raise StageFail("bpt-cache: cached positive bpt must survive a later 0")
    print(_green("  [bpt-cache] transient 0 not cached; positive bpt cached + "
                 "sticky (#209) OK"))


def stage_spec_decode() -> None:
    """#206: the post-forward spec hook attributes accept_lens (= accepted
    tokens/req, ≥1) to per-program decode EMA — NOT 1/req.  Reads
    result.num_correct_drafts_per_req_cpu (= accept_lens − 1), accepted=+1."""
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    reqs = [_Req("A"), _Req("B")]
    result = _types.SimpleNamespace(num_correct_drafts_per_req_cpu=[2, 0])
    s = _sched_self(spec_on=True)
    s._aginfer_last_decode_t = _time.perf_counter() - 0.1
    Scheduler._aginfer_record_spec_decode(
        s, _batch(ForwardMode.DECODE, reqs=reqs), result)
    # accepted A=3 (2+1), B=1 (0+1).  EMA seeds on first sample → ratio 3:1.
    a, b = s._aginfer_decode_ema.get("A"), s._aginfer_decode_ema.get("B")
    if a is None or b is None or abs(a / b - 3.0) > 1e-6:
        raise StageFail(f"spec: accept_lens must drive decode EMA at 3:1 "
                        f"(A=3 accepted, B=1); got A={a} B={b}")

    # result without spec accept counts → no-op (non-spec result shape).
    s2 = _sched_self(spec_on=True)
    s2._aginfer_last_decode_t = _time.perf_counter() - 0.1
    Scheduler._aginfer_record_spec_decode(
        s2, _batch(ForwardMode.DECODE, reqs=reqs),
        _types.SimpleNamespace())
    if s2._aginfer_decode_ema:
        raise StageFail(f"spec: missing accept counts must be a no-op; "
                        f"got {s2._aginfer_decode_ema}")

    # wrapper raise-safety: malformed result must not propagate.
    s3 = _sched_self(spec_on=True)
    s3._aginfer_last_decode_t = _time.perf_counter() - 0.1
    bad = _types.SimpleNamespace(num_correct_drafts_per_req_cpu="notalist")
    try:
        Scheduler._aginfer_record_spec_throughput(
            s3, _batch(ForwardMode.DECODE, reqs=reqs), bad)
    except Exception as e:  # noqa: BLE001
        raise StageFail(f"spec: post-forward wrapper must NOT raise; got {e!r}")
    if not s3._aginfer_throughput_warned:
        raise StageFail("spec: a suppressed error must set the warned flag")
    print(_green("  [spec] accept_lens → per-program decode EMA (≠1/req); "
                 "no-op on missing counts; raise-safe OK"))


_STAGES = [
    ("ema", stage_ema),
    ("inflight", stage_inflight),
    ("decode-counts", stage_decode_counts),
    ("running-view", stage_running_view),
    ("routing", stage_scheduler_routing),
    ("mixed", stage_mixed),
    ("spec-decode", stage_spec_decode),
    ("bpt-cache", stage_bpt_cache),
]


def main() -> int:
    print("=" * 60)
    print("T26 (#200) — sglang throughput / in-flight pure helpers")
    print("=" * 60)
    failed = []
    for name, fn in _STAGES:
        try:
            fn()
        except StageFail as e:
            failed.append(name)
            print(_red(f"  [{name}] FAIL: {e}"))
        except Exception as e:  # noqa: BLE001
            failed.append(name)
            import traceback
            print(_red(f"  [{name}] ERROR: {e}"))
            traceback.print_exc()
    print("=" * 60)
    if failed:
        print(_red(f"FAILED: {', '.join(failed)}"))
        return 1
    print(_green(f"T26 pure-helpers PASS — all {len(_STAGES)} stages green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
