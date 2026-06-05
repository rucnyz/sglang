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
    def __init__(self, program_id, allocated, committed):
        self.program_id = program_id
        self.kv_allocated_len = allocated
        self.kv_committed_len = committed


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
    if not (0.0 < AGINFER_THROUGHPUT_EMA_ALPHA < 1.0):
        raise StageFail("ema: default alpha must be in (0,1)")
    print(_green("  [ema] seed / blend / alpha / malformed-guard OK"))


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
    print(_green("  [decode-counts] 1 tok/req + accumulation OK"))


def stage_running_view() -> None:
    ema = {"A": 100.0, "B": 50.0, "C": 7.0}
    # only A and C are still running → B's stale EMA is dropped.
    v = running_program_view(ema, ["A", "C", "Z"])
    if v != {"A": 100.0, "C": 7.0}:
        raise StageFail(f"running-view: must project onto live pids; got {v}")
    if running_program_view(ema, []) != {}:
        raise StageFail("running-view: no running pids → {}")
    print(_green("  [running-view] projects EMA onto live programs OK"))


_STAGES = [
    ("ema", stage_ema),
    ("inflight", stage_inflight),
    ("decode-counts", stage_decode_counts),
    ("running-view", stage_running_view),
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
    print(_green("T26 pure-helpers PASS — all 4 stages green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
