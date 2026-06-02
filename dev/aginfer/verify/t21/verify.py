"""T21 — PUT /aginfer/program_paused (#181, DESIGN §6 round-6 H2).

Daemon → sglang push of program-state transitions
(REASONING / ACTING / PAUSED / ENDED) into the cache's
per_program_usage entry.  Sglang stores the daemon's view as a
passthrough and echoes it back in the next /aginfer/state dump.

Stages (10):

  A. set_aginfer_program_state (cache-level unit tests)
    A0 valid state stores, ok=True, applied=1
    A1 idempotent re-apply: same (state, pre_pause_state) →
       ok=True, applied=0 (DESIGN §10 R2)
    A2 invalid state string → ok=False, applied=0
    A3 invalid pre_pause_state → ok=False, applied=0
    A4 empty pid → ok=False
    A5 None pre_pause_state is valid (default)

  B. Dump-path echo
    B0 stored state shows up in dict-path per_program_usage[pid]
    B1 stored state shows up in bytes-path per_program_usage[pid]
    B2 pids with NO live units still appear in the dump
       (PAUSED-with-no-residue bookkeeping)

  C. Scheduler handler shape
    C0 update_aginfer_program_paused returns ok=False when the
       tree cache lacks set_aginfer_program_state (legacy
       HiRadixCache without UnifiedRadixCache env var)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, List, Tuple


_HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
sys.path.insert(0, str(_HERE.parent.parent))


def _green(s: str) -> str: return f"\033[32m{s}\033[0m"
def _red(s: str) -> str:   return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


def _new_cache():
    """Construct a UnifiedRadixCache stub sufficient to call
    `set_aginfer_program_state` and `_dump_aginfer_state_dict`."""
    # Use a real instance via importing the class; bypass __init__'s
    # heavy GPU dependencies by allocating with __new__ + manually
    # setting the attrs the methods we test touch.
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache._aginfer_program_states = {}
    return cache


# ---- A. setter unit tests ----


def stage_a0_valid_state_stores() -> None:
    cache = _new_cache()
    ok, reason, applied = cache.set_aginfer_program_state(
        pid="p-arrival", state="REASONING", pre_pause_state=None,
    )
    if not (ok and applied == 1):
        raise StageFail(
            f"first PUT should be ok+applied=1; got ok={ok} "
            f"applied={applied} reason={reason!r}"
        )
    if cache._aginfer_program_states["p-arrival"] != {
        "state": "REASONING", "pre_pause_state": None,
    }:
        raise StageFail(
            f"storage not updated: {cache._aginfer_program_states!r}"
        )


def stage_a1_idempotent_reapply() -> None:
    cache = _new_cache()
    cache.set_aginfer_program_state(
        pid="p", state="PAUSED", pre_pause_state="ACTING",
    )
    ok, reason, applied = cache.set_aginfer_program_state(
        pid="p", state="PAUSED", pre_pause_state="ACTING",
    )
    if not (ok and applied == 0):
        raise StageFail(
            f"idempotent re-apply should be ok+applied=0; got "
            f"ok={ok} applied={applied}"
        )


def stage_a2_invalid_state_rejected() -> None:
    cache = _new_cache()
    ok, reason, applied = cache.set_aginfer_program_state(
        pid="p", state="HUNGRY", pre_pause_state=None,
    )
    if ok or applied != 0:
        raise StageFail(
            f"invalid state should reject; got ok={ok} applied={applied}"
        )
    if "HUNGRY" not in reason:
        raise StageFail(f"reason should mention bad value: {reason!r}")


def stage_a3_invalid_pre_pause_rejected() -> None:
    cache = _new_cache()
    ok, reason, applied = cache.set_aginfer_program_state(
        pid="p", state="PAUSED", pre_pause_state="DREAMING",
    )
    if ok or applied != 0:
        raise StageFail(
            f"invalid pre_pause should reject; got ok={ok}"
        )


def stage_a4_empty_pid_rejected() -> None:
    cache = _new_cache()
    ok, _, applied = cache.set_aginfer_program_state(
        pid="", state="REASONING", pre_pause_state=None,
    )
    if ok or applied != 0:
        raise StageFail(f"empty pid should reject; got ok={ok}")
    # Also non-str pid.
    ok, _, applied = cache.set_aginfer_program_state(
        pid=None, state="REASONING", pre_pause_state=None,  # type: ignore[arg-type]
    )
    if ok or applied != 0:
        raise StageFail(f"None pid should reject; got ok={ok}")


def stage_a5_none_pre_pause_valid() -> None:
    """The default `pre_pause_state=None` is valid for the
    REASONING / ACTING / ENDED states; only PAUSED conventionally
    carries the prior state, but the setter allows any combo."""
    cache = _new_cache()
    ok, _, applied = cache.set_aginfer_program_state(
        pid="p", state="ACTING", pre_pause_state=None,
    )
    if not (ok and applied == 1):
        raise StageFail(f"None pre_pause valid; got ok={ok}")


# ---- B. dump-path echo ----


def stage_b0_real_helper_overlays_state() -> None:
    """#186 audit: call the REAL production overlay helper
    (`_aginfer_overlay_program_states`) — both dump paths invoke
    THIS exact method, so testing it is testing production code,
    not a hand-copied replica.

    Existing program (from unit aggregation) gets its state
    OVERLAID by the daemon's PUT; non-state fields preserved."""
    cache = _new_cache()
    cache.set_aginfer_program_state(
        pid="p1", state="PAUSED", pre_pause_state="ACTING",
    )
    per_program = {
        "p1": {
            "hbm":  {"committed": {"kv": 1024}, "inflight": {}},
            "dram": {"committed": {}},
            "state": "REASONING",   # unit-walk default
            "pre_pause_state": None,
            "unit_hashes": ["h0", "h1"],
        }
    }
    out = cache._aginfer_overlay_program_states(per_program)
    if out is not per_program:
        raise StageFail("overlay should mutate + return the same dict")
    if out["p1"]["state"] != "PAUSED":
        raise StageFail(f"state not overlaid: {out['p1']!r}")
    if out["p1"]["pre_pause_state"] != "ACTING":
        raise StageFail(f"pre_pause not overlaid: {out['p1']!r}")
    if out["p1"]["hbm"]["committed"]["kv"] != 1024:
        raise StageFail(f"hbm clobbered: {out['p1']!r}")
    if out["p1"]["unit_hashes"] != ["h0", "h1"]:
        raise StageFail(f"unit_hashes clobbered: {out['p1']!r}")


def stage_b1_both_dump_paths_call_shared_helper() -> None:
    """#186 audit: the divergence risk between dict-path and bytes-
    path is eliminated by construction — both must call the single
    `_aginfer_overlay_program_states` method.  Pin that via source
    inspection so a future edit that re-inlines one path (and risks
    drift) fails loud here."""
    import inspect
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    for meth in ("_dump_aginfer_state_dict", "_dump_aginfer_state_bytes"):
        src = inspect.getsource(getattr(UnifiedRadixCache, meth))
        if "_aginfer_overlay_program_states" not in src:
            raise StageFail(
                f"{meth} does not call the shared overlay helper — "
                f"the two dump paths can diverge again.  Both MUST "
                f"call self._aginfer_overlay_program_states(per_program)."
            )


def stage_b2_pid_with_no_units_still_appears() -> None:
    """PAUSED program with no live units (all KV evicted) still
    shows up in the dump so the daemon can read its own state."""
    cache = _new_cache()
    cache.set_aginfer_program_state(
        pid="ghost", state="PAUSED", pre_pause_state="REASONING",
    )
    out = cache._aginfer_overlay_program_states({})  # no units cited
    if "ghost" not in out:
        raise StageFail(
            "unit-less PAUSED program should still appear; got "
            f"{list(out)}"
        )
    g = out["ghost"]
    if g["state"] != "PAUSED" or g["pre_pause_state"] != "REASONING":
        raise StageFail(f"ghost state wrong: {g!r}")
    if g["hbm"]["committed"] != {} or g["unit_hashes"] != []:
        raise StageFail(f"ghost should be empty residue: {g!r}")


def stage_b3_ended_no_units_gc() -> None:
    """#186 audit (unbounded-growth fix): an ENDED program with NO
    live units is GC'd from _aginfer_program_states during the dump
    AND not echoed.  Without this the dict grows forever."""
    cache = _new_cache()
    cache.set_aginfer_program_state(
        pid="done", state="ENDED", pre_pause_state="REASONING",
    )
    if "done" not in cache._aginfer_program_states:
        raise StageFail("setup: ENDED state not stored")
    out = cache._aginfer_overlay_program_states({})  # no units
    if "done" in out:
        raise StageFail(
            f"ENDED-no-units program should NOT be echoed; got {out!r}"
        )
    if "done" in cache._aginfer_program_states:
        raise StageFail(
            "ENDED-no-units program should be GC'd from storage; "
            f"still present: {cache._aginfer_program_states!r}"
        )


def stage_b4_ended_with_units_kept() -> None:
    """ENDED program that STILL has residual units IS echoed (the
    daemon needs to see the terminal state while cleanup runs) and
    NOT GC'd."""
    cache = _new_cache()
    cache.set_aginfer_program_state(
        pid="ending", state="ENDED", pre_pause_state=None,
    )
    per_program = {
        "ending": {
            "hbm":  {"committed": {"kv": 512}, "inflight": {}},
            "dram": {"committed": {}},
            "state": "REASONING",
            "pre_pause_state": None,
            "unit_hashes": ["h0"],
        }
    }
    out = cache._aginfer_overlay_program_states(per_program)
    if out.get("ending", {}).get("state") != "ENDED":
        raise StageFail(
            f"ENDED-with-units should be echoed ENDED; got "
            f"{out.get('ending')!r}"
        )
    if "ending" not in cache._aginfer_program_states:
        raise StageFail(
            "ENDED-with-units should NOT be GC'd (cleanup ongoing)"
        )


# ---- C. scheduler handler shape ----


def stage_c0_unsupported_tree_cache_rejected() -> None:
    """When the tree cache lacks `set_aginfer_program_state` (legacy
    HiRadixCache), the scheduler handler returns ok=False with a
    helpful reason naming the cache type."""
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.io_struct import UpdateAginferProgramPausedReq

    class _LegacyCache:
        """No set_aginfer_program_state attribute."""
        pass

    class _Shim:
        tree_cache = _LegacyCache()
        update_aginfer_program_paused = Scheduler.update_aginfer_program_paused

    req = UpdateAginferProgramPausedReq(
        pid="p", state="PAUSED", pre_pause_state="REASONING",
    )
    out = _Shim.update_aginfer_program_paused(_Shim(), req)
    if out.ok:
        raise StageFail(f"unsupported cache should fail; got ok=True")
    if "_LegacyCache" not in out.reason:
        raise StageFail(f"reason should name cache type: {out.reason!r}")
    if out.applied != 0:
        raise StageFail(f"applied must be 0 on failure; got {out.applied}")


# ---- D. HTTP body validation (the #186 coercion-bypass fix) ----


def stage_d0_http_body_validation() -> None:
    """#186 audit: the HTTP route previously did `str(body["pid"])`,
    silently coercing JSON null/number to the strings "None"/"123"
    — bypassing the setter's empty-pid guard.  The route now calls
    `_validate_program_paused_body` which type-checks BEFORE
    coercion.  Pin all the reject + accept cases on that pure
    function."""
    from sglang.srt.entrypoints.http_server import (
        _validate_program_paused_body as _v,
    )

    # Accept: well-formed body.
    pid, state, pre = _v({
        "pid": "p", "state": "PAUSED", "pre_pause_state": "ACTING",
    })
    if (pid, state, pre) != ("p", "PAUSED", "ACTING"):
        raise StageFail(f"valid body mis-parsed: {(pid, state, pre)}")
    # Accept: pre_pause_state omitted → None.
    _, _, pre2 = _v({"pid": "p", "state": "ENDED"})
    if pre2 is not None:
        raise StageFail(f"omitted pre_pause should be None; got {pre2!r}")

    # Reject cases — each MUST raise ValueError.
    bad_bodies = [
        ("non-dict", "not a dict"),
        ("missing pid", {"state": "PAUSED"}),
        ("missing state", {"pid": "p"}),
        ("null pid (coercion bypass)", {"pid": None, "state": "PAUSED"}),
        ("numeric pid (coercion bypass)", {"pid": 123, "state": "PAUSED"}),
        ("empty pid", {"pid": "", "state": "PAUSED"}),
        ("null state", {"pid": "p", "state": None}),
        ("numeric state", {"pid": "p", "state": 5}),
        ("empty state", {"pid": "p", "state": ""}),
        ("numeric pre_pause", {"pid": "p", "state": "PAUSED",
                               "pre_pause_state": 7}),
    ]
    for label, body in bad_bodies:
        try:
            _v(body)
        except ValueError:
            continue
        raise StageFail(
            f"body '{label}' should raise ValueError; got no raise "
            f"(body={body!r})"
        )


# ---- run ----


_STAGES: List[Tuple[str, Callable[[], None]]] = [
    ("A0 valid state stores (applied=1)",           stage_a0_valid_state_stores),
    ("A1 idempotent re-apply (applied=0)",          stage_a1_idempotent_reapply),
    ("A2 invalid state rejected",                   stage_a2_invalid_state_rejected),
    ("A3 invalid pre_pause_state rejected",         stage_a3_invalid_pre_pause_rejected),
    ("A4 empty/None pid rejected",                  stage_a4_empty_pid_rejected),
    ("A5 None pre_pause_state valid",               stage_a5_none_pre_pause_valid),
    ("B0 REAL overlay helper echoes state",         stage_b0_real_helper_overlays_state),
    ("B1 both dump paths call shared helper (source pin)",
                                                    stage_b1_both_dump_paths_call_shared_helper),
    ("B2 unit-less PAUSED program still appears",   stage_b2_pid_with_no_units_still_appears),
    ("B3 ENDED + no units → GC'd, not echoed",      stage_b3_ended_no_units_gc),
    ("B4 ENDED + residual units → kept + echoed",   stage_b4_ended_with_units_kept),
    ("C0 unsupported tree cache → ok=False with type-name reason",
                                                    stage_c0_unsupported_tree_cache_rejected),
    ("D0 HTTP body validation (coercion-bypass fix)",
                                                    stage_d0_http_body_validation),
]


def main() -> int:
    failures: List[str] = []
    for label, fn in _STAGES:
        try:
            fn()
            print(f"  {_green('PASS')}  Stage {label}")
        except StageFail as exc:
            failures.append(label)
            print(f"  {_red('FAIL')}  Stage {label}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(label)
            print(f"  {_red('FAIL')}  Stage {label}: "
                  f"unexpected {type(exc).__name__}: {exc}")
    if failures:
        print(_red(f"\nT21 FAILED ({len(failures)}): {failures}"))
        return 1
    print(_green(f"\nT21 PASS — all {len(_STAGES)} stages green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
