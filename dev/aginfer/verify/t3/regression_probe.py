"""T3 round-3 regression probe.

Standalone probe that exercises ONLY the two BLOCKERs from audit round 3:
  * [A] Session.create_req must forward program_id (BLOCKER 1)
  * [B] Sanitizer must cap recursion depth (BLOCKER 2)

Each check is wrapped in try/except so a failure in one doesn't mask the
other.  Used as a bisect-style demonstration: run pre-fix and post-fix
to prove the fix changes behaviour.

Usage:
    AGINFER_VERIFY_BASE=http://127.0.0.1:30001 \
    AGINFER_VERIFY_MODEL=Qwen/Qwen3-0.6B \
    python dev/aginfer/verify/t3/regression_probe.py
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import requests

BASE = os.environ.get("AGINFER_VERIFY_BASE", "http://127.0.0.1:30001")
MODEL = os.environ.get("AGINFER_VERIFY_MODEL", "Qwen/Qwen3-0.6B")


def fetch_state() -> Dict[str, Any]:
    r = requests.get(f"{BASE}/aginfer/state", timeout=30)
    r.raise_for_status()
    return r.json()


def units_with(state: Dict[str, Any], pid: str) -> List[Dict[str, Any]]:
    return [u for u in state["units"] if pid in u["session_ids"]]


def probe_session_forward() -> str:
    """[A] Session.create_req must forward program_id.

    The OpenAI chat handler does NOT forward ``session_params`` to the
    underlying GenerateReqInput; this probe uses sglang's native
    ``/generate`` endpoint instead, which DOES expose session_params and
    actually exercises ``Session.create_req`` on the scheduler side.

    Path: flush -> open_session -> seed via /generate -> second turn via
    /generate with the same session_id + a program_id.  The second turn
    hits ``Session.create_req`` and surfaces the bug.
    """
    # Flush so we don't see residue tags from earlier (non-session-path)
    # chats in the same sglang process.
    try:
        requests.post(f"{BASE}/flush_cache", timeout=10).raise_for_status()
    except Exception:
        pass  # /flush_cache may not exist on all builds; ignore.

    open_r = requests.post(f"{BASE}/open_session", json={"capacity_of_str_len": 1024}, timeout=30)
    open_r.raise_for_status()
    raw = open_r.text.strip()
    # body is a JSON string literal: "abc-def-..."
    session_id = raw.strip('"')
    if not session_id:
        return f"FAIL: open_session returned empty: {open_r.text!r}"

    seed = {
        "text": "session seed prompt: tell me about prime 19.",
        "sampling_params": {"max_new_tokens": 4, "temperature": 0.0},
        "session_params": {"id": session_id},
    }
    r1 = requests.post(f"{BASE}/generate", json=seed, timeout=60)
    if r1.status_code != 200:
        return f"FAIL: seed /generate -> {r1.status_code}: {r1.text[:200]}"

    body = {
        "text": "session second turn: tell me about prime 23.",
        "sampling_params": {"max_new_tokens": 4, "temperature": 0.0},
        "session_params": {"id": session_id},
        "program_id": "prog-A-SESSION-PROBE",
    }
    r2 = requests.post(f"{BASE}/generate", json=body, timeout=60)
    if r2.status_code != 200:
        return f"FAIL: tagged /generate -> {r2.status_code}: {r2.text[:200]}"

    state = fetch_state()
    if units_with(state, "prog-A-SESSION-PROBE"):
        return "PASS"
    return (
        "FAIL: session-multi-turn request did NOT tag any node — "
        "Session.create_req is silently dropping program_id"
    )


def probe_recursion_dos() -> str:
    """[B] Sanitizer must cap recursion on deeply-nested list program_id.

    Flush the cache first so the buried "deeply-buried" tag we test for
    isn't accidentally already in the tree from an earlier probe run.

    Python's ``json.dumps`` (used by ``requests`` library) ALSO recursion-
    errors on deeply-nested lists — so we can't construct the bomb with
    ``json.dumps(nested_list)``.  Build the JSON payload as raw bytes
    using stringbuilding ``[[...]]`` and ship via the requests bytes
    interface.  This bypasses the client-side JSON encoder and lets the
    bomb actually reach the server's sanitizer.
    """
    try:
        requests.post(f"{BASE}/flush_cache", timeout=10).raise_for_status()
    except Exception:
        pass
    # depth=20: well above the post-fix cap (8) but well below Python's
    # json.loads recursion limit (~150 on this Python build).  At this
    # depth, the JSON parser accepts the payload AND it reaches the
    # sanitizer.  Pre-fix (cap effectively disabled): sanitizer recurses
    # 20 levels and tags "deeply-buried" on the tree.  Post-fix
    # (cap=8): sanitizer returns None at depth 9 and the tag never
    # lands.  The probe asserts "deeply-buried" is NOT in the tree.
    depth = 20
    # Use /generate directly so the program_id reaches the sanitizer
    # via the sglang-native path; building raw JSON bytes also avoids
    # Python's client-side json.dumps recursion limit.
    raw = (
        b'{"text":"recursion probe via raw POST"' +
        b',"sampling_params":{"max_new_tokens":4,"temperature":0.0}' +
        b',"program_id":' +
        b"[" * depth + b'"deeply-buried"' + b"]" * depth +
        b"}"
    )
    try:
        r = requests.post(
            f"{BASE}/generate",
            data=raw,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
    except requests.exceptions.RequestException as exc:
        return f"FAIL: request raised {type(exc).__name__}: {exc!s}"
    if r.status_code >= 500:
        return f"FAIL: server returned {r.status_code} (scheduler crash?)"
    # Sanity: server still healthy after the bomb
    try:
        hh = requests.get(f"{BASE}/health", timeout=10)
    except requests.exceptions.RequestException as exc:
        return f"FAIL: /health raised {type(exc).__name__}: {exc!s}"
    if hh.status_code >= 500:
        return f"FAIL: /health returned {hh.status_code} after recursion bomb"
    # If the request was accepted, the cap means the buried tag should
    # NOT be in any node's session_ids.
    if r.status_code == 200:
        try:
            state = fetch_state()
            if units_with(state, "deeply-buried"):
                return "FAIL: deeply-buried tag bypassed the cap"
        except requests.exceptions.RequestException as exc:
            return f"FAIL: /aginfer/state after bomb raised {exc!s}"
    return "PASS"


def assert_fix_state_restored() -> None:
    """Defensive: catch the case where the bisect demo's revert was
    forgotten in code.  The README documents how to revert each fix
    for the demo; if a maintainer forgets to restore, the probe
    would still silently pass for some pre-fix configs.  Introspect
    the production code state explicitly.
    """
    import inspect

    from sglang.srt.managers.schedule_batch import _PROGRAM_ID_MAX_RECURSION
    from sglang.srt.session.session_controller import SessionController

    assert _PROGRAM_ID_MAX_RECURSION == 8, (
        f"_PROGRAM_ID_MAX_RECURSION={_PROGRAM_ID_MAX_RECURSION} "
        f"(expected 8). The bisect demo's revert was forgotten -- "
        f"restore the cap in schedule_batch.py before re-running."
    )
    # The Session class is defined in session_controller; create_req is
    # a method.  Source must include the program_id forward.
    Session = None
    for _name, _obj in inspect.getmembers(
        SessionController.__module__
        if hasattr(SessionController, "__module__")
        else None
    ):
        pass
    from sglang.srt.session import session_controller as _sc_mod

    Session = getattr(_sc_mod, "Session", None)
    assert Session is not None, "Could not import Session class"
    src = inspect.getsource(Session.create_req)
    # Match the line ONLY if uncommented (commented-out reverts still
    # leave the text in inspect.getsource output).
    import re
    pattern = re.compile(
        r"^[ \t]*program_id\s*=\s*req\.program_id",
        re.MULTILINE,
    )
    assert pattern.search(src), (
        "Session.create_req source does NOT have an uncommented "
        "`program_id=req.program_id` line.  The bisect demo's revert "
        "was forgotten -- restore the line in session_controller.py "
        "before re-running."
    )


def main() -> int:
    print("=== T3 round-3 regression probe ===")
    print()
    print("[fix-state] probing production code for restored fixes ...")
    try:
        assert_fix_state_restored()
        print("    PASS  both round-3 fixes are present in source")
    except AssertionError as exc:
        print(f"    FAIL  {exc}")
        return 2
    print()

    results: List[tuple[str, str]] = []

    print("[A] Session.create_req forwards program_id ...")
    try:
        r = probe_session_forward()
    except Exception as exc:
        r = f"FAIL: exception {type(exc).__name__}: {exc!s}"
    print(f"    {r}")
    results.append(("Session.create_req forwards program_id", r))
    print()

    print("[B] Sanitizer caps recursion on deeply-nested list ...")
    try:
        r = probe_recursion_dos()
    except Exception as exc:
        r = f"FAIL: exception {type(exc).__name__}: {exc!s}"
    print(f"    {r}")
    results.append(("sanitizer recursion cap", r))
    print()

    print("=== summary ===")
    n_fail = 0
    for name, result in results:
        ok = result == "PASS"
        n_fail += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
