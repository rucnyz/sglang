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
    # /open_session currently returns a bare JSON string ("abc-def-..."),
    # but the response shape is not contractually fixed.  Try the JSON
    # parse first; accept dicts {"session_id": ...} or {"id": ...} for
    # forward-compat.  Fall back to text.strip('"') only if json fails.
    session_id = None
    try:
        parsed = open_r.json()
        if isinstance(parsed, str):
            session_id = parsed
        elif isinstance(parsed, dict):
            session_id = parsed.get("session_id") or parsed.get("id")
    except Exception:
        session_id = open_r.text.strip().strip('"')
    if not session_id or not isinstance(session_id, str):
        return f"FAIL: open_session response unparsable: {open_r.text!r}"

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
    # CRITICAL (round-5 audit BLOCKER): if the HTTP layer 400s on the
    # depth=20 payload, the request never reaches our sanitizer and the
    # cap is untested.  Future Python / FastAPI may tighten JSON parse
    # recursion limits; the test would silently become a no-op.  Require
    # status==200 so the sanitizer was actually exercised, OR escalate
    # with a clear "json parser rejected -- pick smaller depth" hint.
    if r.status_code != 200:
        return (
            f"FAIL: server returned {r.status_code} on depth-{depth} bomb; "
            f"the request never reached the sanitizer (JSON parser rejected "
            f"earlier).  The recursion cap is UNTESTED.  Tighten depth "
            f"(or split into two depths, one that lands at 200) and re-run."
        )
    # Sanity: server still healthy after the bomb
    try:
        hh = requests.get(f"{BASE}/health", timeout=10)
    except requests.exceptions.RequestException as exc:
        return f"FAIL: /health raised {type(exc).__name__}: {exc!s}"
    if hh.status_code >= 500:
        return f"FAIL: /health returned {hh.status_code} after recursion bomb"
    try:
        state = fetch_state()
    except requests.exceptions.RequestException as exc:
        return f"FAIL: /aginfer/state after bomb raised {exc!s}"
    if units_with(state, "deeply-buried"):
        return "FAIL: deeply-buried tag bypassed the cap"
    return "PASS"


def _function_passes_kwarg(func, kwarg_name: str, expected_value_src: str) -> bool:
    """AST check: does ``func`` contain a Call whose keyword ``kwarg_name``
    is set to the expression ``expected_value_src``?

    Round-6 audit BLOCKER 1: replaced ``_uncommented_lines`` + regex with
    real AST parsing because regex-matching the textual source had blind
    spots — docstrings, string literals containing ``#``, and other
    non-code occurrences of the fix string would all pass.  AST walks
    only over real keyword-argument nodes.
    """
    import ast
    import inspect
    import textwrap

    try:
        src = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError):
        return False
    expected = expected_value_src.strip()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != kwarg_name:
                continue
            # ast.unparse drops formatting differences but keeps
            # structural identity; this is exactly what we want.
            try:
                got = ast.unparse(kw.value).strip()
            except Exception:
                continue
            if got == expected:
                return True
    return False


def _assignment_present(func, attr_chain: str) -> bool:
    """AST check: does ``func`` contain an assignment whose target's text
    (via ``ast.unparse``) is ``attr_chain``?  Useful for verifying that
    a self.attribute init line is present.
    """
    import ast
    import inspect
    import textwrap

    try:
        src = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError):
        return False
    target = attr_chain.strip()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            for t in targets:
                try:
                    if ast.unparse(t).strip() == target:
                        return True
                except Exception:
                    continue
        elif isinstance(node, ast.AugAssign):
            try:
                if ast.unparse(node.target).strip() == target:
                    return True
            except Exception:
                continue
    return False


def assert_fix_state_restored() -> None:
    """Defensive: catch the case where any production-side fix was
    accidentally reverted (the bisect demo documents how to revert
    each one; this guards against "revert + forgot to restore").

    Introspects five production-code invariants (round-2/3/5/6 BLOCKERs):
      1. ``_PROGRAM_ID_MAX_RECURSION == 8`` (round-3 sanitizer cap)
      2. ``Session.create_req`` calls Req with ``program_id=req.program_id``
         (round-3 session multi-turn fix)
      3. ``MMReceiverBase.create_req`` calls Req with
         ``program_id=recv_req.program_id`` (round-2 EPD-disagg fix)
      4. ``UnifiedTreeNode.__init__`` assigns ``self.session_ids``
         (round-6 -- the tagging contract depends on this attribute
         being initialised; if removed, dump's try/except path silently
         emits empty lists)
      5. ``UnifiedRadixCache._split_node`` does
         ``new_node.session_ids |= child.session_ids`` (round-6 --
         split-merge inheritance; if removed, internal nodes lose
         tags after a radix split)

    All checks use ``ast.parse`` rather than regex.  Round-6 audit
    BLOCKER 1: regex over source text false-positives on docstrings /
    string literals that contain the fix string; AST walking ignores
    those automatically.
    """
    import inspect

    from sglang.srt.managers.schedule_batch import _PROGRAM_ID_MAX_RECURSION

    assert _PROGRAM_ID_MAX_RECURSION == 8, (
        f"_PROGRAM_ID_MAX_RECURSION={_PROGRAM_ID_MAX_RECURSION} "
        f"(expected 8). The bisect demo's revert was forgotten -- "
        f"restore the cap in schedule_batch.py before re-running."
    )

    # 2) Session.create_req
    from sglang.srt.session import session_controller as _sc_mod

    Session = getattr(_sc_mod, "Session", None)
    assert Session is not None, "Could not import Session class"
    assert _function_passes_kwarg(
        Session.create_req, "program_id", "req.program_id"
    ), (
        "Session.create_req does NOT pass `program_id=req.program_id` "
        "as a real keyword argument (AST check).  The bisect demo's "
        "revert was forgotten -- restore the line in "
        "session_controller.py before re-running."
    )

    # 3) MMReceiverBase.create_req (EPD-disagg path).  Class name is
    # sglang-version-dependent; pin the most likely one and fall back
    # to iteration.  Require EXACTLY ONE match so a future stub class
    # doesn't accidentally shadow the real one (round-6 MINOR 3).
    from sglang.srt.disaggregation import encode_receiver as _enc_mod

    pinned = getattr(_enc_mod, "MMReceiverBase", None)
    epd_create_req = None
    if pinned is not None and hasattr(pinned, "create_req"):
        try:
            sig = inspect.signature(pinned.create_req)
            if "recv_req" in sig.parameters:
                epd_create_req = pinned.create_req
        except (ValueError, TypeError):
            pass
    if epd_create_req is None:
        matches = []
        for _name, _obj in inspect.getmembers(_enc_mod, inspect.isclass):
            if _obj.__module__ != _enc_mod.__name__:
                continue
            cr = getattr(_obj, "create_req", None)
            if cr is None or not callable(cr):
                continue
            try:
                sig = inspect.signature(cr)
            except (ValueError, TypeError):
                continue
            if "recv_req" in sig.parameters:
                matches.append((_name, cr))
        assert len(matches) == 1, (
            f"Expected exactly one create_req(self, recv_req: ...) in "
            f"sglang.srt.disaggregation.encode_receiver, got "
            f"{len(matches)}: {[m[0] for m in matches]}.  Class layout "
            f"changed; pin a class name in assert_fix_state_restored."
        )
        epd_create_req = matches[0][1]
    assert _function_passes_kwarg(
        epd_create_req, "program_id", "recv_req.program_id"
    ), (
        "encode_receiver.<create_req> does NOT pass "
        "`program_id=recv_req.program_id` as a real keyword argument "
        "(AST check).  EPD-disagg tag would silently drop; restore "
        "the line in disaggregation/encode_receiver.py before re-running."
    )

    # 4) UnifiedTreeNode.__init__ initialises session_ids
    # (round-6 MINOR 6).  Without this, every node defaults to
    # missing-attr and the dump's try/except silently emits [].
    from sglang.srt.mem_cache.unified_radix_cache import (
        UnifiedRadixCache,
        UnifiedTreeNode,
    )

    assert _assignment_present(
        UnifiedTreeNode.__init__, "self.session_ids"
    ), (
        "UnifiedTreeNode.__init__ does NOT assign `self.session_ids` "
        "(AST check).  Every node would start without the attribute "
        "and the dump path would silently emit empty session_ids.  "
        "Restore the init line in unified_radix_cache.py."
    )

    # 5) _split_node does the union-merge of session_ids
    # (round-6 MINOR 6).  Without this, internal nodes created by
    # radix splits lose all tags carried by the original (longer-
    # prefix) child.
    assert _assignment_present(
        UnifiedRadixCache._split_node, "new_node.session_ids"
    ), (
        "_split_node does NOT assign `new_node.session_ids` "
        "(AST check).  After a radix split, the new internal node "
        "would lose every program tag carried by its child.  Restore "
        "the union-merge line in unified_radix_cache.py."
    )


def main() -> int:
    print("=== T3 regression probe ===")
    print()
    # MINOR 4 (round 6): HEAD /health first so a "sglang isn't running"
    # error is the first thing the user sees, not "PASS [fix-state]"
    # followed by ConnectionError on [A].
    try:
        h = requests.get(f"{BASE}/health", timeout=5)
    except requests.exceptions.RequestException as exc:
        print(
            f"FATAL  cannot reach sglang at {BASE}: {type(exc).__name__}: {exc!s}.\n"
            f"       Launch sglang first; see verify/t3/README.md REPRODUCING."
        )
        return 3
    if h.status_code >= 500:
        print(f"FATAL  /health returned {h.status_code}; sglang is unhealthy.")
        return 3

    print("[fix-state] probing production code for restored fixes ...")
    try:
        assert_fix_state_restored()
        print("    PASS  all 5 production-code invariants present in source (AST)")
    except AssertionError as exc:
        print(f"    FAIL  {exc}")
        print(
            "    note: this checks ON-DISK source.  If you edited the .py\n"
            "          but did NOT restart sglang, the live behaviour may\n"
            "          still be correct (probe [A]/[B] would then PASS).\n"
            "          Always restart sglang after touching production code."
        )
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
