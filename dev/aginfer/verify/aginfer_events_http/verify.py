"""Offline verify for ``PUT /aginfer/events`` (remote-proxy control plane).

Stages:
  A. validate_events_body — well-formed / malformed
  B. (optional, AGINFER_VERIFY_BASE) live HTTP round-trip against a running
     sglang.launch_server with SGLANG_AGINFER_IN_ENGINE=1
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, List, Tuple


def _import_validator():
    # Prefer the workspace checkout over any installed sglang.
    # verify.py → aginfer_events_http → verify → aginfer → dev → sglang
    root = Path(__file__).resolve().parents[4]
    py = root / "python"
    if str(py) not in sys.path:
        sys.path.insert(0, str(py))
    from sglang.srt.mem_cache.aginfer.http_validators import (  # noqa: WPS433
        validate_events_body,
    )

    return validate_events_body


def stage_a0_well_formed(validate_events_body) -> None:
    body = {
        "events": [
            {"kind": "tool_call_start", "session": "p1", "payload": {"x": 1}},
            {"kind": "llm_prefill", "session": "p1"},
            {"kind": "sub_return", "session": "child", "payload": None},
        ]
    }
    out = validate_events_body(body)
    assert len(out) == 3
    assert out[0] == {
        "kind": "tool_call_start",
        "session": "p1",
        "payload": {"x": 1},
    }
    assert out[1]["payload"] == {}
    assert out[2]["payload"] == {}


def stage_a1_malformed(validate_events_body) -> None:
    cases = [
        None,
        [],
        {},
        {"events": "nope"},
        {"events": [{"kind": "x"}]},  # missing session
        {"events": [{"session": "p"}]},  # missing kind
        {"events": [{"kind": "", "session": "p"}]},
        {"events": [{"kind": "x", "session": ""}]},
        {"events": [{"kind": "x", "session": "p", "payload": []}]},
        {"events": [{"kind": 1, "session": "p"}]},
    ]
    for bad in cases:
        try:
            validate_events_body(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def stage_a2_caps(validate_events_body) -> None:
    too_many = {
        "events": [
            {"kind": "llm_prefill", "session": "p"} for _ in range(257)
        ]
    }
    try:
        validate_events_body(too_many)
        raise AssertionError("expected ValueError for oversized batch")
    except ValueError:
        pass
    long_session = {
        "events": [{"kind": "x", "session": "s" * 65}]
    }
    try:
        validate_events_body(long_session)
        raise AssertionError("expected ValueError for long session")
    except ValueError:
        pass


def stage_b0_live_http() -> None:
    base = os.environ.get("AGINFER_VERIFY_BASE", "").rstrip("/")
    if not base:
        print("SKIP stage_b0 (set AGINFER_VERIFY_BASE to enable live HTTP)")
        return
    url = f"{base}/aginfer/events"
    payload = {
        "events": [
            {"kind": "llm_prefill", "session": "verify-events", "payload": {}}
        ]
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read().decode("utf-8"))
        # 400 from validator would be a hard fail; driver-not-armed returns 200
        # with ok=False.
        if exc.code == 400:
            raise AssertionError(f"validator rejected live body: {body}") from exc
        raise
    assert "ok" in body and "applied" in body and "ranks" in body, body
    print(f"  live PUT /aginfer/events -> {body}")


_STAGES: List[Tuple[str, Callable[..., Any]]] = [
    ("A0 well-formed body", stage_a0_well_formed),
    ("A1 malformed bodies", stage_a1_malformed),
    ("A2 size / length caps", stage_a2_caps),
    ("B0 live HTTP (optional)", stage_b0_live_http),
]


def main() -> int:
    validate = _import_validator()
    failed = 0
    for name, fn in _STAGES:
        print(f"==> {name}")
        try:
            if fn is stage_b0_live_http:
                fn()
            else:
                fn(validate)
            print("  PASS")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL: {exc}")
    print(f"done: {len(_STAGES) - failed}/{len(_STAGES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
