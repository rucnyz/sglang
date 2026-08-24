"""aginfer HTTP request validators — self-contained (#251).

Pure functions (no FastAPI deps) that the ``/aginfer/*`` endpoints in
http_server.py call as thin hooks. They are the safety boundary for
daemon-pushed inputs; verify/t21 and verify/t40 unit-test them directly.
"""
from __future__ import annotations

import math


def validate_session_end_body(body):
    """Validate ``POST /aginfer/session_end``.

    The canonical wire field is ``program_id``.  ``session_id`` is accepted as
    an alias when ``program_id`` is absent, and may also be supplied separately
    to close a corresponding SGLang continual-prompting session safely.
    """
    if not isinstance(body, dict):
        raise ValueError("body must be a JSON object")
    program_id = body.get("program_id")
    session_id = body.get("session_id")
    if program_id is None:
        program_id = session_id
    if not isinstance(program_id, str) or not program_id.strip():
        raise ValueError("program_id must be a non-empty string")
    program_id = program_id.strip()
    if len(program_id) > 64:
        raise ValueError("program_id must be at most 64 characters")
    if session_id is not None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string or null")
        session_id = session_id.strip()
    else:
        session_id = program_id
    return program_id, session_id


def validate_program_paused_body(body):
    """T21 (#181) + #186 audit: validate a PUT /aginfer/program_paused body.
    Returns ``(pid, state, pre_pause_state)`` or raises ``ValueError`` with a
    400-able message.

    Type-validate BEFORE coercion — a prior ``str(body["pid"])`` silently turned
    JSON null/number into "None"/"123", bypassing the setter's empty-pid guard.
    """
    if not isinstance(body, dict):
        raise ValueError("body must be a JSON object")
    for k in ("pid", "state"):
        if k not in body:
            raise ValueError(f"missing required field: {k!r}")
    pid = body["pid"]
    if not isinstance(pid, str) or not pid:
        raise ValueError("pid must be a non-empty string")
    state = body["state"]
    if not isinstance(state, str) or not state:
        raise ValueError("state must be a non-empty string")
    pre = body.get("pre_pause_state")
    if pre is not None and not isinstance(pre, str):
        raise ValueError("pre_pause_state must be string or null")
    return pid, state, pre


def validate_hints_body(body):
    """T40 (#184): validate a PUT /aginfer/hints body. Returns the normalized
    list of hint dicts (``[{hash, p_hat, lambda, stamp, n_holders}]``) or raises
    ``ValueError`` with a 400-able message.

    Rejects out-of-range V_u inputs here so a malformed daemon push fails at the
    door rather than silently poisoning the inline scorer's eviction order:
    ``p_hat`` ∈ [0, 1], ``lambda`` ≥ 0, ``stamp`` a non-negative int. Non-finite
    (NaN/inf) is rejected (audit A4). The holder count (S2 / DESIGN §2 fact 1)
    MUST survive validation — dropping it neutralised the whole holder-count
    lever — optional for back-compat (absent ⇒ 0), a non-negative int when present.
    """
    if not isinstance(body, dict):
        raise ValueError("body must be a JSON object")
    hints = body.get("hints")
    if not isinstance(hints, list):
        raise ValueError("'hints' must be a list")
    out = []
    for i, h in enumerate(hints):
        if not isinstance(h, dict):
            raise ValueError(f"hints[{i}] must be an object")
        uhash = h.get("hash")
        if not isinstance(uhash, str) or not uhash:
            raise ValueError(f"hints[{i}].hash must be a non-empty string")
        # bool is an int subclass — reject it explicitly for numerics.
        p_hat = h.get("p_hat")
        if isinstance(p_hat, bool) or not isinstance(p_hat, (int, float)):
            raise ValueError(f"hints[{i}].p_hat must be a number")
        if not math.isfinite(p_hat):
            raise ValueError(f"hints[{i}].p_hat must be finite; got {p_hat}")
        if not (0.0 <= float(p_hat) <= 1.0):
            raise ValueError(f"hints[{i}].p_hat must be in [0, 1]; got {p_hat}")
        lam = h.get("lambda")
        if isinstance(lam, bool) or not isinstance(lam, (int, float)):
            raise ValueError(f"hints[{i}].lambda must be a number")
        if not math.isfinite(lam):
            raise ValueError(f"hints[{i}].lambda must be finite; got {lam}")
        if float(lam) < 0.0:
            raise ValueError(f"hints[{i}].lambda must be >= 0; got {lam}")
        stamp = h.get("stamp")
        if isinstance(stamp, bool) or not isinstance(stamp, int):
            raise ValueError(f"hints[{i}].stamp must be an int")
        if stamp < 0:
            raise ValueError(f"hints[{i}].stamp must be >= 0; got {stamp}")
        n_holders = h.get("n_holders", 0)
        if isinstance(n_holders, bool) or not isinstance(n_holders, int):
            raise ValueError(f"hints[{i}].n_holders must be an int; got {n_holders!r}")
        if n_holders < 0:
            raise ValueError(f"hints[{i}].n_holders must be >= 0; got {n_holders}")
        out.append({
            "hash": uhash,
            "p_hat": float(p_hat),
            "lambda": float(lam),
            "stamp": int(stamp),
            "n_holders": int(n_holders),
        })
    return out
