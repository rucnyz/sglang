"""`fatal(reason, **context)` — shared entry point for deployment-bug
halts (DESIGN §10 "Fatal halts emit forensic state dump", PLAN §4 T43).

Two fault classes (DESIGN §10):

  * **deployment-bug** — schema mismatch, missing required state
    fields, joint_decide DP blow-up, ``peak_bw_bps ≤ 0``, mode-
    switch attempt, hash collision.  "This should never happen in a
    correct deployment."  → ``fatal(...)``: dump forensic state, exit.
    Supervisor decides restart policy.
  * **load** — apply_failed race, sglang briefly slow, transient
    outbound queue depth.  "This is just how the system handles
    bursty workload."  → log + continue.

Call sites (DESIGN §10 + this module's docstring):

  * ``joint_decide`` DP blow-up (``joint_decide_dp_blowup`` — the value
    knapsack's reachable-cell count exceeds ``max_dp_cells``; there is no
    infeasibility path, the value-gated DP can always pick the empty set)
  * ``bytes_at`` τ-not-in-residence assertion (T34)
  * Missing ``link_stats`` / ``tier_holding_cost`` / ``throughput_ema``
    fields in ``/aginfer/state``
  * Cross-rank ``pool_usage[tier].subpools`` key disagreement
  * ``peak_bw_bps ≤ 0``
  * Daemon-attached mode losing daemon mid-run
  * ``HASH_COLLISION`` webhook receipt
  * ``unsupported_tree_cache`` reported by sglang
  * ``per_rank`` empty list

The function MUST NOT itself raise.  Forensic preservation under bug
conditions is the whole point of the helper, so a JSON-serialisation
failure on one context field falls back to ``repr(value)`` for that
field and keeps dumping the rest.
"""
from __future__ import annotations

import dataclasses
import enum
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger("aginfer.daemon.fatal")


def _data_dir() -> Path:
    """Resolve the daemon's data directory.

    Order: ``$AGINFER_DATA_DIR`` env var → ``<sglang-repo>/dev/aginfer/data``.

    The function does not assume the repo layout when ``AGINFER_DATA_DIR``
    is unset: it walks the import path from this file up to find
    ``dev/aginfer/``.  This survives ``pip install -e`` and the
    pytest collector cwd both."""
    env_dir = os.environ.get("AGINFER_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    here = Path(__file__).resolve()
    # ``<sglang>/dev/aginfer/daemon/_fatal.py`` → parents[1] is dev/aginfer.
    return here.parents[1] / "data"


def _to_jsonable(value: Any) -> Any:
    """Best-effort JSON coercion.

    Recursive for dict / list / tuple / set; ``dataclasses.asdict`` for
    dataclass instances; ``.value`` for ``enum.Enum``; ``str(...)`` for
    ``Path`` / ``Exception``; ``repr(...)`` as the final fallback.  Never
    raises — the whole point of fatal() is forensic preservation, so we
    swallow encoding errors and record the repr instead of bombing.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            key = k if isinstance(k, str) else repr(k)
            out[key] = _to_jsonable(v)
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return _to_jsonable(dataclasses.asdict(value))
        except Exception:
            return repr(value)
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    if isinstance(value, Path):
        return str(value)
    # Pydantic v2 BaseModel exposes model_dump; v1 had dict().
    md = getattr(value, "model_dump", None)
    if callable(md):
        try:
            return _to_jsonable(md())
        except Exception:
            return repr(value)
    # Try JSON-encode the thing as-is; if that fails, repr.
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def fatal(reason: str, **context: Any) -> None:
    """Halt the daemon with a forensic state dump (DESIGN §10).

    ``reason`` is a short ``snake_case`` slug used both in the log line
    and the filename.  Conventional values appear in the call-sites
    list above.  Arbitrary ``**context`` keyword arguments are
    serialised under ``context.<key>`` in the forensic JSON.  Common
    keys (by DESIGN §10 contract):

      * ``event``       — the event that triggered the handler
      * ``state``       — the ``/aginfer/state`` snapshot
      * ``candidates``  — candidate sets produced upstream
      * ``dp_inputs``   — DP inputs (``budget``, ``bucket_size``,
                          axes, etc.)

    Effects:
      1. Writes ``<data_dir>/forensic/<reason>_<unix_ts_ns>.json``.
      2. Logs ``logger.critical`` with the file path.
      3. ``os._exit(1)`` — crash-only.

    ``os._exit`` (not ``sys.exit``) is deliberate.  ``sys.exit(1)``
    raises ``SystemExit`` which can be caught by an asyncio
    ``Task.__step`` wrapper, an ``asyncio.gather`` collector, an
    ``asyncio.shield`` wrap, or a user-installed
    ``loop.set_exception_handler``.  In modern CPython the propagation
    usually works (see verify/t164 stage C0), but the crash-only
    contract MUST NOT depend on exception-machinery routing.  ``os._
    exit`` bypasses Python shutdown entirely → process dies immediately.

    Never returns.  Never raises (errors during serialisation degrade
    to ``repr``)."""
    ts = time.time()
    forensic_dir = _data_dir() / "forensic"
    try:
        forensic_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - very degraded env
        logger.critical(
            "fatal(reason=%s): could not create forensic dir %s: %s; "
            "dumping to stderr",
            reason, forensic_dir, exc,
        )
        forensic_dir = None

    # Capture the call-site stack as well as any active exception.  The
    # PLAN explicitly lists assertion / positivity halts that do NOT come
    # from an exception context, so format_stack() is required.
    exc_info = sys.exc_info()
    if exc_info[0] is not None:
        tb_lines = traceback.format_exception(*exc_info)
    else:
        tb_lines = traceback.format_stack()
    # Each list element should be one line for log diffing.
    tb_lines = [line.rstrip("\n") for line in tb_lines]

    payload: Dict[str, Any] = {
        "reason": reason,
        "timestamp_unix": ts,
        "timestamp_iso": time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.gmtime(ts),
        ),
        "pid": os.getpid(),
        "traceback": tb_lines,
        "context": {k: _to_jsonable(v) for k, v in context.items()},
    }

    forensic_path: Optional[Path] = None
    if forensic_dir is not None:
        # Nanosecond-resolution timestamp keeps two fatal()s in the
        # same second from clobbering each other.  PID makes
        # multi-rank concurrent fatals safe too.
        ns = time.time_ns()
        forensic_path = forensic_dir / f"{reason}_{ns}_{os.getpid()}.json"
        try:
            forensic_path.write_text(json.dumps(payload, indent=2))
        except OSError as exc:  # pragma: no cover - very degraded env
            logger.critical(
                "fatal(reason=%s): could not write forensic file %s: %s",
                reason, forensic_path, exc,
            )
            forensic_path = None

    if forensic_path is not None:
        logger.critical(
            "FATAL reason=%s forensic_file=%s pid=%d",
            reason, forensic_path, os.getpid(),
        )
    else:
        # Degraded path: dump payload directly to stderr so the
        # supervisor at least has the traceback.
        logger.critical(
            "FATAL reason=%s (no forensic file written) payload=%s",
            reason, json.dumps(payload, default=repr),
        )
    # Flush logging handlers + stdio so the supervisor's log scrape
    # captures the CRITICAL line before we kill the process.
    for h in list(logging.getLogger().handlers):
        try:
            h.flush()
        except Exception:  # pragma: no cover - degraded env
            pass
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # pragma: no cover
        pass
    os._exit(1)
