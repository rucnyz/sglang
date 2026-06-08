"""Request-trace capture for the deterministic replay benchmark (#231).

Why this exists
---------------
The free-running agentic e2e (harbor / terminus-2) is **not reproducible**
even with ``temperature=0`` + fixed seeds: the same task set generates a
1.5x-varying number of requests/tokens across runs, because the agent's
trajectory diverges from (a) docker tool-execution nondeterminism and
(b) concurrent-batch numerics flipping the argmax token at temp=0.  So
agentic wall-time cannot isolate the daemon's serving-latency effect.

The fix is a **fixed-trace replay**: capture one real request stream
here, then replay it byte-identically against ours/baseline with the
output length FORCED (``max_tokens=output_len`` + ``ignore_eos``).  With
the generated work pinned, any latency delta is attributable to the
daemon alone.  This module is the capture half; ``scenarios/replay`` is
the replay half.

Design
------
* **Opt-in, zero-cost when off.**  ``create_app`` attaches a recorder to
  ``app.state.trace_recorder`` only when ``AGINFER_TRACE_CAPTURE`` names
  an output path.  When unset, the proxy never touches this module.
* **One JSONL line per completed request**, written at completion (when
  the generated length is known):

      {"t": <arrival offset s, float>,   # honoured by the replay scheduler
       "program_id": <str|null>,          # drives the daemon's per-program logic
       "body": {"messages": [...], "model": ...},  # replayed verbatim (prefix reuse)
       "output_len": <int>}               # forced on replay (ignore_eos)

* **Arrival offset** is taken from a monotonic clock at request entry and
  rebased to the first request, so the replay scheduler can reproduce the
  inter-arrival timing (and thus the concurrency/pressure profile).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def count_sse_content_tokens(chunk: bytes, _carry: Dict[str, bytes]) -> int:
    """Count generated tokens in a streamed SSE byte chunk.

    sglang streams OpenAI-style ``data: {...}\\n\\n`` events, one token per
    event in the default per-token streaming mode, each carrying
    ``choices[0].delta.content``.  We count events with a non-empty
    content delta.  ``_carry`` holds a partial trailing line across chunk
    boundaries (httpx ``aiter_bytes`` does not split on event lines).

    This is the replay's ``output_len`` source.  It is exact under
    per-token streaming; if a backend ever coalesces tokens per event it
    becomes a lower bound, which we accept (and document) because the
    replay forces the length regardless — the captured value only needs
    to reproduce the real generation magnitude.
    """
    buf = _carry.get("buf", b"") + chunk
    n = 0
    lines = buf.split(b"\n")
    # Last element is the (possibly partial) trailing line — carry it over.
    _carry["buf"] = lines[-1]
    for line in lines[:-1]:
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[len(b"data:"):].strip()
        if payload == b"[DONE]" or not payload:
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        for ch in obj.get("choices") or ():
            delta = ch.get("delta") or {}
            if delta.get("content"):
                n += 1
    return n


def usage_completion_tokens(body_bytes: bytes) -> Optional[int]:
    """Exact generated length from a non-streamed response's usage block."""
    try:
        obj = json.loads(body_bytes)
    except Exception:
        return None
    usage = obj.get("usage") or {}
    ct = usage.get("completion_tokens")
    return int(ct) if isinstance(ct, (int, float)) else None


_SAMPLING_KEYS = (
    "model",
    "temperature",
    "top_p",
    "top_k",
    "frequency_penalty",
    "presence_penalty",
    "stop",
)


class TraceRecorder:
    """Thread-safe JSONL appender for captured requests.

    The proxy is single-event-loop async, but completions can interleave;
    a plain lock around the append keeps lines intact and lets the t0
    rebase be set exactly once by the first arrival.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._t0: Optional[float] = None
        self._fh = open(path, "a", buffering=1)  # line-buffered
        self.n_written = 0
        logger.info("trace_capture: recording request trace to %s", path)

    def note_arrival(self) -> float:
        """Stamp a request's arrival; returns its offset from the first."""
        now = time.monotonic()
        with self._lock:
            if self._t0 is None:
                self._t0 = now
            return now - self._t0

    def write(
        self,
        *,
        arrival_offset: float,
        program_id: Optional[str],
        body: Any,
        output_len: int,
    ) -> None:
        """Append one captured request.  Never raises into the hot path."""
        try:
            if not isinstance(body, dict):
                return
            slim: Dict[str, Any] = {"messages": body.get("messages")}
            for k in _SAMPLING_KEYS:
                if k in body:
                    slim[k] = body[k]
            rec = {
                "t": round(arrival_offset, 6),
                "program_id": program_id,
                "body": slim,
                "output_len": int(output_len),
            }
            line = json.dumps(rec, ensure_ascii=False)
            with self._lock:
                self._fh.write(line + "\n")
                self.n_written += 1
        except Exception:  # noqa: BLE001 — capture must never break serving
            logger.warning("trace_capture: write failed", exc_info=True)

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except Exception:  # noqa: BLE001
                pass


def recorder_from_env() -> Optional[TraceRecorder]:
    """Build a recorder iff ``AGINFER_TRACE_CAPTURE`` is set, else None."""
    import os

    path = os.environ.get("AGINFER_TRACE_CAPTURE", "").strip()
    if not path:
        return None
    try:
        return TraceRecorder(path)
    except Exception:  # noqa: BLE001
        logger.warning(
            "trace_capture: could not open %s; capture disabled", path,
            exc_info=True,
        )
        return None


__all__ = [
    "TraceRecorder",
    "recorder_from_env",
    "count_sse_content_tokens",
    "usage_completion_tokens",
]
