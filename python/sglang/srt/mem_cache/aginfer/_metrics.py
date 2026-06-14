"""Lightweight structured-event metric logger for the aginfer daemon.

All emitters use a stable prefix `aginfer_metric` so a parser can grep
once and key/value-split the rest.  Format example:

    aginfer_metric event=kv_decide kind=tool_call_start dset_size=3 \\
        outcome=dispatched action_n=1 target=DRAM

Key/value pairs are space-separated; values MUST NOT contain spaces.
For free-form text (rare) use ``_quoted`` suffix in the key and quote
the value.

Performance contract:
* No I/O sync; uses the existing root logger.
* < 1 µs / call typical; safe to use in handlers that fire ~10k× per
  cycle.
* Do NOT use inside the inline-eviction hot path (sglang_adapter
  scorer) — that runs orders of magnitude more often and would taint
  per-trial timings.
"""
from __future__ import annotations

import logging
from typing import Any

_logger = logging.getLogger("aginfer.metric")


def m(event: str, **kv: Any) -> None:
    """Emit one structured metric line.

    Example: ``m("kv_decide", kind="tool_call_start", outcome="dispatched")``
    → ``aginfer_metric event=kv_decide kind=tool_call_start outcome=dispatched``

    None values are dropped.  Floats are formatted with %g (compact).
    """
    parts = [f"event={event}"]
    for k, v in kv.items():
        if v is None:
            continue
        if isinstance(v, float):
            parts.append(f"{k}={v:g}")
        else:
            parts.append(f"{k}={v}")
    _logger.info("aginfer_metric " + " ".join(parts))
