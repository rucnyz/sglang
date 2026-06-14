"""aginfer program-id sanitization — self-contained (#251: thin hook in core).

``Req.__init__`` carries only a thin ``sanitize_program_id(...)`` call; the
adversarial-input-hardened coercion (verify/t3) lives here.
"""
from __future__ import annotations

from typing import Any, Optional

_PROGRAM_ID_MAX_RECURSION = 8


def sanitize_program_id(pid: Any, _depth: int = 0) -> Optional[str]:
    """Best-effort coercion of an aginfer program_id to a stable short string.

    Worst-case rows in verify/t3/README.md: bogus shapes (dict, int, very long
    str, list, whitespace-only, deeply-nested-list) must NOT raise. Pipeline:

      * ``None`` / empty after strip → ``None`` (untagged path).
      * list / tuple → recurse into the first non-empty element; one Req has one
        program_id, never a multi-program tag. A leading ``None`` / empty value
        doesn't silently kill a valid later element (audit round-2 NIT 7).
      * Recursion depth is capped at ``_PROGRAM_ID_MAX_RECURSION`` (8); beyond
        that, return ``None``. Audit round-3 BLOCKER 2: an adversarial client
        could send a 1k-deep nested list, blow the scheduler subprocess's
        recursion limit, and kill the event loop. The pickle layer happily
        survives such structures (its own iterative handling), so the IPC
        delivers the bomb intact; the cap is the defensive line.
      * everything else → ``str(...)``, ``.strip()``, truncate to 64 chars.

    Truncation can silently collide two distinct ids that share the first 64
    chars; documented in t3/README.md. v1 callers either keep ids ≤ 64 chars or
    accept the collision.
    """
    if _depth > _PROGRAM_ID_MAX_RECURSION:
        return None
    if pid is None:
        return None
    if isinstance(pid, (list, tuple)):
        for elem in pid:
            sanitized = sanitize_program_id(elem, _depth=_depth + 1)
            if sanitized is not None:
                return sanitized
        return None
    if not isinstance(pid, str):
        try:
            pid = str(pid)
        except Exception:
            return None
    pid = pid.strip()
    if not pid:
        return None
    return pid[:64]
