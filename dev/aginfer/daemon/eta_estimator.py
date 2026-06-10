"""Online, workload-agnostic per-tool-call ETA estimator (T11 / §1).

S1's demote is value-gated: the §7 value rule only proposes demoting a caller's
idle tail when ``h_(τ,sp)(occ)·bytes·ETA`` (holding relief over the tool gap,
``hold_time = ETA``) exceeds the demote+promote round-trip migration cost. So a
SHORT tool call must NOT trigger a demote and a LONG one should — which requires
a good per-call ETA, *learned online*, not an externally-fed constant.

Granularity matters and pure tool-TYPE is too coarse: ``ls`` is an *argument* of
the ``bash`` tool, fast, while ``sleep 60`` / a build are also ``bash`` but slow.
So the key is a CALL SIGNATURE — ``(tool_name, leading-command-token)`` when the
call carries a command/cmd/script arg (``bash``: ``"ls -la"`` → ``("bash","ls")``),
else ``(tool_name,)``. This is FINE yet generic: it sub-buckets by what the tool
actually does without hardcoding any tool's duration (workload-agnostic — it
learns whatever the workload presents; cf. memory feedback-workload-agnostic-phat).

Learning: an EMA of observed ``TOOL_CALL_START → TOOL_CALL_END`` intervals per key,
updated simultaneously at the specific key, the tool-level fallback, and a global
bucket, so a cold specific-key falls back to coarser-but-populated estimates.
Timestamps come from the events themselves (``enqueue_time``), so the estimator
holds no internal wall-clock — consistent with the tracker's timing-free design
and deterministic under replay.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

_GLOBAL: Tuple[str, ...] = ("__global__",)


def call_signature(tool_name: Optional[str], args: Any) -> Tuple[str, ...]:
    """Fine-grained, workload-agnostic ETA key for a tool call.

    ``(tool_name, leading_token)`` when the call has a command-like arg (so
    ``bash``'s ``ls`` vs ``sleep`` separate), else ``(tool_name,)``.  No tool is
    special-cased and no duration is assumed — only the call's own structure is
    used to pick a bucket; the durations are learned.
    """
    tn = (tool_name or "tool").strip() or "tool"
    sub: Optional[str] = None
    if isinstance(args, dict):
        for k in ("command", "cmd", "script", "code"):
            v = args.get(k)
            if isinstance(v, str) and v.strip():
                # leading executable token, path-stripped: "/bin/ls -la" -> "ls"
                tok = v.strip().split()[0].rsplit("/", 1)[-1]
                # drop a leading env-assignment / sudo wrapper to reach the verb
                if tok in ("sudo", "env", "time", "nohup") and len(v.strip().split()) > 1:
                    tok = v.strip().split()[1].rsplit("/", 1)[-1]
                sub = tok[:64]
                break
    elif isinstance(args, str) and args.strip():
        sub = args.strip().split()[0].rsplit("/", 1)[-1][:64]
    return (tn, sub) if sub else (tn,)


def _keys_for(key: Tuple[str, ...]):
    """Most-specific → coarser fallback chain: (tool,sub) → (tool,) → global."""
    out = [key]
    if len(key) == 2:
        out.append((key[0],))
    out.append(_GLOBAL)
    return out


class ETAEstimator:
    def __init__(self, alpha: float = 0.3, min_obs: int = 3) -> None:
        self.alpha = float(alpha)
        self.min_obs = int(min_obs)
        self._ema: Dict[Tuple[str, ...], float] = {}
        self._n: Dict[Tuple[str, ...], int] = {}
        # pid -> (key, start_ts) for the currently-open tool call
        self._open: Dict[str, Tuple[Tuple[str, ...], float]] = {}

    # ---- observation ----

    def on_tool_call_start(self, pid: str, tool_name: Optional[str],
                           args: Any, ts: float) -> Tuple[str, ...]:
        key = call_signature(tool_name, args)
        self._open[pid] = (key, float(ts))
        return key

    def on_tool_call_end(self, pid: str, ts: float) -> Optional[float]:
        rec = self._open.pop(pid, None)
        if rec is None:
            return None
        key, t0 = rec
        dur = float(ts) - t0
        if dur <= 0.0:
            return None
        for k in _keys_for(key):
            n = self._n.get(k, 0)
            self._ema[k] = dur if n == 0 else (1.0 - self.alpha) * self._ema[k] + self.alpha * dur
            self._n[k] = n + 1
        return dur

    # ---- query ----

    def predict(self, tool_name: Optional[str], args: Any) -> Optional[float]:
        """Learned ETA for this call: the most-specific key with >= min_obs
        observations, else a coarser fallback, else None (cold start — the
        caller should fall back to an event-provided ETA or skip the demote)."""
        for k in _keys_for(call_signature(tool_name, args)):
            if self._n.get(k, 0) >= self.min_obs:
                return self._ema[k]
        return None

    def predict_key(self, key: Tuple[str, ...]) -> Optional[float]:
        for k in _keys_for(key):
            if self._n.get(k, 0) >= self.min_obs:
                return self._ema[k]
        return None

    # ---- introspection (observability / tests) ----

    def stats(self) -> Dict[str, Any]:
        return {
            "keys": len(self._ema),
            "open": len(self._open),
            "table": {"/".join(k): (round(self._ema[k], 3), self._n[k])
                      for k in sorted(self._ema, key=lambda x: -self._n[x])[:20]},
        }
