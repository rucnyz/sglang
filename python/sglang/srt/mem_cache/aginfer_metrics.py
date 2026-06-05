"""Pure measurement helpers for the aginfer T26 throughput / in-flight
instrumentation (#200).

These are PURE functions (no scheduler/model state) so they unit-test
without a GPU.  The scheduler hot-path hooks gather the inputs (a step
duration, the running ``Req`` list) and call these; the cold-path dump
reads the results the scheduler pushes onto the radix cache.

Three quantities feed the daemon's DESIGN §8 admission math:

  * ``decode_per_program[pid]`` — per-program decode throughput
    (tokens/sec), an EMA over decode steps.  Under spec decode the real
    per-step count is ``accept_lens[i]`` (≥ 1), not 1 — measured
    post-forward (#206).  Daemon §8 ``forecast_inflight_demand`` /
    ``future_inflight_savings`` (gated additionally on T11's
    ``E[remaining_tokens]``).
  * ``prefill_bps`` — prefill throughput (bytes/sec), an EMA over EXTEND
    and the prefill portion of MIXED batches (#206).  Daemon §8
    ``marginal_pause_cost`` (the in-flight bytes re-prefilled on resume).
  * ``per_program_usage[pid].hbm.inflight[sp]`` — per-program in-flight
    (undivided current-KV) HBM bytes.  Feeds Daemon §8
    ``marginal_pause_cost`` only; ``pause_relief`` uses the shared-aware
    ``committed`` instead (#205).

Single-stack subpool scope: in-flight bytes attribute to the base
attention subpool (``subpool_name``).  Hybrid SWA/Mamba per-component
in-flight splitting is a refinement for when such a model is deployed
(the verifiable deployment is single-stack — one "full" subpool); same
pragmatic boundary as the dump's single-subpool DRAM today.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

# EMA smoothing for the throughput estimates.  Higher = more responsive
# to the latest step, lower = smoother.  0.3 reacts within a handful of
# steps while damping single-step spikes (a partial decode batch).
AGINFER_THROUGHPUT_EMA_ALPHA = 0.3


def ema_update(prev: Optional[float], sample: float, alpha: float) -> float:
    """Exponential moving average.  First sample (``prev`` is None) seeds
    the average; thereafter ``alpha·sample + (1−alpha)·prev``.

    ``sample`` is clamped to ≥ 0 (a rate can't be negative); a
    non-finite sample is ignored (returns ``prev`` or 0.0) so a bad
    timing read never poisons the EMA."""
    import math
    if not math.isfinite(sample) or sample < 0.0:
        return float(prev) if prev is not None else 0.0
    if prev is None or not math.isfinite(prev):
        return float(sample)
    a = alpha
    return a * float(sample) + (1.0 - a) * float(prev)


def inflight_bytes_by_program(
    reqs: Iterable[Any],
    bytes_per_token: int,
    subpool_name: str,
) -> Dict[str, Dict[str, int]]:
    """Per-program in-flight HBM bytes, keyed ``{program_id: {subpool:
    bytes}}``.

    DESIGN §8 ``inflight[sp]`` = the bytes a running request's active
    decode is using right now, lost + re-prefilled if the program is
    paused (``marginal_pause_cost``) and the signal that p is decoding in
    sp (the forecast gate).  In sglang's unified cache a decoding req
    commits each token immediately (``kv_committed_len`` tracks
    ``kv_allocated_len``), so the in-flight footprint is the running
    request's CURRENT total KV — ``kv_allocated_len × bytes_per_token``
    — not ``allocated − committed`` (which is 0 by construction).

    Requests with ``program_id is None`` (untagged) are skipped (matches
    the dump's ``session_ids`` handling).  ``bytes_per_token ≤ 0`` (cache
    couldn't report it) yields an empty result rather than bogus zeros.

    NOTE: this current-KV measure overlaps the dump's tree-walk
    ``committed`` term on a running req's cached prefix, but the two are on
    DIFFERENT bases — ``inflight`` is the UNDIVIDED full KV, ``committed``
    is the shared-aware ``bytes//n_holders`` per holder.  Since #205 the
    daemon's ``pause_relief`` uses ``committed`` ALONE; this undivided
    ``inflight`` feeds only ``marginal_pause_cost`` (the resume re-prefill
    cost), which wants exactly this full-KV figure.  The uncached slice of
    inflight (not yet in the tree) is a separate relief refinement (#207)."""
    out: Dict[str, Dict[str, int]] = {}
    bpt = int(bytes_per_token)
    if bpt <= 0:
        return out
    for req in reqs:
        pid = getattr(req, "program_id", None)
        if pid is None:
            continue
        cur_tokens = int(getattr(req, "kv_allocated_len", 0) or 0)
        if cur_tokens <= 0:
            continue
        b = cur_tokens * bpt
        sp_bucket = out.setdefault(str(pid), {})
        sp_bucket[subpool_name] = sp_bucket.get(subpool_name, 0) + b
    return out


def decode_tokens_by_program(
    reqs: Iterable[Any],
    counts: Optional[Dict[str, int]] = None,
    per_req_tokens: Optional[Iterable[int]] = None,
) -> Dict[str, int]:
    """Count generated tokens THIS decode step per program.

    Default: 1 token per running req (one normal decode step).
    ``per_req_tokens`` (index-aligned with ``reqs``) overrides that with
    the real generated count for each req — used for speculative decode,
    where a verify step commits ``accept_lens[i]`` (≥ 1) tokens per req,
    not 1 (#206).  A per-req count ≤ 0 contributes nothing.  ``counts``
    accumulates into a running dict across calls when provided.

    Untagged (``program_id is None``) requests are skipped — but they
    still consume an index, so ``per_req_tokens`` stays aligned with the
    full ``reqs`` list.  Used by the decode hot-path / spec post-forward
    hook to derive the instantaneous tokens/sec before the EMA update."""
    out: Dict[str, int] = dict(counts) if counts else {}
    prt = list(per_req_tokens) if per_req_tokens is not None else None
    for i, req in enumerate(reqs):
        pid = getattr(req, "program_id", None)
        if pid is None:
            continue
        n = int(prt[i]) if (prt is not None and i < len(prt)) else 1
        if n <= 0:
            continue
        out[str(pid)] = out.get(str(pid), 0) + n
    return out


def running_program_view(
    ema: Dict[str, float],
    running_pids: Iterable[str],
) -> Dict[str, float]:
    """Project the per-program decode EMA onto the CURRENTLY-running
    programs only.  A program with no running request isn't decoding, so
    its (stale) EMA must not appear in ``decode_per_program`` — the
    daemon's §8 forecast keys on ``inflight[sp] > 0`` anyway, but emitting
    only live programs keeps the dict bounded and honest."""
    live = set(running_pids)
    return {pid: float(v) for pid, v in ema.items() if pid in live}
