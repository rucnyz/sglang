"""aginfer scheduler throughput/EMA hooks (refactor #251 Stage A.2): the T26
per-program decode tokens/sec + prefill bytes/sec EMA sampling and the
runtime-metrics push, as free functions over a Scheduler. Upstream keeps thin
delegators. No-op (do-no-harm) when the cache is not aginfer-capable."""
from __future__ import annotations
import logging
import time
from typing import TYPE_CHECKING, Dict
from sglang.srt.mem_cache.aginfer_metrics import (
    AGINFER_THROUGHPUT_EMA_ALPHA,
    decode_tokens_by_program,
    ema_update,
    inflight_bytes_by_program,
    running_program_view,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:  # annotation-only (lazy via from __future__ import annotations)
    from sglang.srt.managers.schedule_batch import ScheduleBatch

logger = logging.getLogger("sglang.srt.managers.scheduler")

def _aginfer_record_throughput(sched, batch: ScheduleBatch) -> None:
    """T26 (#200): update the per-program decode tokens/sec EMA and the
    prefill bytes/sec EMA from one forward.  Called per ``run_batch``;
    a no-op when the cache isn't aginfer-capable (no
    ``_aginfer_bytes_per_token``).

    Timing uses ``perf_counter`` deltas between consecutive forwards of
    the SAME mode — so interleaved prefills slightly inflate the decode
    gap and the decode rate reads marginally LOW, the safe direction
    (a lower decode_throughput → smaller forecast → less over-pause).

    Mode classification (#200 audit + #206):
    * pure ``DECODE`` → decode EMA (1 token/req) — UNLESS the batch is
      spec-v2 (overlap-on spec), where the real accepted-token count is
      only known post-forward, so decode is recorded by
      ``_aginfer_record_spec_decode`` (from ``process_batch_result``)
      and skipped here to avoid the 1/req undercount.  spec-v1 (e.g.
      ngram, overlap off) does NOT expose accept_lens, so it keeps the
      conservative 1/req here (#206).
    * pure ``EXTEND`` → prefill bytes/sec EMA (``extend_num_tokens``).
    * ``MIXED`` (chunked prefill + running decode) → split via
      ``batch.decoding_reqs``: the prefill reqs' ``extend_input_len``
      feeds prefill_bps, the decode reqs feed the decode EMA (#206).
      ``extend_num_tokens`` is NOT usable here — it covers the whole
      batch (1 per decode req included), which would pollute prefill.
    * ``TARGET_VERIFY`` / ``DRAFT_EXTEND`` are set INSIDE the worker
      forward, never present on this pre-forward batch; if one ever
      reaches here it updates neither (drafts aren't committed output).

    Wrapped in a blanket ``except`` (#200 audit): this is brand-new
    per-forward instrumentation; an unhandled error here would kill the
    scheduler event loop.  Losing a throughput sample is harmless —
    the daemon degrades to its no-signal branch.
    """
    try:
        sched._aginfer_record_throughput_inner(batch)
    except Exception:  # noqa: BLE001
        if not getattr(sched, "_aginfer_throughput_warned", False):
            sched._aginfer_throughput_warned = True
            logger.warning(
                "aginfer throughput measurement raised (suppressed; "
                "metric degrades to no-signal)", exc_info=True)

def _aginfer_record_throughput_inner(sched, batch: ScheduleBatch) -> None:
    cache = getattr(sched, "tree_cache", None)
    if cache is None or not hasattr(cache, "_aginfer_bytes_per_token"):
        return
    fm = getattr(batch, "forward_mode", None)
    if fm is None:
        return
    now = time.perf_counter()
    # A pure-DECODE spec-v2 batch (overlap-on EAGLE) resolves the real
    # per-req accepted-token count (accept_lens) post-forward — recorded
    # by _aginfer_record_spec_decode — so skip the pre-forward decode
    # count for it (else we'd undercount at 1/req).  spec-v1 (e.g. ngram,
    # overlap OFF) does NOT expose that count, and the post-forward hook
    # only fires for is_decode() batches (not MIXED), so both of those
    # keep the conservative 1/req here — never empty, never a regression.
    defer_decode = bool(getattr(batch, "is_spec_v2", False))
    if fm == ForwardMode.DECODE:
        if defer_decode:
            return
        sched._aginfer_update_decode(
            decode_tokens_by_program(batch.reqs), now)
    elif fm == ForwardMode.EXTEND:
        sched._aginfer_update_prefill(
            sched._aginfer_extend_token_count(batch), cache, now)
    elif fm.is_mixed():
        # MIXED = chunked prefill + running decode in one batch.  Split
        # by identity against batch.decoding_reqs (sglang's own metrics
        # discriminator): prefill reqs → prefill_bps, decode reqs →
        # decode EMA (#206).  extend_num_tokens spans the whole batch
        # (1 per decode req), so it can't stand in for the prefill count.
        decode_ids = {id(r) for r in (getattr(batch, "decoding_reqs", None) or [])}
        reqs = list(getattr(batch, "reqs", None) or [])
        prefill_reqs = [r for r in reqs if id(r) not in decode_ids]
        ntok = sum(int(getattr(r, "extend_input_len", 0) or 0)
                   for r in prefill_reqs)
        sched._aginfer_update_prefill(ntok, cache, now)
        # Decode portion is always counted 1/req HERE — even under
        # spec-v2.  The post-forward accept_lens hook only fires for
        # pure is_decode() batches (process_batch_result routes MIXED to
        # the is_extend() branch), so deferring a MIXED batch's decode
        # would drop it entirely.  1/req under-counts the acceptance
        # length for spec-v2 MIXED (rare combo) but is never zero and is
        # never double-counted (the post-forward hook can't reach MIXED).
        decode_reqs = [r for r in reqs if id(r) in decode_ids]
        sched._aginfer_update_decode(
            decode_tokens_by_program(decode_reqs), now)

def _aginfer_extend_token_count(batch: ScheduleBatch) -> int:
    """Prefill-token count for a pure EXTEND batch.  ``extend_num_tokens``
    is reset by result-processing time (overlap scheduling), so fall
    back to the per-req extend lengths, then the input-id tensor."""
    ntok = int(getattr(batch, "extend_num_tokens", 0) or 0)
    if ntok <= 0:
        el = getattr(batch, "extend_lens", None)
        if el:
            ntok = int(sum(el))
    if ntok <= 0:
        ii = getattr(batch, "input_ids", None)
        if ii is not None:
            try:
                ntok = int(len(ii))
            except TypeError:
                ntok = 0
    return ntok

def _aginfer_update_prefill(sched, ntok: int, cache, now: float) -> None:
    """Blend ``ntok × bytes_per_token / dt`` into the prefill_bps EMA,
    timed off the previous prefill-bearing forward."""
    last = sched._aginfer_last_prefill_t
    sched._aginfer_last_prefill_t = now
    if last is None or ntok <= 0:
        return
    dt = now - last
    if dt <= 0.0:
        return
    bpt = int(cache._aginfer_bytes_per_token())
    if bpt <= 0:
        return
    sched._aginfer_prefill_bps_ema = ema_update(
        sched._aginfer_prefill_bps_ema, (ntok * bpt) / dt,
        AGINFER_THROUGHPUT_EMA_ALPHA)

def _aginfer_update_decode(sched, counts: Dict[str, int], now: float) -> None:
    """Blend per-program ``tokens / dt`` into the decode EMA, timed off
    the previous decode-bearing forward."""
    last = sched._aginfer_last_decode_t
    sched._aginfer_last_decode_t = now
    if last is None:
        return
    dt = now - last
    if dt <= 0.0:
        return
    for pid, n in counts.items():
        sched._aginfer_decode_ema[pid] = ema_update(
            sched._aginfer_decode_ema.get(pid), n / dt,
            AGINFER_THROUGHPUT_EMA_ALPHA)

def _aginfer_record_spec_throughput(sched, batch: ScheduleBatch, result) -> None:
    """Raise-safe wrapper for the spec-decode post-forward hook (#206):
    a crash in brand-new instrumentation must never break the result
    path.  Losing a sample degrades the metric to the no-signal branch."""
    try:
        sched._aginfer_record_spec_decode(batch, result)
    except Exception:  # noqa: BLE001
        if not getattr(sched, "_aginfer_throughput_warned", False):
            sched._aginfer_throughput_warned = True
            logger.warning(
                "aginfer spec-decode measurement raised (suppressed; "
                "metric degrades to no-signal)", exc_info=True)

def _aginfer_record_spec_decode(sched, batch: ScheduleBatch, result) -> None:
    """T26 (#206): record spec-v2 decode throughput from the POST-forward
    result.  A verify step commits ``accept_lens[i]`` tokens per req (the
    bonus token + accepted drafts), not 1 — so the pre-forward DECODE
    branch (which skips for spec-v2) would undercount by the acceptance
    length.  ``num_correct_drafts_per_req_cpu`` is ``accept_lens − 1`` per
    req, resolved to a CPU list during ``process_batch_result_decode`` and
    index-aligned with ``batch.reqs``; accepted = that + 1.  Wrapped
    raise-safe by the caller's guard.

    No-op when the result carries no spec accept counts (spec-v1 keeps
    its pre-forward 1/req count; a config that doesn't expose them)."""
    cache = getattr(sched, "tree_cache", None)
    if cache is None or not hasattr(cache, "_aginfer_bytes_per_token"):
        return
    ncd = getattr(result, "num_correct_drafts_per_req_cpu", None)
    if not ncd:
        return
    per_req = [int(n) + 1 for n in ncd]
    counts = decode_tokens_by_program(
        getattr(batch, "reqs", None) or [], per_req_tokens=per_req)
    if counts:
        sched._aginfer_update_decode(counts, time.perf_counter())

def _aginfer_push_runtime_metrics(sched) -> None:
    """T26 (#200): assemble the measured decode/prefill EMAs +
    per-program in-flight bytes and push them onto the radix cache so
    the next ``dump_aginfer_state`` reflects real numbers.  Cold path
    (per dump cadence); a no-op when the cache can't store them."""
    cache = getattr(sched, "tree_cache", None)
    setter = getattr(cache, "set_aginfer_runtime_metrics", None)
    if setter is None:
        return
    from sglang.srt.mem_cache.unified_cache_components import (
        BASE_COMPONENT_TYPE,
    )

    try:
        rb = getattr(sched, "running_batch", None)
        reqs = list(rb.reqs) if rb is not None else []
        running_pids = {
            str(r.program_id) for r in reqs
            if getattr(r, "program_id", None) is not None
        }
        # Prune decode EMAs for programs that are no longer running so
        # the dict stays bounded by the live set (not every program
        # ever seen) — the ONLY prune point, so a never-dumped daemon
        # can't make the dict grow unbounded on the hot path.
        sched._aginfer_decode_ema = {
            p: v for p, v in sched._aginfer_decode_ema.items()
            if p in running_pids
        }
        decode_pp = running_program_view(
            sched._aginfer_decode_ema, running_pids)
        bpt = int(cache._aginfer_bytes_per_token())
        subpool = cache._aginfer_subpool_name(BASE_COMPONENT_TYPE)
        inflight = inflight_bytes_by_program(reqs, bpt, subpool)
    except Exception:  # noqa: BLE001 — never break the state dump
        logger.warning(
            "aginfer runtime-metric push raised; emitting empty",
            exc_info=True)
        decode_pp, inflight = {}, {}
    setter(
        decode_per_program=decode_pp,
        prefill_bps=float(sched._aginfer_prefill_bps_ema or 0.0),
        inflight=inflight,
    )

