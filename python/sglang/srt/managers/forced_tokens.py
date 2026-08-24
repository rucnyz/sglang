# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Overlap-compatible teacher-forcing token override.

Splits the pure, model-free half of token forcing out of the scheduler so it is
unit-testable without booting a worker. The GPU scatter that consumes these
positions lives in ``Scheduler._apply_forced_tokens``.
"""

import logging
import os
from typing import List, Tuple

logger = logging.getLogger(__name__)

# rids already reported, so a stalled req logs once instead of once per step.
_reported_skips = set()

# SGLANG_LOG_FORCED_TOKENS=1 logs one line per forced req at its first override.
# A forced list shorter than max_new_tokens ends the override early and the req
# free-runs the remainder, which no other signal reports (running past the end of
# the list is a normal, silent exit from this loop).
_LOG_FORCED = os.environ.get("SGLANG_LOG_FORCED_TOKENS") == "1"


def _report_skip(req, reason: str, pos: int, total: int) -> None:
    """A forced req that stops being overridden mid-sequence diverges silently:
    the engine keeps sampling and the replay is no longer token-identical. Say so."""
    rid = getattr(req, "rid", None)
    key = (rid, reason)
    if key in _reported_skips:
        return
    _reported_skips.add(key)
    logger.warning(
        "teacher forcing skipped mid-sequence: rid=%s reason=%s forced %d/%d dispatched "
        "-- output will diverge from the trace",
        rid,
        reason,
        pos,
        total,
    )


def forced_override_positions(reqs, chunked_req=None, is_extend=True) -> List[Tuple[int, int]]:
    """For each req that commits a token THIS batch (not finished, not retracted,
    not the batch's still-chunking prefill req) and carries
    ``custom_params["forced_output_ids"]`` with ``forced_dispatched < len(forced)``,
    return ``(batch_index, forced_token)`` and advance that req's
    ``forced_dispatched`` counter.

    Pure bookkeeping over req state (no tensor ops). The counter advances at
    DISPATCH, not at commit, because under ``--enable-overlap`` the commit
    (``output_ids.append``) lags ~1 step behind dispatch, so ``len(req.output_ids)``
    would index the wrong forced position.

    ``chunked_req`` is ``ScheduleBatch.chunked_req`` — the one req whose prefill
    continues past this batch, so it commits nothing. Identity is the only
    dispatch-time-correct signal: ``req.inflight_middle_chunks`` is decremented by
    the batch-RESULT processor, which under overlap can still read > 0 while the
    LAST chunk is being dispatched. Filtering on the counter therefore skipped the
    override for that chunk while the commit path (running one step later, after
    the decrement) still appended the token — the model's own first token leaked
    into the output of any request whose uncached prefix exceeded
    ``chunked_prefill_size``.

    ``is_extend`` gates that check, because ``chunked_req`` goes STALE on a decode
    batch: when the running batch is empty the scheduler adopts the prefill batch
    object itself as the running batch (``running_batch = last_batch``), and
    neither ``merge_batch`` nor ``filter_batch`` clears the attribute. A req that
    was mid-chunk in that batch would then match ``req is chunked_req`` for every
    later decode step and free-run its whole output. A still-chunking req is
    filtered out of the batch before it is adopted, so on a decode batch the
    field carries no information and must be ignored.
    """
    out: List[Tuple[int, int]] = []
    for i, req in enumerate(reqs):
        cp = req.sampling_params.custom_params
        forced = cp.get("forced_output_ids") if cp else None
        pos = req.forced_dispatched
        if not forced:
            if pos > 0:
                # The req was being forced and its forced list is now gone: some
                # other path replaced sampling_params mid-flight.
                _report_skip(req, "custom_params-lost", pos, pos)
            continue
        if pos >= len(forced):
            continue  # forced sequence exhausted; the rest is up to max_new_tokens
        if is_extend and req is chunked_req:
            continue  # mid-prefill chunk: commits nothing, so nothing to override
        if req.finished() or req.is_retracted:
            _report_skip(req, "finished" if req.finished() else "retracted",
                         pos, len(forced))
            continue
        if _LOG_FORCED and (pos < 5 or pos % 500 == 0):
            logger.warning("teacher forcing: rid=%s pos=%d/%d bi=%d/%d",
                           getattr(req, "rid", None), pos, len(forced), i, len(reqs))
        out.append((i, int(forced[pos])))
        req.forced_dispatched = pos + 1
    return out
