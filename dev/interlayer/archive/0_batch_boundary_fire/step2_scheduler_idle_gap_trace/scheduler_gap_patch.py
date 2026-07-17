"""Monkey-patch hook for measuring GPU idle gaps between consecutive
batches in sglang's scheduler. Imported at server boot via
SGLANG_GAP_TRACE_LOG=path.

How it works:
  - Captures one CUDA event right before run_batch launches its first
    kernel (event_started), and one right after process_batch_result
    returns (event_finished).
  - For each pair of consecutive batches:
      gap_us = event_started[i+1].elapsed_time(event_finished[i]) * 1000
    (torch's CUDA event elapsed_time is in milliseconds.)
  - Writes one JSON line per gap to the path in SGLANG_GAP_TRACE_LOG.

The patch is non-invasive: we wrap Scheduler.run_batch and
Scheduler.process_batch_result via monkey-patch at import time. No
source-file edit, no permanent change.

The patch is enabled by SGLANG_GAP_TRACE_LOG=path env var. If unset,
no-op.

Usage (must be imported BEFORE the scheduler starts; e.g., put on
PYTHONPATH and import as a -c stub):

  PYTHONPATH=dev/interlayer/0_batch_boundary_fire/step2_scheduler_idle_gap_trace:$PYTHONPATH \\
  SGLANG_GAP_TRACE_LOG=/tmp/gap.jsonl \\
  python -c 'import scheduler_gap_patch; import sglang.launch_server' ...
"""
from __future__ import annotations

import json
import os
import time

import torch


_LOG_PATH = os.environ.get("SGLANG_GAP_TRACE_LOG")


def _install():
    if _LOG_PATH is None:
        return

    from sglang.srt.managers.scheduler import Scheduler

    log_fp = open(_LOG_PATH, "w", buffering=1)
    print(f"[scheduler_gap_patch] logging GPU gap distribution to {_LOG_PATH}")

    # Persistent state on the Scheduler instance.
    _state = {
        "prev_finished_event": None,
        "prev_finished_wall_ns": None,
        "prev_bs": 0,
    }

    orig_run_batch = Scheduler.run_batch
    orig_process_batch_result = Scheduler.process_batch_result

    def patched_run_batch(self, batch, *args, **kwargs):
        # Record START of this batch on the current stream.
        start_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        start_wall_ns = time.perf_counter_ns()

        # If we have a previous "finished" event, compute the gap.
        prev_finished = _state["prev_finished_event"]
        prev_finished_wall = _state["prev_finished_wall_ns"]
        if prev_finished is not None:
            # Need both events to be done before elapsed_time. The
            # `start_evt.record()` just queued; sync to get accurate
            # wall-time measurement of the gap.
            torch.cuda.synchronize()
            try:
                gap_ms = prev_finished.elapsed_time(start_evt)
            except Exception as e:
                gap_ms = -1.0
            wall_gap_us = (start_wall_ns - prev_finished_wall) // 1000
            log_fp.write(json.dumps({
                "ts": time.time(),
                "gap_us_gpu": int(gap_ms * 1000),
                "gap_us_cpu": int(wall_gap_us),
                "prev_bs": _state["prev_bs"],
                "this_bs": batch.batch_size() if batch is not None else 0,
            }) + "\n")

        result = orig_run_batch(self, batch, *args, **kwargs)
        # Stash for process_batch_result to finalize.
        self._gap_trace_start_evt = start_evt
        self._gap_trace_this_bs = batch.batch_size() if batch is not None else 0
        return result

    def patched_process_batch_result(self, batch, result, *args, **kwargs):
        ret = orig_process_batch_result(self, batch, result, *args, **kwargs)
        # Record FINISH of this batch.
        finish_evt = torch.cuda.Event(enable_timing=True)
        finish_evt.record()
        finish_wall_ns = time.perf_counter_ns()
        _state["prev_finished_event"] = finish_evt
        _state["prev_finished_wall_ns"] = finish_wall_ns
        _state["prev_bs"] = getattr(self, "_gap_trace_this_bs", 0)
        return ret

    Scheduler.run_batch = patched_run_batch
    Scheduler.process_batch_result = patched_process_batch_result


_install()
