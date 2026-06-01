# T23 + T37 — APPLY_FAILED webhook (sglang) + handler (daemon)

PLAN §3 T23 + PLAN §4 T37, paired.  Closes DESIGN §6 L506
fire-and-forget delivery loop: when sglang can't apply an action
the daemon dispatched, it fires an `APPLY_FAILED` webhook back so
the daemon's next `joint_decide` can re-evaluate.

## WHAT WE PROMISED

**Wire payload** (per call, one per failed action — DESIGN §4):

```json
{
  "kind":        "apply_failed",
  "endpoint":    "migrate" | "program_paused" | "hints" | "thresholds",
  "action_id":   "<uuid from the originating request>",
  "reason":      "<skip-class slug from T20 vocabulary>",
  "hash":        "<unit hash>" | null,
  "ts":          <wall>,
  "ts_monotonic": <perf>
}
```

POSTed to `<daemon>/aginfer/event` exactly like the watermark
webhook; the daemon's existing `attach_event_routes` dispatches it
to a registered `EventKind.APPLY_FAILED` handler.

**T23 sglang side.**  Inside `scheduler.migrate_aginfer`, after
`apply_aginfer_migrations` returns, iterate `result["skipped"]` and
call `aginfer_webhook.fire_apply_failed(endpoint, action_id, reason,
hash_)` for each.  The synchronous `/aginfer/migrate` response is
preserved unchanged (cold-start probes and verify/t20 still read
it).  When `aginfer_webhook is None` (no `--aginfer-notify-url`),
fire is a no-op.

**T37 daemon side.**  `attach_apply_failed_handler(router)` (called
from `daemon/main.py` after `attach_admission_controller`) registers
a default handler that:
  1. Logs `aginfer apply_failed received: endpoint=... action_id=... reason=... hash=...`.
  2. Calls `router.observability.record_failure(reason)` — bumping
     the per-reason counter that the `daemon_obs_summary` line
     emits (T42 audit S3).
  3. Returns without further action.  The next event's
     `joint_decide` re-evaluates state and may re-issue a
     superseding migration.  DESIGN §10 idempotency makes re-issue
     safe; no per-action retry bookkeeping.

**Double-count avoidance (historical).**  Before T23, the
synchronous `KvScheduler._record_skips` path bumped the observability
counter for every item in `/aginfer/migrate`'s response.  When sglang
ALSO started firing `APPLY_FAILED` per item under T23, both paths
would have counted — so T23 removed the bump from `_record_skips`
(keeping its per-line `migrate_skipped` log).  Then T36-audit
cleanup removed the entire sync POST path from `KvScheduler`
(`_dispatch_migrate` now requires outbound), and `_record_skips`
went with it.  The webhook is now the SOLE per-skip counter source
by construction — there is no longer a parallel sync path that
could double-count, so the original double-count concern is
structurally impossible.  Per-event grep target is now
`aginfer_metric event=apply_failed reason=... endpoint=...`
(emitted by the T37 handler in `event_router.py`).

## WORST CASE

| Failure mode | How to force | Floor | Assertion |
|---|---|---|---|
| EventKind.APPLY_FAILED missing | grep enum | enum value exists | A0 |
| Handler not registered | POST without attach | webhook would log + skip; counter zero | (covered by negative assertions in A4) |
| Counter goes up by reason, not by hash | mix reasons | per-reason buckets | A1 |
| Unknown endpoint future-compat | endpoint="future_endpoint_42" | still counts | A2 |
| Malformed payload (no reason) | strip reason key | webhook accepted (200); counter unchanged; worker survives | A3 |
| Apply_failed structured-metric line | webhook handler fires | `aginfer_metric event=apply_failed reason=... endpoint=...` lands | A4 |
| Webhook never arrives | sglang config without `--aginfer-notify-url` | `aginfer_webhook is None`; fire is no-op (no crash) | (manual; verified by sglang code path) |
| Live integration | known-bad migrate to live sglang+daemon | webhook lands, handler logs, counter bumps | B0, B1, manual breakdown check |

## HOW WE VERIFY

`verify/t23_t37_apply_failed/verify.py` — 7 stages.  Phase A is
in-process FastAPI; Phase B is opt-in live integration.

```
A0  EventKind.APPLY_FAILED exists with value "apply_failed"
A1  webhook → handler → per-reason counter (mix of 3 reasons)
A2  every endpoint counted (forward-compat: future endpoints land
    for free without handler edits)
A3  malformed payload (no reason) accepted with 200; counter
    unchanged; worker survives for the next valid payload
A4  APPLY_FAILED handler emits structured `event=apply_failed` line
    (post-T36 cleanup: the legacy sync `_record_skips` path no
    longer exists; this is the per-event grep target now —
    `aginfer_metric event=apply_failed endpoint=... reason=...
    action_id=...`).

B0  (opt-in via AGINFER_VERIFY_BASE_SGLANG): drive a known-bad
    migrate (hash="node-99999999") at live sglang; sync response
    must return applied=0 + skipped[reason=not_in_tree]
B1  (opt-in via AGINFER_VERIFY_DAEMON_LOG): wait up to 5 s, grep
    the daemon log for `kind=apply_failed` — proves the webhook
    arrived AND the worker processed it
```

## REPRODUCING

Phase A only (no GPU):

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/t23_t37_apply_failed/verify.py
```

Phase A + B (live sglang + daemon):

```bash
# 1. sglang
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 CUDA_VISIBLE_DEVICES=6 \
  python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --host 127.0.0.1 --port 30002 \
    --tp 1 --mem-fraction-static 0.15 \
    --max-total-tokens 65536 --trust-remote-code \
    --attention-backend flashinfer \
    --enable-hierarchical-cache --hicache-ratio 1.5 \
    --hicache-write-policy write_through \
    --aginfer-notify-url=http://127.0.0.1:9100/aginfer/event \
    --aginfer-heartbeat-s=5 \
  > /tmp/sglang_t23.log 2>&1 &
until grep -q "fired up" /tmp/sglang_t23.log; do sleep 5; done; sleep 8

# 2. daemon (low cadence so we see breakdowns quickly)
cd /scratch/yuzhou/projects/sglang/dev/aginfer
python -m daemon.main \
  --sglang-base-url=http://127.0.0.1:30002 \
  --host=127.0.0.1 --port=9100 \
  --kv-scheduler=enabled --admission-controller=enabled \
  --theta-hi=0.7 --theta-lo=0.55 \
  --observability-summary-every-n=10 \
  > /tmp/daemon_t23.log 2>&1 &
sleep 6

# 3. verify (Phase A+B)
AGINFER_VERIFY_BASE_SGLANG=http://127.0.0.1:30002 \
AGINFER_VERIFY_BASE_DAEMON=http://127.0.0.1:9100 \
AGINFER_VERIFY_DAEMON_LOG=/tmp/daemon_t23.log \
  python dev/aginfer/verify/t23_t37_apply_failed/verify.py

# 4. trigger shutdown summary + grep the breakdown
pkill -TERM -f "daemon.main"; sleep 3
grep "daemon_obs_summary" /tmp/daemon_t23.log | tail -1

# 5. shutdown sglang
pkill -9 -f sglang.launch_server
```

## RESULTS

**PASSED** — all 7 stages.

* date: 2026-06-01
* lines: ~80 in `python/sglang/srt/managers/aginfer_webhook.py`
  (`ApplyFailedPayload` + `fire_apply_failed` + `_send_apply_failed`);
  ~30 in `scheduler.py::_fire_apply_failed_for_skipped`; +6 LoC
  daemon (`EventKind.APPLY_FAILED`, `attach_apply_failed_handler`,
  `main.py` wiring); -3 LoC removing the sync-path counter bump
  from `_record_skips` (T23-era — `_record_skips` itself was later
  deleted in the T36 cleanup).

| Stage | Result |
|---|---|
| A0 EventKind.APPLY_FAILED exists | PASS |
| A1 webhook bumps counter per reason | PASS — 3 distinct reasons aggregated correctly |
| A2 every endpoint counted (forward-compat) | PASS — including `future_endpoint_42` |
| A3 malformed payload ignored, no crash | PASS — second valid payload still counted |
| A4 apply_failed structured metric line lands | PASS — `event=apply_failed endpoint=migrate reason=not_in_tree action_id=a-1` |
| B0 known-bad migrate returns sync skipped[] | PASS (live sglang) |
| B1 daemon log shows apply_failed webhook | PASS (live sglang+daemon) |

### Live integration measurement

3 bad migrates fired at the live sglang, each with a unique
`action_id`.  Daemon shutdown summary one-liner:

```
events_dispatched_total=3
state_fetch_p99_ms=10.39
queue_depth_p99=0
time_in_queue_p99_ms=0.78
n_failure_classes=1
n_failures_total=3
failure_class_breakdown={"not_in_tree":3}
```

Exactly one counter bump per failed action — no double-count.
End-to-end: sglang's `migrate_aginfer` → `fire_apply_failed` per
skip → background httpx POST → daemon's `/aginfer/event` →
`EventBus.emit` → `_event_worker` → `_apply_failed_handler` →
`observability.record_failure("not_in_tree")` → next
`daemon_obs_summary` line reflects it.

Daemon handler log (sample):

```
aginfer apply_failed received: endpoint=migrate
  action_id=phase-b-bad-1 reason=not_in_tree hash=node-99999991
```

Logs:
- `results/20260601_t23_t37_initial_pass.log` — Phase A/B verify
  (7/7 stages green)
- `results/20260601_t23_t37_phase_b_daemon.log` — full daemon log
  from the live integration (3 bad migrates + per-event lines + the
  shutdown summary above)
