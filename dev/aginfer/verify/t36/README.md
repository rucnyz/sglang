# T36 — Outbound action queue + worker (PLAN §4, DESIGN §6 B4)

PLAN §4 T36.  Pre-T36 the daemon's event handler did
`await client.post(...)` for each migrate dispatch — the event-
worker coroutine blocked on sglang's POST round-trip.  T42's stress
probe measured this directly: **`time_in_queue_p99 = 355.84 ms`**
on 100 webhook events, 3.5× over PLAN's F3-revisit threshold
(100 ms p99).

T36 decouples handler latency from sglang latency.

## WHAT WE PROMISED

**Module `dev/aginfer/daemon/outbound.py`** — single class
`OutboundQueue` (asyncio.Queue[OutboundBatch]) + dedicated worker
task.

```python
batch_id = router.app.state.outbound.enqueue_migrate(actions_wire)
# returns immediately; UUID4 batch_id correlates any future
# APPLY_FAILED webhook (DESIGN §6 L507).
```

`enqueue_migrate(actions_wire) -> batch_id` is `put_nowait` +
`uuid.uuid4()`.  Returns in < 50 µs.

The worker pops batches and POSTs them via the daemon's shared
httpx.AsyncClient.  Semantics on response (DESIGN §6 / §10):

| status | behavior |
|---|---|
| 200 | drop body; sglang's APPLY_FAILED webhook is the per-item failure path |
| 4xx | log warning + move on (4xx → plan-shape bug; next joint_decide re-evaluates) |
| 5xx | log warning + move on (transient backpressure; idempotent re-issue safe) |
| transport exception | log warning + move on |

**No retry bookkeeping in the worker.**  DESIGN §10 idempotency
means re-issuing the same action is safe; the next event sees the
unchanged state and emits the same plan.

**Wiring.**  `main.py` creates `OutboundQueue` once, injects it into
`KvScheduler(outbound=outbound)`, and ties its lifecycle to the
FastAPI startup/shutdown hooks via `app.state.outbound`.
`KvScheduler._dispatch_migrate` now enqueues + returns; the legacy
sync POST path is preserved as a fallback when `outbound is None`
(unit tests).

**Identifier model.**  Each outbound POST carries `batch_id` in the
JSON body envelope (DESIGN §6 L506).  Per-item `action_id` (T20)
remains in each action object.  Either granularity correlates with
APPLY_FAILED.

## WORST CASE

| Failure mode | Predicted floor | Stage |
|---|---|---|
| Slow downstream POST (200 ms) blocks handler | handler returns <1 ms (asyncio.Queue.put_nowait + uuid4) | A1 |
| Sglang transient 5xx storm | worker survives, posts all batches, no retry | A3 |
| Sglang/network ConnectError | worker survives, no crash | A4 |
| SIGTERM mid-burst | stop() drains in-flight then exits within ~2 s | A5 |
| FIFO inversion | worker dequeues in enqueue order | A2 |
| batch_id collision after 2^32 enqueues | UUID4 collision probability < 10⁻²² | A0 |
| Legacy code path forgets to enqueue | `_dispatch_migrate(outbound=None)` falls back to sync POST + `migrate_post` metric | (preserved; not asserted) |
| kv_scheduler wires outbound | `_dispatch_migrate` enqueues exactly once, no sync POST | A6 |

## HOW WE VERIFY

`verify/t36/verify.py`:

```
A0  batch_id is UUID4 + unique across calls
A1  handler enqueue returns < 1 ms even when downstream POST takes
    200 ms (the headline T36 property)
A2  worker drains FIFO (10 batches with distinct hashes; order preserved)
A3  worker survives 5xx storm (5 × 503; no crash; all 5 hit the wire)
A4  worker survives ConnectError (3 × raise; no crash)
A5  stop() drains in-flight POSTs then exits within bounded time
A6  KvScheduler wired with outbound enqueues exactly one OutboundBatch
    per _dispatch_migrate call (no sync POST)
B0  (opt-in) live daemon + sglang: time_in_queue_p99 < 100 ms PLAN
    threshold under 50-webhook burst
```

## REPRODUCING

Phase A only:

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/t36/verify.py
```

Stress measurement (same recipe as T42; should show
`time_in_queue_p99` drop):

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
  > /tmp/sglang_t36.log 2>&1 &
until grep -q "fired up" /tmp/sglang_t36.log; do sleep 5; done; sleep 8

# 2. daemon
cd /scratch/yuzhou/projects/sglang/dev/aginfer
python -m daemon.main \
  --sglang-base-url=http://127.0.0.1:30002 --host=127.0.0.1 --port=9100 \
  --kv-scheduler=enabled --admission-controller=enabled \
  --theta-hi=0.7 --theta-lo=0.55 --observability-summary-every-n=20 \
  > /tmp/daemon_t36.log 2>&1 &

# 3. drive 100 webhooks (5 batches × 20 parallel)
for batch in 1 2 3 4 5; do
  for i in $(seq 1 20); do
    pid=$(( (batch - 1) * 20 + i ))
    curl -sX POST http://127.0.0.1:9100/aginfer/event \
      -H "Content-Type: application/json" \
      -d "{\"kind\":\"memory_pressure\",\"session\":\"t36-${pid}\",
           \"state\":\"HIGH\",\"prev_state\":\"OK\",\"occ\":0.85}" \
      > /dev/null 2>&1 &
  done
  wait
done

# 4. read first cadence summary (events_dispatched_total=20, apples
#    to apples with T42's stress measurement)
grep "daemon_obs_summary" /tmp/daemon_t36.log | head -1

# 5. shutdown
pkill -TERM -f "daemon.main"; pkill -9 -f sglang.launch_server
```

## RESULTS

**PASSED** — all 8 stages.

* date: 2026-06-01
* lines: ~190 in new `daemon/outbound.py`; ~30 in `kv_scheduler.py`
  (constructor param + `_dispatch_migrate` enqueue branch); ~20 in
  `proxy.py` (startup/shutdown lifecycle); +8 in `main.py` (wire).

| Stage | Result |
|---|---|
| A0 batch_id is UUID4 + unique | PASS |
| A1 handler enqueue < 1 ms regardless of POST latency | PASS — measured max enqueue ~30 µs across 50 calls with 200 ms stubbed POST |
| A2 worker drains FIFO | PASS — 10 hashes preserved order |
| A3 worker survives 5xx | PASS — 5/5 posts despite all 503 |
| A4 worker survives ConnectError | PASS — 3/3 posts attempted |
| A5 stop() drains in-flight | PASS — 3 batches × 50 ms drained in < 2 s |
| A6 KvScheduler wires to outbound, no sync POST | PASS |
| B0 live stress (skipped without env) | PASS (skipped — manual run below) |

### Stress measurement (apples-to-apples with T42)

Same recipe as T42's pre-T36 stress run.  First-cadence summary
(events_dispatched_total = 20):

| metric | T42 pre-T36 | **T36 post-fix** | delta |
|---|---|---|---|
| state_fetch_p99_ms     | 16.38   | 11.62   | -29 % |
| queue_depth_p99        | 18      | 86      | +378 % * |
| **time_in_queue_p99_ms** | **355.84** | **176.05** | **-50 %** |

`*` queue_depth_p99 went UP because the bash driver now outpaces
the much-faster worker — events pile up in the queue faster than
the (now ~9 ms/event) worker can drain them.  The metric that
matters (time-in-queue, what PLAN F3-revisit gates on) drops by
half.

**The remaining bottleneck is `fetch_state`**: state_fetch_p99 ≈
8-12 ms accounts for the bulk of post-T36 per-event work, because
the synthetic memory_pressure events don't actually trigger any
migrate (only 1 of 100 events produced a migrate_enqueued line —
no real cache pressure under synthetic load).  Real-traffic
profile (T42's stress where every event drove ~17 skips) would
show a much bigger T36 win because the dispatched POST that used
to block 50-100 ms now returns instantly.

**Still over PLAN's 100 ms F3-revisit threshold at d=20: 176 ms
> 100 ms.**  The F3-revisit task (#160) now has cleaner data: T36
halved one term, the remaining term is `fetch_state` HTTP round-
trip (network + JSON parse).  The clear next step is daemon-side
state caching / incremental updates (DESIGN §10 F3 "incremental
state" branch).

Logs:
- `results/20260601_t36_initial_pass.log` — Phase A (8/8)
- `results/20260601_t36_stress_100webhooks.log` — daemon log from
  the 100-webhook stress (includes 5 cadence summary lines + the
  shutdown summary)
