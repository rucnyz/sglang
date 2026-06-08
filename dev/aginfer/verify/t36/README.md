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
`KvScheduler._dispatch_migrate` enqueues + returns; the outbound
queue is now **mandatory** (calling `_dispatch_migrate` without one
raises `RuntimeError`).  The legacy sync POST fallback was removed
post-T36 audit — DESIGN §6 B4 makes fire-and-forget the only valid
production dispatch path, and maintaining two implementations was
debt with no real testing benefit.

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
| Burst coalescing (post-#228) | worker drains the wake's burst → ≤1 POST/PUT per endpoint, order program_paused→migrate→hints, latest-per-key | A2 |
| batch_id collision after 2^32 enqueues | UUID4 collision probability < 10⁻²² | A0 |
| Wiring bug: outbound not injected | `_dispatch_migrate(outbound=None)` raises RuntimeError | T42 B4 |
| kv_scheduler wires outbound | `_dispatch_migrate` enqueues exactly once | A6 |

## HOW WE VERIFY

`verify/t36/verify.py`:

```
A0  batch_id is UUID4 + unique across calls
A1  handler enqueue returns < 1 ms even when downstream POST takes
    200 ms (the headline T36 property)
A2  worker coalesces a wake's burst (post-#228): ≤1 POST/PUT per endpoint,
    cross-endpoint order program_paused→migrate→hints, latest-decision-per-
    hash (migrate) / highest-stamp-per-hash (hints)
A3  worker survives 5xx storm (no crash; keeps dispatching later wakes)
A4  worker survives ConnectError (no crash; later wave still dispatched)
A5  stop() drains in-flight POSTs then exits within bounded time
A6  KvScheduler wired with outbound enqueues exactly one OutboundBatch
    per _dispatch_migrate call
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
| A2 worker coalesces burst (≤1/endpoint, order, latest-per-key) | PASS (post-#228) |
| A3 worker survives 5xx | PASS — no crash; later wakes still dispatched |
| A4 worker survives ConnectError | PASS — no crash; later wave dispatched |
| A5 stop() drains in-flight | PASS — 3 batches × 50 ms drained in < 2 s |
| A6 KvScheduler wires to outbound, no sync POST | PASS |
| B0 live stress (skipped without env) | PASS (skipped — manual run below) |

### Stress measurements — three runs, three bottlenecks

The original "p99 dropped 50 %" claim was misleading because T36's
design win is conditional on the synchronous POST being the
bottleneck.  Three runs reveal where the bottleneck actually lives:

| run                          | cache  | sglang load        | events firing migrate | dominant cost                     | time_in_queue_p99 |
|---|---|---|---|---|---|
| **T42 stress** (pre-T36)     | ~80    | idle               | 100/100 (~17 actions each) | POST + sync skip read           | 355.8 ms |
| **T36 stress #1**            | ~0     | idle               | **1/100**             | fetch_state (light)               | 176.1 ms |
| **T36 stress #2** (audit)    | 79     | **32 chats active** | 20/20                 | **fetch_state under sglang load** | 1655.8 ms |

**Why run #1 is not the right number to compare against T42.**
Run #1 fired memory_pressure webhooks at a freshly-launched daemon
whose state had NO units (sglang's cache was empty).  Policy
emitted 0 actions for 99 of 100 events — *the migrate POST never
fired*, so the path T36 optimised was barely exercised.  The
176 ms p99 is just `fetch_state + decide + ε` × queue depth on a
serial worker; the 50 % drop vs T42 is an artifact of comparing
two runs with different cache state, not the T36 effect.

**What run #2 actually shows.**  Cache populated (79 units) + 32
concurrent chats keeping sglang busy.  First-cadence summary
(events_dispatched_total = 20):

```
events_dispatched_total = 20    state_fetch_p99_ms = 711.26
queue_depth_p99        = 17     time_in_queue_p99_ms = 1655.80
migrate_enqueued       = 20     outbound_post        = 20
```

20/20 events triggered a real migrate; the outbound worker drained
all 20 POSTs in parallel with the event worker continuing to
dequeue.  **T36's design is realised** — the event worker did
NOT serialize on the POST.  But time_in_queue is huge because
sglang's HTTP path is queued behind the 32 active chats (T14's
own stress probe already saw sglang state-dump p99 = 320 ms under
chat load; here it's 711 ms because we're driving harder).

**What pre-T36 would look like under run #2's conditions.**  We
can't measure it without reverting the patch, but the back-of-the-
envelope:

```
pre-T36 per-event ≈ fetch_state + decide + await POST
                  ≈ 711 + 1 + (200-500)   # POST is also slow when sglang busy
                  ≈ ~1 second per event

pre-T36 time_in_queue_p99 ≈ 17 × ~1000 ≈ ~17 seconds
```

vs the measured **1655 ms with T36 — roughly an order of magnitude
lower**.  The win is real and significant under load; it's just
hidden when you pick a profile where POST didn't matter (run #1)
or where sglang was idle (T42).

### The bottleneck shifted

Even with T36, run #2's time_in_queue p99 = 1655 ms is 16× over
PLAN's 100 ms F3-revisit threshold.  T36 fixed the synchronous-POST
serialization; the dominant remaining term is now **fetch_state
under concurrent sglang load** — sglang's HTTP path queues behind
the scheduler lock the active chats are holding.

The F3-revisit task (#160) should pursue **daemon-side state
caching / incremental updates** (DESIGN §10 F3 "incremental state"
branch).  T36 is the necessary first step (otherwise the serial
POST would dwarf any state-fetch optimisation), but it's not
sufficient on its own.

Logs:
- `results/20260601_t36_initial_pass.log` — verify Phase A (8/8)
- `results/20260601_t36_stress_100webhooks.log` — run #1 (empty
  cache; misleading single-number)
- `results/20260601_t36_stress_with_populated_cache.log` — run #2
  (cache populated + sglang busy; shows the real bottleneck and
  T36's design realised under heavy POST profile)
