# T42 — Daemon-side observability logging (PLAN §4)

PLAN.md §4 T42.  Companion to T14 (sglang side).  Four daemon metric
streams the operator needs to spot daemon-side backpressure
empirically before designing the F3 fix:

1. **state-fetch latency** p50 / p95 / p99 — daemon's GET
   `/aginfer/state` wall-clock.
2. **event-queue depth at handler entry** — `bus.queue.qsize()`
   AFTER popping the current event (backlog view).
3. **time-in-queue per event** — `now - event.enqueue_time`.
4. **cumulative failure-class counters** — `migrate_skipped` reasons
   today; APPLY_FAILED breakdown will plug in via the same recorder
   when T23+T37 (#153) lands.

PLAN F3-revisit conditions that key on this aggregator:
- queue depth > 64 sustained → revisit
- time-in-queue p99 > 100 ms → revisit
(symmetric to T14's sglang-side `dump p99 > 50ms` trigger.)

**"Sustained" semantics** (audit G3, this README clarifies for PLAN
L401).  The summary line carries both `queue_depth_p50` and
`queue_depth_p99`.  We read PLAN's "sustained" as **p50 > 64 over the
ring window** (a single spike does NOT count): a p50 above the
threshold means at least half of recent dispatches saw a backlog
deeper than 64.  Operators MAY also alert on p99 spikes for
incident detection, but the F3-revisit task should fire on p50 to
avoid bouncing the design between transient bursts and steady-state
overload.

## WHAT WE PROMISED

**Aggregator location.**  `dev/aginfer/daemon/_observability.py`:
a single `DaemonObservability` instance on every `EventRouter`.  Each
of the three quantile streams is a `_DaemonMetricsRing` (single-
valued bounded ring + on-demand sort-then-pick quantile summary).
Failure-class counters are a simple `dict[str, int]`.

**Capacity = 1024** (≈ 5 min at the daemon's typical 3.4 Hz event
rate).  Deep enough for stable p99; shallow enough that a one-off
spike ages out within the window so the F3-revisit trigger doesn't
stick on a single past tail.

**Emission cadence.**  Every `summary_every_n` handled events (CLI
flag `--observability-summary-every-n`, default 200) the aggregator
fires one `daemon_obs_summary` line via the existing
`_metrics.m()` format.  Plus one final summary on shutdown so the
last partial window isn't lost.

**Per-event log lines unchanged.**  `event_received` still fires per
dispatch (now annotated with `qdepth=N time_in_queue_ms=X`); the
T42 aggregator is the rollup on top.

**Hooks added to existing code paths:**

| File | Hook |
|---|---|
| `events.py` | `Event.enqueue_time` field + `EventBus.emit` stamps it via `dataclasses.replace` if unset |
| `event_router.py` | `fetch_state` wraps `_fetch_state_impl` with a perf_counter; the worker records `(qdepth, time_in_queue_ms)` at dispatch entry |
| `kv_scheduler.py` | `_record_skips()` helper extracted from `_dispatch_migrate`'s skip loop; bumps `observability.failure_class_counts[reason]` when an observability instance is wired |
| `main.py` | `--observability-summary-every-n` CLI flag; passes `router.observability` into `KvScheduler`; emits a final summary on shutdown |
| `proxy.py` | plumbs `observability_summary_every_n` through `create_app` → `EventRouter` |

**Post-commit audit fixes (#162 — subagent audit punch list).**

* **G2 — load-fault counter scope.**  `_record_skips` was the only
  call site bumping `failure_class_counts`; `state_fetch_failed`
  (kv_scheduler line 854) now also routes through
  `observability.record_failure("state_fetch_failed")`.  APPLY_FAILED
  (T23+T37, #153) plugs in via the same recorder when it lands.
* **S3 — per-reason breakdown on the summary line.**  Variable-
  cardinality counter previously folded down to two scalars
  (`n_failure_classes`, `n_failures_total`); now also emitted as a
  space-free compact-JSON `failure_class_breakdown={...}` field on
  the same line.  Operator's grep can recover `not_in_tree=820` etc.
  without round-tripping through `summary_dict()`.
* **S6 — counter-name vs semantics drift.**  Renamed
  `events_handled_total` → `events_dispatched_total` (the increment
  fires at dispatch entry, NOT after handler-success).  `router.
  events_handled` separately tracks handler-success; the two now
  read distinctly when handlers raise.
* **T4 — `summary_every_n <= 0` rejected.**  Construction raises
  ValueError instead of silently making every dispatch emit a
  summary (when summary_every_n=0).

**Daemon contract refinement** (audit #161, surfaced while exercising
the T42 stress path): two T43 positivity checks were too strict for
the pre-T12 / pre-T26 sglang placeholders (`h_max = 0.0`,
`prefill_bps = 0.0`):

  * `h_max_per_byte_sec`: now only fatals on PARTIAL config (some
    positive, some zero) or NEGATIVE.  ALL-zero is a legitimate
    "no operator config yet" cold-start state.
  * `prefill_bps`: now only fatals on NEGATIVE.  "Zero + units > 0"
    used to fatal — but pre-T26 measurement wiring, sglang ships
    placeholder zero even when traffic has run.  The "zero + traffic
    = bug" form re-tightens when T26 ships.

verify/t43 Stage 9/10 updated to cover the new sub-cases; all
16 T43 stages still green.

## WORST CASE

| Failure mode | How to force | Predicted floor | Assertion |
|---|---|---|---|
| Empty aggregator read | `summary_dict()` before any event | contract field set with zeros, never raises | A0, A3 |
| Ring buffer wrap | record 2000 samples cap=32 | window holds last 32; `n_recorded_total = 2000` | A2 |
| Quantile monotonicity | 99 fast + 1 slow | p50 ≤ p95 ≤ p99 ≤ max | A1 |
| Failure-class counter | record same reason N times | counter == N | A4 |
| Summary cadence | summary_every_n=5; record 5 events | exactly 1 summary line | A5 |
| Summary line shape | force `emit_summary` | line carries state_fetch / queue_depth / time_in_queue p99 + n_failure_classes | A6 |
| EventBus stamps enqueue_time | construct Event w/o time, emit, get | dequeued copy has `enqueue_time > 0` | B0 |
| Worker records dispatch | 5 events through real router | both rings hold 5 samples | B1 |
| Fetch latency lands | stub `_fetch_state_impl` with 5 ms sleep × 4 | `state_fetch_lat_ms.max >= 4 ms` | B2 |
| KvScheduler-driven failure recorder | feed 4 synthetic skips | per-reason counter matches | B4 |
| End-to-end log emission | 12 events at every-10 cadence | exactly 1 `daemon_obs_summary` line on log capture | C0 |

## HOW WE VERIFY

`verify/t42/verify.py` runs three phases all in-process; no live
sglang needed.

```
Phase A  in-process unit tests of _DaemonMetricsRing + DaemonObservability
  A0  ring empty summary
  A1  ring record + quantile monotonicity
  A2  ring wraps at capacity
  A3  observability empty summary shape
  A4  failure-class counter increments
  A5  emit_summary cadence (every N events)
  A6  summary line carries contract fields

Phase B  integration with EventRouter + EventBus + Event.enqueue_time
  B0  EventBus.emit stamps enqueue_time
  B1  router records dispatch + time-in-queue (real worker)
  B2  router records state-fetch latency
  B3  router exposes failure-class recorder
  B4  kv_scheduler skips bump observability (production wiring)

Phase C  end-to-end log emission
  C0  real-dispatch summary line lands with correct cadence
```

**Stress probe** (manual, ~30 s of webhook load — proxy bypassed
because of a pre-existing httpcore CancelledError race on
`/v1/chat/completions` under high concurrency; bypassed by hitting
`/aginfer/event` directly).  The result is in the RESULTS block.

## REPRODUCING

Phase A/B/C (no GPU needed):

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/t42/verify.py
```

Stress probe (the headline number; needs a sglang to fetch state from):

```bash
# 1. sglang (any GPU)
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
  > /tmp/sglang_t42.log 2>&1 &
until grep -q "fired up" /tmp/sglang_t42.log; do sleep 5; done; sleep 8

# 2. daemon with low summary cadence so we see emissions on a small run
cd /scratch/yuzhou/projects/sglang/dev/aginfer
python -m daemon.main \
  --sglang-base-url=http://127.0.0.1:30002 \
  --host=127.0.0.1 --port=9100 \
  --kv-scheduler=enabled --admission-controller=enabled \
  --theta-hi=0.7 --theta-lo=0.55 \
  --observability-summary-every-n=20 \
  > /tmp/daemon_t42_stress.log 2>&1 &

# 3. fire 100 synthetic memory_pressure webhooks (bypasses the chat
#    proxy whose httpcore race is a pre-existing separate bug)
for batch in 1 2 3 4 5; do
  for i in $(seq 1 20); do
    pid=$(( (batch - 1) * 20 + i ))
    curl -sX POST http://127.0.0.1:9100/aginfer/event \
      -H "Content-Type: application/json" \
      -d "{\"kind\":\"memory_pressure\",\"session\":\"obs-${pid}\",\"state\":\"HIGH\",\"prev_state\":\"OK\",\"occ\":0.85}" \
      > /dev/null 2>&1 &
  done
  wait
done
sleep 8

# 4. read metrics + trigger shutdown summary
grep "daemon_obs_summary" /tmp/daemon_t42_stress.log
pkill -TERM -f "python -m daemon"
sleep 3
grep "daemon_obs_summary" /tmp/daemon_t42_stress.log | tail -1
```

## RESULTS

**Instrumentation: PASSED** — all 20 verify.py stages green
(7 Phase A + 5 Phase B original + 7 audit-driven stages B5–B11
covering audit punch-list G2/T1/T2/T5/T6/T3/T4 + 1 Phase C).

**PLAN F3-revisit trigger condition `time_in_queue_p99 > 100 ms`:
FIRED** under a modest synthetic stress (100 webhook events
serialised through one worker).

### Stress measurement headline

First `daemon_obs_summary` line at `events_handled_total=20`:

```
state_fetch_n=39         state_fetch_p50_ms=4.78    state_fetch_p99_ms=16.38
queue_depth_n=20         queue_depth_p50=9          queue_depth_p99=18
time_in_queue_n=20       time_in_queue_p50_ms=206   time_in_queue_p99_ms=355.84
n_failure_classes=3      n_failures_total=1616
```

| metric                         | reading      | PLAN threshold | under? |
|---|---|---|---|
| `state_fetch_p99_ms`           | 16.38 ms     | 50 ms (T14)    | ✓ |
| `queue_depth_p99`              | 18           | 64             | ✓ |
| **`time_in_queue_p99_ms`**     | **355.84 ms**| **100 ms**     | **✗** |

The 350 ms time-in-queue p99 is the real finding: every webhook
fires a state-fetch + decide + a migrate POST that the policy fills
with ~80 non-leaf candidates which sglang rejects.  The serial
worker can't drain faster than ~5 events/s while POSTing those
batches.  Queue stayed around depth 9–18 for the full burst; events
sat 200–350 ms on average before dispatch.

### Cumulative migrate-skip breakdown

| reason                          | count |
|---|---|
| `remove_not_leaf`               | 820 |
| `remove_dram_not_host_leaf`     | 780 |
| `remove_hbm_not_device_leaf`    | 101 |
| **total skips / 100 events**    | **1701** |

Every event drives ~17 migrate-action skips, all "this node is not a
leaf so I can't device-evict it" rejections.  Confirms the T20
combined-add-remove bug fix (#157) is doing its job — but more
importantly, the daemon's policy is shooting at non-leaf candidates
on every cycle.  **This is the empirical signal that T34 (multi-axis
DP, #156) needs leaf-aware candidate filtering up front** instead
of trusting sglang's defense-in-depth guard.

### Implications for F3 design

Two distinct backpressure flavours surface together:
- **time-in-queue** dominated by `_dispatch_migrate` POST overhead per
  event — drop-on-full or coalesce would NOT help (each POST is
  load-bearing); proper leaf-aware candidate filtering would
  (smaller batches, fewer skipped actions).
- **migrate-skip storm** is wasted daemon→sglang round-trips.  T34's
  candidate set should exclude non-leaves at decision time, not
  rely on sglang to filter.

Both feed into #156 + #160 (the F3-revisit task).

### Ancillary findings (not blocking T42)

1. **Audit #161 — T43 cold-start fatals**.  Sglang's pre-T12 / pre-T26
   placeholders (`h_max=0`, `prefill_bps=0`) were tripping T43's
   `holding_cost_non_positive` and `prefill_bps_non_positive_with_traffic`
   fatals on the very first daemon event.  Relaxed:
   - h_max: all-zero is cold-start (not a bug); partial-config or
     negative is still a bug.
   - prefill_bps: only negative is a bug until T26 wires real
     measurement; the "zero + units > 0" case re-tightens then.
   Both changes covered by updated verify/t43 Stage 9/10.
2. **Proxy httpcore CancelledError** under 32-concurrent
   `/v1/chat/completions` — pre-existing bug in the proxy's
   downstream httpx pool handling.  Forced the stress probe to use
   the webhook path instead.  Tracked separately as part of #160's
   broader F3 scope.

Logs:
- `results/20260531_t42_initial_pass.log` — first-cut verify (13/13,
  pre-audit)
- `results/20260601_t42_audit_pass.log` — post-audit verify (20/20)
- `results/20260531_t42_stress_100events.log` — full daemon log from
  the 100-webhook stress run (includes both periodic + shutdown
  `daemon_obs_summary` lines)
