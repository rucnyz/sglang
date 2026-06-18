# T24 — HASH_COLLISION webhook + detection (#182, DESIGN §4 + §10)

If sglang's radix-cache DFS ever finds two distinct nodes mapping
to the same hash key, that's a deployment-bug class signal —
either sglang's hash function regressed or the daemon's re-keying
broke.  Probability < 10⁻²² at any practical tree size; if it
fires, the daemon `fatal()`s with a forensic dump so ops can
diagnose.

## STATE PRE-#182

Detection landed earlier in `unified_radix_cache.py:2376-2412`
with collision-pair dedupe (`_aginfer_collision_seen` set).  But
the code only logged "T24 webhook pending" — webhook firing was
never wired.

## WHAT #182 LANDED

| layer | change |
|---|---|
| `unified_radix_cache.apply_aginfer_migrations` | DFS now appends `{key, node_a_summary, node_b_summary}` to a `hash_collisions` list (only for pairs not already in `_aginfer_collision_seen`); list is returned in the result dict |
| `unified_radix_cache._aginfer_node_summary` (new) | Compact summary: `node_id`, `hash_value`, `residence`, `n_tokens`, `session_ids[:8]`, `hit_count` |
| `aginfer_webhook.HashCollisionPayload` (new) | Dataclass mirroring `ApplyFailedPayload` shape |
| `aginfer_webhook.AginferWebhookFirer.fire_hash_collision` (new) | Public API; schedules `_send_hash_collision` on background loop; never blocks the scheduler step |
| `aginfer_webhook._send_hash_collision` (new) | 3-attempt POST with backoff 0.1→0.4 s, same pattern as `_send_apply_failed` |
| `scheduler._fire_hash_collisions` (new) | Iterates result dict's `hash_collisions`, fires one webhook per pair |
| `scheduler.migrate_aginfer` | Calls `_fire_hash_collisions` after `_fire_apply_failed_for_skipped` |
| `daemon/events.py:EventKind.HASH_COLLISION` (new) | Enum value `"hash_collision"` |
| `daemon/event_router._hash_collision_handler` (new) | Calls `fatal('hash_collision', key=…, node_a_summary=…, node_b_summary=…, ts=…, ts_monotonic=…)` |
| `daemon/event_router.attach_hash_collision_handler` (new) | Mirrors `attach_apply_failed_handler` |
| `daemon/main.py` | Wires `attach_hash_collision_handler(router)` at startup |

## STAGES (8)

```
A. Sglang webhook firer
  A0  fire_hash_collision payload shape: POST captured with
      kind=hash_collision + key + both summaries with required
      sub-fields
  A1  _send_hash_collision retries on 5xx (3 attempts; backoff
      0.1 → 0.4 s; total wall < 1 s)
  A2  fire_hash_collision returns < 50 ms even against a dead
      port (scheduler hot path must not block)

B. Scheduler wiring (unit-level)
  B0  _fire_hash_collisions skips entries with missing/empty key
  B1  _fire_hash_collisions no-ops when aginfer_webhook is None
      (sglang launched without --aginfer-notify-url)

C. Daemon side
  C0  EventKind.HASH_COLLISION defined with value "hash_collision"
  C1  _hash_collision_handler calls fatal('hash_collision', ...)
      with full context (key, both summaries, ts, ts_monotonic)

D. Subprocess integration
  D0  Real daemon subprocess + synthetic webhook POST to
      /aginfer/event with kind=hash_collision → daemon exits 1 +
      forensic file `hash_collision_*.json` lands with all 5
      context keys
```

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched-rebase
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/t24/verify.py
```

Runs in ~15 s.  A0-C1 in-process; D0 spawns a real daemon
subprocess to exercise the end-to-end fatal path.

## RESULTS

**PASSED** — all 8 stages.

* date: 2026-06-01
* raw log: `results/20260601_t24_initial_pass.log`

| Stage | Result |
|---|---|
| A0 payload shape | PASS |
| A1 retry-on-5xx (3 attempts) | PASS |
| A2 non-blocking (<50 ms) | PASS |
| B0 skips missing key | PASS |
| B1 no-ops with webhook=None | PASS |
| C0 EventKind defined | PASS |
| C1 handler → fatal() with context | PASS |
| D0 subprocess: end-to-end fatal + forensic | PASS — daemon exits 1, forensic JSON has all 5 ctx keys |

## STATE WORTH RECORDING

Caught during wire-up:

1. `AginferWebhookFirer.__init__` starts a background asyncio loop
   in a daemon thread.  Tests that call `fire_*` immediately after
   construction race the loop startup; the verify uses
   `_wait_firer_loop` to poll `firer._loop.is_running()` before
   firing.

2. uvicorn-based capture servers running in a thread collide with
   the firer's own background asyncio loop.  The verify uses
   `http.server.ThreadingHTTPServer` (stdlib, no asyncio) for the
   in-process capture.  uvicorn works only in the subprocess
   integration (D0) where it owns its own process.

## REGRESSION SANITY

Adjacent webhook-firing verifies all green:
* T22 (thresholds PUT) — 8/8
* T23+T37 (apply_failed) — 7/7
* T36 (outbound queue) — 8/8
* T42 (daemon observability) — 21/21
* T164 (sustained-escalate) — 11/11
