# T164 — Sustained-load escalate-to-fatal (DESIGN §10 sustained tier)

The "running degraded forever" anti-pattern.  When sglang has been
unreachable AND the daemon's outbound queue has been backing up, the
daemon self-kills so the supervisor (systemd / k8s) restarts.  If
sglang is still down, the daemon CrashLoopBackOffs visibly to ops.
Crash-only software (Candea & Fox); silently eating memory + doing
no useful work is NOT a valid state.

Lives in #164.  DESIGN §10 "sustained-escalation tier" (added by
#163) names this contract.

## WHAT WE PROMISED

**Trigger criterion: BOTH must trip simultaneously**:
1. `OutboundQueue.consecutive_failures >= escalate_failures`
   (default 100 — ~3-5 min at typical fire-and-forget rates)
2. `oldest_pending_batch_age_ms >= escalate_oldest_age_s * 1000`
   (default 300 s = 5 min)

If only ONE crosses, escalation does NOT fire.  Rationale:

* Low-traffic dead-sglang: every POST fails fast (ConnectError) →
  consec rockets, but queue drains as fast as failures complete →
  `oldest_age` stays small.  Daemon survives, will reconverge when
  sglang returns.  No memory damage.
* All-success but slow POSTs: `oldest_age` can grow under burst but
  consec is 0 → no escalation.

The both-or-neither contract guards against false positives on
either axis alone.

**Failure classes counted for consec**:
| HTTP / exception | counts as |
|---|---|
| 2xx | success → resets consec to 0 |
| 4xx | failure (plan-shape bug — deployment-class; crashloop reveals) |
| 5xx | failure (transient sglang stall) |
| Transport exception | failure (sglang unreachable) |

**CLI flags** (`daemon/main.py`):
- `--sustained-escalate-fails` (default 100)
- `--sustained-escalate-age-s` (default 300.0)

**fatal() call site** (inside `OutboundQueue._worker_loop`):
`fatal('sglang_sustained_unreachable', sglang_base_url=...,
consecutive_failures=N, oldest_age_ms=X, queue_depth=Q,
escalate_failures_threshold=..., escalate_oldest_age_s_threshold=
...)`.  Forensic JSON dumped to `<data>/forensic/
sglang_sustained_unreachable_*.json` per T43 contract.

**/health observability** (`daemon/proxy.py`):
Daemon's `/health` body now includes:
- `outbound_consecutive_failures: int`
- `outbound_oldest_age_ms: float`

HTTP status stays 200 — the daemon process IS responsive; the
restart signal is the fatal() exit, not health failure.  k8s
readiness probes can grep the body fields and depool independently.

## WORST CASE

| Failure mode | How to force | Floor | Assertion |
|---|---|---|---|
| Consec counter doesn't reset on success | mix 2xx with failures | counter resets to 0 on 2xx | A0 |
| One failure flavor missed | 5xx / 4xx / ConnectError each | all four increment consec | A1 |
| False positive: low-traffic dead-sglang fatal | 10 fresh failures, age ≈ 0 | NO fatal | A2 |
| False positive: high-age but all-success | aged batches, all 2xx | NO fatal (consec=0) | A3 |
| Operator can't grep current state | hit /health | body has both counters | A4 |
| True positive: dead sglang + backlog | aged batches + always-fail stub | fatal + forensic dump + exit 1 | B0 |

## HOW WE VERIFY

`verify/t164/verify.py` — 6 stages.  A0-A4 in-process; B0 spawns a
real subprocess to exercise the actual `fatal()` path (which exits
the process — can't test in the same process).

```
A0  counter resets on 2xx (3 batches: 5xx, 5xx, ok → consec=0)
A1  consec increments on 5xx / 4xx / ConnectError (mix of 4
    flavors, no 2xx → consec=4; thresholds high so no escalation)
A2  high consec alone does NOT escalate (always-503 stub, 10 fresh
    batches, consec >> threshold but oldest_age ≈ 0 → daemon
    survives, returns normally)
A3  high oldest_age alone does NOT escalate (always-200 stub,
    aged batches with old enqueue_ts → oldest_age >> threshold
    but consec stays 0 → no escalation)
A4  /health body carries outbound_consecutive_failures +
    outbound_oldest_age_ms (uvicorn server with preset counters)
B0  subprocess: always-fail stub + aged batches + low thresholds
    → fatal('sglang_sustained_unreachable') → process exit 1
    → forensic JSON file landed with context fields
    (sglang_base_url, consecutive_failures, oldest_age_ms,
    queue_depth, escalate_failures_threshold,
    escalate_oldest_age_s_threshold)
```

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/t164/verify.py
```

No GPU / no sglang launch.  Runs in < 5 s (B0 spawns a brief
subprocess).

## RESULTS

**PASSED** — all 6 stages.

* date: 2026-06-01
* lines: ~50 daemon (`outbound.py` worker loop + `_post_one` + ctor;
  `main.py` 2 CLI flags; `proxy.py` /health body)

| Stage | Result |
|---|---|
| A0 consec resets on 2xx | PASS |
| A1 consec increments on 5xx / 4xx / transport-exc | PASS — final consec = 4 after mixed failures |
| A2 high consec alone does NOT escalate | PASS — 10 failures, no fatal |
| A3 high oldest_age alone does NOT escalate | PASS — 5 aged successes, consec stays 0 |
| A4 /health body carries outbound counters | PASS — `outbound_consecutive_failures=5` + `outbound_oldest_age_ms=123.4` in body |
| B0 subprocess: both thresholds → fatal + forensic | PASS — subprocess exit 1, forensic file with all 6 context keys |

* raw log: `results/20260601_t164_initial_pass.log`
