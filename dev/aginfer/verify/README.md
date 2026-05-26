# aginfer verify suite

> Per-component verification scripts for the design in [`../DESIGN.md`](../DESIGN.md).
> Each task in the TODO list maps to a verify file here. A verify confirms
> three things and only three things:
>
> 1. **Capability** — the component does what we said it would do.
> 2. **Cost** — the component costs what we said it would cost.
> 3. **Worst case** — when the mechanism degrades or partially fails, what
>    is the floor outcome we are mentally pre-committed to? Knowing this
>    up-front means a degraded run does not panic us; we know exactly how
>    far down it can go.
>
> If the actual worst case exceeds the documented floor, the verify has
> caught a regression we did not anticipate, and the design must be
> reopened.

## How to read a verify file

Every file has the same four sections:

```
WHAT WE PROMISED
----------------
Capability:   short, testable sentence(s) about behavior.
Cost ceiling: explicit thresholds (latency / lines / cpu / etc.).

HOW WE VERIFY (happy path)
--------------------------
Mechanism or e2e script that drives the component under normal
conditions; asserts capability + cost.

WORST CASE (forced, must actually run)
--------------------------------------
For each documented degradation mode:
  - Failure injected: concrete fault (e.g. "stub returns 500", "kill
    daemon mid-run", "disable webhook receiver").
  - How to force: command / mock / config flip.
  - Predicted floor: explicit number or qualitative outcome
    (e.g. "per-trial mean ≤ Run H' × 1.10").
  - Assertion: what to check after running with the fault.
Each row is a real test, not narration.

RESULTS
-------
Filled in after running. Date, build sha, raw output paths, pass/fail
on both happy path and worst-case rows, delta vs the predicted floor.
```

If results drift past 2× the cost ceiling, or the actual degraded
outcome is worse than the documented worst case, treat that as a
regression and re-open the TODO row.

## Index

Each task lives in its own folder `tN/` with:
* `tN/README.md` — capability + cost ceiling + worst-case contract
* `tN/verify.py` (or `verify.sh`) — runnable verifier
* `tN/results/` — raw logs, optimization notes, before/after writeups

| order | TODO | Component | Folder | Status |
|---|---|---|---|---|
| 1 | T1 | `GET /aginfer/state` | [`t1/`](t1/) | **passed** (2026-05-25, sha `82d2732d6`) |
| 2 | T2 | `POST /aginfer/migrate` | [`t2/`](t2/) | **passed** (2026-05-25) |
| 3 | T3 | session_id passthrough into tree nodes | [`t3/`](t3/) | pending |
| **4** | **T6** | program_tracker state machine (moved up — T4 depends on it) | [`t6/`](t6/) | pending |
| 5 | T4 | daemon HTTP proxy + paper §4 event emission | [`t4/`](t4/) | pending |
| 6 | T5 | sglang→daemon webhook (transition + 5 s plateau heartbeat) + daemon event router | [`t5/`](t5/) | pending |
| 7 | T7 | kv_scheduler event handlers + ACTING-λ calibration | [`t7/`](t7/) | pending |
| 8 | T8 | admission_controller + watermark sensitivity | [`t8/`](t8/) | pending |
| 9 | T9 | Run K + K-a + J ablation (J validates §9 deployment claim w/o HiCache) | [`t9/`](t9/) | pending |
| 10 | T10 | integration / concurrency / restart / GC + forced-fault verifies | [`t10/`](t10/) | pending |

Per-task `tN/results/` directories hold raw outputs (logs, JSON, harbor
result dirs, optimization writeups) for that task only — no cross-task
mixing, no shared `results/` folder.

## Cross-cutting invariants

All verify files share these checks (audit findings made them explicit):

* **No polling anywhere in the daemon code path.** Grep for
  `asyncio.sleep`, `time.sleep`, `loop.call_later` in the policy,
  scheduler, admission, and event_worker modules. Should return zero.
* **No `decide_periodic()` method on `OursGreedyPolicy`.** All call
  sites use `decide(state, event_kind, decision_set)` from the
  pre-existing simulator interface.
* **Shared-aware aggregation in admission scoring.** Verified in T8 via
  a degenerate state where every program shares the same single unit.
* **Inline scorer NOT replaced.** Run K launches sglang with
  `SGLANG_KV_POLICY_MODULE=baselines.sglang_adapter:ours_greedy_score`
  set. Verified in T9 + T10.

## Pre-committed worst-case floor for Run K

Even if the daemon layers misbehave in every documented way (events
lost, state stale, races), the floor is:

* **kv_scheduler degrades** → equivalent to inline-scorer-only → ≈ Run H'
  (885 s/trial, 30/32 successful).
* **admission_controller degrades** → no program-level back-pressure →
  ≈ Run H' / Run F' (873-885 s/trial).
* **Both degrade** → ≈ inline scorer alone → still Run H' baseline (the
  inline scorer is the load-bearing safety net we explicitly kept).
* **inline scorer disabled too** → ≈ Run F' (873 s).

So the absolute floor on Run K is Run F' (873 s). Anywhere below that
means a regression in the inline path, which is independently verified
by T9 / Run H' provenance.
