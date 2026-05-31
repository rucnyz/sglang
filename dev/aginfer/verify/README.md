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
| 1 | T17 | state-dump schema upgrade (DESIGN §5) — was `t1` | [`t17/`](t17/) | **passed** (2026-05-31, all 9 stages + post-audit hardening; aggregate p99 @ 5K nodes = 28.5 ms) |
| 2 | T20 | `POST /aginfer/migrate` (residence-set) — was `t2` | [`t2/`](t2/) | _pending refresh — slated for #134_ |
| 3 | (infra) | session_id passthrough into tree nodes | [`session_id_passthrough/`](session_id_passthrough/) | **passed** (2026-05-31 post-T17 schema; 14 verify steps + 3-section regression probe) |
| 4 | (infra) | program_tracker state machine | [`program_tracker/`](program_tracker/) | **passed** (2026-05-31 post-T33; 10 verify steps in 213 ms) |
| 5 | (infra) | daemon proxy + paper §4 event emission | [`daemon_proxy_events/`](daemon_proxy_events/) | **passed** (2026-05-31 post-T33; 12 steps + 4 recovery probes; proxy overhead p50=1.49 ± 0.01 ms) |
| 6 | (infra) | sglang→daemon webhook + daemon event router | [`daemon_webhook_router/`](daemon_webhook_router/) | **passed** (2026-05-31 post-T33; Layer A 11 steps; arrival→handler p50=0.80 ± 0.01 ms) |
| 7 | (infra) | kv_scheduler value rule | [`kv_scheduler_value_rule/`](kv_scheduler_value_rule/) | ⚠️ **STALE post-T33** — pre-round-9 API; full rewrite tracked as #146 |
| 8 | (infra) | admission_controller | [`admission_controller/`](admission_controller/) | ⚠️ **STALE post-T33** — pre-round-9 API; rewrite tracked as #146 (shared with kv_scheduler_value_rule) |
| 9 | (moved) | Run K + K-a + J ablation | see [`scenarios/`](../scenarios/) | moved 2026-05-31; old verify/t9 README deleted (was redirect-only + dead links to N3_GAPS.md) |
| 10 | (pending) | integration stress (6 stress flavors A–F) | — | not yet started; tracked as #147 (B partially covered by `verify/t17/` Stage 7) |
| 11 | T11 | empirical p_hat estimator (PLAN §1) | [`t11/`](t11/) | OPEN WORK — calibration task, no verify.py; deliverable lands as a model-selection report in `t11/results/`; tracked under #126 |

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

> ⚠️ **2026-05-29: floor numbers below predate the setting-drift
> discovery and are not directly comparable to current matrix
> results.**  Historical Run F' 873 s and Run H' 885 s were
> measured under sglang default sampling.  Current matrix /
> H'_now runs use `temperature=0.0 seed=42`, which deterministically
> triggers runaway generation for ~1% of LLM requests; those
> outliers dominate trial wall time and the historical floors
> don't transfer.
>
> Empirical floor under current settings is the N=3 baseline
> `1389.3 ± 39.7 s` (kv_off, inline scorer only).  Any
> regression below that *under temperature=0.0 settings* would
> be a real signal.  See `verify/t9/results/N3_matrix_SUMMARY.md`.

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
