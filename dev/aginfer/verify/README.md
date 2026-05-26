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
| 1 | T1 | `GET /aginfer/state` | [`t1/`](t1/) | **passed** (2026-05-25 perf opt; 2026-05-26 bytes-schema rewrite; current head `2416701fa1`) |
| 2 | T2 | `POST /aginfer/migrate` | [`t2/`](t2/) | **passed** (2026-05-25 initial; 2026-05-26 depth+round-3 audits, 21 steps) |
| 3 | T3 | session_id passthrough into tree nodes | [`t3/`](t3/) | **passed** (2026-05-26, audit round-5 done, 14 verify steps + 3-section regression probe) |
| **4** | **T6** | program_tracker state machine (moved up — T4 depends on it) | [`t6/`](t6/) | **passed** (2026-05-26, audit round-1 done, 10 verify steps in ~60 ms) |
| 5 | T4 | daemon HTTP proxy + paper §4 event emission | [`t4/`](t4/) | **passed** (2026-05-26, audit round-1 done, 12 verify steps + 5-run latency, ~370 LoC) |
| 6 | T5 | sglang→daemon webhook (transition + 5 s plateau heartbeat) + daemon event router | [`t5/`](t5/) | **passed** (2026-05-26, audit round-1 done; Layer A 10 steps + Layer B real-GPU watermark test) |
| 7 | T7 | kv_scheduler event handlers + ACTING-λ calibration | [`t7/`](t7/) | **passed** (2026-05-26, 13 verify steps + 22 bisect probes across 5 audit rounds; ~510 LoC daemon code; build 2.7 ms / decide 1.4 ms @ 1k units) |
| 8 | T8 | admission_controller (event-driven pause/resume) | [`t8/`](t8/) | **passed** (2026-05-26, 12 verify steps + 5 bisect probes across 2 audit rounds; ~315 LoC daemon code; single-pause 4 ms @ 32 programs) |
| 9 | T9 | Run K + K-a + J ablation | [`t9/`](t9/) | **K full + K-a: both ~1550 s mean, FAIL <716 s target; admission_controller NOT the cause; kv_off (diagnostic) + J pending** (2026-05-26) |
| 10 | T10 | integration / concurrency / restart / GC + forced-fault verifies + daemon-controlled L3 (DISK) tier via Mooncake (paper §3 4-tier completion) | [`t10/`](t10/) | pending |
| **11** | **T11** | **empirical p_hat / scoring (replace paper §7 hits/age proxy)** | [`t11/`](t11/) | **scoped 2026-05-26; T9 K-full+K-a empirical evidence that §7 1-step greedy V_u is wrong for multi-turn agent workloads; T11a trace harvest → T11b estimator → T11c re-run K** |

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
