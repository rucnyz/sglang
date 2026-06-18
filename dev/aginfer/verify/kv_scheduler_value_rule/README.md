# kv_scheduler_value_rule (post-T33 rewrite, #146)

Contract pin for `daemon/kv_scheduler.py` under the post-T17/T33
schema.  The legacy verify (3300 LoC `verify_pre_t33.py` +
`regression_probe_pre_t33.py`) lives in `legacy/` as a behavioural-
spec reference — it does NOT pass against current code because the
schema changed under it.

## What changed at T33

| surface | pre-T33 | post-T33 |
|---|---|---|
| `_value` arg | `tier: Tier` (single target) | `next_residence: List[Tier]` (residence-set) |
| `Action.assignments` element | `(unit_id, Tier)` 2-tuple | `(unit_id, add_tiers, remove_tiers)` 3-tuple |
| `state.pool_usage` | flat dict `tier_usage[tier]` | nested `pool_usage[tier].subpools[sp]` |
| migrate POST body item | `{hash, target_tier}` | `{hash, add_tiers, remove_tiers, action_id}` |

## SCOPE

What this verify pins (NOT covered by sibling verifies):

* `build_paper_state`: post-T17 schema → `SchedulerState`, including
  multi-rank flatten + unknown-tier skip + positivity invariants
* `_build_decision_set`: paper §4 D_t per EventKind (6 kinds + memory_pressure)
* Program-aware λ + p_hat rules (ACTING-floor clamping, PAUSED inherits, alive-holder p_hat=1.0)
* `_top_k_by_regret`: ascending keep-value (lowest first → best demote candidates)
* `Action.assignments` 3-tuple shape contract
* `assignments_to_wire` 4-key envelope (`hash`, `add_tiers`, `remove_tiers`, `action_id`)
* `KvScheduler._dispatch_migrate` post-T36 outbound-only path
* Robustness: state-fetch raises / empty D_t / policy declines
* Idempotence: same state → same Action
* Latency: `decide(1k units)` mean+3σ < 25 ms

NOT covered here (lives elsewhere):

| concern | lives in |
|---|---|
| state-dump schema itself (post-T17) | `verify/t17/` |
| migrate POST wire compatibility w/ sglang | `verify/t20/` |
| outbound queue + worker mechanics | `verify/t36/` |
| event-router wiring | `verify/t5/` |
| program_tracker state machine | `verify/t6/` |

## STAGES (19)

```
A.  Schema adapter
  A0 pool_usage post-T17 nested schema → TierUsage
  A1 multi-rank per_rank flatten (sum used/cap, dedupe units)
  A2 unknown tier label skipped, logged once
  A3 missing required state field → fatal() (subprocess exit 1)
  A4 partial-zero h_max → fatal() (operator misconfig)
B.  Paper §4 decision-set per EventKind
  B0 6 kinds × correct D_t (SESSION_ARRIVAL → shared; LLM_PREFILL → ∅;
     TOOL_CALL_START/END → caller tail; SUB_DISPATCH_BLOCKING →
     tail+shared; SUB_DISPATCH_ASYNC → shared)
  B1 MEMORY_PRESSURE top-k by ascending regret (10 sentinels among
     100 → all 10 in top-20)
C.  Program-aware λ / p_hat
  C0 ACTING → λ clamped to [1/30, 1/1] (both saturating boundaries)
  C1 PAUSED also inherits ACTING-floor (R2-M1)
  C2 alive holder → p_hat=1.0; all-ENDED → hits/age proxy
D.  Action / dispatch (post-T33)
  D0 Action.assignments is List[Tuple[str, List[Tier], List[Tier]]];
     add/remove disjoint
  D1 assignments_to_wire 4-key envelope, action_id unique per item
  D2 _dispatch_migrate without outbound → RuntimeError (no silent drop)
  D3 _dispatch_migrate routes through OutboundQueue.enqueue_migrate
E.  Robustness
  E0 /aginfer/state fetch raises → handler logs + bows; no migrate
  E1 LLM_PREFILL (empty D_t) → no decide(), no migrate
  E2 policy declines (empty Action) → no migrate enqueued
F.  Idempotence + latency
  F0 same state → same Action across 3 repeated decides
  F1 decide(1k units) mean+3σ < 25 ms (5 trials)
```

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched-rebase
cd /scratch/yuzhou/projects/sglang
python dev/aginfer/verify/kv_scheduler_value_rule/verify.py
```

Runs in ~10 s on the agsched-rebase env (A3/A4 each spawn a subprocess to
exercise `fatal()` → `os._exit(1)`).

## RESULTS

**PASSED** — all 19 stages.

* date: 2026-06-01
* implementation: pure rewrite (`verify.py` ~700 LoC); legacy
  preserved at `legacy/verify_pre_t33.py` + `legacy/
  regression_probe_pre_t33.py` for behavioural-spec reference
* raw log: `results/20260601_post_t33_rewrite_pass.log`

| Stage | Result |
|---|---|
| A0 pool_usage schema → TierUsage | PASS |
| A1 multi-rank flatten | PASS |
| A2 unknown tier skipped | PASS |
| A3 missing field → fatal | PASS — subprocess exit 1, reason `missing_state_field` |
| A4 partial-zero h_max → fatal | PASS — reason `holding_cost_non_positive` |
| B0 paper §4 D_t (6 kinds) | PASS |
| B1 memory_pressure top-k by regret | PASS — all 10 sentinels in top-20 |
| C0 ACTING λ clamping | PASS — both ceil (1.0) and floor (1/30) saturate |
| C1 PAUSED λ-floor | PASS |
| C2 alive vs ended p_hat | PASS |
| D0 Action.assignments 3-tuple | PASS |
| D1 assignments_to_wire envelope | PASS — action_id unique per item |
| D2 dispatch without outbound → RuntimeError | PASS |
| D3 dispatch via OutboundQueue | PASS — migrate_calls=1, queue depth=1 |
| E0 fetch raises → no migrate | PASS |
| E1 LLM_PREFILL empty D_t | PASS — decisions=0, migrate_calls=0 |
| E2 policy declines | PASS |
| F0 idempotence | PASS — 3 identical Actions |
| F1 latency at 1k units | PASS — well under 25 ms mean+3σ |

## CALIBRATION NOTES

* λ_ACTING default 0.2 (= 1/5 s, terminus-2 mean tool duration).
  Clamped to `[1/30, 1/1]` per audit #15 sensitivity sweep.
* MEMORY_PRESSURE top-k = 256 (env-overridable
  `AGINFER_MEMORY_PRESSURE_TOPK`).
* `_top_k_by_regret` returns ASCENDING keep-value (lowest first
  → best demote candidates).  DO NOT "fix" to `items[-k:]` —
  that inverts the policy.

## LATENCY BUDGET HISTORY

| version | budget | actual mean | rationale |
|---|---|---|---|
| pre-T33 (legacy) | mean+3σ < 5 ms at 1k units | ~1.4 ms | 4 target tiers per unit |
| post-T33 (this) | mean+3σ < 25 ms at 1k units | TBD (well under) | 6 transitions per unit per §7 |
| post-T34 sparse DP (future) | target < 10 ms | — | sparsify candidates per program |
