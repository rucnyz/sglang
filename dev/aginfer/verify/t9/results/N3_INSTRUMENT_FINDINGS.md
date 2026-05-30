# T9 — daemon instrumentation results (OURS_full single cycle)

First production-cycle run of the structured-metric instrumentation
(commit ~528962+).  Confirms — with hard counters — what the
4-arm matrix could only suspect: **the daemon's three layers are
structurally inactive on this workload**, in two distinct ways.

## Setup

* 2026-05-29 23:33 → 2026-05-30 00:26 (52 min)
* GPUs 0, 1; TP=2 EP=2; HiCache + Mooncake L3
* Variant: `full` (kv_scheduler ON + admission ON + HiCache ON)
* Harbor: `-l 32 -n 32 -k 1 --ak temperature=0.0 seed=42 max_turns=200`
* Per-trial mean: 1369 s (close to N=3 OURS_full ≈ 1344 ± 55 s)

## Headline numbers (from `daemon.log`'s `cycle_summary`)

```
events_received   = 18 039
events_handled    = 18 039
handler_failures  = 0
kv_decisions      = 12 017       (decide() ran 12k times)
kv_migrate_calls  = 0            (← ZERO migrate POSTs dispatched)
adm_pauses        = 0            (← ZERO programs paused)
adm_resumes       = 0
```

### Event-flow breakdown

| EventKind | count | share |
|---|---|---|
| llm_prefill | 6019 | 33.4 % |
| tool_call_start | 6019 | 33.4 % |
| tool_call_end | 5971 | 33.1 % |
| session_arrival | 30 | 0.2 % |
| **memory_pressure** | **0** | **0.0 %** |
| **pressure_resolved** | **0** | **0.0 %** |

→ sglang **never fired** `memory_pressure` over the whole cycle.

### kv_decide outcome breakdown

| outcome | count | share |
|---|---|---|
| policy_declined | 12 017 | 66.6 % |
| empty_decision_set | 6 022 | 33.4 % |
| **dispatched** | **0** | **0.0 %** |

`empty_decision_set` exactly matches the 6019 `llm_prefill` events
(plus 3 extras for boundary conditions) — per paper §4 table,
`llm_prefill` has an empty D_t by design.

The remaining 12 017 events DO build a non-empty D_t (tool_call
events) but `policy.decide()` returns empty `action.assignments`
on every single one of them.

## Why every decision declines: HBM pressure never materialises

### HBM occupancy time series (G5)

```
samples = 18 039
min = 0.000   max = 0.000   p50 = 0.000   final = 0.000
samples ≥ 0.85 (admission would act):  0  (0.0 %)
samples ≥ 0.70 (sglang fires pressure):  0  (0.0 %)
```

`occ_hbm` reported by daemon is **literally 0** for every sample.

### Cross-check vs sglang's own batch log

Sample from `sglang_v4flash.log`:
```
Prefill batch, ... full token usage: 0.02, swa token usage: 0.00, ...
Decode batch,  ... full token usage: 0.01, swa token usage: 0.00, ...
```

So sglang's **own** KV pool allocator says HBM is **1-2 %** used.
The daemon sees 0 %.

### Root cause of the daemon ↔ sglang divergence

`dump_aginfer_state` (sglang side, `unified_radix_cache.py`)
computes `hbm_used` by **walking the radix tree** and summing
`len(cd.value)` for nodes whose value tensor is non-empty.

But the radix tree only contains **prefix-shareable, committed**
nodes — in-flight decode allocations are NOT in the tree until
the request completes.  Under the current workload (1 % runaway
60 k-token decodes, see `N3_ROOT_CAUSE.md`), **most live KV is
in-flight decode**, not committed prefix.  The two views diverge:

| metric | source | reading |
|---|---|---|
| sglang's `full token usage` | KV pool allocator counter | 1-2 % |
| daemon's `tier_usage.HBM.used_bytes` | radix-tree walk | ~0 % |

→ Daemon believes there's nothing in HBM → V_u computes:
   `r1_saved_prefill = p_hat × Δr(τ, DROP)` ≈ 0 because there's
   no unit to migrate.  Policy declines every event.

## Three failure modes confirmed (or ruled out)

### (b) "Mechanism never triggers" — confirmed for memory_pressure / admission

* `memory_pressure` events: **0** (sglang occ never crossed 0.7)
* `admission_pause` calls: **0**

The admission_controller code path is **never exercised at all**
on this workload.  Even if its theta were aligned (G9) or its V_u
were perfect, it can't do anything because it never gets called.

### (c) "Mechanism fires but doesn't help" — confirmed for kv_scheduler

* `kv_decide` called 18 039 times (= 1 per event)
* All 18 039 outcomes are `policy_declined` (66.6 %) or
  `empty_decision_set` (33.4 %)
* `migrate_post` dispatched: **0**

kv_scheduler's mechanism fires (decide() runs) but always returns
empty action — the V_u math says no migration is worth doing
because daemon's view of HBM is empty.

### (a) "Workload doesn't need it" — also confirmed structurally

The combination of (b) + (c) implies (a): the workload doesn't
generate the pre-conditions either layer needs to do work.

## Implications

* **No matrix re-run will change this picture** as long as
  `temperature=0.0 seed=42` + default KV pool + HiCache + Mooncake
  remain the regime.  The daemon's three layers are simply not
  exercised.
* **The 8.7 % OURS_full-vs-LRU spread in the 4-arm matrix is
  attributable entirely to the inline `ours_greedy_score` scorer
  + GPU-pair noise** — not to anything daemon does, because the
  daemon does nothing.
* **The "fix" is the workload, not the daemon** — per
  `N3_GAPS.md` §4:
  1. cap `max_completion_tokens` (so runaways don't dominate)
  2. shrink `--max-total-tokens` (so HBM crosses theta_hi)
  3. (re-)disable HiCache (Run J — chained, pending) so eviction
     goes to DROP, forcing the migration story

## K-a + Run J answer (chain run 2026-05-30)

Both chained cycles produce the **same null result** as OURS_full:

| metric | OURS_full | K-a | Run J |
|---|---|---|---|
| events_received | 18 039 | 17 684 | 18 043 |
| kv_decisions | 12 017 | 11 782 | 12 020 |
| **migrate_post** | **0** | **0** | **0** |
| **adm_pauses** | **0** | **0** | **0** |
| HBM occ (daemon view, all samples) | 0.000 | 0.000 | 0.000 |
| sglang peak `full token usage` | 0.02 | 0.02 | **0.07** |
| per-trial mean (s) | ~1369 | similar | 1351 |

### What this rules out

* **K-a vs OURS_full**: admission_controller's marginal
  contribution is **exactly 0**, because admission never even
  ran in OURS_full (pause_count = 0).  Disabling something that
  was already a no-op changes nothing.  Confirmed by identical
  daemon decision breakdowns.
* **Run J HiCache-OFF hypothesis**: turning off the DRAM tier
  did NOT push HBM into the pressure regime.  Peak only rises
  from 2 % → 7 %; still nowhere near the 70 % sglang-fires
  threshold.

### Why Run J still couldn't pressure HBM

The default `--max-total-tokens` is ~10 M tokens.  Even with
HiCache OFF (no DRAM spill), the workload's 32-trial concurrent
KV footprint is far smaller than the pool.  Per
`N3_GAPS.md` §3 / §4 the right fix is **both**:

1. `--ak max_completion_tokens=4096` (drop the runaway tail that
   eats 80 % of LLM time)
2. `--max-total-tokens 256K-512K` (shrink pool so HBM occ can
   reach > 0.7)

Either alone is not enough — runaway dominates wall time so
mean wouldn't move; pool-shrink alone leaves runaway-decode
free to OOM the small pool.

## Side bug surfaced (G_NEW)

The radix-tree-walk view of `hbm_used` is a **load-bearing
divergence** from the actual KV pool occupancy.  Until the daemon
sees the real pressure, no scheduling decision will fire.
Options to fix:

1. **Patch sglang's `dump_aginfer_state`** to ALSO report the
   allocator-level counters (`token_to_kv_pool_allocator.used`
   etc.) alongside the radix-tree walk, in a parallel
   `pool_usage` field.  Daemon then keys admission on
   `pool_usage` not `tier_usage`.
2. **Move admission's pressure trigger from sglang's webhook**
   (already keyed on the allocator counter at 0.7) to be the
   sole source of truth — and the daemon should NOT additionally
   check its own `tier_usage` calc (currently it does in
   `_on_pressure` to decide whether to pause).  This makes
   `tier_usage` a "what's in cache" view (correct for V_u
   migration source/target selection) while `pool_usage` becomes
   the "is the pool full" view (correct for admission gating).

This is a real design gap worth its own G-number.  Suggest **G10**
in `N3_GAPS.md`.

## Files

* daemon log: `results/run_K_full_instrument_20260529_233332/daemon.log`
* sglang log: same dir, `sglang_v4flash.log`
* parser:    `verify/t9/parse_daemon_events.py`
* methodology: `verify/t9/methodology.md`
* cross-T:   `verify/t9/results/N3_GAPS.md`
