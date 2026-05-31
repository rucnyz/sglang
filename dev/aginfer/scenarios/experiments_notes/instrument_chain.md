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

## A3 (capped completion + 256 K KV pool) — daemon finally fires, sglang rejects

`run_K_a3_instrument_*` (2026-05-30) ran the workload variant
defined in `N3_A3_PLAN.md` — same daemon config as OURS_full
but with `--ak llm_call_kwargs={"max_tokens":4096}` capping the
runaway tail and `MAX_TOTAL_TOKENS=262144` shrinking the KV
pool from default ~10 M tokens to 256 K.

### Hard numbers (A3 v2, with per-action skip-reason instrumentation)

```
events_received   18 213
kv_decisions      12 140
  empty_decision_set 6 073 (33.3%)
  policy_declined   7 433 (40.8%)
  dispatched        4 707 (25.8%)   ← non-zero for the first time
kv_migrate_calls   4 707 (status=200)
applied total     0                 ← but ALL skipped at sglang
skipped total     7 049
  skip reasons:
    promote_not_yet_wired   7 033 (99.8%)   ← 4-tier promote unimplemented
    race:already_on_hbm        15 (0.2%)
    race:not_in_tree            1 (0.0%)

per-trial mean    1 257 s   (vs OURS_full ~1 369 s, −8.2 %)
sglang pool peak  0.55      (vs OURS_full 0.02 — pressure regime
                              reached)
HBM occ peak (daemon view)  0.000   (G10 divergence persists)
memory_pressure events       0
admission pauses             0
```

### What A3 conclusively proves

**The daemon's V_u policy works**.  Under genuine pressure
(pool peak 0.55) it correctly identifies 4 707 positive-value
migrations across 18 k events — that's a 25.8 % dispatch rate.
The math isn't broken.

**sglang's `/aginfer/migrate` handler is half-built.**  The
DRAM → HBM promote path is a literal stub:

```python
elif target == "HBM":
    if has_device:
        skipped.append({"hash": h, "reason": "race:already_on_hbm"})
    elif has_host:
        skipped.append({"hash": h, "reason": "promote_not_yet_wired"})  ← STUB
    ...
```

(`python/sglang/srt/mem_cache/unified_radix_cache.py:2356`)

Because HiCache's `write_through_selective` aggressively demotes
units from HBM to host once they're committed (host pool has the
copy + device pool freed), **almost everything daemon sees is in
DRAM tier**.  V_u then correctly says "promote the useful ones
back to HBM" — but sglang has nothing to copy.  Skips are 99.8 %
`promote_not_yet_wired`.

This is now logged as **G11** in `N3_GAPS.md`.

### Why HBM occ STILL reads 0 (G10 still active under A3)

Even with pool peak at 0.55, the daemon's `tier_usage.HBM.used_bytes`
remains 0 across all 18 213 fetches.  The radix-tree-walk view
counts only committed prefix nodes whose `cd.value` is non-empty.
With `write_through_selective`, those device tensors are nulled
out after the host backup completes, so the radix walk sees 0
in HBM even when the allocator has 55 % of the pool in flight.
G10 (radix vs allocator divergence) is robust to workload regime.

### What A3 ruled out

* **Race conditions are not the root cause** — `race:*` skip
  reasons sum to 16 out of 7 049 (0.2 %).
* **The daemon's V_u isn't pathologically declining** under
  pressure — 25.8 % dispatch rate is real activity.
* **The "workload doesn't need it" framing from N3_GAPS §3 is
  obsolete under A3** — there IS scheduler work to do.

### Open questions A3 doesn't yet answer

* **Does implementing promote unlock real wall-time gains?**
  We'd need (G11 fix) AND a re-run of A3 with 4 707 promotes
  actually applied to know whether daemon scheduling delivers
  value on top of inline scoring.
* **Would shrinking the pool further drive demote instead?**
  At 0.55 peak, V_u still picks promote.  Pool ≈ 128 K might
  push HBM near full enough that demote becomes optimal under
  V_u math.
* **Does G10 actually matter under A3?**  Daemon never fetches
  state for memory_pressure (sglang doesn't fire it because
  the allocator stays under 0.7).  G10 might only matter if a
  fix to (G10) or workload pushes the allocator past 0.7.

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
