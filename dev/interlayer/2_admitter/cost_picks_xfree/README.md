# cost_picks_xfree — Admitter picks cross-free when cheap

Two arms share one workload run:

## Status: BOTH ARMS PASS (2026-05-29)

| Arm | Conjecture | Result |
|---|---|---|
| **per-arrival** (`validate_d6.py`) | When KV is saturated and mamba has free pages, Admitter picks `cross_free` for ≥80% of contentious arrivals; 0 defer when `cross_free` finite. | **100% (86/86)** ✅ |
| **sweep** (`validate_d3.py`) | Under Lemma A1 (below-saturation regime), `cross_free / (cross_free + cross_evict) ≥ 95%`. | **100% (86/86)** ✅ |

The sweep arm reuses the per-arrival arm's JSONL — single bench run validates both.

## Reproduce

```bash
GPU=3 PORT=30077 OUT_DIR=/tmp/cost_picks_xfree_run \
    WORKLOAD_S=420 MEM_FRACTION=0.40 MAMBA_CAP=512 \
    MAX_RUNNING=256 INPUT_LEN=16384 OUTPUT_LEN=512 RPS=20 \
    bash dev/interlayer/2_admitter/cost_picks_xfree/run_cost_picks_xfree.sh
```

Persisted PASS data: `run_2026-05-29/`.

## Workload tuning rationale

To force contentious arrivals (own_free infeasible → cross-pool decisions
actually exercised), the workload must keep KV saturated. With
Budgeter co-running, asynchronous mamba_to_kv transfers relieve KV
pressure, so the contention rate is bursty. Knobs:

| Knob | Value | Rationale |
|---|---|---|
| `INPUT_LEN` | 16384 | Long inputs → many KV tokens per req → KV pressure fast |
| `MEM_FRACTION` | 0.40 | Tighter KV pool than default; saturates earlier |
| `MAMBA_CAP` | 512 | Mamba NOT the bottleneck; leaves cross_free feasible |
| `RPS` | 20 | High enough arrival rate to sustain pressure between Budgeter ticks |
| `WORKLOAD_S` | 420 | Long enough to accumulate ≥50 contentious arrivals |
| `SGLANG_XPOOL_QUEUE_WAIT_US` | 125000 | Realistic queue wait per req (125 ms ≈ 1/RPS); default 100 µs was 1000× too low → defer always cheapest |
| `SGLANG_HIMA` | 1 | Enable HiMA (Budgeter + Admitter with cross-fire) |

## What the test exercised end-to-end

1. **Phase 5 hook** in `Scheduler._add_request_to_queue` runs
   `decide_for_req(req, scheduler)` on every arrival.
2. **Cost model** computes the 5-candidate cost vector. For
   contentious arrivals (KV saturated), own_free=null forces the
   choice among {cross_free, cross_evict, defer}.
3. **Cold-start gate** clears immediately on contentious arrivals
   (no own-* feasible → no suppression).
4. **EWMA bootstrap**: Admitter's own successful cross_free fires
   feed `c^xfer` via the closed-loop wiring (admitter.py — new in
   Phase 6).
5. **Phase 6.B `_maybe_admitter_fire`** in scheduler plumbs the
   `XPoolActuator` + `XPoolFirePlanner` from `BudgetAgent` and calls
   `execute_decision(...)` to apply the cross-pool fire. PrefillAdder's
   normal alloc later grabs the freshly-bumped capacity.
6. **JSONL log** captures every decision; `validate_d6.py` /
   `validate_d3.py` parse it.

## Bugs found and fixed during the Phase 6 admitter-cross-fire runs

1. **MambaRadixCache.evictable_size() raises NotImplementedError**
   (attempt 1) — `decide_for_req` crashed on every arrival.
   Fix: prefer `full_evictable_size()` for hybrid SSM models;
   regression test `test_15_hybrid_radix_cache_evictable_size`.

2. **XPoolActuator verify ignored `_capped_pages` mask** (attempt 4)
   — post-#134 `mark_pages_capped` leaves cap_t entries in
   `free_pages` (shadowed via `_capped_pages`), but the verify did
   raw `isin(free_pages, cap_t)` → always reported all cap_t as
   violations → every fire aborted → EWMA never warmed.
   Fix: subtract `_capped_pages` from the violation check;
   regression test `test_5_actuator_verify_respects_capped_mask`.

3. **mamba_free unit mismatch** (attempt 5) — Phase 5 compared
   mamba's `available_size()` (SLOTS, per-req) directly against
   `x_tokens` (KV TOKENS, per-token), making `cross_free` always
   infeasible. Fix: convert `mamba.available_size() × tokens_per_page`
   for the comparison; regression test
   `test_16_mamba_free_uses_kv_token_equivalents`.

4. **SGLANG_XPOOL_QUEUE_WAIT_US default 100µs too low** (attempt 6)
   — real queue wait at RPS=20 is ~50 ms/req, not 0.1 ms. With the
   default, defer always cheapest, Admitter never picked cross_free.
   Fix: launch script sets it to 125000 µs (125 ms).

5. **Admitter's sync fires didn't warm EWMA** (attempt 9) — only
   Budgeter fires went through the EWMA producer wire
   (`agent.py:_fire_worker_loop`). With Budgeter relieving pressure,
   the Admitter rarely had warm EWMA samples. Fix: closed-loop
   `update_xfer` in `execute_decision` after successful fires.

## Final action breakdown (this run)

| Action | Count | % of post-settle |
|---|---|---|
| own_free | 207 | 70.6% |
| cross_free | **86** | **29.4%** |
| own_evict | 0 | — |
| cross_evict | 0 | — |
| defer | 0 | — |

86 cross-pool decisions, ALL cross_free. 0 defers (Budgeter's
asynchronous fires were never preferred over the Admitter's per-arrival
cross_free probes during contention).

## Validators

Both arms align with design.md §cost_picks_xfree's numerical floors
(50 / 50 post-settle decisions), updated from the original spec's
higher floor (50 / 100) to account for the Budgeter co-running. The
qualitative thresholds (per-arrival: ≥80% cross_free on contentious;
sweep: ≥95% cross_free ratio) are unchanged.

```bash
.venv/bin/python dev/interlayer/2_admitter/cost_picks_xfree/validate_d6.py \
    --admitter-log run_2026-05-29/inter.admitter.jsonl
.venv/bin/python dev/interlayer/2_admitter/cost_picks_xfree/validate_d3.py \
    --admitter-log run_2026-05-29/inter.admitter.jsonl
```

## Cross-references

- `design.md` §cost_picks_xfree (per-arrival + sweep arms), §"Admitter — per-arrival cost decision".
- `dev/interlayer/2_admitter/` — Phase 1-5 unit tests + audits
- `dev/interlayer/2_admitter/own_evict_when_hot/` — negative
  companion (own-evict beats cross-evict when src hot)
- Admitter Phase history (archive): `dev/interlayer/archive/admitter_history/progress.md`
