# S2 — Shared-prefix retention under scratch churn

## Distinctive driver

A **fleet of agents shares one identical system prefix** (system prompt + tool defs).
Each does divergent scratch work, so per-program scratch churn ≫ the rate at which any
one program re-touches the shared prefix. Under memory pressure, **LRU ages the shared
prefix down its recency order and drops it** — then every program that reuses it eats a
full recompute. DESIGN §2 fact 1: *"a shared prefix used by N programs ≫ a stale scratch;
LRU gets it backwards."* The distinctive lever is the **holder-count term** of the value
rule `V_u` (a unit shared by N programs is worth N× keeping).

## Honest result: opportunity real, holder-count fixed, but NO clean win on this stack

> **This is not a clean win. An earlier draft of this README claimed a 20→14 win; that
> was an N=1 artifact (LRU is high-variance on the weak workload). Rigorous N=3 below
> shows the holder-count helps only a small, regime-narrow amount, for reasons that are
> themselves design-revising evidence.** The real, durable outcomes of this work are
> **two genuine `补齐design` bug fixes** (the holder-count term was completely broken) and
> a precise characterization of why the clean win is blocked.

**Falsification passed** — the opportunity is real: under engineered churn LRU does drop
the shared prefix (e.g. strong workload below: B recomputes it 64/64, every fleet request).

**Two design-completion bugs found + fixed (the holder-count was inert / harmful):**
1. **Storage-layer drop** — `set_aginfer_hints` (sglang) stored only `{p_hat, lambda,
   stamp}`, **discarding `n_holders`**, so the daemon's holder count never reached the
   eviction scorer. The holder-count boost was *completely inert*. (Fixed: store + read it.)
2. **Value-formula gap** — the holder count scaled only the saved-prefill (×N), not the
   *effective reuse rate*. A big shared prefix's occupancy tax (∝ n_bytes) then dominated
   its V_u and the boost **backfired** (17.7 recomputes > LRU 13.7). Fixed: the holder
   count also shortens the hold interval (effective λ ← N·λ), per the design's intent.

Wiring: `ReuseUnit.n_holders` (base.py); `_value` save×N **and** hold_time=1/(N·λ)
(ours_greedy.py); daemon pushes `n_holders` (kv_scheduler.py); `hint_v_u` + storage read it
(sglang_adapter.py / unified_radix_cache.py:set_aginfer_hints). Plus `AGINFER_DISABLE_PROMOTE`
to isolate the eviction lever from S1's promote.

**Rigorous N=3 (after both fixes, promote off — the clean eviction-lever test):**

| regime | LRU | ours (holder-count) | verdict |
|---|---|---|---|
| **strong** pool 49152 | 64/64, std 0 | **64/64, std 0** | **TIE** — no room to keep the 24K prefix |
| **moderate** pool 65536 | 13.7 ± 3.8 | 11 (N=1) | small ~20%, **within noise** |

## Why the clean win is blocked (the design-revising evidence)

There is a **fundamental tension**: a strong, consistent LRU-drop (clean baseline) requires
pressure that *also denies ours the room* to keep the 24K prefix — at pool 49152 keeping it
would starve the *active, locked* KV set, so it is evicted by ours too (tie 64/64). Where
ours *can* keep it (pool 65536), LRU mostly keeps it too, so the divergence is weak/noisy.
And critically, **the V4-Flash multi-tier store is non-functional** (see S1 / FLEET_FINDINGS):
an evicted shared prefix is *DROPPED → full recompute*, not cheaply reloaded from DRAM —
which is the cost the design's holder-count value rule was meant to optimize. **So the
holder-count lever can't deliver its designed magnitude without a working tier:** with a
live DRAM tier the kept-vs-evicted difference is a cheap load-back the value rule can
profitably trade; without it, retention competes one-for-one with the live working set.

## Reproduce

```bash
export AGINFER_ROOT=/path/to/sglang/dev/aginfer ; conda activate agsched-rebase
cd $AGINFER_ROOT/scenarios/replay
python /path/.../s2-shared-prefix-retention/traces/s2_gen.py s2_churn_strong.jsonl 8 8 60 10 16000 4000 14000
T=/path/.../s2-shared-prefix-retention/traces/s2_churn_strong.jsonl
bash run_replay_pressured.sh "$T" 3 1 49152 "a3_kvoff" 1 "lru_score"                       # B: 64/64
AGINFER_DISABLE_PROMOTE=1 bash run_replay_pressured.sh "$T" 3 1 49152 "a3" 1 "aginfer:hint_v_u"  # ours: 64/64 (tie)
python /path/.../s2-shared-prefix-retention/s2_analyze.py <results_dir> a3|a3_kvoff 24000
```

## Bottom line

The holder-count value rule is now **correctly implemented** (two real bugs fixed) and
**functional** (it edges below LRU when there's headroom), but it does **not** produce a
clean S2 win on V4-Flash because (a) a big shared prefix competes one-for-one with the
locked active set under pressure, and (b) the dead multi-tier store turns every eviction
into a full recompute rather than a cheap DRAM reload. **The prerequisite for S2's designed
win is the same working multi-tier store S1 identified as missing.**
