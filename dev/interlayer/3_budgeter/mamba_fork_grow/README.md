# mamba_fork_grow — fork-failure self-heal: grow mamba from KV (Phase 4-b)

## Why

The #312 crash is `assert mamba_value_forked is not None, "Can not alloc mamba
cache"` in `MambaRadixCache.cache_unfinished_req`: a caching fork needs one
mamba slot, the pool is full, and `evict_mamba` finds no UNLOCKED cold cache.
The Budgeter working-set floor (`mamba_drain_floor/`) keeps mamba above this
line — but the floor is a **static** reservation that can't be lowered,
because the dangerous moment (the fork) happens **mid-prefill**, where the
Admitter's **arrival-time** grow has no hook. So the floor's
active+fork+protected reserve was irreducible, and Phase 4's symmetric grow
(2_admitter/) couldn't free the schedulable space it was built for.

## What

Give the fork itself a grow hook. `MambaRadixCache._fork_mamba_with_recovery`
recovers in order:

```
fork → (None) evict cold cache + fork → (None) grow mamba from KV + fork → (None) assert
```

The grow step calls `self._mamba_grow_hook(n_slots)` — a synchronous k2m
cross-pool transfer the Budgeter injects (`BudgetAgent._grow_mamba_from_kv`,
wired in `_ensure_actuator_chain` as `tree_cache._mamba_grow_hook`). It builds
a `kv_to_mamba` FirePlan for `ceil(n_slots / mamba_tokens_per_chunk)` chunks
(rounded to the actuator LCM) and executes it; on grant>0 the fork is retried.

This is **not** the rejected best-effort-skip (we still cache the prefix). It's
"grow and retry". With no hook wired (stock sglang / Budgeter off) the path is
the original evict→assert — **fail-loud, unchanged**.

Crucially, this hook fires at **fork time** (mid-prefill), where the Admitter's
arrival grow can't reach — so it catches exactly the scarcity that crashes
#312, and it lets the Budgeter floor drop (P4.5): bursts now degrade to serial
synchronous grows ("slower"), not a crash.

## VMM safety (why a mid-prefill grow doesn't break the decode CUDA graph)

The grow only `cuMemMap`s physical pages onto **previously-unmapped virtual
addresses beyond the live cap** — VAs the captured graph never referenced — and
never moves or unmaps an in-use slot. CUDA graphs bind to virtual addresses;
VMM keeps those fixed while adding physical underneath. (Contrast: live-
migrating an in-use mamba slot WOULD move a VA the graph reads — that's why
mamba migration is atomic-inert, #269. Growing is purely additive, so safe.)
The KV side only unmaps **free** KV pages (the FirePlan selects them), which the
captured KV graph doesn't read.

## Tests

- `test_fork_grow_recovery.py` (5): the recovery order — free slot → no
  recovery; evict alone suffices → grow not fired; evict empty + grow frees a
  slot → fork succeeds (the #312 case, RED-then-GREEN); no hook → stock assert
  preserved; grow fires but frees nothing → still asserts (no None fork).
- `test_grow_hook_budgeter.py` (4): `_grow_mamba_from_kv` builds `kv_to_mamba`
  of `ceil(need/tps)` chunks, rounds up to the actuator LCM, returns
  True on grant / False on no-chain / abort.

```bash
.venv/bin/python dev/interlayer/3_budgeter/mamba_fork_grow/test_fork_grow_recovery.py
.venv/bin/python dev/interlayer/3_budgeter/mamba_fork_grow/test_grow_hook_budgeter.py
```

## Status / next

Code + unit tests landed. P4.5 (lower the Budgeter floor now that the fork
self-heals) + e2e validation (drain mamba → fork-time scarcity → grow catches
it → no #312 crash, bursts degrade gracefully) is the next step — it needs an
isolated GPU and was gated on the user's go-ahead (the floor-drop was the
audit's C1 risk, now mitigated by this hook).

## Working-set floor (#297, shipped; e2e pending)

The m2k floor is `BudgetAgent._mamba_working_set_floor_slots(m_used, evictable)`
= `(m_used − evictable) + safety_margin` = the LIVE active+protected working set
plus a fixed burst buffer. It does NOT reserve the nominal `max_running` cap.

The earlier #312 floor kept `max_running` because the active-slot alloc path had
no grow hook, so dropping it would open a second crash ("Not enough space for
mamba cache"). That follow-on (the M1 active-slot grow) is now shipped: the
`_mamba_active_grow_hook` (wired into `HybridReqToTokenPool.alloc`) fires a
synchronous k2m grow from idle KV when the live mamba cap is exhausted, so a
burst self-heals instead of being statically reserved. With both the fork grow
and the active-slot grow wired, the floor reserves only the live working set,
freeing the ≈ two-thirds of the pool the static `max_running` cap withheld
(`max_running ≈ pool/3` at ratio 3) — the over-reservation that refused 59/72
m2k fires in the 262k regime. Pinned by `mamba_drain_floor/`
`test_mamba_working_set_floor_297.py`; `test_coupling_bound.py` pins that the
floor tracks the live set, not the cap, at the cc KV-full extreme.

e2e validation pending: everything-on → Budgeter drains mamba to its working set
→ confirm no #312 crash (the grow hooks catch any resulting scarcity) + measure
the perf trade (grow frequency vs latency).

## KV–mamba cache coupling bound (computed via sglang APIs)

`kv_mamba_ratio.py` derives — by calling sglang's own config functions, not by
hand or from the (buggy) boot-log GB strings — how tightly KV and mamba
occupancy are coupled. Every running req and every cached radix node consumes
BOTH pools (1 mamba slot + its KV-token span), cached/evicted as a unit, so
they fill proportionally.

sglang-derived constants (cc config, kv_cache_dtype=auto→bf16, tp=1):
- mamba **51,511,296 B/slot** (`Mamba2CacheParams.mamba_cache_per_req`)
- KV **32,768 B/token** (`DefaultPoolConfigurator` cell-size formula)
- M = 64 mamba slots (max_running capped to `M//ratio` = 21, ratio=3)
- K = 1,827,295 KV tokens (boot-observed — the one capacity not derivable
  config-only; labeled as cross-check input)
- per-slot KV-token ratio `t = KV_tokens/mamba_slots ∈ [1, context_len=262144]`

Bounds:
- **KV full ⟹ mamba occupancy ∈ [10.89 %, 100 %]** — mamba can NOT be driven
  near-empty while KV is full; the floor is `K/context_len ≈ 7` slots (7
  maximal-context active reqs). This proves the "KV full, mamba empty" state is
  unreachable.
- mamba full ⟹ KV ∈ [~0 %, 100 %] (loose at the top because the config-only
  ceiling uses `t_max=context_len`; realized spans are far smaller).
- **mamba is the structurally binding pool**: its 64 slots exhaust ~446× sooner
  than KV even at the minimum 64-token span, so mamba saturates first for any
  realistic per-slot span.

Cross-check: the observed p44_allon point (mamba 0.984, KV 0.64) implies an
average `t ≈ 18,570` tok/slot — inside [1, 262144], exactly the predicted
regime (mamba near-full, KV ~64%). So for cc the Budgeter can treat **mamba as
the lead pressure signal**, and the cross-pool slack KV can lend mamba is
bounded and computable from config.

Run: `.venv/bin/python dev/interlayer/3_budgeter/mamba_fork_grow/kv_mamba_ratio.py`
(builds a 1-rank gloo group on CPU so the production `mamba2_cache_params`
property is callable; no GPU/weights).
