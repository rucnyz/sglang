# HiMA vs upstream sglang: workaround / fallback audit (2026-06-15)

Baseline `3553fd0322` (prelude parent). 8 parallel finders + adversarial per-finding
verify + synthesis. 82 suspects -> 49 flagged -> **20 confirmed design-absent
workarounds**, 34 legit (false positives), 2 needs-human. Confirms the suspicion: most
are debug/bisection scaffolding from closed bug hunts (#319/#320/D8) that leaked into
production hot paths, plus defensive `getattr` against our own guaranteed types.

## Family A: debug / bisection kill-switches leaked into prod (env-gated)
- **`xpool_actuator.py` SGLANG_XPOOL_AUDIT_SKIP_VERIFY / SKIP_MARK** [HIGH]: production-
  reachable switches that DISABLE the cap-barrier mark + leak verify (the #320 atomicity
  safety). Make both unconditional; delete the switches + the __init__ warning loop.
- **`agent.py` SGLANG_DISABLE_ADMISSION_GROW** [HIGH], **SGLANG_XPOOL_WORKER_NOOP** [HIGH]:
  bisection kill-switches (self-labeled) that disable designed unconditional behavior /
  neuter every fire. Delete.
- **SGLANG_DEBUG_ALLOC_FAIL probe blocks** [MED/LOW] in `allocator.py` ([MARK_CAP]/[SETCAP]),
  `common.py` ([EVICT_PROBE]/[ALLOC_FAIL]), `xpool_actuator.py` ([M2K_ACCT], try/except:pass),
  `agent.py` (#319/#320 localizers + SGLANG_DEBUG_FIRE_BUCKETS). The README itself says
  "strip before commit". Delete; keep the real `_kv_grow_hook` on-demand-grow + OOM raise.
- **`agent.py` SGLANG_HIMA_TICK_PROBE** [LOW] (30ms magic in tick() hot path),
  **SGLANG_HIMA_PEAK_DECAY** [LOW] (write-only peak EMA, no consumer),
  **`mamba_radix_cache.py` SGLANG_RADIX_DEBUG** [LOW] (per-call INFO in match/insert). Delete.

## Family B: defensive getattr / hasattr on our OWN guaranteed types
- **`agent.py` _grow_mamba_from_kv / _grow_kv_from_mamba** + **`admitter.py` execute_decision**
  [MED]: `getattr(result, 'aborted'/'granted_pages', default)` against the FirePlanResult
  dataclass. Use direct `result.aborted` / `result.granted_pages` (fail loud on a rename).
- **`agent.py` _maybe_fire usage_kv_active** [MED]: `hasattr`+`try/except` around the
  base-class-guaranteed `full_evictable_size()`; on any error it silently zeroes a
  LOAD-BEARING fire-gating signal (collapses usage_active->usage_inst, the phantom-saturation
  case the signal exists to prevent). Use direct `int(tc.full_evictable_size())`.

## Family C: dead / design-superseded code (zero callers, contradict byte-exact design)
- **`radix_cache.py` evict_pages_in_range** [HIGH] (page-id-range drain; design rules out;
  docstring admits "collateral cache loss"; cites nonexistent T9/§3.2.5). Delete.
- **`mamba_radix_cache.py` evict_mamba_above_slot** [MED] (slot-id-threshold eviction
  overriding sglang order; except:continue). Delete.
- **`mamba_radix_cache.py` estimate_v_prefix_marginal + _lpb_pick_mamba_eviction** [MED]
  (dead island citing nonexistent Eq:vprefix-est). Delete.
- **`xpool_planner.py` qdepth_trigger sub-block + level-triggered watermark fires** [MED]
  (pre-NB direction policy, qdepth block 100% dead) + **both_above_max/min_gap magic
  constants** [MED] (anti-oscillation knobs in a branch the shipped config never runs).
  Retire so arg-max-NB is the sole path. (edge_trigger left as a separate question.)
- **K_big** (`mamba_radix_cache.py insert`/_insert_helper) [HIGH]: ALREADY removed in the
  working tree (see [[project_kbig_suppression_frees_live_kv]]); commit it.

## Family D: one CORRECTNESS bug (not just hygiene)
- **`scheduler.py` _maybe_admitter_fire `tokens_per_page=1024`** [HIGH]: design-absent magic
  constant. On the dominant m2k path `n_pages_needed = ceil(x_tokens / tokens_per_page)`, so a
  wrong 1024 (real granularity `owner_provider.kv_tokens_per_page()`, e.g. 4096) mis-sizes the
  cross-pool KV grow by up to 4x. Replace with the owner_provider value already wired into
  agent.py. NOTE: likely relevant to the agentreplay m2k "no-win" (grow mis-sized).

## NEEDS HUMAN (2; both cost-model behavior, optimality not correctness)
1. **Admitter cold-start gate** (`admitter.py` decide: `not warmed and own_alternative_feasible`):
   exists only in the consolidated-away archive design (dangling "design.md §356"); current
   design.md prescribes a different cold-start (conservative 3000us c^xfer through min-cost +
   EWMA drift). (A) promote the gate back into design.md, or (B) simplify to the cost-driven
   mechanism (drop the hard gate).
2. **`common.py` record_recovery_len_retract**: its output is SILENTLY DEAD - agent._snapshot
   never emits `slow_recovery_len_retract`, so pressure_adapter reads 0.0 and the paper-
   calibrated L-aware retract cost (k_retract=75ms) never engages (falls back to flat scalar).
   Confirm: make it live (rewire snapshot emission + regression assert)? And move the
   `_slow_recovery_len_*_ewma` / `_cumulative_evicted_*` counters to unconditional __init__
   (drop the lazy getattr/hasattr scaffolding)?

## Cleared as legit (34, NOT workarounds) — notable
CappedFreeList (whole module); c_M=0 / L_star=0 / c_m terms (the single-curve design's
EXPECTED calibrated zeros, misread as dead scaffolding); marginal-fire cap `marg` (#321);
mamba working-set floor; cold-start 3000us default; cap_barrier free-only-cap strand guard;
sigterm handler; arena .so build; model_runner TLB warmup; SGLANG_ARENA_SHARED setdefault;
scheduler server-availability catches (logger.exception, not swallowed).
