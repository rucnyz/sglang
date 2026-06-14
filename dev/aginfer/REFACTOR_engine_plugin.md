# Refactor: aginfer daemon → in-engine pluggable scheduler (clean layering)

**Status:** PLAN (feasibility confirmed; nothing implemented yet — for review)
**Date:** 2026-06-13

## 0. Principle (the standard this plan is held to)

Per-unit KV tiering is an **engine** concern — the engine owns the KV bytes, the tiers
(HBM/DRAM/DISK), the radix state, and the eviction callsite. Program-level pause/resume is an
**orchestrator** concern. The external daemon reaching *into* the engine over a bridge was
**wrong-layer coupling** (managing an engine concern from above its abstraction). This refactor
puts each lever in its rightful layer, deletes the cross-process plumbing, and keeps the design's
**one V_u** — realised as a *plugin* (mechanism/policy split), NOT a separate process. Design = ideal:
we take the clean decomposition even though it's more work than keeping the bridge hack.

## 1. Feasibility — FEASIBLE-WITH-WORK (clean)

1. **The engine already owns the whole mechanism.** `dump_aginfer_state()` (unified_radix_cache.py
   :3846) and `apply_aginfer_migrations()` (:2770) are direct in-process calls — no HTTP, no lock
   (the scheduler thread serialises cache access). The entire daemon transport tier
   (proxy/event_router/outbound/webhook) exists **only** to bridge a process boundary that disappears.
2. **A zero-cost-when-busy hook already exists.** `on_idle()` (scheduler.py:3424) + the
   `aginfer_webhook.maybe_fire(occ)` cadence in both event loops (:1526/:1601) — a prefill-safe place
   to run the cadenced decision; pays only when the system is light/idle.
3. **The engine self-derives every §4 event.** `Req.program_id` is already present; request
   arrival/completion use the same `perf_counter` clock the proxy used. The proxy was never an
   oracle — it re-observed boundaries the engine already sees. (Only SESSION_END is not
   self-derivable — see §4.)
4. **The policy core is pure stdlib Python** (`build_paper_state`, `joint_decide`,
   `migrate_candidates`, `ETAEstimator`, `ProgramTracker`, the V_u/cost model) — it moves **verbatim**.

## 2. The clean 3-component architecture

| Component | Owns | Layer |
|---|---|---|
| **Engine plugin `AginferScheduler`** (new, in sglang) | §7 per-unit tiering (reactive scorer is already here; add the proactive migrate) | engine |
| **`aginfer_router`** (exists, in dynamo fork) | §8 program pause/resume (fleet-wide view) + stamps `program_id` + forwards SESSION_END | orchestrator |
| **DELETE**: daemon process + `aginfer_bridge.py` + proxy + outbound + event_router + webhook | — | (cross-process plumbing — gone) |

**One V_u, two timescales, both in-engine, sharing one policy module:** the reactive eviction scorer
(`SGLANG_KV_POLICY_MODULE`, already in-engine) and the proactive `AginferScheduler` plugin read the
**same** V_u module — value-consistency (DESIGN §3) preserved, now without a process boundary.

## 3. Migration map

**MOVE → engine plugin** (pure policy, verbatim): `build_paper_state` + `_build_decision_set` +
`_top_k_by_regret` (kv_scheduler.py); `joint_decide` (joint_decide.py); `migrate_candidates` /
`value_residence` / cost fns (ours_greedy.py); `knapsack` (baselines/knapsack.py); `ETAEstimator`
(eta_estimator.py); `ProgramTracker` sync core (program_tracker.py); `ActionTimeline` /
`PromoteAction` (action_timeline.py); cost constants (costs.py); `Event`/`EventKind` as a thin
internal struct.

**STAYS in engine** (mechanism, now called directly not over HTTP): `dump_aginfer_state` (:3846),
`apply_aginfer_migrations` (:2770), `_aginfer_eviction_score` + `set_aginfer_hints` (:3318/:3203),
`set_aginfer_program_state` / `set_aginfer_runtime_metrics`, `node.session_ids` propagation, the
`on_idle`/webhook cadence sites.

**MOVES → Dynamo router**: `_extract_program_id` (proxy → router stamps it); `pause_candidates` /
`resume_candidates` + the `wait_if_paused` gate (admission = the fleet-wide pause/resume decision).

**DELETE** (transport plumbing that bridged a now-gone process boundary): `EventRouter.fetch_state`
+ HTTP GET `/aginfer/state`; `EventRouter._event_worker`; `OutboundQueue`/`OutboundWorker`; the
proxy; the webhook handlers (APPLY_FAILED collapses to the in-process migrate return dict;
SESSION_END → router signal); `main.py` process wiring; `aginfer_bridge.py` (Dynamo).

## 4. Interfaces after the refactor (minimal, clean)

- **Router → Engine**: (a) `program_id` stamped on each request (engine already reads it); (b) ONE
  `SESSION_END` signal per ended program — rides the request channel (terminal flag on the last
  request), NOT a new HTTP endpoint. **This is the only signal the engine cannot self-derive** (a
  program going idle is indistinguishable from a long tool call without it).
- **Engine internal**: plugin ↔ cache mechanism = in-process direct calls (no RPC).
- **No new external API** on either side. The engine gains zero new HTTP surface; Dynamo needs no new
  migrate/state API. This is the answer to "does Dynamo lack an API" — **after this refactor, no.**

## 5. The two design decisions (both resolved BY the layering principle, not compromises)

1. **Knapsack DP isolation.** The exact 0/1 knapsack (`joint_decide`) can take ~20s worst-case
   (capped at 10⁶ cells, top-k 256). It must never block the loop thread → run it on a **worker
   thread with a ~100ms timeout, pressure-gated** (only when `occ` crosses `theta_hi`); the common
   path drains the action-timeline + applies the cached/incremental decision. Preserves the paper's
   exact-optimality (§9) while guaranteeing zero TTFT cost. (Greedy-approx is the fallback if the
   thread harness proves fragile.)
2. **Admission (pause/resume) → router, NOT the engine plugin.** Pause/resume needs the **fleet-wide
   / cross-rank inflight sum** — a property only the orchestrator has. The per-unit §7 migrate is
   **per-rank-local** (a single worker decides if *this* unit leaves *its* HBM) → engine plugin.
   This split **is** the clean layering we agreed: programs → router, KV bytes → engine.

## 6. Risks / invariants to hold

- **do-no-harm**: plugin fully inert when the flag is off; the default path stays the exact stock
  LRU walk at zero plugin cost. (Below-baseline is a bug, not a tuning gap.)
- **Crash isolation lost** (no separate process): wrap the entire plugin tick in try/except → on a
  policy exception, log + disable aginfer for the session, never kill the engine.
- **Event-clock staleness** under bursty arrival: action-timeline is designed for "past-due fires on
  next drain"; use `perf_counter` as a floor, keep monotonic gating.
- **n_holders correctness** on the single-rank path after dropping `_flatten_per_rank` (S2 boost).

## 7. Staged implementation (each stage independently reviewable + testable)

- **Stage A — lift policy core into sglang, no behaviour change.** Move the pure modules into
  `python/sglang/srt/mem_cache/aginfer_policy/` (or similar). The existing `dev/aginfer/verify/`
  suite must still pass against the moved code (import-path update only). *Review:* diff is a move +
  import rewire; verify suite green.
- **Stage B — wire `AginferScheduler` into the loop (in-process).** Cadence at `on_idle` +
  pressure-gate; read state via direct `dump_aginfer_state`, apply via `apply_aginfer_migrations`,
  hints in-process. *Review:* do-no-harm A/B (flag off == stock) + the eviction-value win
  (ours-engine 39/39 reproduced WITHOUT the daemon).
- **Stage C — knapsack isolation** (worker-thread+timeout, pressure-gated). *Review:* TTFT under
  flood is flat with the plugin on.
- **Stage D — delete daemon/bridge/proxy; move admission + program_id + SESSION_END to the router.**
  *Review:* the router A/B (aginfer_router vs thunderagent_router) runs with NO daemon/bridge in the
  picture; the only router→engine wires are program_id + SESSION_END.

## 8. Review checklist (how to review this)

1. **Layer ownership** — does each piece end up in the layer that *owns* its data? (engine = KV
   bytes/tiers; router = programs.) No component reaches below its abstraction.
2. **Interface minimality** — router→engine is exactly {program_id, SESSION_END}; no new API.
3. **One V_u** — the reactive scorer and the proactive plugin share one policy module (not two
   copies that can drift).
4. **do-no-harm** — flag-off path is byte-identical stock; plugin pays zero when off / busy.
5. **No hidden process** — after Stage D, `grep -r aginfer_bridge|daemon.main|OutboundQueue` is empty
   on the Dynamo path.
6. **Tests** — `dev/aginfer/verify/` green after the move; do-no-harm + eviction-win A/B reproduced
   in-engine without the daemon.

---

## 9. Engine fork delta — KEEP / DELETE / EXTRACT / CLEAN (audited vs clean upstream `ca0b7ea4f6^2`)

Our engine delta = **+4158 / −17 across 22 files** (purely additive — we touch ~0 upstream lines, low
merge-conflict surface). But it's **buried**:

| bucket | lines | % | |
|---|---|---|---|
| **NECESSARY-CORE** (real sglang PR) | ~900 | 22% | clean, well-formed; default path = byte-for-byte stock LRU when unset |
| **DAEMON-PLUMBING** (deletes with refactor) | ~3180 | **76%** | pure HTTP/webhook/IPC transport to the external daemon — no hot path depends on it |
| **CLEANLINESS-FIX** | ~80 | 2% | S2DBG hooks, dup docstring, mid-file imports, daemon flags in core args |

### 9a. DELETE — daemon transport (~3180 lines), in full with the refactor
- `aginfer_webhook.py` (NEW +745) → DELETE ~675 (firing/threading/bootstrap/payloads); **extract ~70**
  (`WatermarkState` + `classify()` hysteresis + `derive_kind()`) → `aginfer/occupancy_detector.py`.
- `http_server.py` (+416) → DELETE all 5 `/aginfer/*` endpoints + 50ms refresh-cache; **extract ~75**
  (`_validate_hints_body` NaN/inf/[0,1] bounds) → `aginfer/validation.py` *iff* external hints persist.
- `unified_radix_cache.py` dump/serialization (~1265 of +1822) → DELETE: `dump_aginfer_state*`,
  `_dump_aginfer_state_impl/_dict`, `_StateDumpMetrics`, `_aginfer_program_states`+`set_aginfer_program_state`
  +overlay, `_aginfer_runtime_metrics` store, `_aginfer_pool_usage/link_stats/tier_holding_cost/
  bytes_per_token/node_summary`, collision payloads. (All echo state back over HTTP; zero engine decisions —
  the in-process plugin reads `_aginfer_hints` / `node.session_ids` / EMA dicts directly.)
- `scheduler.py` HTTP handlers (~290 of +532) → DELETE: `get/migrate/update_*` handlers, webhook init/
  cleanup/firing, `_fire_apply_failed/_fire_hash_collisions`, communicator registrations.
- `io_struct.py` (+111 of +136) → DELETE 10 IPC dataclasses (`GetAginferState*/Migrate*/UpdateThresholds*/
  UpdateProgramPaused*/UpdateHints*`).
- `tokenizer_control_mixin.py` (+91) → DELETE (5 pure pass-through communicator methods).
- `server_args.py` (~48 of +60) → DELETE daemon flags (`aginfer_notify_url/heartbeat_s/theta_*`,
  `bootstrap_thresholds_into_server_args`) — daemon-owned policy, not sglang config.
- `hybrid_cache_controller.py` `_last_load_decline` strings (~10) → DELETE (dump-only diagnostic).

### 9b. KEEP — the clean sglang PR (~900 lines), BUT extract inline → `aginfer/` module (thin-hook principle)
- `full_component.py` (+13/−8) — **the model thin hook**: swaps `eviction_strategy.get_priority(n)` →
  `self.cache._eviction_scorer(n, EvictLayer.*)`. Minimal, LRU-identical when unset. KEEP as-is.
- `unified_radix_cache.py` (~520 necessary, currently INLINE) — the pluggable-policy framework
  (`_load_eviction_scorer/_load_write_through_policy`), `_aginfer_hints` table + accessors, `_aginfer_
  eviction_score` (V_u), `apply_aginfer_migrations`, birth-seed, `node.session_ids` propagation, clear-on-
  evict. → **EXTRACT into `aginfer/` module; leave thin hooks** (a scorer-callsite call + the session_ids
  tag on insert/split + a couple accessors). This is the bulk of the thin-hook work.
- `scheduler.py` (~240 necessary, INLINE) — throughput/EMA state + `_aginfer_record_*`/`_update_*`/
  `record_spec_decode`, `program_id` propagation, the on_idle/cadence site. → **EXTRACT the EMA/throughput
  logic into `aginfer/metrics_hooks.py`; leave thin hooks** (pre/post-forward one-liners + the tick call).
- `aginfer_metrics.py` (+144, whole file) — pure stateless utils → **move into `aginfer/` as-is**.
- `batch_result_processor.py` (+40) — `_aginfer_force_token` teacher-forcing (replay) → KEEP, thin hook.
- `schedule_batch.py` (+55) — `_sanitize_program_id` + `Req.program_id` → KEEP (minimal).
- `swa_component.py` (+42) — host-lock pair + `load_back` race short-circuit → KEEP (**V4 correctness**).
- `hybrid_cache_controller.py` `kv_buffer is None` anchor-skip → KEEP (**V4 correctness**).
- `protocol.py` (+28, dedupe docstring), `io_struct.py` (+13: program_id), `base_prefix_cache.py` (+6),
  `tree_component.py` (+11: `peek_time_counter`), `__init__.py` (+2), `session_controller.py` (+7),
  `encode_receiver.py` (+5), serving/tokenizer (+1×3) — `program_id` threaded through every entrypoint/
  disagg/session handoff. KEEP (minimal).

### 9c. CLEANLINESS-FIX (~80 lines, do anytime — independent of the refactor)
1. DELETE 6 S2DBG debug lines: `unified_radix_cache.py:3252–3256, 3327–3330`.
2. DRY the duplicated 20-line `program_id` docstring in `protocol.py` → one `_PROGRAM_ID_DOC` constant.
3. Hoist mid-file `import os, threading` in `http_server.py:~687` to top (moot if file deleted).
4. Consolidate scattered validators → one `aginfer/validation.py`.
5. Isolate daemon flags out of core `server_args` (subsumed by 9a delete).

### 9d. Net result
After delete + extract: the sglang PR = **~900 lines, ~13 files**, of which the upstream-file footprint
is a **handful of thin `# aginfer hook` lines** + the self-contained `aginfer/` module + the 3 small
retained kernels (`occupancy_detector.py` ~70, `validation.py` ~75, runtime-metrics push as a plugin hook).
**Reviewable, minimal, do-no-harm at the code level.**

---

## 10. Progress log

Canonical engine package: `python/sglang/srt/mem_cache/aginfer/`.

- **Stage A — DONE + verified.** 14-file policy closure (base/costs/knapsack/ours_greedy/sglang_adapter/
  events/program_tracker/eta_estimator/action_timeline/admission_controller/joint_decide/_metrics/
  _admission_math/_fatal) moved `dev/aginfer/{baselines,daemon}/` → the engine package; old paths are
  `sys.modules` alias-shims (value-model shims stay for dev research; pure-policy shims die in Stage D).
  Verify suite green against the moved code.
- **Stage A.2 increment 1 — DONE + verified.** Module-level plugin framework (154 lines: env-symbol
  loaders, the LRU-equivalent `_default_eviction_score`, write-through trigger, birth-seed constants)
  `unified_radix_cache.py:67–220` → `aginfer/cache_policy.py`; upstream = a 15-line re-import hook.
  `_default_eviction_score` **identity preserved** (swa_component's `is not` check holds). t27 17/17, t38 7/7.
- **Stage A.2 increment 2 — DONE + verified.** Contiguous hint-table + value-eviction scorer block
  (157 lines: `set_aginfer_hints` / `get`/`clear_aginfer_hint` / `_aginfer_unit_hash` /
  `_init_aginfer_eviction_scoring` / `_aginfer_eviction_score` / `_aginfer_hint_should_write_through` /
  `_aginfer_seed_birth`) `unified_radix_cache.py:3064–3220` → `aginfer/cache_hooks.py` as free functions
  over `cache`; upstream = 8 thin `(*a,**k)` delegators + 1 import. Folded-in cleanups: `hint_v_u` now
  imported via the in-package path (`...aginfer.sglang_adapter`, no `sys.path` dependence) not
  `baselines.*`; added missing `import os`. **Verify:** engine import OK · delegators wired ·
  `_default_eviction_score` identity preserved · t27 17/17 · t38 7/7 · **s2_holder_scorer PASS** (hint→
  `_v_u_from_unit`→scorer, "no drift") · kv_scheduler_value_rule/joint_decide/admission_controller/
  program_tracker PASS. (`action_timeline` FAIL is a pre-existing daemon ETA `due_time` drift, #235 —
  pulls **none** of the changed modules into its import graph, provably independent.)
- **Stage A.2 increment 3 — DONE (code-move verified by construction; 39/39 full-stack DEFERRED).**
  `apply_aginfer_migrations` (the §7 migrate EXECUTOR, 392 lines, `unified_radix_cache.py:2632–3023`) →
  `aginfer/cache_hooks.py` as a free function over `cache`; upstream = a 2-line delegator. The move is
  **behaviour-preserving by construction**: uniform de-indent + whole-word `self`→`cache` (28 subs, every
  `self` token verified a code ref — def sig + 1 `getattr(self,…)` + all `self.X`; no string/comment
  `self`). **Forward proof: 0 diffs** — every moved line equals `dedent(orig)`+`self→cache`. Tier names
  HBM/DRAM/DISK/HOST are string literals (no import); the only runtime deps imported are `EvictLayer` +
  `BASE_COMPONENT_TYPE` (circular-safe — `unified_cache_components` is already loaded before the
  `cache_hooks` import; the swa→urc back-edge is a lazy in-method import). **Verify:** compile + import OK,
  deps resolve, delegator present, t27/t38/s2_holder_scorer/kv_scheduler_value_rule(25/25)/joint_decide
  all PASS. ⚠️ **DEFERRED gate (task #252):** `apply_aginfer_migrations` has no server-free unit test
  (t20/t24 POST a live `:30001`), so the runtime migrate path is unconfirmed until the next
  `dynamo.sglang` restart runs the daemon→engine round-trip + do-no-harm + 39/39-without-daemon.
- **Stage A.2 increment 4 — DONE (code-move verified by construction + full-import; live EMA DEFERRED).**
  The 8 scheduler EMA/throughput hooks (`_aginfer_record_throughput`(+`_inner`), `_aginfer_extend_token_count`
  [@staticmethod], `_aginfer_update_prefill`/`_update_decode`, `_aginfer_record_spec_throughput`/`_spec_decode`,
  `_aginfer_push_runtime_metrics`; `scheduler.py:3674–3895`, 222 ln) → NEW `aginfer/metrics_hooks.py` as free
  functions over `sched`; upstream = 8 delegators (7 instance + 1 `@staticmethod`). The now-block-only
  `aginfer_metrics` import was **relocated** out of `scheduler.py` (5 names had zero out-of-block refs).
  Move = de-indent + strip the lone `@staticmethod` + whole-word `self`→`sched` (NOT `cache` — bodies use a
  local `cache` var). Annotations made lazy via `from __future__ import annotations`; runtime deps imported
  (`ForwardMode`, the 5 `aginfer_metrics` fns; `BASE_COMPONENT_TYPE` covered by the in-method import).
  **Verify:** 0-diff forward proof (accounting for the stripped decorator) · py_compile · **FULL
  `import scheduler` OK (no circular import)** · 8 delegators present + `_metrics_hooks` wired +
  `aginfer_metrics` relocated + staticmethod preserved · t26/t27/t38/s2_holder_scorer green. ⚠️ DEFERRED
  (task #252): the live decode/prefill EMA populate-under-load (#204) re-confirms on the next restart.

- **Stage A.2 increment 5 — DONE.** The write-through policy RESOLUTION (4 inline lines in
  `unified_radix_cache.__init__`) → `_init_aginfer_write_through(cache)` in `cache_hooks.py` + a thin
  delegator + a one-line callsite — making it the symmetric twin of `_init_aginfer_eviction_scoring`. The
  `_load_write_through_policy` re-import stays in urc only as a re-export for verify/t28. Verify: compile +
  import + `_init_aginfer_write_through` wired + re-exports intact + t28/t38/t27/t21/s2/kv_scheduler/joint_decide green.

**A.2 EXTRACTION COMPLETE.** All separable aginfer logic now lives in the self-contained package; the only
aginfer code left inline in the two big upstream files is (a) thin hooks — ~10+8 delegators, the re-import
block, ~8 callsites, 3 irreducible `node.session_ids` tags — and (b) the ~1265-line daemon dump/state plumbing
that **DELETEs (not moves)** in Stage D when the daemon HTTP path is removed (`set_aginfer_program_state`,
`_aginfer_program_states`, `_aginfer_node_summary`, `dump_aginfer_state*`, …; a few — `_aginfer_bytes_per_token`,
`_aginfer_subpool_name`, `set_aginfer_runtime_metrics` — are cross-used by `metrics_hooks` and stay). The
`node.session_ids` propagation is genuinely irreducible (1–2-line node tagging on tree split/insert) — a thin
hook by nature, not an extraction target.

### A.2 final independent verification (3-lens) — completeness PASS, 1 fix, 1 pre-existing gap tasked

A second adversarial pass on the COMPLETE state (inc1–5 + fixes): **Completeness = PASS** (every inline aginfer
ref is a thin hook, a Stage-D-delete daemon-plumbing method, or one of the 3 cross-used keepers
`_aginfer_bytes_per_token`/`_aginfer_subpool_name`/`set_aginfer_runtime_metrics`); **inc5 + all 3 prior fixes
verified correct** (byte-identical move, `_AGINFER_VALID_STATES` is a class attr, re-exports load-bearing,
future-annotations present); **do-no-harm intact for the default config** (env-unset => `_default_eviction_score`
= `last_access_time` = stock LRU, identity preserved through the re-import chain). Findings:
- **FIXED:** dead `import os` at urc:70 (inc5 orphaned it; stale comment) — removed.
- **#253 (pre-existing, NOT A.2):** the aginfer eviction-scorer plugin replaced `eviction_strategy.get_priority(n)`
  with `_eviction_scorer`, whose default hardwires LRU => `--radix-eviction-policy lfu/slru/priority` silently
  diverges from stock even with env unset. Matches stock only for the default `lru` (what all experiments use).
  A do-no-harm gap for a clean upstream PR; tasked, not folded into A.2.
- **Stage-D note:** the duplicated T5 webhook-firing block in `scheduler.py` (event_loop_normal/overlap) is
  pre-existing daemon-plane code; it deletes (or extracts) in Stage D.

### Round-3 review — #253 fix + dedup confirmed do-no-harm; 4 minor non-regressions

After fixing the above: **#253 (FIXED)** — `full_component._evict_keyfn` now branches on a single
`cache._aginfer_value_aware` flag (set in `_init_aginfer_eviction_scoring`): default path keys on
`eviction_strategy.get_priority` (honors `--radix-eviction-policy`), value path on the scorer. swa's
`_value_ordered()` reads the same flag. **Regression gate:** `verify/radix_eviction_policy` (12/12). The
**dead `import os`** and the **duplicated webhook block** (→ `_aginfer_maybe_fire_webhook()`) are also fixed.
Round-3 (3-lens) verdict: 2 PASS + 1 framing-only; do-no-harm proven at source + runtime —
`LRUStrategy.get_priority == last_access_time == _default_eviction_score`, so lru + value paths are
byte-identical no-ops; only non-LRU defaults change (the fix). 4 minor non-regressions, all dispositioned:
(1) swa value-heap is #248's code (framing); (2) #253 boundary already documented; (3) **MambaComponent +
SWA are LRU-list-only — STOCK behavior** (Mamba is byte-for-byte unchanged vs HEAD; honoring non-LRU there
would be a feature *beyond* stock, not do-no-harm); (4) `full_component` heap tuple `(key, n)` matches
**stock** `radix_cache.py` and is crash-safe via `UnifiedTreeNode.__lt__` — the swa `n.id` tiebreaker is
#248's own choice, not required here.

### A.2 adversarial review (5-lens panel + per-finding verify) — 1 CRITICAL fixed, 2 cleanups, 1 mis-attribution

Ran an adversarial review of inc1–4 (5 independent lenses → verify each finding). Lenses B (imports/
circular), C (callsites), D (identity invariants) returned **CLEAN**. 5 findings, all triaged + resolved:

- **CRITICAL (FIXED + gated):** inc3's block boundary swept the trailing CLASS attribute
  `_AGINFER_VALID_STATES` into `cache_hooks.py` module scope while its consumer `set_aginfer_program_state`
  stayed inline → `AttributeError` on the program-state path. The 0-diff forward-proof validated the
  TRANSFORM but **not the BOUNDARY** (the swept line was a class attr, not part of `apply_aginfer_migrations`);
  compile/import can't catch runtime attribute resolution. FIX: restored `_AGINFER_VALID_STATES` as a class
  attribute (`unified_radix_cache.py:2635`), deleted the orphan. **Regression gate: t21** (exercises
  `set_aginfer_program_state` 17×) green + now in the standing set. (Lesson: also run the test for the
  methods left BEHIND adjacent to a cut, and grep the moved module for stray module-level `NAME =` — only
  `logger` should remain. Verified: both new modules have only `logger` at module scope.)
- **Minor (FIXED):** `cache_hooks.py` lacked `from __future__ import annotations` (bare `UnifiedTreeNode`
  in a local annotation — safe per PEP 526 but inconsistent with `metrics_hooks`) → added. 3 dead re-imports
  in the urc hook block (`_AGINFER_HINT_SCORER_SPEC`/`_AGINFER_BIRTH_LAMBDA`/`_AGINFER_BIRTH_STAMP` — no inline
  use, no external consumer) → deleted; kept `_AGINFER_BIRTH_PHAT` (read by verify/t27 via `_urc.`).
- **Mis-attribution (NO CHANGE):** the newborn birth-seed `p_hat` 1.0→`_AGINFER_BIRTH_PHAT`=0.1 is
  **PRE-EXISTING WIP** — the reviewer diffed against HEAD, which predates it; the inc1/inc2 scripts did only
  de-indent + self→X + byte-copy and so CANNOT have introduced it. do-no-harm floor intact (the seed fires
  only on the opt-in `aginfer:hint_v_u` path; env-unset = stock LRU).

Post-fix verify (all green): compile + full `import scheduler` + t21/t26/t27/t28/t38/s2_holder_scorer/
kv_scheduler_value_rule(25)/joint_decide. The 925-line, 3-file extraction stands clean + reviewed.

**Conclusion — re-sequence.** Because the remaining cache-method KEEP code is tangled with the
DELETE-in-Stage-D plumbing AND has no unit-level gate, the clean order is **Stage B (in-engine
`AginferScheduler` plugin) + Stage D (delete daemon transport) BEFORE the final cache-method extraction** —
deleting the ~1265 interleaved dump/state lines de-tangles the region, and Stage B/D need the live stack
anyway. All three remaining pieces share one gate: restart `dynamo.sglang` → do-no-harm (default == stock
LRU) + ours-engine **39/39 reproduced without the daemon**. That is the next focused, live-stack effort.
