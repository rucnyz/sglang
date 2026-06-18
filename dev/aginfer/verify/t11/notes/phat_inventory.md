# p_hat / lambda_rate Touch-Point Inventory for T11

Goal: When swapping in a new empirical p_hat estimator (T11), identify all code paths that depend on `p_hat`, `lambda_rate`, and the `hits/age` proxy. This document enumerates every touch point so an estimator-swap touches all necessary sites consistently.

---

## 1. TOUCH-POINT INVENTORY

### 1.1 Inline Scorer (sglang hot path: `drive_eviction`)

**File:** `/scratch/yuzhou/projects/sglang/python/sglang/srt/mem_cache/unified_radix_cache.py`

- **Module-level pluggable scorer loader (lines 53–106)**
  - `_load_eviction_scorer()`: Loads `SGLANG_KV_POLICY_MODULE` env var → callable
  - Default fallback: `_default_eviction_score()` → returns `node.last_access_time` (stock LRU)
  - Emits canonical startup log `[aginfer] kv_policy_loaded=...`

- **UnifiedTreeNode fields (lines 114–142)**
  - `node.hit_count` (int): accumulated reuse count
  - `node.last_access_time` (int): set on creation via `get_and_increase_time_counter()`
  - `node.session_ids` (set): program holders (used by daemon to tag ACTING/PAUSED)
  - These feed the inline scorer at call time (no pre-computed storage of p_hat/lambda)

- **Instance scorer registration (lines 339–342)**
  - `self._eviction_scorer = _load_eviction_scorer()` in `UnifiedRadixCache.__init__`

**File:** `/scratch/yuzhou/projects/sglang/python/sglang/srt/mem_cache/unified_cache_components/full_component.py`

- **drive_eviction() call site (lines 126–146)**
  - Line 131: `score_fn = self.cache._eviction_scorer`
  - Lines 132–135: Builds heap of `(score_fn(n, EvictLayer.DEVICE), n)` for all evictable device leaves
  - Line 145: Re-scores parent on dynamic eviction: `score_fn(x.parent, EvictLayer.DEVICE)`
  - **Decision:** min-heap: lowest score (smallest future value) is popped first
  - **Execution:** hot path on every memory pressure event in the token loop

- **drive_host_eviction() call site (lines 148–167)**
  - Line 152: `score_fn = self.cache._eviction_scorer`
  - Lines 153–156: Builds heap for evictable host leaves
  - Line 166: Re-scores parent: `score_fn(x.parent, EvictLayer.HOST)`
  - **Decision:** same min-heap semantics (evict lowest-value first)

**Signature expected by sglang:**
```python
def scorer(node: UnifiedTreeNode, layer: EvictLayer) -> float
```
- Must read `node.hit_count`, `node.last_access_time`, optionally `node.session_ids`
- Must compute dynamically at call time (no cached fields on node)
- Lower score → higher eviction priority

### 1.2 Baseline Policy: `ours_greedy_score` (sglang adapter)

**File:** `/scratch/yuzhou/projects/sglang/dev/aginfer/baselines/sglang_adapter.py`

- **_node_to_unit() (lines 63–84)**
  - Line 66: `age = max(1, current_counter - int(getattr(node, "last_access_time", 0)))`
  - Line 67: `hits = int(getattr(node, "hit_count", 0))`
  - Line 69: `p_hat = min(1.0, hits / age) if age > 0 else 0.0`
  - Line 71: `lam = max(1e-3, hits / age) if age > 0 else 1e-3`
  - Lines 73–84: Package into `ReuseUnit(p_hat=p_hat, lambda_rate=lam, ...)`
  - **Execution:** called on every `ours_greedy_score()` call (hot path)

- **ours_greedy_score() (lines 135–160)**
  - Line 145: Get current time counter: `now = _current_time_counter()`
  - Line 147: Convert node → unit via `_node_to_unit(node, layer_name, now)`
  - Lines 150–152: Compute saved-prefill value: `save_prefill = u.p_hat * (reload_cost(DROP) - reload_cost(tier))`
  - Line 157: Compute expected reuse interval: `hold_time = 1.0 / u.lambda_rate if u.lambda_rate > 0 else 1e6`
  - Line 158: Holding tax: `hold = h_base * u.n_bytes * hold_time`
  - Line 159: Return value: `value = save_prefill - hold`
  - **Decision:** Value-based heap key (higher V_u stays in cache)

- **recency_freq_score() (lines 128–132) [ablation]**
  - Line 132: Returns `float(hits) / float(age)` (the raw proxy)
  - Not used in Run H, but shows the hits/age pattern

### 1.3 Daemon: kv_scheduler

**File:** `/scratch/yuzhou/projects/sglang/dev/aginfer/daemon/kv_scheduler.py`

- **build_paper_state() (lines 227–365)**
  - **hits/age proxy computation (lines 304–310)**
    - Line 305: `hits = int(raw["hit_count"])`
    - Line 306: `age = max(1, now_counter - last_access)`
    - Line 309: `p_hat = min(1.0, hits / age)` — clamped to [0, 1]
    - Line 310: `lam = max(1e-3, hits / age)` — floored at 1e-3
  - **Lambda override for ACTING/PAUSED programs (lines 320–342)**
    - Line 322: Check if any holder is in ACTING or PAUSED state
    - Lines 332–335: Override `lam` with calibrated floor `_clamp_lambda_acting(lambda_acting)`
    - Line 340: Pick ANY ACTING holder's floor λ
  - **Units populated (lines 343–354)**
    - Line 351: `p_hat=p_hat`
    - Line 352: `lambda_rate=lam`
  - **Execution:** every paper §4 event, after fetching `/aginfer/state`

- **_top_k_by_regret() (lines 400–438) — decision_set builder**
  - **Regret scoring (lines 432–436)**
    - Line 432: `saved = u.p_hat * (rho_disk - rho_hbm) * u.n_tokens`
    - Line 434: `hold = costs.h_base[Tier.HBM] * u.n_bytes`
    - Line 435: `score = saved - hold`
  - **Decision:** Sort ascending, return first k (lowest-value units are demote candidates)
  - **Execution:** called for `MEMORY_PRESSURE` / `PRESSURE_RESOLVED` events
  - **Note:** Inline proxy (no lambda term in this lightweight ordering)

### 1.4 Daemon: admission_controller

**File:** `/scratch/yuzhou/projects/sglang/dev/aginfer/daemon/admission_controller.py`

- **_value_at_current_tier() (lines 90–110)**
  - Line 102: `save_prefill = u.p_hat * (reload_cost(DROP) - reload_cost(tier))`
  - Line 109: `hold_time = 1.0 / u.lambda_rate if u.lambda_rate > 0 else 1e6`
  - Line 110: Return: `save_prefill - h * u.n_bytes * hold_time`
  - **Matches:** `OursGreedyPolicy._value` line for line (B1 audit fix: restores holding tax)
  - **Execution:** called for every unit in `shared_aware_prog_scores()`

- **shared_aware_prog_scores() (lines 113–136)**
  - Line 132: Calls `_value_at_current_tier(u, state, costs, pi_u)` for each unit
  - Line 133: Divides by holder count: `share = v / len(u.holders)`
  - **Decision:** Aggregate program scores; lowest-score program is pause victim
  - **Execution:** on `MEMORY_PRESSURE` event in admission._on_pressure loop

- **_on_pressure() (lines 193–237)**
  - Line 214: `scores = shared_aware_prog_scores(sched_state)`
  - Pauses lowest-score program(s) until HBM occupancy < theta_hi

### 1.5 Baseline Policy: OursGreedyPolicy

**File:** `/scratch/yuzhou/projects/sglang/dev/aginfer/baselines/ours_greedy.py`

- **_value() (lines 80–91)**
  - Line 82: `save_prefill = u.p_hat * (reload_cost(DROP) - reload_cost(tier))`
  - Line 90: `hold_time = 1.0 / u.lambda_rate if u.lambda_rate > 0 else 1e6`
  - Line 91: Return: `save_prefill - h * u.n_bytes * hold_time`
  - **Decision:** Value per unit; used in decide() to rank tiers
  - **Execution:** called for every unit in every candidate tier during decide()

- **decide() (lines 98–126)**
  - Lines 108–126: Greedy loop over decision_set
  - Line 111: Score current tier: `best_score = self._net_value(u, u.tier, state)`
  - Lines 112–120: Evaluate all candidate tiers, pick argmax
  - **Decision:** Tier assignment per unit (HBM / DRAM / DISK / DROP)

### 1.6 ReuseUnit data class

**File:** `/scratch/yuzhou/projects/sglang/dev/aginfer/baselines/base.py`

- **ReuseUnit fields (lines 36–54)**
  - Line 49: `p_hat: float = 0.0` — "predicted reuse prob over Delta t"
  - Line 50: `lambda_rate: float = 0.0` — "Poisson reuse rate (Section 7)"
  - **Default:** both 0.0 (first-touch or unobserved units)
  - **Constructor:** populated by `_node_to_unit()` (adapter) or `build_paper_state()` (daemon)

---

## 2. BEHAVIORAL COUPLING

### 2.1 Same p_hat value drives multiple decisions

| Decision path | Uses p_hat | Uses lambda | Couples with |
|---|---|---|---|
| Inline eviction (`drive_eviction`) | ✓ (via `ours_greedy_score`) | ✓ | Daemon migrate decisions |
| Daemon migrate (`_top_k_by_regret`) | ✓ | ✗ (proxy only) | Inline eviction |
| Daemon migrate (`_value`) | ✓ | ✓ | Admission scoring |
| Admission pause (`_value_at_current_tier`) | ✓ | ✓ | Daemon migrate (`_value`) |

### 2.2 Critical consistency requirements

1. **Inline ↔ Daemon p_hat parity**
   - Inline scorer computes: `hits / age` (lines 63–71 in adapter)
   - Daemon computes: `hits / age` (lines 309–310 in kv_scheduler)
   - **Must match:** Both use the same `max(1, counter - last_access_time)` for age
   - **Risk:** Inline and daemon clocks drifting → different p_hat values → inconsistent eviction vs. migration

2. **lambda_rate override consistency**
   - Inline scorer: uses raw `hits/age` proxy (no override; always REASONING lambda)
   - Daemon: overrides to `_LAMBDA_ACTING_FLOOR` if any holder is ACTING/PAUSED
   - **Issue:** Inline eviction doesn't know program state → may score high-hit ACTING prefixes as valuable, but admission pauses the whole program
   - **Mitigation:** Paper §7 designed this: inline is a safety net; daemon admission is the primary control

3. **Holding-time semantics (h × b_u / λ)**
   - Appears in: `ours_greedy_score` (line 158), `_value` (line 91), `_value_at_current_tier` (line 110)
   - Uses: `hold_time = 1.0 / lambda_rate if lambda_rate > 0 else 1e6`
   - **Assumption:** lambda is reuse rate (events/tick); 1/λ is expected time until next use
   - **T11 impact:** If new estimator changes meaning of λ (e.g., from Poisson rate to probability per tick), holding-time math breaks
   - **Safe estimate:** Keep λ in units of "reuse events per observation window" (same as hits/age)

---

## 3. DEFAULT VALUES & FIRST-TOUCH BEHAVIOR

### 3.1 p_hat defaults

| Context | Default | Trigger | Semantics |
|---|---|---|---|
| ReuseUnit creation | `0.0` | Initial value in dataclass | Unobserved unit |
| _node_to_unit (adapter) | `0.0` if age=0 | New node (age=1 min) | No historical reuse |
| _node_to_unit (adapter) | `min(1.0, hits/age)` | age > 0 | Frequency proxy, clamped |
| build_paper_state (daemon) | `min(1.0, hits/age)` | all ages, age >= 1 | Same proxy |

### 3.2 lambda_rate defaults

| Context | Default | Trigger | Semantics |
|---|---|---|---|
| ReuseUnit creation | `0.0` | Initial value | No reuse signal |
| _node_to_unit (adapter) | `1e-3` if age=0 | New node | Floor (avoid div-by-zero) |
| _node_to_unit (adapter) | `max(1e-3, hits/age)` | age > 0 | Frequency proxy, floored |
| build_paper_state (daemon) | `max(1e-3, hits/age)` | REASONING programs | Raw proxy |
| build_paper_state (daemon) | `_clamp_lambda_acting(0.2)` | ACTING/PAUSED | Calibrated override |

### 3.3 Edge cases under new estimator T11

| Scenario | Current behavior | T11 must handle |
|---|---|---|
| First-touch node (no history) | p_hat=0, λ=1e-3 | Must not crash; default conservatively |
| Zero age (same tick as insert) | p_hat=0 (by max(1, ...)) | Clamp age, same as now |
| High-hit old prefix (hits=1000, age=10000) | p_hat=0.1, λ=0.1 | Smooth transition; avoid order-of-magnitude jumps |
| Single hit (hits=1, age=1) | p_hat=1.0, λ=1.0 | May underestimate λ (one sample); T11 can smooth |
| No holders (orphan unit) | Still scored (but shouldn't be paused) | admission filters by holders list |

### 3.4 Smooth handoff checklist

When integrating T11 estimator:

1. **Maintain range [0, 1] for p_hat**
   - Clamping is relied on in `_top_k_by_regret` (line 432: `p_hat * cost`, cost is signed)
   - Inline scorer returns float; heap comparison is stable for any value

2. **Keep λ > 0 (strictly positive)**
   - Holding-time calculation: `1.0 / λ` must not hit div-by-zero
   - Current floor: `max(1e-3, ...)` guarantees λ >= 1e-3
   - T11 must maintain invariant or update all call sites

3. **Preserve ACTING override in daemon**
   - Line 340: `lam = program_lambda[next(...)]` — T11 estimator is daemon-side input
   - Inline scorer has no state; cannot apply ACTING override
   - **Gap:** Inline eviction doesn't know ACTING state → may keep high-hit prefixes that admission later pauses
   - **Design:** This is intentional (paper §9: dual-layer redundancy)

4. **Match inline ↔ daemon age calculation**
   - Both use: `max(1, counter - last_access_time)`
   - Current counter: inline calls `_current_time_counter()`, daemon uses fresh `/aginfer/state` time_counter
   - **Clock sync:** sglang and daemon may have drifting counters over a long run
   - **Audit:** Measure clock skew; if > 1000 ticks (≈seconds), may impact p_hat agreement

---

## 4. ESTIMATOR-SWAP CHECKLIST

When T11 supplies a new empirical p_hat / λ estimator:

### Phase 1: Compute-time wiring

- [ ] Update `/scratch/yuzhou/projects/sglang/dev/aginfer/baselines/sglang_adapter.py::_node_to_unit()` lines 69, 71
  - Compute new `p_hat` (must be in [0, 1])
  - Compute new `lam` (must be > 0; recommend keeping >= 1e-3)
- [ ] Ensure new estimator reads only `hits`, `age`, optionally `session_ids` from node
  - Must work at every call to `ours_greedy_score` (hot path, strict latency)

### Phase 2: Daemon-side wiring

- [ ] Update `/scratch/yuzhou/projects/sglang/dev/aginfer/daemon/kv_scheduler.py::build_paper_state()` lines 309, 310
  - Match adapter's new p_hat / λ computation exactly
  - Test that two paths agree on the same unit
- [ ] Verify daemon lambda override still applies (line 340; should not need change)

### Phase 3: Validation

- [ ] Regression test: inline eviction produces same order as daemon _top_k_by_regret on same state snapshot
  - Use verify/t11/notes/phat_inventory.md as trace guide
- [ ] Latency check: `_node_to_unit()` must be < 1 µs per call (hot path)
- [ ] Clock sync check: compare inline `_current_time_counter()` vs. daemon state's time_counter at same moment
  - Acceptable skew: < 100 ticks (≈seconds on slow machine)

### Phase 4: Empirical validation

- [ ] Run K-full with new estimator; compare hit rate vs. T8 baseline
- [ ] Audit log: check `[aginfer] kv_policy_loaded=...` to confirm estimator loaded
- [ ] No regressions in memory pressure response (pause/resume latency should not change)

---

## 5. FILE SUMMARY TABLE

| File | Lines | Function | p_hat/λ role | T11 impact |
|---|---|---|---|---|
| `baselines/sglang_adapter.py` | 69, 71 | `_node_to_unit` | Compute proxy | **CRITICAL:** Update here |
| `baselines/sglang_adapter.py` | 150, 157 | `ours_greedy_score` | Use in value | No change (calls _node_to_unit) |
| `baselines/ours_greedy.py` | 82, 90 | `_value` | Compute value | No change (uses ReuseUnit fields) |
| `daemon/kv_scheduler.py` | 309, 310 | `build_paper_state` | Compute proxy | **CRITICAL:** Update here |
| `daemon/kv_scheduler.py` | 432, 434 | `_top_k_by_regret` | Score units | No change (uses ReuseUnit fields) |
| `daemon/admission_controller.py` | 102, 109 | `_value_at_current_tier` | Compute value | No change (uses ReuseUnit fields) |
| `baselines/base.py` | 49, 50 | `ReuseUnit` dataclass | Storage | No change (fields already generic) |
| `python/sglang/srt/mem_cache/unified_radix_cache.py` | 68–105 | `_load_eviction_scorer` | Loader | May need to update env var docs |
| `python/sglang/srt/mem_cache/unified_cache_components/full_component.py` | 131, 152 | `drive_eviction`, `drive_host_eviction` | Call scorer | No change (calls pluggable scorer) |

---

## 6. CRITICAL CODE PATHS

### 6.1 Inline eviction (hot path)

```
sglang: UnifiedRadixCache.drive_eviction() [synchronized]
  └─ FullComponent.drive_eviction() [lines 126–146]
     └─ for each evictable device leaf:
        score = self.cache._eviction_scorer(node, EvictLayer.DEVICE)  [line 133]
        └─ ours_greedy_score(node, layer) [adapter line 135]
           └─ _node_to_unit(node, layer_name, now) [line 147]
              ├─ age = max(1, counter - node.last_access_time) [line 66]
              ├─ hits = node.hit_count [line 67]
              ├─ p_hat = min(1.0, hits/age) [line 69]  ← T11 swap point 1
              └─ lam = max(1e-3, hits/age) [line 71]   ← T11 swap point 2
           └─ save_prefill = u.p_hat * (R_DROP - R_tier) [line 150]
           └─ hold = h_base * u.n_bytes / u.lambda_rate [lines 157–158]
           └─ return save_prefill - hold [line 159]
        └─ min-heap: smallest score evicted first
```

### 6.2 Daemon migrate (serial worker)

```
daemon: EventRouter.handle(event) [serial queue]
  └─ KvScheduler.handle(event) [line 542]
     └─ build_paper_state(state_json, ...) [line 559]
        └─ for each unit in state_json["units"]:
           ├─ age = max(1, now_counter - last_access_time) [line 306]
           ├─ hits = raw["hit_count"] [line 305]
           ├─ p_hat = min(1.0, hits/age) [line 309]  ← T11 swap point 1
           ├─ lam = max(1e-3, hits/age) [line 310]   ← T11 swap point 2
           └─ if any holder ACTING: lam = floor(lambda_acting) [line 340]
        └─ decision_set = _build_decision_set(event, units, ...) [line 356]
           └─ if MEMORY_PRESSURE: decision_set = _top_k_by_regret(units, k) [line 472]
              └─ for each HBM unit:
                 score = u.p_hat * (R_DISK - R_HBM) - holding [line 435]
                 └─ sort ascending, return first k
     └─ policy.decide(sched_state) [line 573]
        └─ OursGreedyPolicy._value(unit, tier) [line 80]
           ├─ save_prefill = u.p_hat * (R_DROP - R_tier) [line 82]
           └─ hold = h * u.n_bytes / u.lambda_rate [line 91]
```

### 6.3 Daemon admission (serial worker, post-migrate)

```
daemon: AdmissionController._on_pressure(event) [lines 193–237]
  └─ loop up to max_pauses_per_event times:
     └─ shared_aware_prog_scores(sched_state) [line 214]
        └─ for each unit:
           v = _value_at_current_tier(u, sched_state, ...) [line 132]
           ├─ save_prefill = u.p_hat * (R_DROP - R_tier) [line 102]
           └─ hold = h * u.n_bytes / u.lambda_rate [line 110]
           └─ share = v / len(u.holders) [line 133]
     └─ victim = min(eligible.items(), key=score) [line 230]
     └─ tracker.pause(victim) [line 231]
```

---

## 7. REGRESSION PROBE ENTRY POINTS

For T11 validation, trace execution via:

1. `/scratch/yuzhou/projects/sglang/dev/aginfer/verify/t8/verify.py` — end-to-end V_u hand-derivation
   - Calls `_value_at_current_tier` and expects specific numeric result
   - Update numeric assertions when p_hat/λ change

2. `/scratch/yuzhou/projects/sglang/dev/aginfer/verify/t7/verify.py` — daemon ordering checks
   - Calls `_top_k_by_regret` and checks sort order
   - May need to update hit_count fixtures to exercise new estimator

3. `/scratch/yuzhou/projects/sglang/dev/aginfer/verify/t7/regression_probe.py` — adapter proxy derivation
   - Hand-computes expected V_u; compare against policy.decide output
   - Critical: verify inline ↔ daemon agreement

---

## END OF INVENTORY

**Key takeaway:** T11 estimator changes are confined to TWO compute sites:
1. **Inline:** `baselines/sglang_adapter.py::_node_to_unit()` lines 69, 71
2. **Daemon:** `daemon/kv_scheduler.py::build_paper_state()` lines 309, 310

All other code paths consume the p_hat and lambda_rate fields via the generic ReuseUnit interface. Keep inline ↔ daemon computation identical for consistency.
