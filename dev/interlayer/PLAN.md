# Implementation plan

Roadmap for landing the design in `design.md`. This file is the **forward
roadmap**; the completed history lives in git + each folder's `README.md`
(and the per-area docs under `0_page_state_machine/`, `1_dyn_admission_cap/`,
`2_admitter/`, `3_budgeter/`, `4_e2e/`). The task tracker now holds only
**actionable** (pending) work — completed tasks were pruned (their
falsification records are in the folder READMEs / commit messages).

Reading order: this file → `design.md` → the folder README for the area →
the task `#N` in TaskList for execution detail.

---

## Shipped so far (one-line-per-milestone record)

**Phase 0 — audits.** `0_page_state_machine/`, `1_dyn_admission_cap/`,
`2_admitter/` audited against `design.md` + production; doc + naming aligned;
all prod-vs-spec gaps routed and closed. Fire path is no-defense fail-fast
(layer-0 invariant, no defensive sync / no setAccess revoke).

**Phase 1 — substrate.** Arena `grow → mapped slot IDs` single ID-flow;
`_capped_pages`/`_capped_slots` invariants fail-fast (KV + mamba); mamba
`live_size` derived from `_capped_slots`; padded slot-0 (chunk 0) kept mapped;
`clear()` preserves unmapped headroom; cap-barrier verify on the worker thread.

**Phase 2 — M+Drain prims.** LPB eviction policy across RadixCache variants
(`--radix-eviction-policy lpb`); boot probes for `c_m` (migrate), `c^xfer`
(self-reversing fire, 90.7 µs/page), `κ_i`; policy-aware `c^evict` predictor
(`predict_evict_cost_us`, byte-identical to the real evict walk) wired for
`dst=kv` on the live `MambaRadixCache`. Boot-validated on Qwen3.5-9B / 1×H200.

**Phase 3 / 3.5 — Admitter + fire_planner + migration feasibility.** 7-action
cost program + three-stage page-selection knapsack (anywhere-free → Drain-
expansion → Migration-expansion). Migration feasibility = scattered-free-slot
consolidation: **mamba is atomic-inert at tp=1/fp32** (the single-GPU bench
corner), **fragmentable at tp≥2 / bf16**; **KV is always fragmentable**. The
migration dst budget is the scattered free slots (single source of truth
shared by Admitter feasibility ↔ planner selection). Mamba-source cold-cache
Drain (`cross_evict`) closed end-to-end; KV-source cold-cache Drain (k2m)
shipped symmetrically.

**Cross-fire grow arc (Phase 4/5).** Direction-correctness (never grow by
draining the more-constrained pool); grow-side eviction signal (fire k2m when
mamba sheds hot cache); cross-fire mamba growth made durable; grow gated on
LPB (no-regression on real cc traces); **A1: KV pool made growable**
(dynamic-cap port to the KV allocator) + the `available_size` double-count
fix + bounded drain-victim walk.

**KV live-slot migration (`cross_migrate(src=kv)`) — mechanism COMPLETE,
gated OFF.** `TokenToKVPoolAllocator.migrate_slot` (byte-move via
`move_kv_cache` + Convention-A free swap); pool-agnostic `_run_stage0`
(validate-then-apply, cuda.sync brackets); `req_to_token` rewrite; vectorized
KV cost-order walk. Fail-closed behind `SGLANG_XPOOL_KV_MIGRATE` (default off)
**and** a real `can_migrate_slot()` capability gate; `enable_kv_cache_copy`
wired via single-source-of-truth `kv_live_migration_enabled()`; `BudgetAgent`
requests `allow_migrate=True` (self-gated). **Empirically proven safe**:
captured-graph replay equivalence on flashinfer (#291) + FA3, with
triton/aiter kernel-identity-asserted (all CUDA backends covered). Three
adversarial audit rounds fixed test-first: locked-prefix exclusion,
validate-then-apply, capability gate, mamba-migration fail-closed.
Folder: `0_page_state_machine/kv_migrate_slot/`.

**Phase 5 headline (in progress).** The 4-cell cc_traces_headline ran; the
`cross_evict`/k2m cold-cache **drain captures +3.3 pp cache_hit** (0.688→0.721)
at conc≈22 with better TTFT/tps — but short of the static `mamba256→160`
envelope (+20 pp). The mechanism is correctness-complete and 3c-confirmed;
scaling it is tuning (#290), and the migration lever's value is unmeasured
(#295). Folder: `4_e2e/cc_traces_headline/`.

**Phase 6 — agentreplay migration + first-principles cleanup (2026-06-15).**
RQ1 moved to the agentreplay token-exact CC harness (Qwen3.5-9B, cc_qwen3p5_9b.jsonl); synthetic bench_serving / reproduce-table harnesses are retired (#328).
K_big snapshot suppression was removed (#325): it was a Path-A heuristic absent from design, and it freed a still-running request's suffix KV, the agentreplay sys-arm crash root cause.
Then four first-principles gaps were closed (commit on HiMA): the m2k mamba floor now reserves the LIVE working set `(m_used - mamba_evictable_size) + margin` instead of the nominal `max_running` cap (#297, which over-reserved ~2/3 of the pool and aborted 59/72 m2k fires in the 262k run); the planner collapsed to its single arg-max NB path (dead `edge_trigger` / `_net_benefit_ok` removed); dead regime-label cost logging removed (migrate `c_m` curve untouched); and mamba eviction now consumes one `_iter_mamba_victims` generator shared by `evict_mamba` and the cost predictor, mirroring the KV side.

---

## What's next (the actionable backlog)

Five tracks. **A** (headline win) and **C** (calibration) are the near-term
critical path — the win is mechanism-complete but modest, and #276 is the
likeliest reason. **B** (BOCPD) is the largest remaining *design* gap and runs
in parallel. **D/E** harden coverage/safety. **F/G** are conditional / hygiene.

### Track A — Close & scale the headline win (Phase 5)

The mechanism works. Post the arena `need_sort` fix, the cc recurrent-bound cell
(cap=16, Qwen3.5-9B, N=3 request-bounded) already shows **cache_hit +50 pp
(0.31→0.81), out-TPS +77%, P99 TTFT −38%**, zero-downside at slack. The static
ceiling (cap=256, unconstrained mamba) is **0.936**, so \sys closes ~79% of the
baseline→ceiling gap — ~13 pp headroom remains. Now push toward the ceiling and
broaden the RQ1 workload set. (RQ1 = three cc-trace workloads: recurrent-bound
[done], high-concurrency/short [#317], dynamic/shifting [#318].)

| Task | Purpose | Target (falsification) | Dep |
|---|---|---|---|
| #290 | Scale the cross-fire win toward the static envelope — tune fire magnitude (`PAGES_PER_FIRE`), cadence (`XPOOL_COOLDOWN_S`), NB margin. Baseline PF=8/COOL=12 → cache_hit 0.81; ceiling 0.936. | N=3 paired-delta at cap=16/12: cache_hit moves from 0.81 toward the 0.936 static ceiling with **no regression** + bounded over-harvest (#268), OR a measured reason it plateaus. | calib (#276) helps |
| #317 | RQ1 workload 2 — construct a high-concurrency/short-session workload from the cc trace (naturally recurrent-bound: mamba fills, KV slack, no artificial cap). | N=3 \sys vs default; out-TPS / P99 TTFT + cache_hit; fills the 2nd RQ1 row. | — |
| #318 | RQ1 workload 3 — construct a dynamic/shifting workload from the cc trace (binding pool flips across phases; \sys re-balances at each transition). | N=3 \sys vs default; per-phase out-TPS / P99 TTFT + transfers; fills the 3rd RQ1 row. | — |
| #323 | **Metric-significance A/B** (backs paper §3.2 pricing-eviction + `sec:eval`): `--eviction-policy lpb` vs `lru`, Budgeter on, cross-pool-pressure cc workload. Two layers: necessity (LRU has no native eviction cost, so the cost model is required to compare evict vs transfer) and optimality (LPB order lower-bounds `c^evict(X)` vs LRU `n_b≡1`/recency victims). | N=3 paired-delta: LPB arm lower `c^evict` + better P99 TTFT / out-TPS / cache_hit, both arms priced≡evicted; OR a measured reason the order does not move e2e. | #318/#321 |

**#290 outcome (2026-06-10): fire-tuning is exhausted; the win is warm-up-bound,
not knob-bound.** Swept `PAGES_PER_FIRE` {8,16,32} × `COOLDOWN_S` {12,6,4} at
cap=16 (cc, Qwen3.5-9B). Request-bounded N=3 at the most aggressive useful config
(PF=16/COOL=6) gives inter cache_hit **0.819 ± 0.010** vs the PF=8 default's
**~0.806** — equal within rep variance (the directional single-run 0.864 was a
short-workload artifact, did not survive the full 784-request set). Evidence the
plateau is *not* a tunable deficit: PF 8→16→32 all land ~0.81, and the
recurrent-pool usage after growth is only 6–14% (`mamba_usage` mean 0.14, last
quarter 0.06) — the pool is neither fire-limited nor capacity- nor NB-limited.
The residual gap to the static ceiling (cap=256 → 0.936) was attributed to an
**irreducible dynamic warm-up transient**.
**SUPERSEDED (2026-06-15):** this #290 run predated the #297 floor fix, so its m2k fires were crippled by the nominal-cap floor (the 262k run later showed 59/72 m2k fires aborting on that floor, mamba idle).
The "fire-tuning exhausted, win is warm-up-bound" conclusion must be re-measured on the fixed code (working-set floor + agentreplay 262k); treat the cap=16 synthetic numbers here as historical.
| #295 | Validate the **KV live-migration** lever earns an e2e win (or bound when it does). Migration uniquely frees a whole page when free+drain can't. | Idle-GPU A/B `SGLANG_XPOOL_KV_MIGRATE` 1 vs 0: migrations fire (>0); a k2m fire the OFF baseline **refuses** now succeeds; cache_hit +X pp where X is *predicted* from pages consolidated; out_tps within tol. Negative/bounded result documented if drain always suffices. | #271 ✅ |
| #100 | Graceful degradation under double-saturation (complement to #295 — checks migration is *harmless*, not a win). | Crash-free; `throughput_inter ≥ 0.95 × off`; Lemma-A1 break non-fatal. | — |
| #102 | Super-capacity burst graceful defer. | Crash-free; `total_admitted_inter == off`; `queue_p99` ratio ∈ [0.9, 1.1]. | — |

### Track B — BOCPD continuous-time Budgeter (Phase 4, largest design gap)

The Budgeter today fires on a threshold/NB rule; `design.md` specifies a
continuous-time Bayesian change-point trigger. This track lands that.

| Task | Purpose | Target | Dep |
|---|---|---|---|
| #201 | Implement vanilla BOCPD per `design.md` §"Trigger rule" — 6 params (`λ_iter`, `σ_obs`, `μ_0`, `σ_0²`, `cooldown_min`, `amortize_horizon`); per-pool posterior, marginalized mean, cooldown gate. | compiles + S3/S5/S7 sub-second smoke pass | — |
| #202 | Run the 8 falsification harnesses (replay_eq, phase_shift, spike_rejection, fire_step_neutrality, bimodal_dt, slow_drift, cp_burst, disable). | 4 hard pre-ship gates pass: S1 stationary, S2 phase-shift, S4 fire-step neutrality, S6 slow-drift bound | #201 |

**#276 outcome (2026-06-10): κ_M=0 is the EXPECTED hybrid-calibration state, not
a bug — resolved.** Ran the real `calibrate.sh` probe on Qwen3.5-9B / H200 (clean
GPU): fitted `c_KV(L) = 1.02e-07·L² + 0.0246·L + 5.97` ms (κ_KV fully
non-degenerate), `c_M = 0` (M_ALPHA=M_BETA=0 — the hybrid forward folds the total
prefix recompute into κ_KV, the documented "HYBRID CAVEAT"), L*=0; saved to
`dev/eval/cost_model/kappa_fit.json`. This validates the current drain-gate design
on the real model: the gate keys on κ_KV non-degenerate (sound), and κ_M=0 is
correct, not the #276 "bug" (that was the OLD gate failing closed on κ_M=0, since
fixed). The 35B builtin default underestimated 9B's κ_KV ~13× (γ 0.44 vs 5.97), so
deployments should load the model-specific JSON. **The win is robust to κ**:
request-bounded N=1 with the 9B κ gives inter cache_hit **0.782** (vs the 35B-κ
baseline 0.806, within noise; +66% out-TPS, ZERO-DOWNSIDE PASS) — and despite the
9B κ firing **698** transfers (vs 69–282 under the builtin, because the larger
κ_KV raises grow-benefit), cache_hit did not rise. This independently corroborates
the #290 finding: the win is **warm-up-transient-bound**, not fire- or
κ-pricing-bound. Deployments should use `kappa_fit.json`; the headline win stays
~+50 pp.

### Track C — Cost-model calibration (cross-cut; gates win *quality*)

| Task | Purpose | Target | Dep |
|---|---|---|---|
| #276 | **κ_M fits to zero** (`M_ALPHA=M_BETA=0`) → the reuse-aware drain cost collapses to ~0, so `_cross_drain_allowed` falls closed and the drain win is capped. Re-calibrate `c_M` (mamba recompute curve). **Likely the main reason #290's win is modest.** | non-degenerate κ_M from a real prefill/recompute probe; drain cost prices a hot mamba cache as expensive; #290 win improves | — |
| #217 | `c^xfer` drift detector with a one-shot warning (#210 Gap 4) — detect when the boot-probed transfer cost drifts from observed. | warns once when observed c^xfer departs the probed value beyond a band | — |
| #157 | `w_q` calibration from observed queue dynamics. | `w_q` fit from real queue traces, not a constant | — |

### Track D — Admitter cost-program coverage (Phase 3/4)

| Task | Purpose | Target | Dep |
|---|---|---|---|
| #184 | Refuse-rate counter + α > α_max stress workload (paper line 109): planner increments a counter when even Migration-expansion can't satisfy `n`. | refuse-rate ≥ 1/s on synthetic high-α trace; 0 on α ≤ 0.85 | — |
| #185 | `cov_action_coverage` 7-action × 4-cell sweep ({below-sat, fragmenting, saturated, all-LIVE-burst} × {LRU, LPB}); assert each action ≥ 1 % in some cell + below-sat zero-downside (migrate ≤ 0.1 %). | passes `design.md` §cov_action_coverage decision rule | #184 |
| #158 | Workload sweep feeding #185's coverage cells. | cells exercise the intended regimes | — |

### Track E — Cross-fire correctness / safety hardening

| Task | Purpose | Target | Dep |
|---|---|---|---|
| #268 | Cross-fire **over-harvest**: the Admitter can starve the mamba batch under heavy KV pressure (grow-side too aggressive). | bound the per-fire harvest so the source pool's running batch isn't starved; out_tps regression bounded | — |
| #285 | **both-full guard** keys on cache-inclusive occupancy; should key on **active-vs-cold** so a both-pools-full-of-COLD-cache state still allows a beneficial cross-fire. | guard admits a fire when both pools are full of cold (evictable) cache but blocks when both are genuinely active | — |

### Track F — Conditional architecture (Phase 6; only if measurements demand)

| Task | Trigger | Purpose |
|---|---|---|
| #175 | tps regression ≥ 10 % at headline, new crash class, or mechanism-not-engaging despite high-α | Unified per-ID cap architecture — single arena-driven capacity tracking; removes the asymmetric high-water vs per-ID mechanism in `MambaPool`. |
| #203 | #202 S6 fails (long-batch drift → wrong-direction fires) | BOCPD within-segment random walk (Saatçi et al. 2010): add `σ_q²`, track drift without false CPs. |
| #261 | a sliding-window-attention model needs cross-pool | SWARadixCache + LPB support (separate TreeNode hierarchy). |
| #274 | the fixed free→drain→migrate order leaves value on the table | Planner cost-merge of drain ↔ migrate (vs the current fixed cumulative order). |

### Track G — Hygiene / correctness gaps (cross-cut)

| Task | Purpose | Target |
|---|---|---|
| #272 | Chunk-sizing: `tokens_per_chunk = max(1, chunk//per_token)` silently clamps while `MultiTensorArena` requires `chunk % per_token == 0` — a >2 MiB per-token config boot-crashes in the arena instead of erroring clearly. | a >2 MiB per-token config gets a clear pre-arena error (or auto-sized chunk) |
| #161 | Test coverage for the allocator `need_sort=True` + `free_group` paths. | both paths covered |
| #266 | `design.md`: replace residual `file.py:NNN` line-number refs with function/class names (per the no-line-numbers-in-docs rule). | no line-number refs remain |
| #230 | Stale `fire_only` references in `verify/D6/README.md` + `verify/D11/README.md`. | refs updated to current mechanism names |

---

## Tracking conventions

- **Every task has a falsification criterion derived from `design.md`.** Write
  the failing test FIRST (per the bug-workflow discipline), then satisfy it.
- **`pending` = eligible to start** unless it lists a `blocked by`.
- **Bench discipline**: any A/B (#290, #295, #100, #102) runs only on an idle
  GPU — check `nvidia-smi` free mem + util first; contended runs invalidate the
  comparison.
- **Perf targets are pinned, not "no-regression"**: a perf task asserts the
  design target (the envelope, the predicted Δ), not merely `≥ baseline`.
- **Spec changes go in `design.md`; pure re-orderings go here.**

## Start-of-next-session checklist

1. Re-read `design.md` §"Status" for the area you're implementing.
2. `TaskList` → pick a pending task (lowest ID / critical-path first); confirm
   it isn't blocked.
3. Read the area `README.md` + the `design.md` scenario for the falsification
   criterion.
4. Write the failing test FIRST. Implement. Verify the criterion. Update the
   folder README + this PLAN if the roadmap shifts.
