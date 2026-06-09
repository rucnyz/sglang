# Scenario 5 — Eviction & Hint-Latency Characterization

**Purpose.** Comprehensively *characterize and document* how the
design-ideal aginfer system behaves across pressure regimes, workload
shapes, and the hint-latency dimension — so we **understand and record**
(for the paper and future reasoning) where each mechanism contributes and
what its latency budget is.

This is **measurement + documentation, not a feature-cut decision.** The
architecture stays aim-for-design-ideal; this suite instruments it. A
result like "LRU is as good as hint-steering in regime X" is a *recorded
fact about regime X*, not a license to delete the steering — the ideal
design keeps the mechanism and we document where it does/doesn't bind.

---

## 1. The three eviction-related layers (what we characterize)

The design-ideal system frees / moves KV through three layers with very
different latency profiles. Naming them precisely is the whole point:

| layer | who | latency | role |
|---|---|---|---|
| **L1 inline eviction** | sglang radix cache | **0** (synchronous in the alloc path) | frees HBM the instant a forward pass needs space; *unavoidable* — it cannot wait for the daemon |
| **L2 hint steering** | daemon → `PUT /aginfer/hints` → sglang inline scorer (`ours_greedy_score`, #177) | hint *freshness* (≈ outbound dispatch, now ~100 ms) | makes L1 evict by V_u (p_hat·reuse) instead of blind LRU |
| **L3 imperative migrate** | daemon → `POST /aginfer/migrate` (§6/§9) | dispatch latency (now ~100 ms p99, #228) | explicit cross-tier transitions, predictive promote, joint multi-subpool relief |

Plus **admission** (pause/resume) as an orthogonal pressure-relief lever.

**Empirical context that motivates the suite** (from the post-#228 a3
run, A3 saturation, HBM 99%): `migrate_enqueued ≈ 6` while the pool churns
continuously — i.e. **L1 inline is already the workhorse in saturation and
L3 imperative is a rounding error there.** So the daemon's saturation value
is hypothesized to be *L2 hint-steering of L1*, and the open question is how
much that steering is worth and how fresh hints must be. This suite turns
those hypotheses into measured, documented facts.

---

## 2. Understanding questions (UQ)

| | question | isolates |
|---|---|---|
| **UQ1** | How much does L2 hint-steered inline eviction reduce re-prefill / cache-miss vs sglang-default **LRU**, per regime? | value of steering |
| **UQ2** | **Hint-freshness latency budget** — how stale can hints be before the steering benefit decays? (the delay gradient) | the inline latency requirement |
| **UQ3** | L3 imperative-migrate contribution *vs* L1+L2 — in which regimes does imperative carry real load (not ~6)? | value of the imperative path |
| **UQ4** | Which **pressure regime** makes daemon steering matter at all (vs LRU sufficing)? | when the daemon binds |
| **UQ5** | How does **reuse structure** (imminent / delayed / shared-prefix / none) modulate UQ1–UQ4? | robustness across workloads |
| **UQ6** | Under **allocation bursts**, how much inline eviction is forced + does it preempt/stall? | confirms L1 is structurally unavoidable |

---

## 3. Dimensions swept

- **D1 pressure**: under-subscribed (HBM headroom) · critical (~θ_hi) · over-subscribed (A3 saturation).
- **D2 reuse pattern**: imminent (fast tool round-trip) · delayed (long tool) · shared-prefix (agents share a system prompt) · none (one-shot).
- **D3 eviction policy**: `lru` (sglang default) · `ours_greedy_score` (hint-steered) · `const_v_u` (ablation: steering present but uniform score — isolates "does the *content* of V_u matter").
- **D4 hint freshness**: immediate · 50 ms · 100 ms · 250 ms · 500 ms · 1 s · ∞ (never pushed = stale-frozen). **The D4 gradient is the headline UQ2 measurement.**
- **D5 imperative migrate**: on · off.
- **D6 admission**: on · off.
- **D7 burstiness**: steady · synchronized tool-return bursts.

Full cross-product is intractable and unnecessary. Each **slice** below
fixes all but 1–2 dimensions to isolate one UQ.

---

## 4. Scenario slices

| slice | fixes | sweeps | answers |
|---|---|---|---|
| **S1 policy value** | pressure=saturated, reuse=imminent | D3 (lru / ours / const_v_u) | UQ1 — does steering beat LRU; does V_u content matter |
| **S2 hint-latency budget** | pressure=saturated, reuse=imminent, D3=ours | **D4 (freshness gradient)** | **UQ2 — the inline latency requirement** |
| **S3 imperative value** | pressure=saturated, D3=ours-fresh | D5 (migrate on/off) | UQ3 |
| **S4 regime sweep** | reuse=imminent | D1 × D3 (lru vs ours) | UQ4 — where the daemon binds |
| **S5 reuse sensitivity** | pressure=critical, D3=ours-fresh | D2 (reuse patterns) | UQ5 |
| **S6 burst stress** | pressure=critical | D7=bursts, D3 ∈ {lru, ours} | UQ6 — forced inline + preemption |

---

## 5. Two evaluation tiers

### Tier 1 — controlled harness (`harness/`, deterministic, CPU, seconds/arm)
A **trace-replay simulator** of the radix-cache eviction layer: a synthetic
workload generator emits a token-accurate event trace (program arrivals,
prefills, decode steps, tool calls/returns with known reuse) sized to a
configurable HBM capacity; an eviction engine replays it under each
(D3 × D4 × D5 × D1 × D2) arm and counts the outcome metrics. No GPU.

This is where **comprehensive coverage** lives — the full matrix runs in
seconds, every arm deterministic and reproducible, so we can sweep D4 at
fine granularity and repeat across seeds. It models L1 (capacity-driven
eviction), L2 (scorer reads a hint table with injectable staleness), L3
(periodic imperative migrate at a configurable latency), and the
lock-protection of actively-decoding nodes.

*Fidelity caveat (documented honestly):* the harness models the
*scheduling logic and timing*, not GPU kernels — it answers "which unit
gets evicted when, and was it about to be reused," not absolute latency.
Tier 2 grounds it.

### Tier 2 — e2e real-stack (`e2e/`, GPU, ~30 min/arm)
The **headline arms** on the real harbor/terminus-2 stack via `run_k.sh`
variants, to validate Tier 1 on real hardware: `lru`, `ours-fresh`,
`ours-delayed-{250,500}ms`, `ours-migrate-off`. Measures real TTFT (tool-
return), throughput, radix cache-hit, and the inline-evict-vs-imperative
volume split. N≥3 cycles/arm, mean+std (per the latency-claim discipline).

The **D4 e2e arms require the hint-delay-injection knob** (a daemon env
`AGINFER_HINT_DELAY_MS` that holds hints that long before enqueue — small,
env-gated, off by default; also the #230 batching lever).

---

## 6. Metrics

| metric | measures | layer |
|---|---|---|
| **re-prefill tokens** (primary) | recompute waste from evicting a unit that was then reused | eviction quality |
| evict-of-reuse-imminent count | the specific failure steering should prevent | L2 quality |
| radix prefix cache-hit rate | how well hot prefixes are retained | overall |
| p99 TTFT on tool-return | a re-prefill delays the next turn | user-visible |
| throughput (tok/s) | aggregate | user-visible |
| inline-evict tokens **vs** imperative-migrate tokens | mechanism attribution (the "6 migrates" question) | L1 vs L3 |
| preemption / queue-stall count | the must-evict-now wall | L1 forced |

---

## 7. Arm → config mapping

- **D3** via `SGLANG_KV_POLICY_MODULE` (unset = LRU; `baselines.sglang_adapter:ours_greedy_score` = steered; a `const_v_u` adapter for the ablation).
- **D4** via `AGINFER_HINT_DELAY_MS` (Tier 2) / harness `hint_delay_ms` (Tier 1).
- **D5** via `DAEMON_KV` enabled/disabled (imperative migrate path).
- **D6** via `DAEMON_ADMISSION`.
- **D1** via `MAX_TOTAL_TOKENS` + workload size (set total KV demand vs pool).
- **D2 / D7** via the workload generator (tool-gap distribution, phasing, prompt sizes).

---

## 8. What each outcome documents (interpretation guide)

- **S1: ours ≪ lru** → steering materially improves eviction quality; record the regime + magnitude. **ours ≈ lru** → in this regime LRU's recency proxy already captures reuse; document that the steering's value is regime-specific (design keeps it for the regimes where it binds).
- **S2 (the key one): re-prefill flat then rising across D4** → the knee is the **hint-latency budget**; its physical meaning is the unlocked gap between tool-return and re-prefill-lock. This is the number that says how low the L2/inline latency requirement actually is — measured, not guessed.
- **S3: migrate-off ≈ migrate-on under saturation** → confirms L3 is supplementary there (consistent with ~6 migrates); migrate-off degrading in *under-subscribed* regimes (where the daemon proactively pre-demotes with slack) → documents where imperative carries load.
- **S4**: the pressure threshold where ours starts beating lru = where the daemon begins to bind.
- **S6**: nonzero forced inline-evict + preemptions under bursts = empirical proof L1 is structurally unavoidable (you cannot route burst eviction through a 100 ms imperative path).

All findings land in `RESULTS.md` per slice (authoritative per-task doc),
mean+std for any latency/throughput claim, N≥3.
