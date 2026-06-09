# Scenario 5 — RESULTS (Tier-1 controlled harness)

Reproduce: `python harness/sweep.py` (deterministic, CPU, ~1 s).
Self-test: `python harness/verify.py` (6 invariants, must stay green).

**Fidelity caveat (read first).** Tier-1 models the *scheduling logic and
timing* (which unit is evicted when, and was it about to be reused), **not
GPU kernels**. Absolute magnitudes are model-dependent and intentionally
pessimistic about LRU (it evicts on every allocation, so parked
about-to-return contexts are always its victims). The **qualitative trends**
below are the findings; Tier-2 e2e grounds the absolute numbers. The
architecture stays design-ideal — these results *map where each mechanism
binds*, they do not license cutting any of it.

---

## Headline findings

### UQ1 + UQ4 — value of hint-steering vs LRU, by pressure regime
| regime (pool / demand) | LRU hit | ours-fresh hit | re-prefill gain |
|---|---|---|---|
| under (75%) | 0.11 | 0.55 | **+49%** |
| critical (50%) | 0.04 | 0.23 | **+20%** |
| tight (33%) | 0.00 | 0.07 | +7% |
| saturated (20%) | 0.00 | 0.03 | +3% |

**Finding:** the daemon's steering value is **strongly regime-dependent — it
binds most where there is headroom to make a smart keep/evict choice, and
fades toward LRU under saturation.** This *quantitatively explains the A3
observation* (saturated → `ours ≈ baseline`, `migrate ≈ 6`): at 99% HBM there
is almost nothing the policy can do better than LRU because almost nothing
fits regardless. The daemon earns its keep in the **under-/critically-loaded**
regimes, not the saturated one.

### UQ2 — hint-freshness latency budget (the headline)
The `ours` re-prefill rises monotonically with hint delay and converges to
the LRU ceiling. **Knee ≈ 8 steps** in both the critical and tight regimes —
and 8 steps is exactly the workload's `phat_lead` (the predictable-reuse
lead time).

**Finding:** the **hint-latency budget = the predictable-reuse lead time** —
the gap between "reuse becomes predictable" (p_hat starts rising) and "reuse
happens" (the tool returns and re-locks). Hints staler than that can't
protect an about-to-return prefix; the inline eviction races them. This is
the measured answer to *"how low must the inline/hint latency be"*: **fresh
to within one reuse-lead-time, not arbitrarily fast.** For agentic tool
round-trips this lead time is short → hints are genuinely latency-sensitive
(consistent with the earlier "100 ms batching is not unconditionally
harmless" analysis).

### UQ3 — imperative migrate contribution
| regime | migrate OFF | migrate ON | Δ |
|---|---|---|---|
| all four | — | — | ≈ 0 (±1 ctx) |

**Finding:** as an *eviction* mechanism, imperative migrate adds ~nothing on
top of inline steering in every regime (consistent with the live `migrate ≈ 6`).
**Caveat:** the harness models migrate only as proactive demote — it does
**not** model imperative migrate's *other* roles (cross-tier promotion,
predictive load-ahead, joint multi-subpool relief), which are where that path
is expected to earn value. So this says "imperative-as-eviction is
redundant with inline steering," **not** "imperative is useless." Promotion
value needs its own slice (Tier-2, future).

### UQ5 — reuse-structure sensitivity (critical regime)
| reuse pattern | gain (ours vs LRU) |
|---|---|
| imminent, frequent (short tool gap) | **+41%** |
| delayed (long tool gap) | +11% |
| high-churn (many rounds) | +12% |
| one-shot (no reuse) | +100% (ours drops the never-reused ctx LRU clings to) |

**Finding:** steering shines when reuse is **predictable and frequent**
(short tool round-trips — the dominant agentic pattern) and on **one-shot**
contexts (it knows they won't reuse and frees them, where LRU holds them by
recency). It helps least when reuse is far-future (long tool gaps) — there
the lead time is long so even LRU's recency is an okay proxy.

---

## What this documents (for the paper / decisions)

1. **The daemon's eviction-steering value is a function of pressure**, peaking
   under headroom and fading at saturation — so do-no-harm at saturation
   (which we proved: 0 rejects, latency ↓12×) is the *right* bar there, and
   the *benefit* story lives in the under-/critically-loaded regimes.
2. **Inline (hint-steered) eviction is the latency-critical path**, with a
   budget of one reuse-lead-time; the imperative migrate path (now ~100 ms,
   #228) is fast enough because it is *not* the eviction workhorse — it is the
   promotion / cross-tier / predictive path, whose value needs its own e2e
   slice.
3. The architecture stays design-ideal; these are the regimes/limits we now
   *understand and have on record*.

---

## Tier-2 (e2e real-stack) — next increment

Headline arms on harbor/terminus-2 via `run_k.sh`, N≥3, mean+std:
`lru` · `ours-fresh` · `ours-delay-{250,500}ms` · `ours-migrate-off`, swept
across the under/critical/saturated pressure regimes (via `MAX_TOTAL_TOKENS`).
Requires the `AGINFER_HINT_DELAY_MS` knob (daemon, env-gated). Grounds the
Tier-1 trends in real TTFT / throughput / cache-hit and confirms the latency
budget on hardware.
