# S1 on a realistic high-concurrency fleet — findings (the honest scope)

This is the result of running S1 the way it should be evaluated: **real Claude Code
agent traces, replayed at fleet concurrency, against a *correct* baseline.** It both
(a) confirms the S1 mechanism where the substrate works and (b) precisely locates why
it does **not** generalize to a saturated fleet on V4-Flash. Read alongside
[`RESULTS.md`](results/RESULTS.md) (the controlled win) — this file is the realistic-
fleet companion.

## Workload
`scenarios/replay/traces/combined_all.jsonl` — **90 programs** (30 from the captured
`a3real.jsonl` + 60 from local CC session transcripts, `convert_cc_traces.py`), real
per-turn prompt sizes / output lengths / tool-gap timing, a shared ~16K system+tools
prefix across all programs (cross-program KV sharing). Replayed teacher-forced
(`max_tokens=output_len`, `temp=0`) so every arm does identical token work. Pool =
131072 (HBM) → working-set ≫ cache (the realistic over-subscribed fleet).

## What we measured (single trial each; the gaps are large + consistent)

| arm / lever | eviction scorer | write policy | cache-hit | note |
|---|---|---|---|---|
| baseline B | LRU (`lru_score`) | selective | **49.0 %** | true LRU baseline |
| ours-local | `ours_greedy_score` (local p_hat) | selective | 48.4 % | ≈ LRU |
| ours-program-aware | `aginfer:hint_v_u` (daemon p_hat) | selective | 47.5 % | ≤ LRU |
| B + full write-through | LRU | `write_through` | 42.1 % | overhead, no tier gain |

**Baseline opportunity:** ~21.9M of 42.4M prompt tokens are re-prefilled (51 %), of
which **~20M (47 %) is *recoverable* eviction-reuse** (a prefix that was cacheable,
got evicted, then reused) — measured by `analyze_opportunity.py`.

## Three findings

1. **The opportunity is real and large** (~20M recoverable tokens, 47 %).

2. **The eviction-ORDER lever is moot here.** LRU ≥ ours-local ≥ ours-program-aware.
   When working-set ≫ cache, almost everything churns regardless of order, and
   recency is a *strong* reuse predictor for agentic prefixes (the next turn extends
   the most-recent prefix). Value-aware eviction does not beat LRU on the fleet.
   *(Note: this was only measurable after fixing an experiment bug — `run_k.sh` forced
   `ours_greedy_score` for BOTH `a3` and `a3_kvoff`, so the value-eviction lever was
   in both arms and canceled; every prior "ours ≈ a3_kvoff / do-no-harm" result is
   explained by that. `a3_kvoff` was never LRU.)*

3. **The tier-RETENTION lever — the only one that can capture the 20M — does not
   engage, blocked UPSTREAM of the daemon.** Selective write-through is
   `hit_count ≥ threshold`, so under churn nodes are evicted before any hit and never
   written. Forcing `--hicache-write-policy write_through` does not help: at runtime
   **0 mooncake-DISK writes** and ~0 host write-backs, with startup logging
   `storage backend does not support layer` — **V4-Flash's hybrid `swa` + `deepseek_v4_c4`
   paged-pool KV layout isn't supported by the mooncake storage write path.** The DISK
   tier is effectively dead for this model, so evicted prefixes are dropped, not
   retained → recompute. There is no functioning tier for the value-aware manager to
   operate on.

## The full matrix (2 regimes × 3 levers) — nothing beats LRU; eviction order is zero-sum

| cache-hit | LRU (`lru_score`) | ours-local (`ours_greedy_score`) | ours-program-aware (`aginfer:hint_v_u`) |
|---|---|---|---|
| **gap1** (saturated) | **49.0 %** | 48.4 % | 47.5 % |
| **parking** (10–30 s gaps) | ~47 % | 47.6 % (a3_kvoff) / 46.8 % (a3) | **47.1 %** |

All six cells land in a **46.8–49.0 % noise band**; LRU(gap1) is the highest. The
decisive cell — **program-aware value-in-eviction *at the parking regime*** (where the
parked prefix is cold-by-recency but reuse-imminent-by-program-state, and value-in-
eviction sidesteps the non-leaf-demote problem entirely) — is **also flat (47.1 %)**.
**Root:** at working-set ≫ cache, eviction order is a **zero-sum game** — any policy
holds the same ~47 % of the reuse, just a different 47 %; protecting the parked prefix
merely evicts another reused one. **No eviction policy can beat LRU when the cache is
the bottleneck.** The only lever that raises the captured fraction is *expanding the
effective cache* (the tier), which is dead (above). This is why ours ≈ baseline
across every regime and lever — confirmed, not a single-regime artifact.

## Tested + falsified: "more parking (HBM headroom) will fix it" — it doesn't

Hypothesis: gap1 saturated HBM, so a parking regime (10–30s gaps → ~18 of 90
concurrently active → headroom + stable demote-leaf targets) would let the
demote/promote engage. **Result (`combined_park.jsonl`): still flat** — ours 46.8%
vs baseline 47.6%, re-prefill reduction 0.4%. Headroom DID improve (promote-declines
2876 → 106), but **the demote still fails — 5714 `remove_hbm_not_device_leaf`
(more than gap1).** Root: in a **shared-prefix** agentic workload (all programs share
the 16K system block + common prefixes), a parked program's tail node is a **non-leaf**
(other programs' continuations are its children), and you cannot evict/migrate a node
that has children. **The explicit node-migrate demote plane is structurally
incompatible with shared-prefix fleets, regardless of the gap regime.** This also
explains why the controlled 41–91 % wins worked: a *single* program's prefix is a
leaf with no sharing, so the demote/promote can touch it; a fleet's shared prefixes
are not.

## The unifying root cause: HBM saturation defeats the mechanism (confirmed)

The deeper, confirmed reason all three levers fail is a **cascade rooted in HBM
saturation**, which is inherent to a high-concurrency fleet with short gaps:
* `running-req` sat at **81–86 of 90 programs concurrently active** — HBM is full.
* the daemon's predictive promote declines with the exact reason
  **`promote_load_back_declined:evict_short: needed=21760 > evicted=0`** — there is no
  HBM headroom to land a promoted prefix.
* the explicit demote (which would free that headroom) fails
  (`remove_hbm_not_device_leaf` ×5368) — under fleet churn the target node gains a
  child (another program's continuation) between the daemon's state dump and the
  apply (a TOCTOU the synchronous in-process eviction doesn't have, but the
  in-process value scorer doesn't beat LRU anyway).
→ **promote needs room ← demote can't free room ← saturation.** This is exactly the
documented **GPU/HBM-idle premise**: S1's predictive promote re-stages a prefix
during the *idle* gap; at fleet saturation the gap isn't idle (other programs fill
HBM), so the lever can't fire. The realistic win regime is **moderate concurrency
with genuine tool-parking** (HBM idles during gaps) — not a saturated fleet.

## Conclusion

On a realistic saturated V4-Flash fleet, **ours ≈ HiCache+LRU (~48 %)** — and the
reason is **not** the scheduler. It is that the multi-tier KV store doesn't populate:
the eviction-value lever genuinely doesn't beat LRU, and the tier-retention lever is
blocked by a HiCache↔mooncake layer-layout incompatibility for hybrid-attention
models. The 41–91 % controlled wins in [`RESULTS.md`](results/RESULTS.md) are **real
but forced** — the explicit warm (`/generate(max_new_tokens=0)`) drives the storage
prefetch+load-back for one chosen prefix, so the tier works *on demand for that
prefix*. Passive fleet replay never triggers that path, and the DISK backend can't
hold V4's layout anyway.

## And the moderate regime (6 programs) — where S1 should most win — ours HARMS

`cc6_park.jsonl` (6 real CC programs, parking gaps, sum-peak 298K vs 131K pool = 2.3×
over — the regime that *matches the controlled win's structure*): **ours is worse on
every metric.** TTFT p50 **2417 ms (ours) vs 1764 ms (baseline)** — 37 % slower;
cache-hit **45.4 % vs 54.2 %**; re-prefill **+19 %**. The a3 daemon logged 847
`remove_hbm_not_device_leaf`, 93 promote-declined, 72 evict_short. **The daemon's
failed-migrate churn + admission disruption actively degrade sglang's own cache** —
ours violates even do-no-harm here. (The value-in-eviction *alone*, without the
explicit migrate plane, is ≈ baseline — neutral, not winning; the explicit migrate
plane is what harms.)

## Bottom line across the FULL real-trace exploration

Across **every regime (saturated fleet / parking fleet / moderate 6-program) and
every lever (LRU / local-value / program-aware value-in-eviction / daemon
promote+demote)**, on real CC-trace replay, **aginfer's value-aware scheduling never
beats sglang HiCache+LRU** — best case it equals it (do-no-harm), worst case the
explicit-migrate churn harms it (cc6_park, −37 % TTFT). The controlled synthetic wins
(41–91 %) — which used a *single isolated leaf prefix* hand-forced onto a working
tier — **do not reproduce on real shared-prefix CC traces.** The design's central
premise (value-aware > LRU for agentic KV) is not supported empirically on realistic
workloads on this stack; the predictive-promote *mechanism* is real but its benefit
is confined to the controlled isolated-prefix microbenchmark.

## UPDATE (N=3, rigorous) — value-aware EVICTION *does* beat LRU at MODERATE concurrency

The earlier "never beats LRU" was measured on the heavy 90-program fleet (everything
churns → eviction-order moot) and an N=1 cc6_park run that *included the harmful
write-through*. Re-measured properly — **N=3, write-through dropped, fresh LRU
baseline, mean±std** — on `cc6_park` (6 real CC programs, 2.3× over-subscribed, the
MODERATE regime):

| metric (cc6_park, N=3) | ours (a3 + `hint_v_u`) | baseline (LRU) | verdict |
|---|---|---|---|
| **cache-hit** | **71.6 % ± 3.4** | **55.8 % ± 5.5** | **ours WINS +16pt (separable)** |
| re-prefill tokens | ~1.53 M | ~2.57 M | **ours −42 %** |
| resume-TTFT | 2481 ± 45 ms | 2173 ± 325 ms | within noise (ours slightly worse) |
| makespan / e2e / ttft | — | — | all within noise |

**Honest reading:** a **real, separable GOODPUT win — 42 % fewer re-prefilled tokens** —
because value-aware eviction keeps the reused prefixes LRU drops. It is **NOT a latency
win**: every latency/makespan metric is within noise (the daemon's promote/migrate/
admission adds resume-path overhead; the gap-paced replay masks compute savings in
wall-time). So the supported claim is **value-aware eviction > LRU on goodput at
moderate over-subscription** — NOT the S1 predictive-promote *latency* claim (still
unsupported on realistic traces). The regime matters: at the 90-program fleet the same
lever is moot (heavy churn); at 2.3× over-subscription *which* prefixes you keep
matters. (Write-through with hint-awareness was implemented but is HARMFUL here — 37 %
hit, and caused intermittent scheduler stalls — so it is OFF.) Reproduce:
`run_replay_pressured.sh cc6_park.jsonl 3 1 131072 "a3" 1 aginfer:hint_v_u` vs
`... "a3_kvoff" 1 lru_score`; analyze with `analyze_opportunity.py` + `resume_ttft.py`.

### Attribution + validation (what actually causes the win)

* **It's the program-aware eviction, not the promote.** Trial c2 fired **0 promotes**
  yet hit 70.9% (same as promote-heavy c1/c3) → the win is `hint_v_u` keeping the
  reuse-imminent prefix in HBM (a TRUE compute saving: prefix stays cached → no
  recompute, no promote cost), not the predictive promote shifting compute into gaps.
* **It requires the daemon's program-aware foresight.** Isolation (N=3) —
  `a3_kvoff + ours_greedy_score` (local hits/age value, NO daemon hints): **50.8 % ±
  2.2 ≈ LRU 55.8 %** (even slightly below). Local value-eviction does NOT beat LRU;
  the win is +21pt over local value, entirely from the daemon. The daemon's event-driven
  p_hat (knowing which program will resume from the tool-call lifecycle) is what lets
  the scorer keep the right prefixes. This is the core aginfer thesis, confirmed.
* **Why latency is neutral despite −42% prefill:** resume-TTFT is queueing-bound (a
  resuming program queues behind others at moderate concurrency), so per-request
  latency is similar even though ours does far less prefill *compute*. **The win is
  throughput/capacity (compute saved), not per-request latency.** GPU was often idle
  (454/their samples at 0 running-req), confirming the moderate-headroom regime.

## Honest claim scope for the paper

* **What holds:** when the reuse-imminent prefix sits on a *functioning* tier, program-
  aware predictive promote moves the re-acquisition off the resume critical path →
  41–91 % faster TTFT (controlled, N=3) — the mechanism is correct and measured.
* **What does NOT yet hold:** that this wins on a *saturated fleet* on V4-Flash —
  because the substrate's multi-tier store is non-functional there.

## The real prerequisite (the next work, in order)

1. **Make the tier populate under load** — fix HiCache↔mooncake for V4-Flash's hybrid
   pools (or use a storage backend / `hicache_mem_layout` that supports them), so
   write-through reaches DRAM **and** DISK without blocking prefill. *(This is sglang/
   mooncake infra, upstream of the daemon.)*
2. **Then** layer ours' differentiator: **predictive, value-aware write-through** —
   retain reuse-imminent prefixes *before* eviction (by the daemon's value), which the
   hit-count-reactive default structurally cannot under churn.
3. Re-measure ours vs LRU on `combined_all` with the tier alive.

## Reproduce
```bash
# baseline (true LRU)         vs   ours value-eviction        vs   program-aware
bash scenarios/replay/run_replay_pressured.sh traces/combined_all.jsonl 1 1 131072 "a3_kvoff" 1 lru_score
bash scenarios/replay/run_replay_pressured.sh traces/combined_all.jsonl 1 1 131072 "a3_kvoff" 1 ours_greedy_score
bash scenarios/replay/run_replay_pressured.sh traces/combined_all.jsonl 1 1 131072 "a3"       1 aginfer:hint_v_u
python scenarios/replay/analyze_opportunity.py <results_dir> 16000
# launch with dangerouslyDisableSandbox (process ops); see memory sandbox-blocks-process-ops
```
