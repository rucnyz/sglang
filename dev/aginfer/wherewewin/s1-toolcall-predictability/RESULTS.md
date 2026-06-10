# S1 — predictive promote across tool gaps: RESULTS

**Claim.** When an agent is parked in a tool call, its idle KV prefix is demoted
(HBM→DRAM→DISK) under memory pressure. A reactive HiCache baseline (**B**) re-acquires
that prefix on the next prefill — on the resume critical path. Aginfer (**ours**) has
**program-aware foresight**: it knows the program will resume *this* prefix at ~`T+ETA`,
so it predictively promotes it back into HBM *during the idle gap* — moving the
re-acquisition off the critical path.

## Headline (end-to-end, measured)

`s1_disk_microbench.py` — flood a 50K victim prefix out of HBM+DRAM, then time its
resume TTFT (N=3, V4-Flash tp2, GPUs 5,6):

| config | B (reactive) | ours (pre-staged) | win |
|---|---|---|---|
| DRAM ratio 1.5, flood 8×50K | 3180 ms | 284 ms | **91% / 2.9 s** |
| DRAM ratio 0.5, flood 4×50K | 3127 ms (cached=0 → recompute) | 299 ms (cached=49920 → HBM hit) | **90% / 2.8 s** |

ours pre-stages the prefix during the gap → resume hits HBM (~300 ms, only new tokens
computed). B has no foresight → the victim is lost under pressure → it re-prefills the
whole 50K (~3.1 s) on the critical path.

## Why the magnitude is a spectrum (offline tier-link characterization)

`link_characterize.py` (GPU5, real disk, KV≈1.17 KB/token):

| prefix | DRAM→HBM | DISK→HBM |
|---|---|---|
| 12K | 0.44 ms | 4.7 ms |
| 24K | 0.82 ms | 48.7 ms |
| 50K | 1.28 ms | 78.9 ms |
| 100K | 2.63 ms | 141 ms |

The win = the cost S1 moves off the critical path, which depends on where the evicted
prefix lands — **controllable via tier-2 (DRAM, `--hicache-ratio`) and tier-3 (mooncake)
sizes** (HBM itself can't be shrunk: V4-Flash deadlocks below a min pool). **All three
tiers count** (the lever is the same at every tier; only the magnitude differs):
- prefix dropped → B recomputes → win ≈ full prefill (**~2.8 s @50K**);
- prefix on DISK → B pays DISK load_back → win ≈ **50–140 ms**;
- prefix only reaches DRAM → win ≈ DRAM load_back (**~0.5–2.6 ms on B300**).

**The DRAM win is small on B300 but is NOT to be dismissed** — it is exactly the
load_back cost `= bytes / interconnect_bandwidth`, so it **scales as 1/bandwidth and
grows on slower hardware**. B300 has fast host↔device (~32–57 GB/s here); on PCIe-gen3
/ no-NVLink boxes (~3–8 GB/s) the same 12K–100K prefix load_back is ~2.8–40 ms — a
real, always-positive saving (蚊子小也是肉). S1 is correct and useful at every tier;
the headline magnitude just happens to be largest under deep eviction on this box.

## Aggregate space-freeing (the capacity / throughput half)

The per-resume TTFT win above is only one half. The other half: **when a program enters
a tool call whose ETA is long enough to be worth it, its idle prefix becomes worth
evicting — freeing a large block of HBM at once**, and the predictive promote brings it
back before the resume, so it's harm-free.

**The eviction is value-gated, NOT unconditional.** It is the §7 cost-benefit emerging
naturally from the value rule: a demote is taken iff `h_(τ,sp)(occ)·bytes·ETA` (holding
relief over the gap, `hold_time = ETA`) exceeds the demote+promote round-trip migration
cost. A short tool — e.g. `ls` (~ms) — has too short a gap to cover the round-trip, so
its demote value is negative and `migrate_candidates` simply never proposes it; no
special-casing needed. Only tool calls whose ETA clears the round-trip get demoted (and
the longer/slower the tool, the more worth it). This needs a per-tool ETA estimate
(T11's `p_hat`/ETA estimator).

- Per program: a 50K-token prefix ≈ **58 MB** (1.17 KB/token) freed on tool-call entry;
  a 100K prefix ≈ 117 MB.
- In an agentic workload a large fraction of programs are tool-parked at any instant
  (reasoning↔tool loops). S1 evicts exactly those — **program-aware** (it knows which
  are parked) and **proactive** (ahead of need) — so the freed HBM goes to active work:
  more concurrent programs admitted, longer contexts, bigger batches → higher throughput
  / lower queueing. Effective concurrency rises roughly by the parked-fraction.
- vs a reactive HiCache baseline: B eventually evicts under pressure too, but blindly
  (LRU-ish) and just-in-time; S1 frees the *right* (parked) prefixes *earlier*, and —
  unlike B — guarantees their return is pre-staged. This is the same program-aware
  foresight, applied to capacity instead of latency.

(Magnitude of the throughput gain needs the high-concurrency regime to measure cleanly;
the per-program freed footprint and the mechanism are established here.)

## What's proven vs what remains

- **Mechanism proven**: whole-chain demote applies leaf-inward (`probe_mimic.py`
  applied=5/5); predictive-promote logic dispatches (mimic, and §3 action-timeline).
- **Autonomous ETA learning + value-gating proven LIVE** (#239): the daemon learns the
  per-command ETA online — `bash/ls`=0.8s vs `bash/sleep`=20s, learned *separately* at
  command-token granularity (`ls` is a `bash` argument), no hardcoding — and value-gates
  the demote: under pressure `ls` is **declined every time** (gap can't cover the
  migration round-trip), `sleep` is **dispatched as whole chains** (13× in a 9-program
  run). `daemon/eta_estimator.py`; gating observable via the `kv_decide eta=… cmd=…`
  metric.
- **Saturation yield proven LIVE (do-no-harm, #240)**: the daemon's explicit demote
  loses the lock-race to sglang's own write-through / reactive-eviction lock at the apply
  moment (`remove_hbm_not_device_leaf:locked`). The daemon **self-measures** whether its
  demotes land (hash still HBM-resident a dump later → didn't) and **yields** below a 0.4
  apply-rate EMA. Live: yield fired 29×, **`:locked` churn 157 → 0**, daemon clean — the
  5× do-no-harm regression eliminated. sglang's V_u-guided reactive eviction does the
  demote; the daemon keeps the promote. A recovery drift re-probes once pressure eases.
- **Design finalized**: single V_u over two execution timescales (sync reactive scorer +
  async proactive migrate), proactive-first/reactive-fallback, saturation yield
  (DESIGN §3/§9, committed).
- **Win measured** end-to-end (above), via **manual pre-access** standing in for the
  daemon's automatic promote.
- **Autonomous DISK→HBM promote DONE + measured (#238)**: the original "wire
  `prefetch_from_storage` into the apply" scope was MISCONCEIVED — a DISK-evicted prefix
  *leaves the radix tree* (`_evict_host_leaf` → `_remove_leaf_from_parent`), so the
  node-based migrate plane can't reach it. The clean path needs **zero** sglang
  storage-controller change: a `/generate(input_ids, max_new_tokens=0)` is `is_prefill_only`
  — it runs the native prefix-match → storage-prefetch → load_back (KV → HBM), skips
  decode, retains it cached. Uniform across DRAM/DISK/dropped. Plumbed end-to-end: the
  proxy/driver registers each session's prefix (`POST /aginfer/session_prefix`); the
  action-timeline promote fires `warm_to_hbm` (prefer warm; per-unit DRAM `load_back`
  migrate is the fallback). **Measured fully daemon-driven** (`auto_victim_warm.py`, N=3):
  daemon warms a fully-evicted 50K victim → resume `cached=49920` = 274 ms vs B recompute
  3094 ms = **91% / 2.82 s**, `via=warm` 3/3.
- **Clean STABLE multi-program live win (#241)**: 6 concurrent programs, realistic
  establish→park→resume (a burst of memory pressure, then idle gaps) with a dense event
  stream (`live_clean.py`, N=3). **ours TTFT 1251±71 ms vs B 2109±314 ms = 41% faster, every
  cycle ours wins**; ours `cached ≈ 29881/30000` (full hit) vs B 0 (recompute) every cycle.
  Three diagnosis-driven fixes got here from an earlier −27% *unstable* result: (1) a
  `WARM_LEAD` floor (the warm is a prefill, not a ~0.1 ms transfer, so it must fire
  seconds early); (2) a dense event clock (so the action-timeline fires the warm on time —
  production posts events per request); (3) the realistic agentic STRUCTURE (burst-then-
  idle, not a continuous flood that removes the GPU idle the warm needs). **Honest finding:
  the S1 win is clean under the GPU-idle premise (real tool-parking); a compute-saturated
  synthetic regime hides it because the warm — an extra prefill — has no free GPU and its
  cached-saving is eaten by queue (the documented "spare GPU during the gap" caveat).**

## Honest caveats

- B recomputes (cached=0) here because the victim isn't retained on mooncake under the
  flood; with a larger tier-3 it would DISK-load instead (smaller win ~50–140 ms). The
  robust core is the **foresight**, not the exact magnitude.
- The win assumes **spare GPU during the gap** (true when the holder is genuinely
  tool-parked) so the pre-stage is ~free.
