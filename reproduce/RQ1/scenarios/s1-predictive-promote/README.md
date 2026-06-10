# S1 — Predictive promote across tool gaps

## Workload (the distinctive driver)

An agent program runs a **reason ↔ tool-call loop**. While it is parked in a tool call
(waiting on a slow tool — a web fetch, a build, a human), its KV prefix is **idle** and,
under memory pressure from other programs, gets **evicted** (HBM → DRAM → DISK, or
dropped). When the tool returns, the program **resumes the *same* prefix**.

This is the canonical agentic situation: **a known prefix, a known idle gap, a known
imminent reuse** — and crucially, **the GPU is idle during the gap** (the program isn't
computing while its tool runs).

## Claim

Ours has **program-aware foresight**: it knows program *p* will resume *this* prefix at
≈ `T + ETA`, so during the idle gap it **predictively promotes** the prefix back into HBM
(using the free GPU) — moving the re-acquisition **off the resume critical path**.

- **B** (default HiCache) is reactive + program-blind: it re-acquires the prefix on the
  next prefill, **on the critical path** → the resume pays the full load-back / recompute.
- **TA** (ThunderAgent) is HiCache-unaware and never promotes → same critical-path cost as
  B for the cache, plus it can only *pause* (withhold), not *pre-stage*.

## Mechanism (Ours, as implemented)

1. **Fine-grained online ETA estimator** — learns each tool's duration online at
   *command-token* granularity (`bash/ls` ≈ 0.8 s vs `bash/sleep` ≈ 20 s, learned
   separately; no hardcoding). Used to **value-gate** the demote: a short tool (`ls`) can't
   cover the migration round-trip → not demoted; a long tool → demoted.
2. **Demote** the idle prefix during the gap (frees HBM), with a **saturation yield**
   (do-no-harm): if the explicit demote can't land — it loses the lock race to sglang's own
   eviction — the daemon yields rather than churn.
3. **Predictive warm** at ≈ `T + ETA − warm_lead`: the daemon re-prefills the prefix with
   `max_new_tokens=0` (`is_prefill_only`) — sglang's native prefix-match → storage-prefetch
   → load-back stages the KV into HBM and skips decode. This **uniformly reaches
   DRAM/DISK/dropped** prefixes (a DISK-evicted prefix has *left the radix tree*, so the
   node-based migrate plane cannot reach it; the prefill-only warm can).

## Knobs (workload, not pool)

| Knob | Meaning | Headline value |
|---|---|---|
| `MAX_TOTAL_TOKENS` | HBM KV-pool (pressure) — **never below the V4-Flash min** | 262144 |
| `--hicache-ratio` | DRAM tier size (where an evicted prefix first lands) | controls win magnitude |
| mooncake size | DISK tier (retain vs drop the evicted prefix) | controls win magnitude |
| prefix length | per-program KV (eviction footprint + win size) | 30K (live), 50K (controlled) |
| gap / ETA | tool-call duration (idle window for the warm) | 12–16 s |
| concurrency | programs (aggregate KV → eviction) | 6 (live) |

`AGINFER_WARM_LEAD_S` (daemon, default 2.5) — the warm is a *prefill*, so it must fire
seconds before the resume, not the ~0.1 ms a pure transfer would need.

## Reproduce

```bash
export AGINFER_ROOT=/path/to/sglang/dev/aginfer
conda activate agsched-rebase
bash scripts/stack_up.sh            # wait for "[s1-stack] READY"
bash scripts/run_live_ab.sh         # headline live A/B (Ours vs B), N=3
bash scripts/run_controlled.sh      # controlled full-eviction magnitude (91%)
bash scripts/run_microbench.sh      # per-resume B-vs-prestaged
python scripts/link_characterize.py # offline tier-link bandwidths (CUDA_VISIBLE_DEVICES=5)
```

## Measured results (8× B300, V4-Flash tp2, GPUs 5,6)

**Headline — clean STABLE live win** (`run_live_ab.sh`, 6 programs, establish→park→resume, N=3):

| arm | resume TTFT | cached / 30000 |
|---|---|---|
| **Ours** | **1251 ± 71 ms** (every cycle wins) | ≈ 29881 (full hit) |
| B | 2109 ± 314 ms | 0 (recompute) |

→ **41 % faster TTFT, stable** (±71 ms / 6 % CoV).

**Controlled magnitude** (`run_controlled.sh`, fully-evicted 50K, daemon-driven, N=3):
Ours **274 ms** (cached 49920) vs B **3094 ms** (recompute) = **91 % / 2.82 s**, `via=warm` 3/3.

**Offline tier-link** (`link_characterize.py`): DRAM→HBM 0.4–2.6 ms; DISK→HBM 49–141 ms
(12K–100K). The win = the cost moved off the critical path → spectrum: prefix dropped →
recompute (~2.8 s); on DISK → ~50–140 ms; only DRAM → ~0.5–2.6 ms (scales as 1/bandwidth →
larger on slower interconnects).

## Honest caveats

- **The win is gated on the GPU-idle premise.** The predictive warm is an *extra* prefill;
  it pays off only when the gap has free GPU (real tool-parking). A compute-saturated
  synthetic regime (continuous background flood) hides the win — the warm competes for GPU
  and its cached-saving is eaten by queue. The headline regime uses the realistic
  burst-then-idle structure.
- **Latency is multi-run.** The first single live run showed 51 %; N=3 revealed −27 %
  (unstable) — the 51 % was a lucky run. The clean 41 % is *after* three diagnosis-driven
  fixes (warm-lead floor, dense event clock, burst-idle structure) and holds at N=3.
- **TA arm: set up + analytically TA ≈ B; direct run blocked by interface** — TA proxies
  only `/v1/chat/completions`, not the token-level `/generate` the prefix-stable S1 driver
  needs (404 on `/generate`). Since TA never promotes (HiCache-blind, only pauses), its
  resume recomputes the evicted prefix **exactly like B** → the Ours-vs-B win *is* the win
  over TA's prefix handling. A direct TA run needs a chat-interface 3-arm driver. See
  [`THUNDERAGENT.md`](THUNDERAGENT.md).
- B recomputes (cached=0) here because the victim isn't retained on mooncake under the
  flood; a larger DISK tier would DISK-load instead (smaller win, ~50–140 ms). The robust
  core is the **foresight**, not the exact magnitude.

See [`results/RESULTS.md`](results/RESULTS.md) for the full write-up + raw numbers.
