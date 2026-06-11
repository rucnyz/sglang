# S1 — Program-aware KV scheduling across tool gaps

## Workload (the distinctive driver)

An agent program runs a **reason ↔ tool-call loop**. While it is parked in a tool call
(waiting on a slow tool), its KV prefix is **idle** and, under memory pressure from other
programs, gets **evicted**. When the tool returns, the program **resumes the *same*
prefix**. The canonical agentic situation: **a known prefix, a known idle gap, a known
imminent reuse**.

## What we set out to claim — and what realistic-trace evidence actually shows

> **This package previously headlined a predictive-promote *latency* win (41–91 %). That
> number is from a hand-forced microbenchmark and does NOT reproduce on realistic agent
> traces. The corrected, validated result below is a program-aware *eviction goodput*
> win. The microbench is kept as "mechanism in isolation" at the bottom, honestly
> labeled.**

**Original claim (predictive promote → latency):** during the gap, Ours predictively
promotes the evicted prefix back into HBM (using free GPU), moving the re-acquisition off
the resume critical path → faster resume TTFT.

**What rigorous realistic-trace replay showed (the correction):** replaying real Claude
Code agent traces at concurrency, the predictive-promote *latency* win **does not hold** —
the promote is not even the active lever, and resume-TTFT is within noise. The real,
validated win is **program-aware EVICTION**: Ours keeps the reuse-imminent prefixes that
LRU evicts. It is a **goodput** (compute-saving) win, not a latency win. Full cross-regime
investigation: [`FLEET_FINDINGS.md`](FLEET_FINDINGS.md).

## Headline result — realistic CC traces, N=3 rigorous

`cc6_park` — 6 real Claude Code programs, parking gaps (10–30 s), sum-peak ≈ 298 K vs HBM
pool 131072 = **2.3× over-subscribed** (the moderate-concurrency regime with genuine idle
headroom). Teacher-forced replay (both arms do identical token work). **N=3, mean ± std:**

| arm | cache-hit | re-prefill tokens |
|---|---|---|
| **Ours** (daemon program-aware hints + `hint_v_u` eviction) | **71.6 % ± 3.4** | **~1.53 M (−42 %)** |
| **B** (default HiCache + LRU) | 55.8 % ± 5.5 | ~2.57 M |
| local value-eviction (`ours_greedy`, **no daemon**) | 50.8 % ± 2.2 | ≈ B |

**Ours re-prefills 42 % fewer tokens** by keeping the reused prefixes LRU drops. Three
things validated, not just claimed:

1. **Real + separable** — gap 16 pt ≫ combined std.
2. **Entirely the daemon's program-aware foresight** — local value-eviction (no daemon
   hints) is ≈ LRU (50.8 %); the full +21 pt comes from the daemon knowing *which program
   will resume* from the tool-call event stream. **The core aginfer thesis, confirmed.**
3. **True compute saving, not an artifact** — it's eviction-keeping (a trial that fired
   **0 promotes** had the identical 70.9 % win → it's `hint_v_u` keeping the prefix in
   HBM, not the promote shifting compute into the gap).

**Honest scope:**
- **Goodput/capacity, NOT per-request latency.** Resume-TTFT is queueing-bound (a
  resuming program queues behind others), so latency/makespan are within noise; the win is
  42 % less prefill *compute* — a throughput/capacity benefit.
- **Regime-specific.** At the heavy 90-program fleet, Ours ≈ LRU: everything churns →
  eviction-order is moot, and the V4-Flash multi-tier store is non-functional (mooncake
  layer incompatibility, 0 DISK writes). At 2.3× over-subscription, *which* prefixes you
  keep matters. See [`FLEET_FINDINGS.md`](FLEET_FINDINGS.md).

## Mechanism (as implemented)

The active winning lever is the **program-aware eviction scorer**:
1. The daemon tracks each program's tool-call lifecycle and pushes a per-unit value hint
   (program-aware `p_hat` — reuse-imminence from the event stream) to sglang.
2. sglang's eviction scorer (`SGLANG_KV_POLICY_MODULE=aginfer:hint_v_u`) reads those hints
   and keeps high-value (reuse-imminent) prefixes — so a parked program's prefix, *cold by
   recency* but *reuse-imminent by program state*, survives eviction where LRU drops it.

The daemon also implements demote / predictive-promote and an online ETA estimator, but on
realistic traces those are **not** the source of the measured win.

## Reproduce

```bash
export AGINFER_ROOT=/path/to/sglang/dev/aginfer
conda activate agsched-rebase
# (process ops need an unsandboxed shell; GPUs 5,6 free)
cd $AGINFER_ROOT/scenarios/replay

# build the moderate-concurrency real-CC workload (one-time)
python convert_cc_traces.py        # ~/.claude transcripts -> traces/cc/<program>.jsonl
#   then assemble traces/cc6_park.jsonl (6 biggest programs, 10-30s parking gaps; see FLEET_FINDINGS.md)

# Ours (program-aware eviction) — N=3
bash run_replay_pressured.sh traces/cc6_park.jsonl 3 1 131072 "a3"       1 "aginfer:hint_v_u"
# Baseline B (true LRU) — N=3
bash run_replay_pressured.sh traces/cc6_park.jsonl 3 1 131072 "a3_kvoff" 1 "lru_score"
# (optional attribution) local value-eviction, no daemon
bash run_replay_pressured.sh traces/cc6_park.jsonl 3 1 131072 "a3_kvoff" 1 "ours_greedy_score"

# analyze: cache-hit / re-prefill, and resume-TTFT (queueing-bound -> within noise)
python analyze_opportunity.py <ours_results_dir> 16000
python resume_ttft.py <ours_results_dir> <base_results_dir>
```

Note: do **not** set `SGLANG_WRITE_THROUGH_MODULE=aginfer:hint_write_through` — the
hint-aware write-through is implemented but is **harmful** here (37 % hit) and caused
intermittent scheduler stalls; keep it OFF.

## Microbenchmark — the mechanism in isolation (NOT the realistic-trace result)

For completeness: the predictive-promote *mechanism* does produce a large latency win in a
**hand-engineered** setting — an isolated 50 K prefix, a deliberate flood to evict it, then
a guaranteed-idle gap during which the daemon warms it back: Ours **274 ms** (HBM hit) vs
B **3094 ms** (recompute) = **91 %**, `via=warm` 3/3 (`run_controlled.sh`, N=3). This shows
the warm path works, but its conditions — a single isolated leaf prefix, forced eviction,
guaranteed idle GPU, no shared-prefix tree — **do not occur in realistic agentic serving**,
which is why it does not reproduce above. Treat it as a mechanism check, not a headline.
Details + the GPU-idle-premise caveats: [`results/RESULTS.md`](results/RESULTS.md).

## TA (ThunderAgent) arm

Set up; analytically TA ≈ B (TA only pauses, never promotes/evicts/migrates — HiCache-blind).
A direct TA run needs a chat-interface 3-arm driver. See [`THUNDERAGENT.md`](THUNDERAGENT.md).
