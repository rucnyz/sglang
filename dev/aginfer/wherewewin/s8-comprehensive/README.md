# S8 — Comprehensive (everything at once, end-to-end)

**Distinctive driver:** not a single signal — **all of them, together**, in one
realistic agentic workload, with the daemon running **every lever jointly** over
the full 4-tier hierarchy. This is the "all-on" end-to-end scenario: the closest
thing to a real deployment, and the strongest aggregate claim if ours wins it.

S1–S7 each isolate one driver to get a clean, attributable win. S8 is the
opposite: it accepts that many things move at once and asks the integration
question — **when tool gaps, shared prefixes, compaction, program/sub-agent
lifecycle, fan-out spikes, and genuine overload all happen concurrently, does the
joint scheduler beat B and ThunderAgent end-to-end?** Harder to attribute, but it
is the headline number a paper actually reports.

## The situation (workload)
One workload that exhibits all of S1–S7 simultaneously, driven by Claude Code
agents:
- typed **tool calls** with predictable ETAs (S1) and the occasional long
  build/test;
- a **fleet-shared system/tool-def prefix** across all agents (S2);
- periodic **context compaction** in long agents (S3a) and a stream of programs /
  sub-agents that **end** (S3b);
- **blocking** sub-agent dispatches (S6) and **background** fan-outs (S7);
- enough concurrency that the **running set periodically overloads** HBM (S5);
- a working set large enough to keep the **DISK** tier continuously in play.

Knobs: concurrency, mix of long/short tools, compaction rate, sub-agent rate,
arrival burstiness; tiers sized (below) so the full hierarchy is pressured.

## What our framework does
`joint_decide` (§9) runs the **union** action space every event — value eviction,
proactive demote, predictive promote (ETA-timed, §3 action-timeline), drop-on-
death, value-gated pause, and DISK cascade — under one `V_u` and one byte budget,
balancing them against each other (it cannot, e.g., promote and pause the same
bytes; the DP resolves the trade). The whole point of a *joint* decision rule is
this regime, where the levers interact.

## Why we win
Each lever's edge (S1–S7) is present, and — the integration claim — they
**compound and are de-conflicted**: HBM freed by a drop-on-death (S3) or a
blocking-parent demote (S6) is exactly the room a predictive promote (S1) or a
fan-out spike (S7) needs; the value rule keeps the shared prefix (S2) resident
through all of it; pause (S5) sheds load only when migration can't relieve. A
reactive cache handles each in isolation and late; a joint, anticipatory,
value-aware scheduler handles them together and ahead.

## Why vanilla sglang+HiCache cannot
It has no joint decision and no program/lifecycle awareness — it reacts to each
pressure independently and by recency, with no cross-lever budgeting and no
anticipation. Under the combined load its reactive eviction + on-access load_back
+ preemption stack up.

## Why ThunderAgent cannot
Its only lever is HBM-keyed admission throttling (HiCache-blind, no migrate, no
promote, no value, no drop). In a workload that needs all six levers, it
contributes at most one — and an over-conservative one.

## Measurement plan
Arms: B / TA / ours (all HiCache, full 4-tier). **Headline = end-to-end makespan /
goodput** plus the TTFT distribution, over the combined workload, N≥3 mean±std.
Secondary (attribution): per-lever activity counts on the ours arm (migrates,
promotes, drops, pauses) to show the joint scheduler actually exercised them.

## 4-tier sizing (required)
All four tiers continuously in play: HBM (pool), DRAM (`--hicache-ratio`), DISK
(`global_segment_size`), DROP reachable. Size the combined working set above
HBM+DRAM+DISK so both **cache pressure** (DISK→DROP recompute) and **live
pressure** (running-set preemption) occur during the run — that is the regime
where all levers fire at once.

## Honest status & falsification
- **Status / known hardness:** this is the **hardest to design and attribute** —
  many things move together, so a win is end-to-end, not per-lever, and a loss is
  ambiguous (which lever underperformed?). It also depends on the unbuilt pieces
  (predictive promote, compaction hook, event-carried demand). **Parked here on
  purpose**: run it *after* S1–S7 are individually understood, as the capstone
  integration test, not first.
- **Falsifies the win if:** ours ≈ B/TA end-to-end even though individual levers
  win in S1–S7 — which would mean the levers don't compound (or conflict) under
  joint load; the per-lever activity counts + an ablation (turn levers off one at
  a time) would then localize why.
