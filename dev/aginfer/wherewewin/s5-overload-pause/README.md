# S5 — Value-gated pause under true overload

**Claim:** when the **live decode set** genuinely exceeds HBM, we pause the
**lowest-value** programs (preserving their state) so the rest run un-thrashed,
and resume by value. ThunderAgent also pauses but keyed to an **HBM-only,
HiCache-blind** estimate (over-pauses); B has no pause and thrashes (preempt +
recompute).

## The situation (workload)
Genuine over-subscription of the **running** set: many programs simultaneously
decoding (short/no tool gaps), offered faster than served, so the live KV of the
running batch alone exceeds the pool → sglang must preempt/queue. (Open-loop high
arrival rate, or a decode-storm; NOT the long-tool-gap regime where programs are
idle and there is nothing to pause.)

Construct with Claude Code: a burst of agents all in dense reasoning (few/short
tools) arriving faster than capacity. Knob: arrival rate vs serving rate; pool
small relative to the concurrent live set.

## What our framework does
`MEMORY_PRESSURE` / forecast → admission generates `Pause(lowest V_u program)`;
joint_decide sheds the least-valuable load, preserving its KV in DRAM/DISK;
`_greedy_resume` re-admits by value when room frees.

## Why we win
Under true overload, doing *all* the work at once thrashes (preemption =
retract + recompute). Cleanly deferring low-value work raises high-value goodput
and cuts p99. Our pause is **tier-aware**: it won't pause a program HiCache could
still serve from DRAM. Metric: **goodput** and **p99 latency** under sustained
overload.

## Why vanilla sglang+HiCache cannot
B cannot shed offered load at all — it has no program-pause concept. It admits
everything and relies on preemption/recompute, which thrashes under real overload.

## Why ThunderAgent cannot (match us)
TA *does* pause, but its capacity estimate is **HBM-only** (`max_total_num_tokens`)
and HiCache-unaware, so it pauses based on HBM occupancy even when HiCache could
keep serving from DRAM → **over-conservative**, withholding work needlessly. It
also can't preserve/pre-stage the paused program's KV (no migrate/promote).

## Measurement plan
Arms: B / TA / ours (all HiCache). Metric: goodput (completed within deadline) and
p99 e2e under sustained over-arrival. Expected order: ours > TA > B (B thrashes;
TA over-pauses).

## 4-tier sizing (required)
**Live-pressure** mode. DISK + DRAM are still enabled (full 4-tier) — the cache
hierarchy is in place — but here the *binding* constraint is the live set, not the
cache: size the pool so the **running decode set > HBM**, so sglang reaches the
4th state = **preemption** (retract + recompute of in-flight work) — that is the
cost our clean value-pause (state preserved down the tiers) avoids. Tune below
total collapse (my 35%-timeout run was over-pressured): preemption must occur and
be measurable on B without every request timing out. Confirm via B's logs that
preemption/retraction actually fires.

## Honest status & falsification
- **Status:** admission pause is implemented but **dormant** in every regime
  tested so far (`pauses=0`) — sglang did not preempt because only the *evictable
  cache* overflowed, not the *live* set. This scenario's whole job is to build the
  regime where the **live decode set** overflows (so pause is actually needed).
- **Falsifies the win if:** sglang+HiCache handles the overload gracefully (chunked
  prefill + tier spill) without thrashing → B doesn't lose → nothing to beat. Must
  first demonstrate B actually thrashes/preempts under the constructed load.
