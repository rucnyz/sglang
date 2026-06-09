# S6 — Blocking sub-agent

**Distinctive driver:** the **sub-agent** — an *optional* agent feature (only some
workloads use it). A blocking sub-agent dispatch is not just "another tool gap";
its NEW properties are: (1) the parent idle window is **much longer** than any
tool (whole child run) so demoting the parent tail frees a **large** block of HBM
for a long time; (2) the child runs a **fresh, independent** context (own system
prompt + task — it does **not** inherit the parent's prefix), and (3) on return
the child's whole context **dies** at once.

**Reuses, does not re-argue, S1.** The demote-during-idle + predictive-promote
mechanism is S1's basic capability applied to this longer idle window — we cite it,
not re-derive it. The *new* wins here are the **magnitude** (big HBM freed for a
long window lets the child + others avoid DROP/recompute) and the **child-context
death** (a large, instant drop, = S4's drop-on-end on the child's session).

## The situation (workload)
Agents that dispatch **blocking** sub-agents (`run_in_background:false`): the
parent suspends (no in-flight request) while the child runs many turns; the child
builds a **fresh, independent** context (its own system prompt + task — it does
**not** inherit the parent's prefix); on return the child context dies and the
parent resumes with `[P ⧺ M]`.

Construct with Claude Code: a parent agent that does a real blocking `Agent(...)`
fan-in (e.g. dispatch an Explore agent and wait). Knob: child run length (=parent
idle window), parent prefix size, number of parents.

## What our framework does
- `SUB_DISPATCH_BLOCKING` → parent is idle ⇒ demote parent tail HBM→DRAM/DISK
  (frees a large block during a long window), and schedule a predictive promote
  for when the child is expected to return.
- `SUB_RETURN` → child's context is dead (S4 drop) **and** parent's tail should be
  HBM-ready for the resume.

## Why we win
The parent-idle window during a blocking child is the **longest, most confident
demote opportunity** in agentic serving (far longer than a tool call), so the
freed HBM materially helps the child + other programs; and the parent resume is
the cleanest predictive-promote target. Two metrics: **HBM freed during the child
run** (→ child/throughput benefit) and **parent-resume TTFT** (→ promote benefit).

## Why vanilla sglang+HiCache cannot
It only demotes the idle parent tail reactively under pressure (late), and
promotes it back on-access (load_back on the parent's resume TTFT). It has no
event telling it "the parent will be idle for a long, bounded time."

## Why ThunderAgent cannot
No backend KV action — it neither frees the parent's HBM nor pre-stages it. Its
only move is to withhold the parent's next prefill (which is already absent during
the block), so it does nothing here.

## Measurement plan
Arms: B / TA / ours (all HiCache). Metrics: (a) effective concurrent capacity /
child-run throughput given the freed HBM; (b) parent-resume TTFT. Expected: ours
frees HBM during the block and serves the parent resume from HBM.

## 4-tier sizing (required)
Cache-pressure mode. Tiers tight enough that during the long parent-idle the
parent tail is demoted **all the way to DISK** (long window allows it), so the
predictive promote is a DISK→HBM pre-stage; and so that the HBM freed by demoting
the parent tail is what lets the **child avoid a DROP/recompute** it would
otherwise suffer under B. The win is then on two 4th-state costs at once (parent
resume recompute avoided + child recompute avoided). Confirm via B that the parent
resume or child pays a DISK load_back / recompute.

## Honest status & falsification
- **Status:** parent-tail demote on dispatch is expressible; predictive promote is
  **not yet firing** (`promotes=0`) — shares S1's build.
- **Falsifies the win if:** parents are few / prefixes small (freed HBM doesn't
  help anyone), or child run length is so variable that the promote can't be timed
  (then promote at `SUB_RETURN` still beats on-access, but the margin is smaller).
