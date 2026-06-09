# S1 — Tool-call predictability (the basic capability)

**Distinctive driver:** the **tool call** — the one structural feature *every*
agent has. A tool call is a predictable off-GPU window: we know it started
(`TOOL_CALL_START`), roughly how long it lasts (tool type → ETA), and that a
follow-up LLM call is near-certain (the loop continues). That predictability is
what lets us **demote the idle tail during the tool and predictively promote it
back before the resume**.

**This scenario establishes the basic demote+predictive-promote mechanism.** It is
a *building block* the later scenarios reuse (S6 reuses it on a longer sub-agent
idle window; S8 on a DISK hop). Those scenarios should NOT re-argue this mechanism
— they emphasize their own distinctive driver. S1 owns the foundational claim.

**Claim:** when an agent resumes after a tool call, its prefix KV is **already
back in HBM** because we pre-staged it timed to the tool's ETA — so the resume
prefill pays **~0 load_back latency**. HiCache and ThunderAgent both eat the
DRAM→HBM (or DISK→HBM) load_back on the critical path.

## The situation (workload)
Many concurrent agents in a reasoning↔tool loop, under enough pool pressure that
idle prefixes are demoted to DRAM during tool calls. Each agent: emit a request,
go off-GPU for a tool of **known type** (read/grep ≈ ms; build/test/docker ≈
seconds–minutes), then resume with the tool result appended.

Construct with Claude Code: agents whose tool calls are tagged with the tool kind
(so the proxy gets a `TOOL_CALL_START` carrying an ETA), and a tail of long,
high-variance tools (build/test) where the demote→resume window is wide. Knobs:
concurrency, fraction of long-gap tools, prefix size (bigger prefix → bigger
load_back → bigger win).

## What our framework does
- `TOOL_CALL_START(ETA)` → demote caller's exclusive tail HBM→DRAM (frees HBM for
  others during the gap) **and schedule the promote-back for
  `T_start + ETA − load_back_latency`** (DESIGN §7).
- The promote lands the prefix in HBM just as the next prefill arrives → resume
  hits HBM directly.

## Why we win
The load_back of a resuming agent's prefix (potentially thousands of tokens) is
real DRAM→HBM bandwidth time on the TTFT critical path. We move it **off** the
critical path by doing it ahead, paid out of the idle tool window. Metric:
**post-tool-gap TTFT** and **load_back bytes that landed on the critical path**.

## Why vanilla sglang+HiCache cannot
HiCache promotes **on access** — the request is already waiting when the load_back
starts. Its prefetch is `best_effort` and **not tool-ETA-aware** (it has no idea
a tool is running or when it ends), so it cannot reliably pre-stage before the
request arrives.

## Why ThunderAgent cannot
It never promotes anything — it only withholds prefills. The resume request, once
admitted, hits whatever tier HiCache left the prefix in, paying the same load_back
as B.

## Measurement plan
Arms: B / TA / ours (all HiCache). Metric: TTFT of the **first prefill after each
tool gap**, mean + p99; secondary: makespan. Expected: ours TTFT < {B, TA} by the
load_back time; gap grows with prefix size and gap length.

## 4-tier sizing (required)
Cache-pressure mode. Size DRAM (`--hicache-ratio`) and DISK
(`global_segment_size`) small enough that, during long tool gaps, demoted
prefixes are pushed past DRAM **onto DISK** — so the predictive promote is a
**DISK→HBM** pre-stage (the slowest load_back, hence the biggest win), and any
prefix NOT promoted in time is DROPped and **recomputed** on resume. Confirm via
B's logs that some resumes pay a DISK load_back or a recompute; otherwise the
window is too small to matter.

## Replay architecture (decided 2026-06-09, evidence-based)
The faithful replay for S1 (and the campaign) is **pure token-space `/generate`
+ explicit event injection**, NOT the chat-path replay_driver. Three findings
forced this:
1. **Predictive-promote plane built** (`daemon/action_timeline.py` + kv_scheduler
   `_schedule_promote_back`/`fire_due_action`, wired in event_router/main; unit
   test `verify/action_timeline` PASS). On `TOOL_CALL_START` carrying
   `tool_eta_s` it schedules a promote-back at `T_start+ETA−load_back`, fired by
   the event-stream-clocked due-action heap, belief-validated (program still
   ACTING + tail still DRAM/DISK) → `Migrate(→HBM)`.
2. **The proxy emits `TOOL_CALL_START` with NO ETA** (proxy.py:455). So the
   promote can only fire if the ETA is injected explicitly → the replay must
   POST events to `/aginfer/event` itself (carrying `tool_eta_s`), not rely on
   the proxy's auto-emission.
3. **The model's default chat template is NOT prefix-stable across turns** — its
   generation-prompt header (`add_generation_prompt=True`) differs from the
   in-conversation assistant header, so even the server gets poor multi-turn KV
   reuse via the chat path (prep's 528/528 turns failed prefix-stability). Pure
   token-space replay (Part-B-proven: build ONE growing token sequence per
   program, force each assistant segment, append) is faithful BY CONSTRUCTION
   and sidesteps the template entirely.

So the S1 driver: per program, build a growing token sequence from the CC trace;
send each turn as `/generate(input_ids, forced_output_ids, program_id)` to sglang
directly; between turns POST `TOOL_CALL_START{tool_eta_s}` / sleep ETA /
`TOOL_CALL_END` to the daemon. Requests carry `program_id` so sglang tags the KV
units' `session_ids` (the daemon's `_units_for_session` needs this).

## Honest status & falsification
- **Status:** DESIGN now specifies the realization (§3 action-timeline plane: a
  due-action heap clocked by the event stream, belief-validated at fire; the old
  "no internal timer" invariant was revised to "belief event-sourced; actions
  time-scheduled but belief-validated"). So this is no longer blocked on a design
  contradiction — it is **`promotes=0` because the action-timeline plane is not
  yet implemented**. That implementation (carry tool ETA on the event; due-action
  heap; degrade to promote-at-TOOL_CALL_END when ETA is unreliable) is the #1
  build, and it unlocks S6 and S8 too.
- **Falsifies the win if:** measured on-resume load_back is negligible vs prefill
  (then there's nothing to hide), OR HiCache's best-effort prefetch already
  overlaps it well (then our margin is small). Both are empirical questions to
  settle with the load_back/TTFT measurement.
