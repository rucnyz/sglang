# S3 — Drop-on-death (compaction + program/sub-agent end)

**Distinctive driver:** **KV that is provably dead** — content the agent will
never reuse — detected from a lifecycle event, not guessed from recency. Two
triggers, one mechanism:
- **(a) Context compaction** — a long agent summarises old turns and drops the
  originals; the dropped span's KV is orphaned and dead.
- **(b) Program / sub-agent end** — a task finishes (`SESSION_END`) or a
  sub-agent returns (`SUB_RETURN`); its whole session-scoped KV is dead.

(These were two scenarios; they share the lever and the metric, so they are one.)

## The situation (workload)
A churn of dying KV under memory pressure: long agents that periodically compact,
and/or a high turnover of short programs and sub-agent fan-outs that finish. The
question is how fast the dead KV is reclaimed and whether holding it forces live
work to suffer.

Construct with Claude Code: (a) agents that hit a context threshold and emit a
compaction (proxy surfaces a `CONTEXT_COMPACTED` naming the dropped span); (b) a
stream of short sessions ending via `/aginfer/session_end` and sub-agent fan-outs
that complete. Knobs: compaction frequency × dropped-span size; session turnover ×
footprint; concurrency — pool sized so the dead KV's occupancy actually matters.

## What our framework does
On `SESSION_END`, the Dynamo router sends `POST /aginfer/session_end` to every
worker that served the program. Each worker fans the request to every DP rank;
the scheduler then waits for in-flight requests, native SGLang session locks,
and asynchronous HiCache work to drain across the TP/CP cache group. It removes
the ending program from shared nodes and deletes exclusively-held leaves
bottom-up, releasing their HBM and DRAM allocations immediately instead of
waiting for memory pressure or an LRU sweep.

The operation is idempotent and acknowledges completion only after every rank
has completed the same lifecycle barrier. `SUB_RETURN` can use this path when a
child is represented as its own terminal session. `CONTEXT_COMPACTED` still
needs a range-aware lifecycle signal and is not implemented.

## Why we win
**Proactive vs reactive timing.** Dead KV is reclaimed the instant it dies, so
live programs keep more HBM/DRAM throughout the window between death and the next
eviction sweep. Under 4-tier pressure the sharp form is: holding a corpse in
HBM/DRAM forces a **live** reused unit down to DISK or to DROP+recompute — so the
corpse's occupancy is paid by someone else's re-prefill.

## Why vanilla sglang+HiCache cannot
HiCache has no notion that the span is *logically* dead — only "not recently
accessed." LRU **will** age it out (this is its job), so the honest edge is **how
much sooner**, and that it won't keep a corpse ahead of a live unit in recency
order. If LRU's age-out is already prompt under pressure, the win shrinks.

## Why ThunderAgent cannot
It does not evict. Dead-KV reclamation is entirely HiCache's job under TA. (TA's
`/programs/release` frees only its router-side bookkeeping — zero backend call —
so on the backend the KV stays until LRU.)

## Measurement plan (lead with the user-visible metric)
Arms: B / TA / ours (all HiCache, 4-tier). **Headline = arrival/next-prefill TTFT
and throughput at fixed pool** (the user-visible effect); secondary = time-
integrated HBM/DRAM held by dead KV (the internal cause). Expected: ours reclaims
on the event → live work keeps fast tiers → lower TTFT; B/TA hold corpses until
age-out.

## Full 4-tier follow-up
Cache-pressure mode, **all four tiers live** (HBM, DRAM, DISK enabled; DROP
reachable). Size HBM+DRAM+DISK below the live+dead working set so an un-reclaimed
corpse pushes a live reused unit across the **DISK→DROP** frontier (synchronous
DISK load_back, or recompute). Confirm via B's logs that live units are stranded
on DISK / recomputed while corpses sit resident in HBM/DRAM.

The current physical lifecycle deletion covers the radix-tree allocations in
HBM and DRAM. Per-key deletion from an external HiCache storage backend is a
separate follow-up; until that exists, do not claim that `SESSION_END` deletes
backing-store bytes.

## Honest status & falsification
- **Status:** trigger (b) `SESSION_END` is implemented end-to-end for PP=1 and
  HBM+DRAM, including TP/CP synchronization, shared-prefix preservation,
  deferred in-flight cleanup, and duplicate delivery. Pipeline parallelism,
  decode-side asynchronous KV offload, external-storage deletion, and trigger
  (a) `CONTEXT_COMPACTED` remain explicit follow-ups.
- **Falsifies the win if:** under pressure LRU evicts dead KV about as fast as our
  event-driven drop (it's the least-recent anyway) → negligible gap. The test must
  show LRU actually keeps dead KV ahead of live units, or that the reclaim-latency
  gap moves the user-visible metric (TTFT / throughput).
