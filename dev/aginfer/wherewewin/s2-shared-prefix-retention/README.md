# S2 — Shared-prefix retention under scratch churn

**Claim:** under heavy per-program scratch churn, LRU ages the **fleet-shared
system/tool-def prefix** down its recency order and demotes it (then load_backs it
repeatedly). We keep it pinned in HBM by **holder-count value**, so every program
that touches it hits HBM.

## The situation (workload)
High concurrency, where each program generates lots of unique recent scratch
(reading different files, divergent reasoning) between its touches of the common
prefix. The shared prefix (system prompt + tool definitions, identical across all
same-type agents) is touched by everyone but, at any instant, is "less recent"
than the flood of fresh scratch.

Construct with Claude Code: a fan-out of agents sharing one system prompt + tool
set, each doing divergent exploratory work (many distinct file reads) so the
scratch churn rate ≫ the per-program prefix touch rate. Knob: scratch-churn rate
vs concurrency; pool sized so HBM must demote *something*.

## What our framework does
Inline `V_u` eviction key: a unit's value includes its **holder count** (shared by
N programs ⇒ N× the saved-prefill term). The shared prefix outranks any single
program's stale scratch, so under pressure the scratch is demoted first and the
prefix stays HBM-resident.

## Why we win
DESIGN §2 fact 1: "shared prefix used by 32 programs ≫ a 30-turn-old scratch; LRU
gets it backwards." When value and recency **diverge**, recency evicts the
high-value prefix. Metric: **load_back bytes of the shared prefix** and the
**TTFT of prefills that reuse it**.

## Why vanilla sglang+HiCache cannot
Its eviction key is `last_access` only — it has no holder-count term. Under scratch
churn the shared prefix becomes the least-recently-touched large unit and gets
demoted; the next program to need it eats a load_back.

## Why ThunderAgent cannot
It does not evict or place at all; whatever HiCache (LRU) does to the prefix is
what the program sees.

## Measurement plan
Arms: B / TA / ours (all HiCache). Metric: shared-prefix load_back count/bytes and
reuse-TTFT. Expected: ours keeps prefix in HBM (≈0 prefix load_back); B repeatedly
demotes+load_backs it.

## 4-tier sizing (required)
Cache-pressure mode. Size HBM+DRAM+DISK **below** the total churn working set so
that a shared prefix LRU demotes does not merely sit in DRAM but is eventually
**DROPped** under the scratch flood → the next program to touch it pays a **full
recompute**, not a load_back. That escalates the L1 penalty from ms (load_back) to
a re-prefill, which is where value-retention decisively beats recency. Confirm via
B's logs that the shared prefix actually reaches DROP/recompute.

## Honest status & falsification
- **Status:** value eviction (`ours_greedy_score`) is implemented. A prior test on
  the a3real trace **tied B on cache-hit rate** — but that trace's shared prefix
  was constantly hot (recency ≈ value), the non-divergent case. This scenario
  must **engineer the divergence** (churn rate ≫ prefix touch rate).
- **Falsifies the win if:** even with high churn, the prefix stays recent enough
  that LRU keeps it too (no divergence) → tie. The test must first verify (via
  B's logs) that LRU actually demotes the prefix; if it doesn't, there is no win
  to be had here and that should be stated.
