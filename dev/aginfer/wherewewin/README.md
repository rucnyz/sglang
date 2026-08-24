# Where We Win — verification campaign

A catalogue of **distinct agentic scenarios where aginfer's framework should beat
a strong cached baseline**, and the plan to verify each one empirically, one at a
time. This is the index; each `sN-*/README.md` details one scenario's workload,
the theoretical win, why the competition cannot match it, and how to measure it.

## The two competitors (verified capability ceilings)

- **B = vanilla sglang + HiCache + LRU.** Reactive (demotes only on pressure),
  recency-keyed eviction (`last_access`), on-access best-effort load_back, and
  **program-blind** (manages bytes, not programs).
- **ThunderAgent (TA).** Verified from source: its "pause" makes **zero backend
  calls** — pure router-side admission control that withholds a program's next
  prefill, keyed to **HBM-only `max_total_num_tokens`**. It is **HiCache-unaware**
  (no model of DRAM/disk), never migrates, never promotes, never evicts.

- **Ours.** sglang's decision pipeline: event-driven, per-unit value `V_u`
  (tier + holder-count + reuse-prob aware), residence-set migration across all 4
  tiers `{HBM, DRAM, DISK, DROP}`, value-gated program pause/resume, and
  **predictive (ETA-timed) promote**.

The unifying thesis: **every win below is a case where "proactive + value-aware +
program-aware" beats "reactive + recency + program-blind".** A scenario only wins
where the reactive/recency baseline is actually suboptimal — so each scenario
names the precise condition that makes that true, and a falsification test.

## How to use this folder

Go through the scenarios **one at a time**. For each: construct the workload
(driven by **Claude Code agents**, so any tool pattern / lifecycle signal is
producible — do not pre-limit by "can we detect X"), run the 3 arms (B / TA /
ours, **all with HiCache on**), measure the named metric, and record the result
in that scenario's README. If a scenario turns out **not** to win, write down
exactly why — a clear negative is a valid, useful outcome.

## Organizing principle: distinctive driver, not mechanism (avoid overlap)

Scenarios are split by the **distinctive agent characteristic** that creates the
situation, **not** by the mechanism (lever) used — because mechanisms are shared
**building blocks** that recur across scenarios. Concretely:

- **S1 (tool calls)** is the *basic* scenario: it establishes the
  demote+**predictive-promote** mechanism off the universal tool-call signal.
- Every later scenario emphasizes its **own** new driver (shared prefix /
  compaction / session-end / overload / sub-agents / deep tiering) and **reuses**
  S1's basic capability silently rather than re-arguing it. Do not re-present
  "predictive promote on a predictable gap" as a new win — that's S1.
- Optional characteristics (e.g. sub-agents) appear **only** in the scenarios that
  use them; a scenario that doesn't need them omits them.

So each `sN/README.md` leads with its **distinctive driver** and the *new* thing it
buys; shared building blocks (predictive promote, value eviction, drop-on-death)
are cited, not re-derived.

## Scenario index

All seven run **full 4-tier** (HBM/DRAM/DISK/DROP, see the requirement below).
S1–S7 each isolate one driver for a clean, attributable win; **S8 is the
all-on capstone** (every driver + lever together). (Numbering keeps S5/S6/S7;
the former S3 "compaction" and S4 "session-end" are merged into S3 "drop-on-death",
and the former S8 "DISK cascade" is now the global 4-tier requirement + folded
into S1/S8 — so there is no standalone S4.)

| # | Scenario | Distinctive driver | Mechanism(s) used | Headline metric | Status |
|---|---|---|---|---|---|
| **S1** | [Tool-call predictability (basic)](s1-toolcall-predictability/) | tool call (universal) | demote + predictive promote — **established here** | post-gap TTFT / load_back | promote not yet firing |
| **S2** | [Shared-prefix retention under scratch churn](s2-shared-prefix-retention/) | fleet-shared system prefix | value eviction | shared-prefix recompute / TTFT | tied on cache-hit once; retest on TTFT |
| **S3** | [Drop-on-death (compaction + end)](s3-drop-on-death/) | provably-dead KV (compaction; program/sub-agent end) | drop-on-death | arrival/next-prefill TTFT, throughput | SESSION_END A/B passed on TP4 HBM+DRAM; compaction and storage deletion open |
| **S5** | [Value-gated pause under overload](s5-overload-pause/) | overload + heterogeneous runtimes | admission pause | goodput / p99 under overload | dormant (pauses=0); needs live overload |
| **S6** | [Blocking sub-agent](s6-blocking-subagent/) | blocking sub-agent (very long idle + child lifecycle) | *reuses S1* demote+promote, + drop | freed-HBM benefit + parent-resume TTFT | promote not firing |
| **S7** | [Background fan-out](s7-background-fanout/) | background fan-out (predictable concurrency spike) | proactive demote / pause (event-carried forecast) | spike p99 / no-thrash | forecast term added; needs impl |
| **S8** | [Comprehensive (everything at once)](s8-comprehensive/) | **all of S1–S7 together** | the full joint_decide union | end-to-end makespan / goodput | capstone; hardest to attribute; run last |

## Measurement discipline (applies to all)

- **Metric ≠ cache-hit rate.** With HiCache, a DRAM hit and an HBM hit both count
  as "hit" — cache-hit is blind to the tier difference that is the whole point.
  Measure **load_back bytes**, **TTFT**, **makespan/goodput**, **p99**.
- **All three arms run HiCache** (the user's constraint: win in the HiCache
  scenario). Only the scheduler differs.
- **Same offered load** across arms; for closed-loop, fixed-trace replay; report
  N≥3 mean±std.
- **Replay must be teacher-forced** (force the captured *output token ids*, not
  just the length). Length-only replay re-prefills each turn's output segment in
  the next turn (forced content ≠ real), an artifact that dilutes the TTFT /
  cache-reuse signal these scenarios measure. The faithful mechanism + its
  empirical no-op proof (timing + sglang-state identical) is a **prerequisite**:
  see [`harness/teacher_forcing/`](harness/teacher_forcing/) (task #234). Trust
  wherewewin's TTFT/cache numbers only after that PASSes.
- Each scenario states its **falsification**: the result that means "no win here".

## Make every workload genuinely 4-tier (force DROP / preemption) — REQUIRED

A wrong scheduling decision's **cost escalates with tier depth**: a wrong
tier-placement costs a `load_back` (ms); a wrong **DROP** (eviction from *all*
tiers) costs a **full re-prefill / recompute** (the most expensive outcome). So a
3-tier-or-under-pressured workload (everything stays cached somewhere) caps the
penalty at load_back and yields only a small win — exactly the regime where
cache-hit tied. **The decisive win requires the full 4-tier `{HBM, DRAM, DISK,
DROP}` to be genuinely pressured so the 4th state actually occurs**, because that
is where value-aware (drop the low-value, recompute-cheap unit) and recency (drop
the least-recent, which can be a high-value cold unit) diverge under maximal
penalty.

**DISK is always enabled** (every arm launches with the mooncake store on), so the
hierarchy is genuinely 4-deep in all scenarios — never a 2-tier HBM/DRAM setup.
Two pressure modes — both must reach the 4th state, **tuned so it happens but the
system does not collapse** (my earlier 35%-timeout run was over-pressured):

- **Cache pressure** (S1, S2, S3, S6): size tiers so the **reusable cached working
  set > HBM + DRAM + DISK** → reused prefixes get DROPped and recomputed on reuse.
  Knobs: small `--hicache-ratio` (DRAM), small `global_segment_size` (DISK), pool,
  concurrency × context size.
- **Live pressure** (S5, S7): size pool so the **running decode set > HBM** →
  sglang preempts → pause/admission has something to relieve.
- **S8** drives **both** at once (combined workload over the full 4 tiers).

**Gate before comparing:** first confirm on **baseline B's logs** that DROP/
recompute (or preemption) is *actually occurring* — else the hierarchy isn't
pressed to the decisive frontier and the comparison is uninformative.
