# Workload Characterisation: swebenchpro / terminus-2 / 200-turn

Source: 3 sampled trials from Run K (`harbor_jobs/2026-05-26__14-16-56/`):
`instance_ansible__ansible-0ea40e__DrXc7Ht`, `..._1a4644__meVAZEe`, `..._5c225d__h8z3iB4`.
Per-trial conversation reconstructed from `agent/trajectory.json` (full message
history) and `agent/episode-N/debug.json` (the exact message list sent on turn N).

## 1. Turn structure

Every sampled trial **hit the `max_turns=200` cap exactly** (201 steps =
1 initial user prompt + 200 agent steps). The cap is hard-coded in the harbor
config (`config.agent.kwargs.max_turns=200`); no trial finished early. So turn
count is **constant=200**, not a tunable random variable — what grows is
**prompt length per turn**.

Prompt-token growth across the 200 turns (one trial, representative):

| metric | ep 0 | ep 199 | mean | median | max-delta | min-delta |
|---|---|---|---|---|---|---|
| `0ea40e` | 1931 | 14291 | 10090 | 10211 | +3513 | +41 |
| `1a4644` | 1583 | 10629 | 6481  | 6549  | +710  | +41 |
| `5c225d` | 1481 | 11166 | 6762  | 6709  | +587  | +41 |

Per-turn growth: median **+41 tokens**, mean ~45-62, occasional bursts up to
+3500 when a tool emits a large file dump. Total context stays well below
DeepSeek-V4-Flash's window; no summarisation fired (`summarization_count=0`
across all three).

## 2. Prefix structure — strictly monotonically extending

Verified directly: for each trial, every turn N's message list is **byte-identical
to turn (N-1)'s list plus exactly one `[assistant, user]` pair appended**
(199/199 boundaries, 0 mismatches). No truncation, no summarisation, no
re-ordering on this workload. This is the cleanest possible case for prefix
reuse: **the entire previous turn's KV is reusable, plus one short new tool
output (~41 tokens median) is the only new prefill**.

## 3. Tool call vs reasoning vs synthesis

terminus-2 uses a uniform JSON-shell protocol — every turn the model emits
`{analysis, plan, commands, task_complete}` and the harness replies with the
tmux pane diff. There is **no separate "tool-call" vs "reasoning" mode**;
every turn is structurally identical (user tool output → assistant JSON).
Median `completion_tokens=1` (frequent empty/parse-error turns) interleaved
with rare synthesis turns up to 3.8k tokens.

## 4. Inter-turn timing (600 gaps across 3 trials)

| p10 | p50 | p75 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| 0.62s | 1.11s | 3.24s | 13.96s | 22.81s | 88.60s | 223.38s |

44% of gaps are <1s (back-to-back tool calls), 87% <10s, 3.7% are ≥30s
(long generations or LLM thinks). So the workload is **mostly bursty
(sub-second cadence) with a heavy-tail of ~minute-scale stalls** — exactly
the regime where eviction during a long stall is the dominant risk.

## 5. Cross-trial sharing

Comparing initial prompts across the 3 trials: **first ~3013 chars (≈ 750
tokens) are byte-identical** — the system prompt, JSON-protocol spec, and the
boilerplate "You are an AI assistant…" preamble. Divergence starts at the
`<pr_description>` body which is task-specific. So there is a real
**shared-platform unit (~750 tokens)** with very high cross-trial reuse,
plus a per-task tail (~1.5–5k tokens of PR text) that only a single trial reuses
across its own 200 turns.

## Implications for T11's estimator design

The workload is **a single regime**: a strictly monotonically extending
conversation, +40 tok/turn at ~1Hz median, with a ~750-tok shared system
prefix and a per-trial private trunk. This argues for a **two-bucket p_hat**:

1. **Per-session trunk units**: p_hat≈1.0 while the session is active because
   the next turn provably reuses every prior token. Right time horizon is the
   inter-turn gap distribution (p90≈14s, p99≈90s) — a Hawkes/Pareto long-tail
   fit is overkill; a session-active step ("active → p=1, idle>p99 → p=0")
   works as the baseline.
2. **Shared platform units** (~750-tok system prompt): p_hat is high and
   roughly constant under concurrent multi-trial load. A simple global
   hit-rate counter (recent prefix matches / admissions) captures it.

Not needed: per-tool-class clustering (uniform protocol), summarisation-aware
logic (didn't fire), or short-window burst detection (monotonic extension
makes next-turn reuse deterministic from the trace). The 200-turn hard cap
means the estimator never has to predict end-of-program. **Recommendation**:
ship the session-active + shared-platform-prior baseline (T11b-1 + T11b-2)
first; defer Hawkes/Pareto (T11b-3) unless Run K-a/J shows a more
heterogeneous workload than swebenchpro/terminus-2.
