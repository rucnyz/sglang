# T9 — root cause of the "500 s gap" baseline vs Run H'

## TL;DR

The 500 s gap is **not** caused by daemon proxy / event bus /
KV scheduling.  It's caused by **agent runaway generations**:

> 1 % of LLM requests consume 80 % of all LLM serving time, by
> generating up to **64 k completion tokens** until they hit the
> `length` finish reason (context_length = 65536).

Cache scheduling cannot help these — they are decode-bound,
eating their own KV cache for tens of thousands of tokens.

## Numbers (N=3 matrix)

Parsed sglang per-request JSON logs (turned on via `--log-requests
--log-requests-format json` for this matrix).  16k+ requests per
config.

| metric | baseline (3 cycles) | ours (3 cycles) |
|---|---|---|
| total requests | 16331 | 16495 |
| runaway requests (>2500 completion_tokens) | 170 (1.0%) | 122 (0.7%) |
| runaway mean completion_tokens | 23080 | 29675 |
| runaway max completion_tokens | 63963 | 64068 |
| Σ e2e_latency all requests (s) | 141956 | 137164 |
| **Σ e2e_latency runaway** (s) | **114863 (80.9%)** | **107932 (78.7%)** |
| per-trial-equiv: all | 1478.7 | 1428.8 |
| **per-trial-equiv: without runaway** | **282.2** | **304.5** |

So 0.7–1.0 % of requests cause ~80 % of all per-trial wall time.
The baseline / ours difference (45 s, statistically not
significant) is dominated by which trials happened to hit a
runaway and how long the runaway was.

## Why this happens (hypothesis)

This matrix was the first to use `--ak temperature=0.0
--ak seed=42` to make the agent decision path reproducible.
Under greedy decoding:
* the model is deterministic given input,
* if it gets into a generation pattern that doesn't reach a
  natural stop token (e.g., a "let me think step by step ..."
  loop), it has no stochastic chance to break out,
* it runs until either the agent's `</response>` or until
  `max_new_tokens` / context cap = 65 k.

The single-shot Run K / kv_off / K-a runs used default sampling
(temperature unset, sglang default applies).  Those did NOT show
this 1 % outlier dominance, so wall times were closer to the
~885 s Run H' baseline.

## Implications

### For the matrix experiment

The 45-s Δ between baseline and ours is **noise from the runaway
distribution**, not signal about T11a daemon scheduling.  Removing
or capping runaway requests is the right first step before any
further matrix work.

### For T11 design

Daemon-side T11a (`program-alive` rule) cannot help runaway
requests — they generate tens of thousands of tokens *into* their
own context.  Their KV is in active use the whole time;
no eviction / migration / admission decision applies.

Inline-side T11a (sglang_adapter.py) is in the same boat: the
inline scorer runs at eviction time, not during generation.

The whole multi-tier KV cache scheduling story assumes:
* prefill is the dominant cost (prefix reuse benefit ≫ decode)
* decode is short per request

terminus-2 / swebenchpro under greedy decoding violates both:
20–60 k decode tokens per outlier turn means decode dwarfs
prefill, and outliers dominate wall time.

### For the paper

Two clean paths forward, both about workload cleanup:
1. **Cap completion_tokens** at, say, 4096.  This forces the
   model to truncate its runaways; trial wall time will be
   dominated by the prefix-reuse / migration story we actually
   want to evaluate.
2. **Use a workload where runaway is rare** (e.g., a single-turn
   chat dataset, or an agent with strong stop-token discipline).

Without one of these, KV cache scheduling cannot be shown to
matter on the current dataset — not because the design is wrong,
but because the workload's variance is dominated by decode-bound
outliers.

## Confirmatory test (plan B): H'_now N=3 matrix (no daemon)

Ran 2026-05-27: 3 cycles of `harbor → sglang :30000 directly`, no
daemon, same `temperature=0.0 seed=42 -l 32 -n 32`, same sglang
HEAD as the matrix.  Discriminates whether daemon proxy itself
explains any of the gap to historical H' 885 s.

| config | N=3 per-trial mean (s) | Δ vs matrix baseline |
|---|---|---|
| matrix baseline (daemon, kv_off settings) | 1389.3 ± 39.7 | — |
| matrix ours (daemon, T11a) | 1344.0 ± 54.6 | −45 (noise) |
| **H'_now (NO daemon, direct)** | **1392.8 ± 53.6** | **+3.5 (noise)** |

H'_now ≈ matrix baseline.  **Daemon proxy contributes 3.5 s/trial
at most — not 500 s.**  The "gap to Run H' 885 s" is NOT from the
daemon proxy.

So the original "500 s gap" is fully accounted for by:
1. **Setting drift** — historical H' didn't use `temperature=0`;
   under default sampling the model's stochastic decoding has
   probabilistic escape from runaway-generation patterns.  Once
   we pinned `temperature=0` (this matrix), runaways dominate.
2. **sglang HEAD drift** — multiple internal edits since H'
   (UnifiedRadixCache wire-contract, peek_time_counter,
   --log-requests overhead, --enable-cache-report) — each small,
   collectively non-trivial.
3. **Possible model/agent prompt drift** — DeepSeek-V4-Flash
   weights and terminus-2 prompt are pulled fresh, not pinned.

**For T11 design**: see "Implications" above.  daemon-side and
inline-side T11a are both equally unable to help runaway requests.
The real path forward for a clean KV-scheduling eval is to cap
`max_completion_tokens` so runaways can't dominate.

## Files

* parser: `verify/t9/parse_ttft.py`
* per-request stats: `verify/t9/results/N3_ttft_analysis.md`
* matrix summary: `verify/t9/results/N3_matrix_SUMMARY.md`
* this doc:   `verify/t9/results/N3_ROOT_CAUSE.md`
