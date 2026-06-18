# Prelude paper — evaluation settings (the canonical plan)

This document is the source-of-truth for every benchmark / experiment that goes into the paper. Each setting names the workload, the comparison cells, the metrics, the dataset source, and the expected paper figure / table. **When the paper or eval design changes, edit this file first; the runner scripts in `dev/eval/` follow.**

Drawn from `prelude-paper/evaluation.tex` (Q1–Q5) plus the §Ablations section. Every TODO in the paper has a corresponding setting block here.

## Hardware + global setup

- **Hardware:** 1× NVIDIA H200 (143 GB) per engine, isolated. CUDA 13.2 / driver 595.58. Single-GPU unless noted.
- **Engine commit:** `rucnyz/sglang@prelude` (this repo, fork). vLLM and unmodified SGLang are baselines.
- **Quality gate:** all settings emit token-identical output at `temperature=0` vs vLLM under the same prompt. KS test at `temperature>0`.
- **Dataset sources:** prefer `/data/yuzhou/projects/aproj/vllm/pd_exp/` infrastructure (Alpaca, ShareGPT, LongBench-v2, WildChat) — these are pre-vetted and the dataset utilities at `pd_exp/dataset_utils.py` and `pd_exp/serve/generate_distribution_shift_dataset.py` already produce the right shapes.
- **Models:** Qwen3 dense (0.6B–32B), Qwen3-MoE (30B-A3B), Qwen3-Next (80B-A3B, hybrid DeltaNet+MoE), Qwen3.5 hybrid DeltaNet (0.8B–35B), Gemma3 (1B/4B/12B). Implementation track currently focuses on **Qwen3.5-35B-A3B (TP=1, BF16)** as the primary hybrid target.

## The four-cell joint ablation (Q1 headline)

Every experiment below is one of:
- **(L1=0, L2=0):** stock-engine baseline. Default `MambaRadixCache` + `HiMambaRadixCache` + `strip_thinking_cache`. Pools sized at startup at the average-phase-mix optimum.
- **(L1=1, L2=0):** Layer 1 on (heterogeneous granularity + hits-per-byte LRU). Layer 2 off (static partition).
- **(L1=0, L2=1):** Layer 1 off (engine-default cache). Layer 2 on (planner consumes engine's existing tier-cost-dominated $V_\text{prefix}'$).
- **(L1=1, L2=1):** full system.

Whenever a setting reports four numbers, this is the cell ordering.

---

## Setting 1 — 24-hour phase-shift trace (Q1, headline)

**Paper section:** §6.2 (`evaluation.tex` "Headline").
**Goal:** prove $(L_1=1, L_2=1)$ strictly dominates the other three cells across phases that genuinely shift the binding pool. Report sustained throughput, P99 TTFT, KV-pool utilization, prefix-cache hit rate over time.

**Trace structure (synthesized; each phase compressed from 8h to ~5min for tractable runtime):**
- **Phase A** — classification + multi-LoRA + shared system prompt
  - Source: `pd_exp/dataset_utils.py:load_alpaca_prompts` + 32 rank-16 LoRA adapters (`max_loras_per_batch` workload from Sweep 2)
  - Distribution: input ~ 512 tokens (1.5K-token shared prefix + 100-token query), output ~ 16
  - Concurrency: 64; prompts/phase: 1500
- **Phase B** — short-form rerank with K∈[4,16]
  - Source: ShareGPT first-turn excerpts truncated to 256–768 tokens
  - Distribution: input ~ 512, output ~ 1–16 (`max_tokens=K` for various K)
  - Concurrency: 96; prompts/phase: 4000
- **Phase C** — long-context multi-turn chat with K∈[256,2048]
  - Source: `pd_exp/multiturn/export_dataset.py --dataset wildchat --num-conversations 200 --min-turns 8`
  - Distribution: input grows turn-by-turn from 1K to 16K tokens; output ~ 256–2K
  - Concurrency: 16; prompts/phase: 1200
- **Transition:** 5 minutes between phases with linear traffic mix from previous to next.

**Cells:** all 4 (L1×L2 ablation).

**Metrics (per phase + cross-phase):**
- `input_throughput`, `output_throughput` (tokens/s, sustained)
- `mean_ttft_ms`, `p99_ttft_ms`, `mean_tpot_ms`, `p99_tpot_ms`
- `prefix_hit_rate_pct` (from `cached-token / new-token` ratio in scheduler logs)
- `kv_pool_util_max`, `mamba_pool_util_max`
- `xpool_kv_to_mamba_count`, `xpool_mamba_to_kv_count` (Layer 2 actuator decisions)
- Time-series of `v_prefix_marginal` (Layer 1 reporter)

**Expected paper figure:** time-series plot of throughput per cell across the trace, plus a 4-row × N-metric table.

**Implementation:** `dev/eval/01_phase_shift_trace.sh` (TODO).
- Prerequisite: `dev/eval/datasets/phase_a.jsonl`, `phase_b.jsonl`, `phase_c.jsonl` (generated once via `pd_exp/serve/generate_distribution_shift_dataset.py` and `pd_exp/multiturn/export_dataset.py`).
- Runtime: ~80 min per cell × 4 cells = **~5.5 h end-to-end**.

---

## Setting 2 — $V_\sigma$ characterization sweeps (Q2)

**Paper section:** §6.3 ("Empirical $V_\sigma$ Curves Across Pools"), Tables \ref{tab:sweep1} and \ref{tab:sweep2}.
**Goal:** quantify the static-default-is-wrong claim by sweeping one knob per pool and showing $V_\sigma$ slope changes by orders of magnitude.

### Setting 2.1 — KV ↔ DeltaNet (Sweep 1)

- **Model:** Qwen3.5-35B-A3B
- **Workload:** prefill-heavy random-uniform, 1000 prompts, 1024 input / 256 output, request-rate=32. Source: `sglang.bench_serving --dataset-name random` is fine (synthetic random-token prompts; no shared prefix needed for this sweep).
- **Sweep:** `mamba_full_memory_ratio ∈ {0.1, 0.3, 0.5, 0.7, 0.9}` (5 points)
- **Cells:** baseline only (no Layer 1/2). This sweep characterizes the underlying $V$, not our system.
- **Metrics:** `input_throughput`, `output_throughput`, `mean_ttft_ms`, `mamba_usage_peak`, `full_token_usage_peak`. Match Table \ref{tab:sweep1}.
- **Reference numbers (paper):** at ratio=0.9 → 7648 input TPS, 13.6s mean TTFT, 0.66 mamba_usage. At ratio=0.1 → 3039 input TPS, 70s mean TTFT — 2.5× throughput swing across the sweep.

**Implementation:** `dev/eval/02_sweep_kv_dn.sh` (TODO). Runtime: ~1.5 h (5 server boots × ~15 min each).

### Setting 2.2 — KV ↔ LoRA (Sweep 2)

- **Model:** Qwen3-4B + 32 rank-16 LoRA adapters (any open-source LoRA set, or generate synthetic ones)
- **Workload:** uniform random adapter selection per request, real Alpaca prompts
- **Sweep:** `max_loras_per_batch ∈ {1, 2, 4, 8, 16, 32}` (6 points)
- **Metrics:** `input_throughput`, `mean_ttft_ms`, `p99_ttft_ms`, `kv_usage_peak`. Match Table \ref{tab:sweep2}.
- **Reference numbers (paper):** at `max_loras=1` → 5652 TPS, 7s TTFT. At `max_loras=32` → 7556 TPS, 74ms TTFT — 95× TTFT swing.

**Implementation:** `dev/eval/03_sweep_lora.sh` (TODO). Requires populating 32 LoRA adapters first (`dev/eval/datasets/loras/`).

### Setting 2.3 — Prefix cache on multi-turn (Sweep 3)

- **Model:** Qwen3-8B (full attention, no mamba — naive RadixCache baseline)
- **Workload:** 64-client 6-round multi-turn shared-prefix; system prompt 1024 tokens shared across each client's 6 rounds.
- **Sweep:** `mem_fraction_static ∈ {0.30, 0.40, 0.50, 0.65, 0.80}` (5 points)
- **Metrics:** `input_throughput`, `mean_ttft_ms`, `prefix_hit_rate_pct`. Expected: all five points report ~75.8% hit rate (V_prefix' is flat — failure mode the paper highlights).

**Implementation:** `dev/eval/04_sweep_prefix.sh` (TODO).
- Source: `pd_exp/multiturn/export_dataset.py --dataset wildchat --num-conversations 64 --min-turns 6`.
- Runtime: ~1 h.

---

## Setting 3 — Layer 1 $V_\text{prefix}'$ repair (Q3)

**Paper section:** §6.4 ("Layer 1 Repairs the $V_\text{prefix}'$ Signal on Hybrid Models"). Three sub-experiments labelled Q3.A / Q3.B / Q3.C in the paper.
**Goal:** show that Layer 1's two refinements (heterogeneous granularity + hits-per-byte LRU) repair all three signal-quality failures (faithful, smooth, stable).

### Setting 3.A — Faithful slope on smooth traffic (Q3.A)

- **Model:** Qwen3-Next-80B-A3B (or fall back to Qwen3.5-35B-A3B if Qwen3-Next isn't available; document the substitution)
- **Workload:** multi-turn shared-prefix (same as Setting 2.3), no perturbations
- **Sweep:** $m_\text{prefix}$ across the same range as Sweep 3 (5 points)
- **Cells (4 cache configurations):**
  1. **naive RadixCache** (no mamba state recovery — equivalent to $L_1=0, L_2=0$ on a non-mamba RadixCache)
  2. **engine default `MambaRadixCache`** with `page_size=1` + `HiMambaRadixCache` (host tier on)
  3. **engine alternative** `enable_mamba_extra_buffer=True`, `page_size=8K`, host tier off
  4. **Layer 1** ($K_\text{big}=8K$, $K_\text{small}=512$, hits-per-byte LRU)
- **Metrics:** `v_prefix_marginal` (the estimator from `MambaRadixCache.estimate_v_prefix_marginal`), `prefix_hit_rate_pct`, `per_hit_prefill_saving_tokens`.
- **Expected shapes:**
  1. flat (state-equality failure)
  2. tier-cost-dominated slope (faithful failure)
  3. step-function with cliffs at multiples of 8K (smooth failure)
  4. smooth and high — Layer 1 wins

**Implementation:** `dev/eval/05_layer1_faithful.sh` (TODO).
- **Blocker:** cells (3) and (4) require Layer 1's "heterogeneous granularity" code (Phase 3.d, NOT YET IMPLEMENTED). Today we have hits-per-byte LRU (3.a) and the V_prefix' reporter (3.b) but not big-page-only snapshot policy. Need to implement before this setting can run.

### Setting 3.B — Stability under cold-burst perturbation (Q3.B)

- **Model:** Qwen3-Next or Qwen3.5-35B-A3B
- **Workload:** synthesized 10-min trace
  - 4 min multi-turn shared-prefix (build phase)
  - 2 min cold un-shared random prompts (burst phase) — sourced from independent 10K-token Alpaca prompts via `generate_distribution_shift_dataset.py`
  - 4 min multi-turn returning (recovery phase)
- **Cells:**
  1. Layer 1 heterogeneous tree + recency LRU (the partial-Layer-1 failure case)
  2. Layer 1 heterogeneous tree + hits-per-byte LRU (full Layer 1)
- **Metrics (time-series):**
  - $\hat V_\text{prefix}'(m)$ at fixed $m$ over the 10-min trace
  - Within-burst $\hat V_\text{prefix}'$ floor
  - Post-burst recovery time-to-pre-burst-baseline
- **Expected:** (1) collapses during burst, slow recovery; (2) maintains flat $\hat V_\text{prefix}'$ across the perturbation.

**Implementation:** `dev/eval/06_layer1_stability.sh` (TODO).
- **Blocker:** same Phase 3.d blocker. Without heterogeneous granularity, both cells reduce to identical behavior.

### Setting 3.C — Composed Layer 1 → Layer 2 effect (Q3.C)

- **Workload:** the 24-hour phase-shift trace from Setting 1
- **Cells:** the four cache configurations from Setting 3.A combined with Layer 2 ON
- **Metrics:**
  - Number of cross-pool reallocation decisions Layer 2 issues per phase
  - Fraction of decisions reversed within the hysteresis window (thrash indicator)
  - Per-decision throughput delta on the post-decision interval
- **Expected:** under engine defaults, Layer 2 either does not act or thrashes; under Layer 1 the budgeter acts decisively, infrequently, with positive delta.

**Implementation:** `dev/eval/07_layer1_compose.sh` (TODO). Depends on Setting 1 + 3.A.

---

## Setting 4 — Estimator accuracy and convergence (Q4)

**Paper section:** §6.5.
**Goal:** quantify $\hat V_\sigma'$ vs ground truth $V_\sigma'$.

- **Workload:** held-out fragments of the 24-h trace
- **Ground truth:** exhaustive sweep of $m_\sigma$ on the held-out fragment (essentially Setting 2 repeated on a phase slice)
- **Metrics:**
  1. Steady-state relative error: $|\hat V_\sigma' - V_\sigma'|/V_\sigma'$
  2. Lag time after a 5-min phase transition until $\hat V_\sigma' \in [V_\sigma' \pm \epsilon]$
  3. Downstream effect on budgeter: overshoot, settling time, oscillation amplitude

**Implementation:** `dev/eval/08_estimator.sh` (TODO). Runtime: ~2 h.

---

## Setting 5 — Path-axis dispatcher on $K=1$ (Q5, secondary)

**Paper section:** §6.6.
**Goal:** Table \ref{tab:headline} — dispatcher beats vLLM and unmodified SGLang at $K=1$.

### Setting 5.A — Dense Qwen3-4B headline

- **Model:** Qwen3-4B
- **Workload:** 512-token random input, `max_tokens=1`
- **Concurrency sweep:** {1, 8, 32, 96}
- **Cells:**
  1. vLLM (latest stable)
  2. unmodified SGLang
  3. **Ours** (with path-axis dispatcher enabled)
  4. Ours with dispatcher OFF (= disabled, fallback to chunked-prefill — see ablation below)
- **Metrics:**
  - Throughput (req/s) at concurrency=96
  - P50 / P95 latency at concurrency=1
- **Reference numbers (paper):** vLLM 134.2 req/s; SGLang 151.7; Ours 186.7. P50 @ c=1: 18.1ms / 20.8ms / 15.4ms.

**Implementation:** `dev/eval/09_path_dense.sh` (TODO).
- **Blocker:** path-axis dispatcher itself is **NOT YET IMPLEMENTED**. Dev work needed before this can run.

### Setting 5.B — Hybrid extension

- **Models:** Qwen3.5-1.5B fine-tuned for sequence classification, Qwen3-Next-80B-A3B fine-tuned similarly
- **Cells:**
  1. Ours with dispatcher ON
  2. Ours with dispatcher OFF
  3. vLLM on the hybrid model directly
  4. unmodified SGLang on the hybrid model directly
- **Goal:** show the dispatcher bypasses both the paged-attention loop AND the DeltaNet slot pool simultaneously.

**Implementation:** `dev/eval/10_path_hybrid.sh` (TODO). Depends on Setting 5.A.

---

## Ablations

**Paper section:** §6.7.

### A1 — Layer 1 sub-features

- Decompose Layer 1 into `heterogeneous granularity alone` / `hits-per-byte LRU alone` / `full Layer 1` (3 cells, all on top of stock engine baseline).
- Cross with two workload modes: smooth multi-turn (Setting 3.A workload) and cold-burst (Setting 3.B workload).
- **Expected:** heterogeneous-only repairs shape but breaks under cold-burst; LPB-only is stable but inherits step-function shape; only the combination passes both.

**Implementation:** `dev/eval/A1_layer1_sub.sh` (TODO).

### A2 — Layer 1 big-page granularity

- Sweep $K_\text{big} \in \{2K, 4K, 8K, 16K\}$ at fixed $K_\text{small}=512$.
- Workload: Setting 3.A multi-turn.
- **Metric:** snapshot memory footprint vs cache-hit yield. Optimum depends on typical shared-prefix length.

**Implementation:** `dev/eval/A2_kbig_sweep.sh` (TODO). **Blocked on Phase 3.d.**

### A3 — Layer 2 hysteresis

- Sweep $\Delta_\text{hyst} \in \{0\%, 1\%, 5\%, 10\%, 20\%\}$ on Setting 1's trace.
- **Expected:** 0% thrashes, 20% lags; ours uses 5% as default.

**Implementation:** `dev/eval/A3_hyst.sh` (TODO).

### A4 — Layer 2 control interval

- Sweep $\tau \in \{5, 15, 30, 60, 300\}$s on Setting 1's trace.
- Default: 30s.

**Implementation:** `dev/eval/A4_tau.sh` (TODO).

### A5 — Layer 2 actuator: VMM chunk size

- Sweep VMM `chunk_size ∈ {64MB, 256MB, 1GB}`.
- Default: 256MB. We currently use 64MB.
- **Metric:** wasted bytes on shrink/grow + bitmap overhead. Need to verify 256MB is actually the right operating point. Original 2e.5.6.3.b ~6% steady-state TTFT claim has been superseded by the 5-trial bisection (RESULTS.md "Arena structural cost" block, paper §sec:eval-arena-cost): +7.15% mean TTFT with 3.4× higher trial-to-trial variance than baseline. Open question: does coarser chunk size (1 GiB / 4 GiB) reduce that variance? Hypothesis (queued for variance-source session) is that GPU TLB pressure on the 25 GiB cuMemMap range with 2 MiB pages is the source, in which case bigger chunks → fewer TLB entries → lower variance.

**Implementation:** `dev/eval/A5_chunk_size.sh` (TODO).

### A6 — Path-axis dispatcher: $K=1$ vs $K=2$

- Force chunked-prefill (dispatcher OFF) at `max_tokens=1`.
- **Expected:** recovers gap to unmodified SGLang at `max_tokens=1`; unchanged at `max_tokens=512`.

**Implementation:** `dev/eval/A6_k1_vs_k2.sh` (TODO).

---

## Quality preservation

**Paper section:** §6.8.
**Goal:** confirm the system never trades quality for latency.

- **Test 1:** at `temperature=0`, exact-match every output token vs vLLM under the same prompt. Pass criterion: byte-identical full sequences.
- **Test 2:** at `temperature>0`, two-sample Kolmogorov-Smirnov test on output-token distributions vs vLLM. Pass: $p > 0.05$.
- **Test 3:** per-task accuracy on a held-out classification set (Qwen3.5-1.5B classifier).
- **Test 4:** ROUGE-L on XSum.
- **Test 5:** prefetcher accuracy in the PF-LLM case study (Section 7).

**Implementation:**
- `dev/eval/Q1_token_identical.sh` (= our `dev/2e/16_kv_mamba_xfer_equiv.sh` and `17_kv_mamba_xfer_coordinated.sh` already cover this for Layer 2; needs extension to Layer 1).
- `dev/eval/Q2_ks_test.sh` (TODO).
- `dev/eval/Q3_classify_acc.sh` (TODO).
- `dev/eval/Q4_rouge_xsum.sh` (TODO).

---

## Implementation status

| Setting | Description | Status | Blocker |
|---|---|---|---|
| 1 | 24-h phase-shift trace | **DONE v9 + 3-trial variance bands (2026-05-01)** — adaptive K_BIG fixed Phase A regression. (1,1) cell beats baseline -29% mean TTFT / -62% P99 on Phase C. With variance bands joint cell ≈ L1-only (Δ < combined σ); L2 = no-regression mechanism, doesn't add measurable value over L1-alone on this trace. Honest paper framing landed. → paper Fig 7 + tab:headline-v9 + tab:variance-bands + tab:contribution-attribution. | demonstrating L2-positive workload is paper's flagged work-in-progress |
| 2.1 | KV↔DN sweep | **DONE PASS** — paper Table 1 updated | — |
| 2.2 | KV↔LoRA sweep | **DONE PASS** — paper Table 2 updated, 192× swing | — |
| 2.3 | Prefix sweep | **DONE PASS** — paper Table 3 updated, V_prefix flat | — |
| 3.A | V_prefix' faithful | **DONE PARTIAL (3-arm)** — default best on TTFT, all configs in 80-83% hit band | host-tier-on arm + pressured-pool workload future work |
| 3.B | V_prefix' stability | **DONE PASS + 3-trial variance bands (2026-05-01)** — LPB recovery -18% TTFT vs recency (paper tab:q3b). 4-cell ablation tab:q3b-4cell variance run shows joint cell L11 209.0±6 ms ≈ L10 207.4±4.4 ms (Δ < combined σ); same pattern as v9-auto. L1 carries -26% recovery TTFT win; L2 marginal value below noise floor. | same |
| 3.C | Composed L1+L2 | **DONE** — Layer 2 invariant to L1 on stress trace (21 transfers all cells); shows clean separation of concerns | feedback-loop workload follow-up |
| 3.D | LPB-vs-recency on GSP | **DONE PASS** — paper tab:lpb-gsp added, -19.77% TTFT | — |
| 4 | Estimator accuracy | **DONE QUANTITATIVE** — proxy saturation-blind; SGLANG_XPOOL_QDEPTH_TRIGGER fallback rule landed (unit tests PASS, gated). E2e on Phase 1+2+3: workload doesn't dual-saturate so new rule never activates; deeper per-pool admission signal marked follow-up. | broader workload needed to exercise new rule |
| 5.A | Path-axis dense | not started | **path-axis dispatcher implementation** |
| 5.B | Path-axis hybrid | not started | depends on 5.A |
| A1 | L1 sub-features | **DONE PASS** — cross-workload table, LPB dominant on smooth, full Layer 1 wins on cold-burst → paper tab:a1 | — |
| A2 | K_big sweep | **DONE INFORMATIVE** — K_big=0 best on prefix-friendly GSP; tradeoff only when snapshot mem pressured | — |
| A3 | Hysteresis sweep | **DONE INFORMATIVE** — workload too monotone for thrash | depends on 1's actual phase-shift trace |
| A4 | Tau sweep | **DONE PASS** — smooth monotone curve, paper tab:a4 | — |
| A5 | VMM chunk-size sweep | **DONE MAJOR** — 1GB chunks fix TTFT, paper §6.7 updated | — |
| A6 | K=1 vs K=2 | not started | depends on 5.A |
| Q1 | Quality preservation (temperature=0 token-identity) | **DONE PASS** — 50/50 byte-identical, default vs full prelude | — |
| Q2 | Quality preservation (temp=1.0 KS test) | **DONE PASS** — KS p=0.362, cosine 0.985, len within 0.2% | — |
| Q3 | Quality preservation (classification accuracy) | **DONE PASS** — 49/50 = 98% on both arms, 50/50 byte-identical | — |
| Q4 | Quality preservation (ROUGE-L vs wildchat ref) | **DONE PASS** — mean ROUGE 0.124 vs 0.148, KS p=0.306, t-test p=0.055 | XSum substitute |

## Order of execution (proposed) — STATUS as of 2026-05-01

The original priorities (Setting 2.1, A5, 2.3, Setting 1) are all DONE. The
2026-05-01 NeurIPS-strengthening session added:

1. **Tier 1 (figures, paper `76b753e`):** 6 figures tied to paper narrative.
2. **Tier 2 (paper `ef09782` → `198a9a5`):** L1×L2 contribution attribution +
   tab:contribution-attribution; vLLM cross-engine baseline tab:vllm-vs-sglang;
   static-best partition baseline tab:static-best (single ratio=0.9 beats
   dynamic on v9 Phase A by 1.9× — honest finding); 3-trial variance bands on
   Setting 1 v9-auto headline + Fig 7 + tab:variance-bands.
3. **Tier 3 (paper `0376ec4`):** gate-retune negative results — MAMBA_HIGH
   sweep + NET_BENEFIT enable + COOLDOWN sweep; all configs fire 15 transfers
   on v9 Phase A regardless. Reframed as "v9-auto = L1 headline + L2
   no-regression test"; Q3.B is L2's actual win trace.
4. **Tier 4 (paper `aaa837c`):** Q3.B 4-cell variance bands — joint cell
   209.0±6 ms ≈ L1-only 207.4±4.4 ms; L2's marginal value below trial-to-trial
   variance on Q3.B too. Final paper claim: L1 carries the measurable
   end-to-end win on every workload tested; L2 = no-regression mechanism whose
   marginal value over L1 is below variance at our measurement budget.
   Demonstrating L2-only positive delta requires admission-pressured workload
   (no paused/retracted reqs in current traces because stock cache evicts
   aggressively) — flagged as work-in-progress.
5. **Path-axis dispatcher (Settings 5.A, 5.B, A6):** still BLOCKED.

## Open implementation work (blocking eval)

These need development before the paper-grade eval can run:

1. **Phase 3.d: heterogeneous granularity in MambaRadixCache.** Required for Settings 3.A, 3.B, A1, A2. Estimated ~400 LoC + integration test.
2. **Path-axis dispatcher.** Required for Settings 5.A, 5.B, A6. Estimated ~600 LoC + integration test.
3. **Layer 2 default-on with planner consuming real signals.** Currently behind several env flags. To run Setting 1 cleanly we want a single `SGLANG_PRELUDE=1` switch that turns on the full stack with sane defaults.
4. **Eval orchestration.** Each setting needs a runner script in `dev/eval/` that:
   - Loads or generates the prompts
   - Boots the server with the right flag combination
   - Sends the workload
   - Captures metrics into a structured JSONL
   - Generates the paper-format table or figure

## Where datasets come from

| Dataset | Source location | Generator |
|---|---|---|
| Alpaca prompts | HuggingFace `tatsu-lab/alpaca` (cached) | `pd_exp/dataset_utils.py:load_alpaca_prompts` |
| ShareGPT V3 (multi-turn) | `/data/yuzhou/.cache/huggingface/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/.../ShareGPT_V3_unfiltered_cleaned_split.json` | `pd_exp/dataset_utils.py:load_sharegpt_prompts` |
| WildChat (multi-turn long) | HuggingFace `allenai/WildChat-1M` | `pd_exp/dataset_utils.py:load_wildchat_conversations` |
| LongBench-v2 (long-context) | HuggingFace `THUDM/LongBench-v2` | `pd_exp/dataset_utils.py:load_longbench_prompts` |
| NuminaMath (reasoning) | HuggingFace `AI-MO/NuminaMath-CoT` | `pd_exp/dataset_utils.py:load_numina_math_prompts` |
| Phase-mixed JSONL (varied input/output) | generated | `pd_exp/serve/generate_distribution_shift_dataset.py` |
| GSP (shared-prefix synthetic) | generated in-bench | SGLang's built-in `--dataset-name generated-shared-prefix` |

For all settings, we keep one regenerated copy in `dev/eval/datasets/` and reuse across runs to ensure reproducibility.

---

## Append-only changelog

- **2026-04-30** — Initial draft. Settings 1–5 + 6 ablations + quality preservation. Drawn from `prelude-paper/evaluation.tex` Q1–Q5 section + §Ablations. Marked Phase 3.d and path-axis dispatcher as critical blockers.
- **2026-05-01** — NeurIPS strengthening session: 6 figures (Fig 1-7), L1×L2 contribution attribution, vLLM baseline, static-best partition baseline, 3-trial variance bands on v9-auto and Q3.B, gate-retune sweeps (negative). Final honest framing: L1 carries lift; L2 = no-regression mechanism; demonstrating L2-positive workload is paper-flagged work-in-progress. Phase 3.d FIXED earlier today (commits b37bbc82e + 325f25334).
