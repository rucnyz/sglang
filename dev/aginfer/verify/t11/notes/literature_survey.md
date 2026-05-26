# T11 — Cache Reuse Prediction: Literature Survey (Decision-Grade)

Context: paper §7 uses `p_hat = min(1, hits/age)` (Poisson-rate MLE,
uniform prior). On swebenchpro / terminus-2 / 200-turn rollouts this
is ~1.76× slower than plain LRU (Run K ≈ 1559 s vs H' 885 s). We need
a better reuse estimator. Below per approach: signal / fit on
multi-turn agent traces / cost.

## 1. Classical adaptive replacement

- **LRU-K (O'Neil 1993).** Signal: K-th-most-recent reference time;
  Backward-K interval. Captures *frequency through recency-of-history* —
  exactly what `hits/age` misses ("young+hot" vs "young+cold").
  **Agent fit: strong** — 200-turn prefix reuse leaves a long K=2/K=3
  tail. **Cost: ~150 LoC**, per-node K-timestamp ring, O(log N) heap,
  online.
- **ARC (Megiddo & Modha 2003).** Two LRU lists (T1/T2) + ghost lists
  B1/B2; self-tunes split via ghost hits. **Agent fit: medium** —
  partition has no probability semantics, V_u can't consume it directly.
  **Cost: ~300 LoC**, O(1).
- **LIRS (Jiang & Zhang 2002).** Signal: *reuse distance* between two
  consecutive references (IRR), not recency. Classifies Low-IRR (hot)
  vs High-IRR (cold). **Agent fit: strong** — IRR is exactly the
  distribution we want over "ages in HBM, revisited at turn 47".
  **Cost: ~500 LoC**, O(1) but intricate stack maintenance.
- **CAR/CART.** Clock-based ARC; pure perf hack, no new signal. Skip.
- **SLRU / Segmented LRU.** Probationary+protected, two-touch promotion.
  Only 1-bit history. **Agent fit: weak.** ~50 LoC.
- **TinyLFU / W-TinyLFU (Einziger 2017).** Count-min sketch frequency +
  aging. **Agent fit: medium** — frequency only, no recency; admission
  filter in front of window LRU. ~400 LoC.

Signal-quality order: **LIRS ≈ LRU-K > TinyLFU > ARC > SLRU**.

## 2. Learning-based replacement

- **LeCaR (Vietri 2018) / CACHEUS (2021).** Online RL: experts (LRU,
  LFU) blended by Hedge/EXP3, ghost-cache regret signal. **Ceiling =
  max(LRU, LFU)** — agent fit medium at best. ~200 LoC, fully online.
- **DLIRS / DLRFU.** LIRS+LFU mixtures; marginal over plain LIRS.
- **Hawkeye (Jain & Lin ISCA'16) / Glider (MICRO'19).** Train ML model
  (PC-hash SVM / LSTM) to **imitate Belady's MIN** on replay. Signal:
  learned cache-friendly/averse label per access. **Agent fit: very
  strong** — Belady is the optimum; if a model can predict "this prefix
  is reused 30 turns later" we win by definition. ~1k LoC + training
  pipeline; offline fit, online table-lookup inference.
- **Parrot (ICML'20) / LRB (Song NSDI'20).** Learned next-access-time
  regressor; LRB shipped in Akamai CDN. Same family. **Agent fit: very
  strong** — exactly our problem shape. ~1.5k LoC, GBDT, hourly retrain.
- **MAB framing.** Each (UnitType, Scope) = an arm, reward = hit.
  Assumes stationarity per arm, breaks on phase shifts. **Weak.** ~100 LoC.

Belady-imitation (Hawkeye/Glider/LRB) is the theoretical ceiling.

## 3. Self-exciting / temporal-point-process

- **Hawkes process.** λ(t) = μ + Σ α·e^{−β(t−t_i)}. Past hits *raise*
  future intensity — the exact multi-turn prefix story (more hits → more
  likely next-turn re-hit). Closed-form
  p_hat(τ) = 1 − exp(−∫λ). **Agent fit: very strong, principled.**
  ~300 LoC (EM/MLE on event stream); recursive O(1) online update.
- **Pareto / power-law inter-arrival.** Heavy-tailed memoryless;
  matches CDN reuse tails. **Agent fit: medium** — catches tail, misses
  burstiness.
- **Renewal / hyper-exponential.** K-phase mixture of exponentials —
  generalizes current Poisson assumption. ~150 LoC EM, online.

Hawkes is the theoretical primary.

## 4. LLM KV cache / agent-specific

- **SGLang RadixCache (SOSP'24).** Pure LRU on radix leaves; no
  prediction. Current baseline.
- **vLLM PagedAttention.** LRU; sibling prefix sharing only across
  concurrent reqs.
- **Mooncake (OSDI'24).** Conductor-side reuse prediction via
  **forward-window lookahead from request schedule**, not per-block
  history — assumes you know next-batch tokens.
- **HiCache / LMCache.** Hierarchical (HBM/DRAM/SSD), LRU + admission
  threshold; no learned p_hat.
- **Preble / multi-LoRA routing.** Static affinity, not reuse modeling.

Net: **no published LLM KV system uses a probabilistic per-block reuse
model.** T11 is new ground; closest prior art is Mooncake's lookahead
(different signal — schedule, not history).

## Recommended ordering (ideal-over-pragmatic)

Per memory:feedback-design-ideal-over-pragmatic, order is **by
theoretical correctness, not by LoC**:

1. **Hawkes process per (UnitType, Scope)** — *first*. Self-exciting
   matches multi-turn prefix bursts; gives a real p_hat(τ); O(1) inline
   (cached μ,α,β + last-event timestamp), O(N log N) daemon refit.
   This is what §7 V_u was *trying* to be.
2. **LRU-K with K=3** — *strong baseline / sanity floor*. No model
   assumption; reads K-th-back timestamp directly. If Hawkes can't beat
   LRU-3, the fit is bad.
3. **Belady-imitation regressor (LRB-style)** — *upper bound*. Train on
   T11a traces to predict next-access time; ship GBDT as offline
   estimator. The ceiling; right §7 reformulation if Hawkes is too
   parametric for phase changes.
4. **LIRS-style IRR histogram** — *fallback / interpretability*.
   Non-parametric, sliding window, smoothed; matches T11b plan #1 but
   with IRR (not raw inter-access) as the variable.

Skip: LeCaR/CACHEUS (ceiling = max(LRU, LFU)), MAB (non-stationary),
ARC (no probability semantics), SLRU/TinyLFU (insufficient signal).

For paper §7: replace Poisson-rate proxy with **Hawkes conditional
intensity**; fall back to **Belady-imitation** if the parametric form
breaks on phase shifts (terminus-2 tool-call bursts vs swebenchpro
edit loops).
