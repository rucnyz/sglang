# c_σ(L) recovery-cost curves — findings & paper revision

## Setup

H200 BF16, Qwen3.5-35B-A3B real config (40 layers: 30 linear-attention DeltaNet
+ 10 full-attention; head_dim=256, GQA 16:2 on attn; lin_num_v_heads=32, lin
key/value head_dim=128). Microbench times bare attention prefill kernel
(sgl_kernel.flash_attn_varlen_func, causal=True) and bare DeltaNet recurrent
kernel (sgl FLA Triton chunk_gated_delta_rule), both in single-sequence prefill
mode, sweeping L ∈ [128, 16384].

## Headline data (stack-level recovery wall-clock)

| L     | per-tok attn (µs) | per-tok GDN (µs) | c_KV stack (ms) | c_M stack (ms) | ratio |
|-------|-------------------|------------------|-----------------|----------------|-------|
| 128   | 0.254             | 2.267            | 0.326           | 8.707          | **26.7×** |
| 256   | 0.136             | 1.132            | 0.348           | 8.691          | **25.0×** |
| 512   | 0.079             | 0.556            | 0.404           | 8.537          | **21.1×** |
| 1024  | 0.049             | 0.278            | 0.498           | 8.540          | **17.1×** |
| 2048  | 0.037             | 0.164            | 0.767           | 10.064         | 13.1× |
| 4096  | 0.053             | 0.120            | 2.162           | 14.733         | 6.8× |
| 8192  | 0.089             | 0.099            | 7.251           | 24.293         | 3.4× |
| 16384 | 0.186             | 0.088            | 30.496          | 43.345         | 1.4× |

## Parametric fit

Stack-level recovery time (10 attn layers + 30 linear layers, ms):

```
c_KV(L) = α_KV · L² + β_KV · L + γ_KV
  α_KV = 1.19e-07 ms/token²   (L² attn compute — dominates at large L)
  β_KV ≈ 0
  γ_KV = 0.44 ms              (10× per-layer kernel-launch overhead)

c_M(L)  = α_M · L + β_M
  α_M = 2.17e-03 ms/token     (asymptotic per-token GDN scan)
  β_M = 6.99 ms               (30 layers × per-chunk setup overhead)
```

**Crossover L\* = 21,780 tokens** — solve c_KV(L*) = c_M(L*).
- L < L*: c_M dominates → cheaper to evict KV, expensive to evict mamba.
- L > L*: c_KV dominates → cheaper to evict mamba, expensive to evict KV.

## Why this is the *right* shape (not a problem)

The two curves having different functional forms with a real crossover is
exactly what we want for an L-aware budgeter:

1. **Below crossover (typical agent regime, L ≤ K_big = 8K).** Mamba snapshots
   are 3-27× more expensive to recover than KV. Layer 2 should prefer evicting
   KV when both pools are pressured. This matches the design goal.

2. **Above crossover (very long re-prefill, L > 22K).** Attention's L² compute
   overtakes the linear scan; KV evictions become more expensive. Layer 2
   should prefer evicting mamba. The same code, with the same parametric
   c_σ(L) curves and the live recovery-length distribution, gets this right
   without hand-coding either direction.

3. **Same model, different deployment regime.** A workload with very long
   contexts and small mamba snapshot intervals would hit the inverted regime;
   a workload with short replies stays in the dominant regime. The
   parametric form lets a single code base serve both.

The "constant 10-30×" hand-wave was always going to be wrong somewhere.
The crossover-driven cost model is more honest *and* gives the budgeter a
natural mechanism to read the workload.

## Paper changes (committed)

- **motivation.tex** §sec:motivation-l2: replaced "10-30× per-token" with
  parametric c_σ(L) form, two regimes, L* crossover, references
  Figure~\ref{fig:cost-curves} (= fig10_cost_curves_fit.pdf).
- **design.tex** §sec:design-l2-vhat: c_σ are now functions of L; per-pool
  V_σ' estimators evaluate at live mean recovery length \bar{L}_σ; explicit
  parametric forms for KV (α·L² + γ), M (α·L + β), LoRA (constant).
- **design.tex** §sec:design-l2-firegate: gate prose updated to say the
  asymmetry direction itself depends on \bar{L}_σ; gate inverts above L*
  without recoding.
- **design.tex** Stage-0 calibration: added probe (4) "Recovery-cost-curve
  probe" that produces (α_KV, γ_KV, α_M, β_M, L*); refs dev/eval/cost_model/.
  Section~\ref{sec:eval-margin} numbers updated to include curve fit values.

## Files

- Plot: `figures/fig10_cost_curves_fit.{pdf,png}` (paper repo)
- Microbench raw data: `figures/data/recovery_cost_NVIDIA_H200.json`
- Fit JSON (consumed by budgeter at boot): `figures/data/cost_curve_fit.json`
- Reproduce: `dev/eval/cost_model/README.md`

## Implications for #61

The budgeter's c_σ multiplier is **not a scalar** — it's a parametric curve
loaded from `cost_curve_fit.json` at boot, plus an online estimate of the
live mean recovery length per pool. Implementation steps:

1. Add `cost_curve_fit.json` to the budgeter's calibration JSON loader;
   store (α_KV, γ_KV, α_M, β_M) as planner state.
2. Track per-pool running mean recovery length \bar{L}_σ via EWMA over
   observed retraction lengths (KV) and snapshot-miss recovery distances (M).
3. In the V_σ' estimators, multiply the rate signal by c_σ(\bar{L}_σ) instead
   of a fixed scalar.
4. In the gate, use the same c_σ(\bar{L}_σ) so net-benefit calculation is
   consistent.
