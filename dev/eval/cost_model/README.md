# Recovery-cost microbench + Stage-0 calibration

## Stage-0 calibration (the one a deployment actually runs)

Run before the first `sglang.launch_server` on a new (GPU, model) tuple. Caches
to `~/.cache/sglang/cost_calibration/<gpu>__<model>.json`; subsequent calls
hit the cache and return in milliseconds.

```bash
# Pin env vars in the current shell, then launch the server.
eval "$(CUDA_VISIBLE_DEVICES=2 bash dev/eval/cost_model/stage0_calibrate.sh)"
echo $SGLANG_CSIGMA_LSTAR    # → 21780-ish on H200/Qwen3.5-A3B

# Or run from python directly with human-readable output.
CUDA_VISIBLE_DEVICES=2 .venv/bin/python dev/eval/cost_model/stage0_calibrate.py
```

Env vars emitted (SGLANG_CSIGMA_* convention, mirrors SGLANG_BUDGETER_*):

- `SGLANG_CSIGMA_JSON` — full record path
- `SGLANG_CSIGMA_KV_{ALPHA,BETA,GAMMA}` — c_KV(L) = α·L² + β·L + γ
- `SGLANG_CSIGMA_M_{ALPHA,BETA}` — c_M(L) = α·L + β
- `SGLANG_CSIGMA_LSTAR` — crossover length
- `SGLANG_CSIGMA_{MODEL,DEVICE}` — for sanity-check at server boot

Force re-run with `--force`. The python script `--print-env` mode produces
stdout suitable for `eval`. All progress goes to stderr.

Currently ships dim presets for: `Qwen/Qwen3.5-35B-A3B`. Add presets for
other models by extending the `MODELS` dict in `stage0_calibrate.py`.

---

## Recovery-cost microbench (c_KV vs c_M)

Empirical validation of the paper's cross-pool recovery-cost asymmetry claim.
Times the bare attention prefill kernel (FlashAttention varlen, causal) and
the bare DeltaNet recurrent-derivation kernel (Triton chunk_gated_delta_rule)
at varying recovery length L, on Qwen3.5-35B-A3B's real layer dims.

## Reproduce

```bash
cd /scratch/yuzhou/projects/sglang

# 1. Run the microbench (≈30s on H200, no model load)
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u dev/eval/cost_model/bench_recovery_cost.py \
  --out-dir dev/eval/cost_model

# 2. 3-panel plot: per-tok kernel, stack-level total, ratio
.venv/bin/python -u dev/eval/cost_model/plot_recovery_cost.py \
  --in-json dev/eval/cost_model/recovery_cost_NVIDIA_H200.json \
  --out-dir dev/eval/cost_model

# 3. Fit parametric curves c_KV(L) = α·L² + β·L + γ, c_M(L) = α·L + β
#    Solves for crossover L*; writes cost_curve_fit.json that the budgeter
#    loads at boot.
.venv/bin/python -u dev/eval/cost_model/fit_cost_curves.py \
  --in-json dev/eval/cost_model/recovery_cost_NVIDIA_H200.json \
  --out-dir dev/eval/cost_model

# 4. Sync to paper repo (fig9 = 3-panel, fig10 = parametric fit)
cp dev/eval/cost_model/recovery_cost.{pdf,png} \
   /data/yuzhou/projects/hybrid-inference/figures/fig9_recovery_cost.{pdf,png}
cp dev/eval/cost_model/cost_curves_fit.{pdf,png} \
   /data/yuzhou/projects/hybrid-inference/figures/fig10_cost_curves_fit.{pdf,png}
cp dev/eval/cost_model/recovery_cost_NVIDIA_H200.json \
   dev/eval/cost_model/cost_curve_fit.json \
   /data/yuzhou/projects/hybrid-inference/figures/data/
```

To sweep different L, pass e.g. `--lengths 64 128 256 512 1024 2048`. Defaults
to a log grid 128…16384.

## Outputs

- `recovery_cost_NVIDIA_H200.json` — raw timings (median + IQR per L)
- `recovery_cost.png/.pdf` — 3-panel descriptive plot:
  1. per-token kernel cost (one layer) vs L, log-log
  2. stack-level total recovery wall-clock (10 attn + 30 linear) vs L
  3. ratio c_M/c_KV vs L, regime-dependent
- `cost_curves_fit.png/.pdf` — parametric fit overlay with crossover L\*
  marked. **This is what fig 10 in the paper points to.**
- `cost_curve_fit.json` — fit coefficients (α_KV, β_KV, γ_KV, α_M, β_M, L\*).
  **Loaded by the budgeter at boot for the c_σ(L) lookup table.**
- `FINDINGS.md` — full writeup of the regime structure and what changed in
  the paper.

## Key result

Stack-level recovery wall-clock on Qwen3.5-35B-A3B / H200 BF16:

```
c_KV(L) = 1.19e-7 · L² + γ_KV ms        (γ_KV ≈ 0.44 ms, β_KV ≈ 0)
c_M(L)  = 2.17e-3 · L + 6.99 ms
crossover L* = 21,780 tokens
```

Below L\*: c_M dominates → prefer evicting KV (typical agent regime).
Above L\*: c_KV dominates → prefer evicting mamba (very-long-context regime).
The budgeter consumes the parametric form, not a constant ratio.

## Configuration

Real Qwen3.5-35B-A3B dims (from HF config):
- 40 layers: 30 linear-attention + 10 full-attention (every 4th)
- hidden=2048, attn head_dim=256, GQA 16:2
- linear: num_v_heads=32, num_k_heads=16, key/value head_dim=128, conv_kernel=4
- BF16 weights/activations, FP32 mamba ssm state

Both kernels run in prefill mode over a single sequence (cu_seqlens=[0,L]).
