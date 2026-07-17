# cost profile for Qwen/Qwen3.5-9B on H200 — generated 2026-06-01T23:40:08
# κ_i recompute curve (persistent truth, offline bench fit):
export SGLANG_CSIGMA_KV_ALPHA=1.092994e-07
export SGLANG_CSIGMA_KV_BETA=2.469174e-02
export SGLANG_CSIGMA_KV_GAMMA=6.442200e+00
export SGLANG_CSIGMA_M_ALPHA=0.0
export SGLANG_CSIGMA_M_BETA=0.0
export SGLANG_CSIGMA_LSTAR=0.0
export SGLANG_CSIGMA_MODEL=Qwen/Qwen3.5-9B
export SGLANG_CSIGMA_DEVICE=H200
# c^xfer cross-pool fire wall — cold-start SEED (runtime EWMA drifts on top):
export SGLANG_XPOOL_NB_CHUNK_COST_INIT_US=90.1908
# c_m mamba per-slot copy — fixed-HW constant (env-precedence; boot probe skips):
export SGLANG_CM_MAMBA_PER_SLOT_US=114.464
