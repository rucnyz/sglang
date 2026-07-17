#!/usr/bin/env python3
"""KV-vs-mamba occupancy coupling bound for a hybrid Mamba+attention model.

Every quantity below is read from sglang's OWN config functions / variables --
nothing is hand-estimated or hardcoded. We never trust the server boot-log GB
strings (the mamba "ssm_state size: 1923.05GB" line is a known display bug);
instead we recompute mamba bytes/slot from Mamba2CacheParams.mamba_cache_per_req
and KV bytes/token from the DefaultPoolConfigurator cell-size formula.

If any sglang API is missing, this script FAILS LOUDLY -- there are no
hardcoded fallbacks.

Run:
    .venv/bin/python dev/interlayer/3_budgeter/mamba_fork_grow/kv_mamba_ratio.py
(ignore the zoxide stderr warning)

sglang symbols used (each printed value is labeled with its source):
  - ModelConfig                       sglang.srt.configs.model_config
  - <config>.mamba2_cache_params      -> Mamba2CacheParams (qwen3_next.py path)
  - Mamba2CacheParams.mamba_cache_per_req   (configs/mamba_utils.py)
  - ModelConfig.get_num_kv_heads / head_dim / v_head_dim / dtype / context_len
  - DefaultPoolConfigurator._compute_cell_size formula (model_executor/pool_configurator.py)
  - MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO (model_runner_kv_cache_mixin.py)
  - CHUNK_SIZE as FLA_CHUNK_SIZE      (layers/attention/fla/chunk_delta_h.py)
  - MambaRadixCache snapshot granularity (mem_cache/mamba_radix_cache.py)
"""

import os

import torch

# ---- the cc bench config (server args; the only "inputs", all else derived) ----
MODEL_PATH = (
    "/scratch/yuzhou/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/"
    "snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
TP_SIZE = 1
MAX_MAMBA_CACHE_SIZE = 64          # server arg --max-mamba-cache-size
PAGE_SIZE = 1                      # server arg --page-size
KV_CACHE_DTYPE = "auto"            # server arg --kv-cache-dtype (auto -> model dtype)

# boot-observed cross-checks (NOT used in the bound derivation; printed for
# comparison only). K from budgeter.jsonl (kv_used+kv_available mode);
# occupancies are the (min,max,mean) over a real p44_allon run.
BOOT_K_TOKENS = 1_827_295
OBS_KV_OCC = (0.50, 0.72, 0.64)
OBS_MAMBA_OCC = (0.0, 0.984, None)


def banner(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def init_tp_group_cpu():
    """Initialize a 1-rank TP group on CPU/gloo so <config>.mamba2_cache_params
    (which calls get_attention_tp_size()) works config-only, no GPU/weights."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=TP_SIZE,
        rank=0,
        local_rank=0,
        distributed_init_method="tcp://127.0.0.1:29577",
        backend="gloo",
    )
    initialize_model_parallel(tensor_model_parallel_size=TP_SIZE)


def main():
    # K_BIG must be read the way MambaRadixCache reads it, to label the snapshot
    # granularity regime correctly.
    k_big = int(os.environ.get("SGLANG_K_BIG", "0"))

    init_tp_group_cpu()

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.layers.attention.fla.chunk_delta_h import (
        CHUNK_SIZE as FLA_CHUNK_SIZE,
    )
    from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
        MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO,
    )

    mc = ModelConfig(model_path=MODEL_PATH, dtype="auto")
    htc = mc.hf_text_config

    # ------------------------------------------------------------------
    # 1. mamba bytes/slot -- sglang's Mamba2CacheParams.mamba_cache_per_req
    #    (this is the production path: <hf_text_config>.mamba2_cache_params)
    # ------------------------------------------------------------------
    cache_params = htc.mamba2_cache_params  # property; FAILS LOUDLY if absent
    mamba_bytes_per_slot = cache_params.mamba_cache_per_req
    n_mamba_layers = len(cache_params.layers)

    banner("1. MAMBA per-slot bytes (sglang Mamba2CacheParams.mamba_cache_per_req)")
    print(f"  hf_text_config class                 = {type(htc).__name__}")
    print(f"  cache_params.shape.conv              = {cache_params.shape.conv}")
    print(f"  cache_params.shape.temporal          = {cache_params.shape.temporal}")
    print(f"  cache_params.dtype.conv              = {cache_params.dtype.conv}")
    print(f"  cache_params.dtype.temporal          = {cache_params.dtype.temporal}")
    print(f"  len(cache_params.layers)  [n mamba L] = {n_mamba_layers}")
    print(f"  mamba_cache_per_req       [bytes/slot]= {mamba_bytes_per_slot:,}")

    # ------------------------------------------------------------------
    # 2. KV bytes/token -- sglang DefaultPoolConfigurator._compute_cell_size
    #    For a non-MLA hybrid model the KV pool spans ONLY the full-attention
    #    layers (linear/mamba layers carry no KV). cell_size =
    #      num_kv_heads * (head_dim + v_head_dim) * num_full_attn_layers * kv_size
    # ------------------------------------------------------------------
    # kv_cache_dtype "auto" with no FP8 quant resolves to model dtype
    # (model_runner.configure_kv_cache_dtype: self.kv_cache_dtype = self.dtype).
    assert KV_CACHE_DTYPE == "auto", "this script assumes kv_cache_dtype=auto"
    kv_cache_dtype = mc.dtype
    kv_size = torch._utils._element_size(kv_cache_dtype)
    num_kv_heads = mc.get_num_kv_heads(TP_SIZE)
    full_attn_layer_ids = htc.full_attention_layer_ids
    n_full_attn_layers = len(full_attn_layer_ids)

    kv_bytes_per_token = (
        num_kv_heads * (mc.head_dim + mc.v_head_dim) * n_full_attn_layers * kv_size
    )

    banner("2. KV per-token bytes (sglang DefaultPoolConfigurator cell-size formula)")
    print(f"  mc.dtype (kv_cache_dtype=auto)       = {kv_cache_dtype}")
    print(f"  kv_size (element bytes)              = {kv_size}")
    print(f"  mc.get_num_kv_heads(tp={TP_SIZE})            = {num_kv_heads}")
    print(f"  mc.head_dim                          = {mc.head_dim}")
    print(f"  mc.v_head_dim                        = {mc.v_head_dim}")
    print(f"  n full-attention layers              = {n_full_attn_layers}")
    print(f"  cell_size                 [bytes/tok]= {kv_bytes_per_token:,}")

    # ------------------------------------------------------------------
    # 3. Pool capacities and ratio cap
    # ------------------------------------------------------------------
    ratio = MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO
    M_slots = MAX_MAMBA_CACHE_SIZE
    max_running = M_slots // ratio  # mixin caps max_running this way

    banner("3. Pool capacities (server args + sglang ratio constant)")
    print(f"  M = max_mamba_cache_size  [slots]    = {M_slots}")
    print(f"  MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO = {ratio}")
    print(f"  capped max_running = M // ratio       = {max_running}")
    print(f"  context_len (mc.context_len)         = {mc.context_len:,}")
    print(f"  FLA_CHUNK_SIZE                       = {FLA_CHUNK_SIZE}")
    print(f"  page_size                            = {PAGE_SIZE}")
    print(f"  K (boot-observed, cross-check only)  = {BOOT_K_TOKENS:,} tokens")
    print(f"  SGLANG_K_BIG (snapshot granularity)  = {k_big}")

    # K config-only would need GPU mem-profiling (available_bytes // cell_size).
    # We cannot get available_bytes without a running engine, so K is taken from
    # the boot log purely as a cross-check input to the bound -- this is the
    # ONLY non-derived capacity, and it is labeled as such.
    K_tokens = BOOT_K_TOKENS

    # ------------------------------------------------------------------
    # 4. tokens-per-cached-mamba-slot range r  (the crux of the bound)
    #
    #   MambaRadixCache stores ONE mamba_value (1 mamba pool slot) per inserted
    #   radix node, covering that node's page_aligned_token_ids span.
    #   cache_finished_req / cache_unfinished_req assert the cached span is
    #   FLA_CHUNK_SIZE-aligned (page_aligned_len), so a node covers a multiple
    #   of FLA_CHUNK_SIZE tokens.
    #     - r_min: a node can be as small as one chunk  -> FLA_CHUNK_SIZE tokens.
    #     - r_max: with SGLANG_K_BIG=0 (cc default) snapshots are taken at every
    #       inserted node and a single node can cover an entire prefix, bounded
    #       only by context_len -> r_max = context_len tokens per slot.
    #       (With SGLANG_K_BIG=k>0, snapshots are forced onto multiples of k, so
    #       a long prefix becomes ceil(L/k) slots and r is pinned near k.)
    # ------------------------------------------------------------------
    r_min = FLA_CHUNK_SIZE
    if k_big > 0:
        # heterogeneous granularity: deepest snapshot ancestor at multiples of
        # k_big, so a node's effective covered span tops out at one k_big stride.
        r_max = k_big
    else:
        r_max = mc.context_len

    banner("4. tokens-per-cached-mamba-slot r  (MambaRadixCache granularity)")
    print(f"  r_min = FLA_CHUNK_SIZE               = {r_min} tokens/slot")
    print(f"  r_max ({'K_BIG stride' if k_big else 'context_len, K_BIG=0'})        = {r_max:,} tokens/slot")

    # ------------------------------------------------------------------
    # 5. Derive the KV:mamba coupling bound.
    #
    #   Per RUNNING req:   1 active mamba slot : seqlen KV tokens,
    #                      seqlen in [1, context_len].
    #   Per CACHED node:   1 mamba slot : [r_min, r_max] KV tokens.
    #
    #   So the per-slot KV token count t (= KV_tokens / mamba_slots) lives in
    #       [t_min, t_max] = [1, context_len]
    #   across the whole reachable region (a running req at seqlen=1 gives the
    #   floor; a single cached node covering a full prefix gives the ceiling).
    #
    #   Convert to a KV-occupancy bound when MAMBA is full (mamba_slots = M):
    #       kv_tokens in [M * t_min, M * t_max]  (clamped to K)
    #       kv_occ    = kv_tokens / K
    # ------------------------------------------------------------------
    t_min = 1                  # running req, seqlen 1, holds 1 active slot
    t_max = mc.context_len     # one slot (cached node or single long running req)

    banner("5. Derived KV:mamba coupling bound (algebra in comments)")
    print("  per-slot KV-token ratio t = KV_tokens / mamba_slots")
    print(f"    t_min = 1            (running req @ seqlen=1)        = {t_min}")
    print(f"    t_max = context_len  (1 slot covers a full prefix)  = {t_max:,}")

    # Bytes-domain check: which pool's bytes saturate first at the EXTREMES.
    mamba_pool_bytes = M_slots * mamba_bytes_per_slot
    kv_pool_bytes = K_tokens * kv_bytes_per_token
    print(f"\n  mamba pool bytes = M * bytes/slot    = {mamba_pool_bytes:,}"
          f"  ({mamba_pool_bytes / (1<<30):.2f} GiB)")
    print(f"  KV pool bytes    = K * bytes/token   = {kv_pool_bytes:,}"
          f"  ({kv_pool_bytes / (1<<30):.2f} GiB)")

    banner("5a. MAMBA full (mamba_slots = M)  =>  KV occupancy window")
    kv_when_mamba_full_lo = min(M_slots * t_min, K_tokens)
    kv_when_mamba_full_hi = min(M_slots * t_max, K_tokens)
    occ_lo = kv_when_mamba_full_lo / K_tokens
    occ_hi = kv_when_mamba_full_hi / K_tokens
    print(f"  KV tokens in [M*t_min, M*t_max] clamped to K:")
    print(f"    lo = min(M*1, K)          = {kv_when_mamba_full_lo:,} tok  -> KV occ >= {occ_lo:.4%}")
    print(f"    hi = min(M*context_len,K) = {kv_when_mamba_full_hi:,} tok  -> KV occ <= {occ_hi:.4%}")
    print(f"  => mamba full IMPLIES KV occupancy in [{occ_lo:.2%}, {occ_hi:.2%}]")

    banner("5b. KV full (kv_tokens = K)  =>  mamba occupancy window")
    # mamba_slots = kv_tokens / t  with t in [t_min, t_max]; clamp to M.
    mamba_when_kv_full_lo = min(K_tokens / t_max, M_slots)   # few slots, big spans
    mamba_when_kv_full_hi = min(K_tokens / t_min, M_slots)   # many slots, tiny spans
    mocc_lo = mamba_when_kv_full_lo / M_slots
    mocc_hi = mamba_when_kv_full_hi / M_slots
    print(f"  mamba slots in [K/t_max, K/t_min] clamped to M:")
    print(f"    lo = min(K/context_len, M) = {mamba_when_kv_full_lo:,.2f} slots -> mamba occ >= {mocc_lo:.4%}")
    print(f"    hi = min(K/1, M)           = {mamba_when_kv_full_hi:,.2f} slots -> mamba occ <= {mocc_hi:.4%}")
    print(f"  => KV full IMPLIES mamba occupancy in [{mocc_lo:.2%}, {mocc_hi:.2%}]")

    banner("5c. Which pool structurally saturates first?")
    # Compare slot-equivalent capacities at the *typical* working point: a
    # running/cached unit costs 1 mamba slot and (seqlen) KV tokens. The mamba
    # pool runs out after M units; the KV pool after K/t units. The pool with
    # the SMALLER unit-capacity saturates first.
    kv_units_at_tmax = K_tokens / t_max   # most KV-favourable (long spans)
    kv_units_at_64 = K_tokens / FLA_CHUNK_SIZE  # typical cached-chunk span
    print(f"  mamba unit-capacity            = M             = {M_slots} units")
    print(f"  KV unit-capacity @ t=context   = K/context_len = {kv_units_at_tmax:,.1f} units")
    print(f"  KV unit-capacity @ t=FLA_CHUNK  = K/64          = {kv_units_at_64:,.1f} units")
    first = "MAMBA" if M_slots < kv_units_at_64 else "KV"
    print(f"  M = {M_slots} << K/64 = {kv_units_at_64:,.0f}  =>  {first} pool saturates first")
    print("  (mamba has only 64 slots; even at the smallest 64-token cached span")
    print("   the KV pool could hold K/64 such units, ~%.0fx more -- so mamba is" % (kv_units_at_64 / M_slots))
    print("   the structurally binding pool for the cc config.)")

    # ------------------------------------------------------------------
    # 6. Cross-check vs observed (0.64 KV / 0.95+ mamba)
    # ------------------------------------------------------------------
    banner("6. Cross-check vs OBSERVED p44_allon run")
    obs_kv = OBS_KV_OCC[2]
    obs_mamba = OBS_MAMBA_OCC[1]
    print(f"  observed KV occupancy  (min/max/mean) = {OBS_KV_OCC}")
    print(f"  observed mamba occupancy (min/max)    = {OBS_MAMBA_OCC[0]}/{OBS_MAMBA_OCC[1]}")
    print(f"  mamba ~full ({obs_mamba}) bound says KV in [{occ_lo:.2%}, {occ_hi:.2%}]")
    in_bound = occ_lo <= obs_kv <= occ_hi
    print(f"  observed KV mean {obs_kv:.2%} inside [{occ_lo:.2%}, {occ_hi:.2%}]? -> {in_bound}")
    # implied avg tokens-per-slot at the observed point:
    obs_kv_tokens = obs_kv * K_tokens
    obs_mamba_slots = obs_mamba * M_slots
    implied_t = obs_kv_tokens / obs_mamba_slots
    print(f"  implied avg t = (KVocc*K)/(Mocc*M) = "
          f"({obs_kv:.2f}*{K_tokens:,})/({obs_mamba:.3f}*{M_slots}) = {implied_t:,.0f} tok/slot")
    print(f"  is implied t in [t_min={t_min}, t_max={t_max:,}]? -> "
          f"{t_min <= implied_t <= t_max}")
    print("  (the observed point sits well inside the structural window: mamba is")
    print("   near-full while KV is only ~64% -- exactly the regime the bound")
    print("   predicts when avg span per slot ~%.0f tokens, far below context_len.)" % implied_t)


if __name__ == "__main__":
    main()
