"""
Phase 2e — kernel-level arena-cost isolation.

Hypothesis: the prelude/arena structural slowdown is concentrated in
`fused_moe` expert-dispatch kernels because those touch BOTH
`cudaMalloc`-backed model weights AND `cuMemMap`-backed KV-style
storage in a single launch (TLB / HBM channel locality cost).

This test deliberately strips away every other system effect and just
times pure CUDA kernels under two backing-allocator regimes:

    cudaMalloc baseline (torch.zeros / torch.empty)         vs.
    arena VMM (cuMemAddressReserve + cuMemCreate + cuMemMap)

across four kernels:

    fused_moe forward  - touches weights (cudaMalloc) + acts (we vary)
    flash-attn forward - touches Q (cudaMalloc) + KV (we vary)
    rmsnorm            - touches hidden_state only (we vary)
    pure GEMM          - touches A and B (both cudaMalloc; sanity)

Per cell we run 50 warmup + 200 timed iters with cuda.Event.

Hypothesis predictions:
    t_moe_vmm  / t_moe_cm  ~ 1.05-1.10  (the cross-region cost)
    t_attn_vmm / t_attn_cm ~ 1.00-1.02  (KV only; one region)
    t_rms_vmm  / t_rms_cm  ~ 1.00       (no KV at all - control)
    t_gemm_vmm / t_gemm_cm ~ 1.00       (both cudaMalloc - control)

Verdict:
    CONFIRMED if t_moe_vmm/t_moe_cm > 1.04 AND t_attn_vmm/t_attn_cm < 1.02.
    REFUTED   if t_moe_vmm/t_moe_cm <= t_attn_vmm/t_attn_cm.
    PARTIAL   otherwise.

NOT a system test: no scheduler, no model load, no server. Mock
weights and routing decisions. Run directly:

    CUDA_VISIBLE_DEVICES=7 python /scratch/yuzhou/projects/sglang/dev/2e/40_arena_kernel_isolation.py

Optional env:
    SMOKE=1                       run with tiny iters to confirm wiring
    SGLANG_ARENA_FROM_BLOB=1      use at::from_blob path inside arena
                                  (set automatically by this script)
"""
from __future__ import annotations

import json
import logging
import os
import statistics
import sys
import sysconfig
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

# Ensure arena uses the from_blob path (the path we actually run in
# production; the MemPool path is a separate mechanism with its own
# overhead).
os.environ.setdefault("SGLANG_ARENA_FROM_BLOB", "1")

# sglang's JIT kernels (silu_and_mul etc.) shell out to `ninja` via
# tvm_ffi. If we were invoked via an interpreter outside the venv that
# vendors ninja, the subprocess won't find it on PATH. Prepend the
# venv's bin dir so the JIT can resolve ninja.
_venv_bin = os.path.join(sys.prefix, "bin")
if os.path.isfile(os.path.join(_venv_bin, "ninja")):
    os.environ["PATH"] = _venv_bin + os.pathsep + os.environ.get("PATH", "")
# Also try the Python user/site scripts dir.
_scripts = sysconfig.get_path("scripts")
if _scripts and _scripts not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _scripts + os.pathsep + os.environ.get("PATH", "")

import torch

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("arena_kernel_isolation")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Constants — Qwen3.5-35B-A3B text_config (verified via AutoConfig).
# We do NOT load weights; only use these to size mock tensors.
# ---------------------------------------------------------------------------

QWEN35_A3B = dict(
    hidden_size=2048,
    moe_intermediate_size=512,
    num_attention_heads=16,
    num_key_value_heads=2,
    num_experts=256,
    num_experts_per_tok=8,
    head_dim=256,
    num_hidden_layers=40,
)

# Decode-shaped batch (Qwen3.5-A3B typical). 32 tokens decode batch is
# representative; for fused_moe the tokens land in the MoE input.
BATCH_TOKENS = 32           # decode-style "batch" (each row = one token)
NUM_KV_TOKENS = 4096        # how much KV we allocate (paged backing)
PAGE_SIZE = 64

DTYPE = torch.bfloat16

# Iter counts — overridable via SMOKE=1.
SMOKE = os.environ.get("SMOKE") == "1"
WARMUP_ITERS = 5 if SMOKE else 50
TIMED_ITERS = 20 if SMOKE else 200

# GEMM sanity-check shapes from the spec.
GEMM_M, GEMM_N, GEMM_K = 1024, 4096, 4096


# ---------------------------------------------------------------------------
# Allocation helpers.
# ---------------------------------------------------------------------------

def alloc_cudamalloc(shape: Tuple[int, ...], dtype: torch.dtype, device: str) -> torch.Tensor:
    """Plain torch.zeros backed by the caching allocator (cudaMalloc)."""
    return torch.zeros(shape, dtype=dtype, device=device)


class _ArenaTensorFactory:
    """Wrap MultiTensorArena so we can allocate fresh VMM-backed tensors
    on demand. MultiTensorArena groups N_LAYERS x N_KINDS sub-tensors
    sharing one VA arena; here we map each requested logical tensor
    onto its own (layer, kind) slot. The wrapper hides the bookkeeping.
    """

    def __init__(self, device_id: int):
        self.device_id = device_id
        # Pool of pre-built arenas keyed by (per_token_shape, dtype). We
        # carve a 64-slot arena per shape and hand out slots round-robin.
        self._arenas: Dict[Tuple[Tuple[int, ...], torch.dtype, int], "object"] = {}

    def _get_or_make_arena(self, per_token_shape, dtype, max_tokens):
        from sglang.srt.arena.multi_tensor_arena import MultiTensorArena

        # Element size and per-token bytes drive chunk_bytes alignment.
        elsz = torch.tensor([], dtype=dtype).element_size()
        per_token_bytes = elsz
        for d in per_token_shape:
            per_token_bytes *= d
        # 32 MiB chunk; bump if per-token bytes don't divide it.
        chunk_bytes = 32 * 1024 * 1024
        while chunk_bytes % per_token_bytes != 0:
            chunk_bytes *= 2
        # Round max_tokens up to chunk boundary.
        tokens_per_chunk = chunk_bytes // per_token_bytes
        if max_tokens % tokens_per_chunk != 0:
            max_tokens = ((max_tokens // tokens_per_chunk) + 1) * tokens_per_chunk

        key = (tuple(per_token_shape), dtype, max_tokens)
        if key not in self._arenas:
            arena = MultiTensorArena(
                device_id=self.device_id,
                n_layers=8,
                n_kinds=2,
                per_token_shape=tuple(per_token_shape),
                dtype=dtype,
                max_tokens=max_tokens,
                init_tokens=max_tokens,  # full backing — we want all bytes mapped
                chunk_bytes=chunk_bytes,
            )
            self._arenas[key] = {"arena": arena, "next_slot": 0}
        return self._arenas[key]

    def alloc(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Return a tensor of `shape` aliased onto a fresh VMM slot.

        Heuristic: treat the leading dim as N_TOKENS, the rest as
        per_token_shape. The arena allocates (max_tokens, *per_token).
        We slice [:n_tokens] off the first dim before returning.
        """
        n_tokens = shape[0]
        per_token_shape = shape[1:]
        info = self._get_or_make_arena(per_token_shape, dtype, n_tokens)
        arena = info["arena"]
        slot = info["next_slot"]
        info["next_slot"] += 1
        layer = slot // arena.n_kinds
        kind = slot % arena.n_kinds
        if layer >= arena.n_layers:
            raise RuntimeError(
                f"arena out of slots for shape={shape} dtype={dtype}; "
                f"increase n_layers in _get_or_make_arena"
            )
        full = arena.tensor(layer, kind)  # (max_tokens, *per_token_shape)
        return full[:n_tokens].contiguous() if False else full[:n_tokens]

    def cleanup(self):
        for info in self._arenas.values():
            try:
                info["arena"].cleanup()
            except Exception as e:
                logger.warning("arena cleanup failed: %s", e)


# Module-level factory — initialized in main().
_FACTORY: _ArenaTensorFactory | None = None


def alloc_vmm(shape: Tuple[int, ...], dtype: torch.dtype, device: str) -> torch.Tensor:
    assert _FACTORY is not None, "call setup_factory() first"
    return _FACTORY.alloc(shape, dtype)


# ---------------------------------------------------------------------------
# Timing.
# ---------------------------------------------------------------------------

@dataclass
class TimingResult:
    name: str
    samples_ms: List[float]

    @property
    def mean(self) -> float:
        return statistics.fmean(self.samples_ms)

    @property
    def p50(self) -> float:
        return statistics.median(self.samples_ms)

    @property
    def p99(self) -> float:
        if len(self.samples_ms) < 100:
            return max(self.samples_ms)
        s = sorted(self.samples_ms)
        return s[int(0.99 * len(s)) - 1]


def time_kernel(name: str, fn: Callable[[], None], warmup: int, timed: int) -> TimingResult:
    # Warmup.
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: List[float] = []
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]
    for i in range(timed):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    for i in range(timed):
        samples.append(starts[i].elapsed_time(ends[i]))
    return TimingResult(name=name, samples_ms=samples)


# ---------------------------------------------------------------------------
# Kernel: fused_moe.
# ---------------------------------------------------------------------------

def build_fused_moe_inputs(alloc_act, device, cfg=QWEN35_A3B):
    """Build the inputs for fused_experts_impl. Weights and routing are
    cudaMalloc-backed; only the *input hidden_states* (the activation
    that crosses MoE) is the variable. This matches what would happen
    in production — weights are model params (cudaMalloc), KV is arena;
    here we use the activation as a stand-in for the arena-backed
    intermediate that fused_moe touches alongside the weights.
    """
    H = cfg["hidden_size"]
    I = cfg["moe_intermediate_size"]
    E = cfg["num_experts"]
    K = cfg["num_experts_per_tok"]

    # hidden_states must be contiguous and bf16/fp16/fp32. (M, H).
    M = BATCH_TOKENS
    hidden_states = alloc_act((M, H), DTYPE, device).contiguous()
    # If the alloc path returned a non-contiguous view we make a clone
    # but only of the right shape so we still measure the alloc-path
    # tensor for the kernel call. For arena VMM the slice is contiguous
    # by construction (full first dim).
    assert hidden_states.is_contiguous(), \
        f"hidden_states not contiguous after alloc: shape={hidden_states.shape} stride={hidden_states.stride()}"

    # w1: (E, 2*I, H) for gated SiLU, w2: (E, H, I). Plain cudaMalloc.
    w1 = torch.randn((E, 2 * I, H), dtype=DTYPE, device=device).contiguous()
    w2 = torch.randn((E, H, I), dtype=DTYPE, device=device).contiguous()

    # Routing — fixed assignment, cudaMalloc-backed.
    topk_weights = torch.full((M, K), 1.0 / K, dtype=torch.float32, device=device)
    topk_ids = torch.zeros((M, K), dtype=torch.int32, device=device)
    for k in range(K):
        topk_ids[:, k] = torch.arange(M, device=device, dtype=torch.int32) * (k + 1) % E

    return hidden_states, w1, w2, topk_weights, topk_ids


def make_fused_moe_fn(alloc_act, device):
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts_impl

    hidden_states, w1, w2, topk_weights, topk_ids = build_fused_moe_inputs(alloc_act, device)

    def run():
        fused_experts_impl(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation="silu",
            is_gated=True,
        )
    return run


# ---------------------------------------------------------------------------
# Kernel: FlashAttention.
# ---------------------------------------------------------------------------

def make_flash_attn_fn(alloc_kv, device, cfg=QWEN35_A3B):
    """flash_attn_with_kvcache decode: q is (B, 1, H_q, D), KV is paged
    cache (num_pages, page_size, H_kv, D). Q is always cudaMalloc; KV
    is the variable. page_table (cudaMalloc) tells which pages each row
    consults.
    """
    from sgl_kernel.flash_attn import flash_attn_with_kvcache

    H_q = cfg["num_attention_heads"]
    H_kv = cfg["num_key_value_heads"]
    D = cfg["head_dim"]

    B = BATCH_TOKENS                # 32 active sequences
    seqlen_per_seq = NUM_KV_TOKENS // B
    pages_per_seq = (seqlen_per_seq + PAGE_SIZE - 1) // PAGE_SIZE
    num_pages = B * pages_per_seq

    # Q: (B, 1, H_q, D). Decode: one query token per seq.
    q = torch.randn((B, 1, H_q, D), dtype=DTYPE, device=device).contiguous()

    # KV cache layout: (num_pages, page_size, H_kv, D). This is the
    # tensor we vary by allocation path.
    kv_shape = (num_pages, PAGE_SIZE, H_kv, D)
    k_cache = alloc_kv(kv_shape, DTYPE, device)
    v_cache = alloc_kv(kv_shape, DTYPE, device)
    # Fill with random data so we measure real reads, not zeros (which
    # in some kernels short-circuit).
    with torch.no_grad():
        k_cache.copy_(torch.randn(kv_shape, dtype=DTYPE, device=device))
        v_cache.copy_(torch.randn(kv_shape, dtype=DTYPE, device=device))

    # page_table: (B, pages_per_seq) -> page indices into cache.
    page_table = torch.zeros((B, pages_per_seq), dtype=torch.int32, device=device)
    for b in range(B):
        for p in range(pages_per_seq):
            page_table[b, p] = b * pages_per_seq + p

    cache_seqlens = torch.full((B,), seqlen_per_seq, dtype=torch.int32, device=device)

    def run():
        flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            softmax_scale=1.0 / (D ** 0.5),
            causal=True,
            ver=3,
        )
    return run


# ---------------------------------------------------------------------------
# Kernel: RMSNorm.
# ---------------------------------------------------------------------------

def make_rmsnorm_fn(alloc_act, device, cfg=QWEN35_A3B):
    H = cfg["hidden_size"]
    M = BATCH_TOKENS

    x = alloc_act((M, H), DTYPE, device).contiguous()
    with torch.no_grad():
        x.copy_(torch.randn((M, H), dtype=DTYPE, device=device))
    weight = torch.ones((H,), dtype=DTYPE, device=device)

    def run():
        # Keep it simple — torch's functional rms_norm is good enough
        # for the control measurement.
        torch.nn.functional.rms_norm(x, normalized_shape=(H,), weight=weight, eps=1e-6)
    return run


# ---------------------------------------------------------------------------
# Kernel: pure GEMM.
# ---------------------------------------------------------------------------

def make_gemm_fn(alloc_a, device):
    M, N, K = GEMM_M, GEMM_N, GEMM_K
    a = alloc_a((M, K), DTYPE, device).contiguous()
    with torch.no_grad():
        a.copy_(torch.randn((M, K), dtype=DTYPE, device=device))
    b = torch.randn((K, N), dtype=DTYPE, device=device).contiguous()

    def run():
        torch.matmul(a, b)
    return run


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def setup_factory(device_id: int):
    global _FACTORY
    _FACTORY = _ArenaTensorFactory(device_id=device_id)


def teardown_factory():
    global _FACTORY
    if _FACTORY is not None:
        _FACTORY.cleanup()
    _FACTORY = None


def run_all() -> Dict[str, TimingResult]:
    device = "cuda"
    results: Dict[str, TimingResult] = {}

    # Order kernels so the heavy fused_moe (which compiles Triton on
    # first launch) runs first — its compile cost is absorbed in warmup
    # for both alloc paths.
    kernels = [
        ("fused_moe", make_fused_moe_fn),
        ("flash_attn", make_flash_attn_fn),
        ("rmsnorm", make_rmsnorm_fn),
        ("gemm", make_gemm_fn),
    ]

    for kname, make in kernels:
        for path_name, alloc_fn in [
            ("cudaMalloc", alloc_cudamalloc),
            ("vmm", alloc_vmm),
        ]:
            label = f"{kname}.{path_name}"
            logger.info("building inputs for %s", label)
            try:
                fn = make(alloc_fn, device)
            except Exception as e:
                logger.error("setup failed for %s: %s", label, e)
                raise
            logger.info("timing %s (warmup=%d, timed=%d)", label, WARMUP_ITERS, TIMED_ITERS)
            results[label] = time_kernel(label, fn, WARMUP_ITERS, TIMED_ITERS)
            # Free the closure so its tensors are eligible for GC
            # before the next path's allocations land.
            del fn
            torch.cuda.synchronize()

    return results


def fmt_ms(x: float) -> str:
    return f"{x:.4f}"


def emit_table(results: Dict[str, TimingResult]) -> Tuple[str, Dict[str, float]]:
    rows: List[str] = []
    rows.append("| kernel | path | mean ms | p50 ms | p99 ms |")
    rows.append("|---|---|---|---|---|")
    kernels = ["fused_moe", "flash_attn", "rmsnorm", "gemm"]
    means: Dict[str, float] = {}
    for k in kernels:
        for path in ("cudaMalloc", "vmm"):
            r = results[f"{k}.{path}"]
            means[f"{k}.{path}"] = r.mean
            rows.append(
                f"| {k} | {path} | {fmt_ms(r.mean)} | {fmt_ms(r.p50)} | {fmt_ms(r.p99)} |"
            )
    rows.append("")
    rows.append("| kernel | vmm/cudaMalloc mean ratio |")
    rows.append("|---|---|")
    ratios: Dict[str, float] = {}
    for k in kernels:
        ratio = means[f"{k}.vmm"] / means[f"{k}.cudaMalloc"]
        ratios[k] = ratio
        rows.append(f"| {k} | {ratio:.4f}x |")
    return "\n".join(rows), ratios


def verdict(ratios: Dict[str, float]) -> str:
    moe = ratios["fused_moe"]
    attn = ratios["flash_attn"]
    if moe > 1.04 and attn < 1.02:
        return "Hypothesis CONFIRMED"
    if moe <= attn:
        return "Hypothesis REFUTED"
    return "Hypothesis PARTIAL"


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 2
    torch.backends.cudnn.benchmark = True

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(f"== arena kernel isolation ==  cuda_visible={visible}  smoke={SMOKE}  "
          f"warmup={WARMUP_ITERS}  timed={TIMED_ITERS}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Qwen3.5-A3B config (mocked weights): {QWEN35_A3B}", flush=True)

    # fused_experts_impl reaches into get_global_server_args() to check
    # enable_deterministic_inference. Install a default ServerArgs so
    # that lookup works without needing to actually start a server.
    try:
        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    except Exception as e:
        logger.warning("failed to install global ServerArgs (fused_moe may fail): %s", e)

    # Touch CUDA so the runtime is initialized before MultiTensorArena
    # ctypes-loads its driver-API helper.
    torch.empty(1, device="cuda")
    torch.cuda.synchronize()

    setup_factory(device_id=torch.cuda.current_device())

    t0 = time.time()
    try:
        results = run_all()
    finally:
        teardown_factory()

    elapsed = time.time() - t0

    print("", flush=True)
    print(f"## Results  (wall: {elapsed:.1f} s)", flush=True)
    table, ratios = emit_table(results)
    print(table, flush=True)
    print("", flush=True)
    print(f"## Verdict\n{verdict(ratios)}", flush=True)

    # Save raw timings.
    raw = {
        "config": QWEN35_A3B,
        "batch_tokens": BATCH_TOKENS,
        "num_kv_tokens": NUM_KV_TOKENS,
        "page_size": PAGE_SIZE,
        "warmup_iters": WARMUP_ITERS,
        "timed_iters": TIMED_ITERS,
        "smoke": SMOKE,
        "samples_ms": {k: r.samples_ms for k, r in results.items()},
        "summary": {
            k: {"mean": r.mean, "p50": r.p50, "p99": r.p99}
            for k, r in results.items()
        },
        "ratios": ratios,
        "verdict": verdict(ratios),
    }
    out_path = "/tmp/arena_kernel_isolation.json"
    with open(out_path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\nraw timings -> {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
