"""
Microbench: empirical c_KV vs c_M per-token recovery cost for Qwen3.5-35B-A3B.

Goal: validate the paper's c_M ~ 10-30x c_KV claim with direct kernel timings.

Recovery semantics:
  c_KV(L): wall-clock to re-derive L tokens of attention KV state
           = L tokens through ONE FA prefill kernel (varlen, causal),
             aggregated over the model's 10 full-attention layers.
  c_M(L):  wall-clock to re-derive L tokens of DeltaNet recurrent state
           starting from a saved snapshot
           = L tokens through ONE chunk_gated_delta_rule kernel,
             aggregated over the model's 30 linear-attention layers.

Both kernels run in prefill mode over a single sequence; we sweep L on a log grid
and report per-token wall-clock cost. The cost asymmetry in the paper is structural
(parallel matmul-bound attention vs sequential chunked scan), so we expect the
ratio to be roughly invariant across L (modulo small-L overhead and tile effects).

Output: dev/eval/cost_model/recovery_cost_<gpu>.json + a matplotlib PNG plot.
"""
import argparse
import json
import os
import sys
import time

import torch

# Qwen3.5-35B-A3B real config
N_LAYERS_TOTAL = 40
N_LAYERS_ATTN = 10
N_LAYERS_LINEAR = 30
HIDDEN = 2048
N_QHEADS = 16
N_KVHEADS = 2
HEAD_DIM = 256
LIN_NUM_QHEADS = 16
LIN_NUM_VHEADS = 32
LIN_KEY_DIM = 128
LIN_VALUE_DIM = 128
DTYPE = torch.bfloat16
DEVICE = "cuda"


def time_fn_cudagraph(fn, n_warmup=5, n_iter=20):
    """Time fn via repeated launches with cuda events; returns median ms."""
    # Warmup
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_iter):
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        fn()
        ev_end.record()
        torch.cuda.synchronize()
        times.append(ev_start.elapsed_time(ev_end))
    times.sort()
    return times[len(times) // 2], times[len(times) // 4], times[3 * len(times) // 4]


def bench_attn_prefill(L: int):
    """One full-attention layer's prefill kernel cost on L tokens (single sequence).

    Models the FA-prefill compute that re-derivation of an L-token KV miss pays
    *per attention layer*. Includes only the attention kernel itself (q, k, v are
    given); QKV proj / out proj / MoE are not the asymmetry target and are
    approximately equivalent on the linear path too.
    """
    from sgl_kernel.flash_attn import flash_attn_varlen_func

    q = torch.randn(L, N_QHEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    k = torch.randn(L, N_KVHEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    v = torch.randn(L, N_KVHEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

    cu_seqlens_q = torch.tensor([0, L], dtype=torch.int32, device=DEVICE)
    cu_seqlens_k = torch.tensor([0, L], dtype=torch.int32, device=DEVICE)

    def fn():
        flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=L,
            max_seqlen_k=L,
            causal=True,
        )

    return time_fn_cudagraph(fn)


def bench_gdn_prefill(L: int):
    """One DeltaNet layer's chunk_gated_delta_rule kernel cost on L tokens.

    Models the recurrent re-derivation cost that a snapshot miss pays *per linear
    layer*. The kernel processes L tokens with chunked parallel scan, but the
    chunk dimension has sequential data dependency.
    """
    from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule

    q = torch.randn(1, L, LIN_NUM_QHEADS, LIN_KEY_DIM, dtype=DTYPE, device=DEVICE)
    k = torch.randn(1, L, LIN_NUM_QHEADS, LIN_KEY_DIM, dtype=DTYPE, device=DEVICE)
    v = torch.randn(1, L, LIN_NUM_VHEADS, LIN_VALUE_DIM, dtype=DTYPE, device=DEVICE)
    g = torch.nn.functional.logsigmoid(
        torch.randn(1, L, LIN_NUM_VHEADS, dtype=DTYPE, device=DEVICE)
    )
    beta = torch.sigmoid(torch.randn(1, L, LIN_NUM_VHEADS, dtype=DTYPE, device=DEVICE))

    pool_size = 8
    pool = torch.randn(
        pool_size,
        LIN_NUM_VHEADS,
        LIN_KEY_DIM,
        LIN_VALUE_DIM,
        dtype=DTYPE,
        device=DEVICE,
    ) * 0.1
    # K-contiguous layout (matching SGLang's storage)
    pool = pool.transpose(-2, -1).contiguous().transpose(-2, -1)
    cache_indices = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    cu_seqlens = torch.tensor([0, L], dtype=torch.long, device=DEVICE)

    def fn():
        chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=pool,
            initial_state_indices=cache_indices,
            cu_seqlens=cu_seqlens,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )

    return time_fn_cudagraph(fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384],
    )
    parser.add_argument("--out-dir", default="dev/eval/cost_model")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    gpu_tag = name.replace(" ", "_").replace("/", "_")
    print(f"Device: {name} (SM {cap[0]}{cap[1]}); dtype={DTYPE}")
    print(f"Model: Qwen3.5-35B-A3B  (10 attn layers, 30 linear layers)")
    print()
    header = (
        f"{'L':>6}  "
        f"{'attn_us':>10}{'attn_us/tok':>14}  "
        f"{'gdn_us':>10}{'gdn_us/tok':>13}  "
        f"{'tot_KV_ms':>11}{'tot_M_ms':>11}{'ratio':>9}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for L in args.lengths:
        # one-layer microbench
        attn_med, attn_q1, attn_q3 = bench_attn_prefill(L)
        gdn_med, gdn_q1, gdn_q3 = bench_gdn_prefill(L)
        # per-layer in microseconds
        attn_us = attn_med * 1000.0
        gdn_us = gdn_med * 1000.0
        # per-token cost (microseconds)
        attn_us_per_tok = attn_us / L
        gdn_us_per_tok = gdn_us / L
        # full-stack recovery wall-clock for L tokens
        tot_kv_ms = attn_med * N_LAYERS_ATTN
        tot_m_ms = gdn_med * N_LAYERS_LINEAR
        ratio = tot_m_ms / tot_kv_ms if tot_kv_ms > 0 else float("nan")
        rows.append(
            dict(
                L=L,
                attn_us_med=attn_us,
                attn_us_q1=attn_q1 * 1000.0,
                attn_us_q3=attn_q3 * 1000.0,
                gdn_us_med=gdn_us,
                gdn_us_q1=gdn_q1 * 1000.0,
                gdn_us_q3=gdn_q3 * 1000.0,
                attn_us_per_tok=attn_us_per_tok,
                gdn_us_per_tok=gdn_us_per_tok,
                tot_kv_ms=tot_kv_ms,
                tot_m_ms=tot_m_ms,
                stack_ratio=ratio,
                n_layers_attn=N_LAYERS_ATTN,
                n_layers_linear=N_LAYERS_LINEAR,
            )
        )
        print(
            f"{L:>6}  "
            f"{attn_us:>10.1f}{attn_us_per_tok:>14.3f}  "
            f"{gdn_us:>10.1f}{gdn_us_per_tok:>13.3f}  "
            f"{tot_kv_ms:>11.3f}{tot_m_ms:>11.3f}{ratio:>9.2f}x"
        )

    out = dict(
        device=name,
        sm=f"{cap[0]}{cap[1]}",
        dtype=str(DTYPE),
        config=dict(
            n_layers_total=N_LAYERS_TOTAL,
            n_layers_attn=N_LAYERS_ATTN,
            n_layers_linear=N_LAYERS_LINEAR,
            hidden=HIDDEN,
            n_qheads=N_QHEADS,
            n_kvheads=N_KVHEADS,
            head_dim=HEAD_DIM,
            lin_num_qheads=LIN_NUM_QHEADS,
            lin_num_vheads=LIN_NUM_VHEADS,
            lin_key_dim=LIN_KEY_DIM,
            lin_value_dim=LIN_VALUE_DIM,
        ),
        rows=rows,
    )
    out_path = os.path.join(args.out_dir, f"recovery_cost_{gpu_tag}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
