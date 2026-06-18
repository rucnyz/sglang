"""#294 part (b) — captured-graph KV-migration replay invariant, FA3 backend.

The flashinfer proof (#291, test_kv_captured_replay.py) captured
`create_flashinfer_kv_indices_triton`. The TRITON backend reuses that exact
kernel (so #291 covers it). The FA3 backend (FlashAttentionBackend) uses a
DIFFERENT fill on decode replay — `normal_decode_set_metadata` (the fused
`_fused_metadata_kernel_ps1_no_swa` triton kernel), which gathers
`page_indices = req_to_token[pool_idx, stride]` and writes
`metadata.page_table = page_indices // page_size`. This test proves the same
safety property for THAT kernel: a captured decode graph re-derives the page
table from `req_to_token` on every replay, so a between-replay KV-slot
migration (move_kv_cache + req_to_token rewrite) is byte-transparent.

We capture a real CUDA graph over the production `normal_decode_set_metadata`
(page_size=1, no-SWA fast path — the configuration the cross-pool migration
primitive requires) PLUS a KV gather over the resulting page_table, with
`req_to_token` the captured-input tensor mutated between replays. Protocol
mirrors #291: baseline (req→s) / neg-control (req→d pre-move, must differ) /
migrated (move_kv_cache(d←s)+req→d, must equal baseline).

Exit 0 = FA3 captured-graph replay equivalence CONFIRMED. Needs real CUDA.

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python \\
    dev/interlayer/0_page_state_machine/kv_migrate_slot/test_kv_captured_replay_fa3.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
import torch

DEVICE = "cuda:0"


def _make_kv_pool(size=64, layer_num=2, head_num=4, head_dim=64):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    return MHATokenToKVPool(
        size=size, page_size=1, dtype=torch.float16,
        head_num=head_num, head_dim=head_dim, layer_num=layer_num,
        device=DEVICE, enable_memory_saver=False, enable_kv_cache_copy=True,
    )


def main() -> int:
    from sglang.srt.layers.attention.flashattention_backend import (
        normal_decode_set_metadata,
    )
    print("\n=== #294(b): FA3 captured-graph KV-migration replay invariant ===\n")
    torch.manual_seed(0)
    pool = _make_kv_pool()
    layers = list(range(pool.start_layer, pool.start_layer + pool.layer_num))
    kbs = [pool.get_key_buffer(lid) for lid in layers]
    vbs = [pool.get_value_buffer(lid) for lid in layers]
    for buf in kbs + vbs:
        buf.zero_()
    for slot in range(pool.size + 1):
        for buf in kbs + vbs:
            buf[slot] = float(slot + 1) * 0.05
    torch.cuda.synchronize()

    bs, seq_len, max_ctx = 1, 5, 16
    s, pos, d = 10, 2, 20
    page_size = 1
    # Captured-input metadata (page_size=1 fast path of normal_decode_set_metadata).
    req_to_token = torch.zeros((bs, max_ctx), dtype=torch.int32, device=DEVICE)
    req_to_token[0, :seq_len] = torch.tensor([3, 4, s, 5, 6],
                                             dtype=torch.int32, device=DEVICE)
    req_pool_indices = torch.zeros(bs, dtype=torch.int64, device=DEVICE)
    seq_lens = torch.tensor([seq_len], dtype=torch.int64, device=DEVICE)
    strided_indices = torch.arange(max_ctx, dtype=torch.int64, device=DEVICE)
    # Buffers the fill WRITES (persistent; captured graph reads page_table).
    page_table = torch.zeros((bs, max_ctx), dtype=torch.int32, device=DEVICE)
    cache_seqlens_int32 = torch.zeros(bs, dtype=torch.int32, device=DEVICE)
    cu_seqlens_k = torch.zeros(bs + 1, dtype=torch.int32, device=DEVICE)
    max_seq_pages = (seq_len + page_size - 1) // page_size

    pt64 = torch.empty(seq_len, dtype=torch.int64, device=DEVICE)
    ksel = torch.empty((seq_len,) + kbs[0].shape[1:], dtype=kbs[0].dtype, device=DEVICE)
    vsel = torch.empty_like(ksel)
    out_scalar = torch.zeros(1, dtype=torch.float32, device=DEVICE)

    def decode_forward():
        # (1) production FA3 fill: req_to_token -> page_table (page_size=1).
        normal_decode_set_metadata(
            cache_seqlens_int32, cu_seqlens_k, page_table, req_to_token,
            req_pool_indices, strided_indices, max_seq_pages, seq_lens,
            0, page_size, None, None,
        )
        # (2) gather KV named by page_table[0, :seq_len] and reduce.
        pt64.copy_(page_table[0, :seq_len])
        out_scalar.zero_()
        for kb, vb in zip(kbs, vbs):
            torch.index_select(kb, 0, pt64, out=ksel)
            torch.index_select(vb, 0, pt64, out=vsel)
            out_scalar.add_(ksel.float().sum())
            out_scalar.add_(vsel.float().sum())

    decode_forward()  # warmup / JIT before capture
    torch.cuda.synchronize()
    stream = torch.cuda.Stream(device=DEVICE)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(g, stream=stream):
            decode_forward()
    torch.cuda.synchronize()
    print("[capture] graph over normal_decode_set_metadata (FA3) + KV gather")

    def replay():
        g.replay()
        torch.cuda.synchronize()
        return float(out_scalar.item())

    req_to_token[0, pos] = s
    out_s = replay()
    print(f"[baseline]  req_to_token[pos]={s}  -> {out_s:.4f}")

    req_to_token[0, pos] = d
    out_d = replay()
    print(f"[neg-ctrl]  req_to_token[pos]={d}  -> {out_d:.4f} "
          f"(|Δ baseline|={abs(out_d - out_s):.4f})")
    if abs(out_d - out_s) < 1e-2:
        print("  ERROR: baseline vs unmigrated-dst indistinguishable — the "
              "captured graph isn't reading page_table (test can't discriminate).")
        return 3

    pool.move_kv_cache(
        torch.tensor([d], dtype=torch.int64, device=DEVICE),
        torch.tensor([s], dtype=torch.int64, device=DEVICE),
    )
    torch.cuda.synchronize()
    out_d2 = replay()
    diff = abs(out_d2 - out_s)
    print(f"[migrated]  move_kv_cache(d<-s), req_to_token[pos]={d} -> "
          f"{out_d2:.4f} (|Δ baseline|={diff:.6f})")

    TOL = 1e-2
    if diff > TOL:
        print(f"\n=== #294(b) FA3 REFUTED ===")
        print(f"FA3 captured-graph replay diverges from baseline by {diff:.6f} "
              f"(> tol {TOL}) after migration — page_table would be baked-stale.")
        return 1
    print(f"\n=== #294(b) FA3 CONFIRMED ===")
    print(f"FA3's normal_decode_set_metadata re-derives page_table from "
          f"req_to_token on replay: a between-replay KV migration is "
          f"byte-transparent (|Δ|={diff:.6f} <= {TOL}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
