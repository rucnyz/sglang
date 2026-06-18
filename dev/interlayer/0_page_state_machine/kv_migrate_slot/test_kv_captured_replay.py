"""#291 — the owed rigorous proof: a CAPTURED decode graph picks up a
between-replay KV-slot migration (move_kv_cache + req_to_token rewrite).

This is the KV analog of the mamba `step2_migrate_slot_replay_invariant`
spike, and it closes the gap the feasibility spike (`test_kv_migrate_replay`
part B) left explicitly owed: B ran the flashinfer index fill DIRECTLY
(outside any graph) and only proved it is data-driven. The load-bearing
safety property — that a captured decode graph, on REPLAY, re-derives
`kv_indices` from `req_to_token` (so indices are NOT baked-stale at capture,
and a between-replay `req_to_token` rewrite propagates to what attention
reads) — was code-read only. Here we PROVE it by capture+replay.

We capture a real CUDA graph over BOTH production steps the attention path
runs each replay:
  1. `create_flashinfer_kv_indices_triton(req_to_token → kv_indices)` — the
     real production index-fill kernel (what `init_forward_metadata_replay_
     cuda_graph → indices_updater_decode.update → call_begin_forward` runs),
  2. a GATHER that reads the KV bytes named by `kv_indices` (a faithful
     stand-in for attention's KV read) and reduces to a scalar.
`req_to_token` is the captured-input tensor we mutate IN PLACE between
replays — exactly as mamba mutates `ssm_state_indices`. Capturing the fill
INSIDE the graph proves the STRONGEST form (indices re-derive even if the
fill itself is captured); production runs the fill just outside the graph
into the same persistent `kv_indices` buffer, so it is a fortiori safe.

Protocol (mirrors the mamba spike):
  - baseline   : req_to_token[pos]=s            → out_s
  - neg-control: req_to_token[pos]=d (pre-move) → out_d, assert out_d != out_s
                 (the captured graph IS sensitive to the slot identity, and
                  the rewrite reached the gather — not a dead/baked index)
  - migrated   : move_kv_cache(d<-s); req[pos]=d → out_d2, assert out_d2==out_s
                 (d now holds s's bytes; replay reads the migrated slot)

Exit 0 = captured-graph replay equivalence CONFIRMED → live-KV migration is
empirically safe under flashinfer decode (the #271 step-5 enable gate's
remaining empirical blocker). Needs real CUDA.

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python \\
    dev/interlayer/0_page_state_machine/kv_migrate_slot/test_kv_captured_replay.py
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
        device=DEVICE, enable_memory_saver=False,
        enable_kv_cache_copy=True,  # initializes _kv_copy_config for move_kv_cache
    )


def main() -> int:
    from sglang.srt.layers.attention.flashinfer_backend import (
        create_flashinfer_kv_indices_triton,
    )
    print("\n=== #291: captured-graph KV-migration replay invariant ===\n")
    torch.manual_seed(0)
    pool = _make_kv_pool()
    layers = list(range(pool.start_layer, pool.start_layer + pool.layer_num))
    kbs = [pool.get_key_buffer(lid) for lid in layers]
    vbs = [pool.get_value_buffer(lid) for lid in layers]
    # Distinct, well-separated bytes per slot so the gather's scalar is
    # sensitive to exactly which slot kv_indices names.
    for buf in kbs + vbs:
        buf.zero_()
    for slot in range(pool.size + 1):
        for buf in kbs + vbs:
            buf[slot] = float(slot + 1) * 0.05
    torch.cuda.synchronize()

    bs, seq_len, max_ctx = 1, 5, 16
    s, pos, d = 10, 2, 20  # migrate the 3rd token's slot s -> d
    # Persistent (captured-input) metadata. req_to_token is the one we mutate.
    req_to_token = torch.zeros((bs, max_ctx), dtype=torch.int32, device=DEVICE)
    base = torch.tensor([3, 4, s, 5, 6], dtype=torch.int32, device=DEVICE)
    req_to_token[0, :seq_len] = base
    req_pool_indices = torch.zeros(bs, dtype=torch.int64, device=DEVICE)
    paged_kernel_lens = torch.tensor([seq_len], dtype=torch.int64, device=DEVICE)
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int64, device=DEVICE)
    kv_indptr[1:] = torch.cumsum(paged_kernel_lens, dim=0)
    kv_indices = torch.empty(seq_len, dtype=torch.int32, device=DEVICE)
    kvi64 = torch.empty(seq_len, dtype=torch.int64, device=DEVICE)
    ksel = torch.empty((seq_len,) + kbs[0].shape[1:], dtype=kbs[0].dtype, device=DEVICE)
    vsel = torch.empty_like(ksel)
    out_scalar = torch.zeros(1, dtype=torch.float32, device=DEVICE)

    def decode_forward():
        # (1) production index fill: req_to_token -> kv_indices.
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token, req_pool_indices, paged_kernel_lens,
            kv_indptr, None, kv_indices, max_ctx,
        )
        kvi64.copy_(kv_indices)
        # (2) gather the KV bytes named by kv_indices and reduce (attention
        #     KV-read stand-in). Reads the pool's persistent buffers; sensitive
        #     to which slot each index names.
        out_scalar.zero_()
        for kb, vb in zip(kbs, vbs):
            torch.index_select(kb, 0, kvi64, out=ksel)
            torch.index_select(vb, 0, kvi64, out=vsel)
            out_scalar.add_(ksel.float().sum())
            out_scalar.add_(vsel.float().sum())

    # Warmup (JIT the triton fill before capture), then capture.
    decode_forward()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream(device=DEVICE)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(g, stream=stream):
            decode_forward()
    torch.cuda.synchronize()
    print("[capture] graph over create_flashinfer_kv_indices_triton + KV gather")

    def replay():
        g.replay()
        torch.cuda.synchronize()
        return float(out_scalar.item())

    # --- baseline: req_to_token[pos] = s ---
    req_to_token[0, pos] = s
    out_s = replay()
    print(f"[baseline]  req_to_token[pos]={s}  -> {out_s:.4f}")

    # --- neg-control: point at d BEFORE migrating (d holds its own bytes) ---
    req_to_token[0, pos] = d
    out_d = replay()
    print(f"[neg-ctrl]  req_to_token[pos]={d}  -> {out_d:.4f} "
          f"(|Δ baseline|={abs(out_d - out_s):.4f})")
    if abs(out_d - out_s) < 1e-2:
        print("  ERROR: baseline vs unmigrated-dst indistinguishable — the "
              "captured graph isn't reading kv_indices (test can't discriminate).")
        return 3

    # --- migrate d<-s (the real primitive) + keep req_to_token[pos]=d ---
    pool.move_kv_cache(
        torch.tensor([d], dtype=torch.int64, device=DEVICE),
        torch.tensor([s], dtype=torch.int64, device=DEVICE),
    )
    torch.cuda.synchronize()
    out_d2 = replay()
    diff = abs(out_d2 - out_s)
    print(f"[migrated]  move_kv_cache(d<-s), req_to_token[pos]={d} -> "
          f"{out_d2:.4f} (|Δ baseline|={diff:.6f})")

    TOL = 1e-2  # fp16 sums over a few slots
    if diff > TOL:
        print(f"\n=== #291 REFUTED ===")
        print(f"After migrating slot bytes s->d and rewriting req_to_token, "
              f"the captured-graph replay diverges from baseline by {diff:.6f} "
              f"(> tol {TOL}). Captured kv_indices would be baked-stale — "
              f"live-KV migration is UNSAFE under this decode path.")
        return 1
    print(f"\n=== #291 CONFIRMED ===")
    print(f"Captured decode graph re-derives kv_indices from req_to_token on "
          f"replay: a between-replay KV migration (move_kv_cache + req_to_token "
          f"rewrite) is byte-transparent to the output (|Δ|={diff:.6f} <= "
          f"{TOL}). The #271 step-5 enable gate's empirical blocker is cleared "
          f"for flashinfer decode.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
