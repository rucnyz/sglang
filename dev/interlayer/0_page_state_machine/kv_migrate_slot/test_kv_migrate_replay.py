"""#271 feasibility spike — KV-slot migration (Stage-3, src=kv).

The "genuinely-valuable migration": relocate a LIVE KV token-slot's data
(s→d, across ALL layers) and rewrite the owning request's pointer, so a
cross-pool k2m fire can consolidate KV free pages by moving live slots off a
to-be-transferred page. This is the KV analog of MambaPool.migrate_slot
(step2_migrate_slot_replay_invariant), and the two empirical unknowns it
de-risks BEFORE any planner/actuator wiring are:

  A. DATA PRIMITIVE — does the existing `MHATokenToKVPool.move_kv_cache`
     relocate a single slot s→d BYTE-EXACTLY across every layer's k AND v
     buffer? (A migrate_slot wrapper = move_kv_cache([d],[s]).)

  B. POINTER PROPAGATION — does rewriting `req_to_token[req,pos]` s→d
     actually reach the attention kernel? For flashinfer decode the captured
     graph re-runs `create_flashinfer_kv_indices_triton(req_to_token, ...)`
     into the captured kv-indices buffer EVERY replay (verified by code-read
     of init_forward_metadata_replay_cuda_graph → indices_updater_decode.
     update → call_begin_forward), so a req_to_token rewrite propagates —
     KV indices are NOT baked-stale at capture. This spike confirms the
     triton index-fill picks up the rewrite.

Exit 0 = the data primitive (A) is byte-exact and the index fill (B) is
data-driven → live-KV migration is LIKELY feasible with mamba-style sync.
The load-bearing captured-graph replay equivalence (the graph re-runs the
index fill on replay) is code-read only here; the rigorous capture+replay
test is owed (#291). Needs real CUDA.

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python \\
    dev/interlayer/0_page_state_machine/kv_migrate_slot/test_kv_migrate_replay.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
import torch

DEVICE = "cuda:0"


def _make_kv_pool(size=64, layer_num=3, head_num=4, head_dim=64):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    return MHATokenToKVPool(
        size=size, page_size=1, dtype=torch.float16,
        head_num=head_num, head_dim=head_dim, layer_num=layer_num,
        device=DEVICE, enable_memory_saver=False,
        enable_kv_cache_copy=True,   # initializes _kv_copy_config for move_kv_cache
    )


def test_A_move_kv_cache_relocates_slot_byte_exact():
    """A: move_kv_cache([d],[s]) copies slot s → d byte-exactly across ALL
    layers' k and v buffers. This is the migrate_slot data primitive."""
    pool = _make_kv_pool(size=64, layer_num=3)
    s, d = 10, 20
    torch.manual_seed(0)
    # Write distinct known KV at s and (different) at d, every layer.
    saved_k, saved_v = [], []
    for lid in range(pool.start_layer, pool.start_layer + pool.layer_num):
        kb = pool.get_key_buffer(lid)
        vb = pool.get_value_buffer(lid)
        kb[s] = torch.randn_like(kb[s])
        vb[s] = torch.randn_like(vb[s])
        kb[d] = torch.randn_like(kb[d])  # d starts DIFFERENT from s
        vb[d] = torch.randn_like(vb[d])
        saved_k.append(kb[s].clone())
        saved_v.append(vb[s].clone())
        assert not torch.equal(kb[d], kb[s]), "fixture: d must differ from s pre-move"

    pool.move_kv_cache(
        torch.tensor([d], dtype=torch.int64, device=DEVICE),
        torch.tensor([s], dtype=torch.int64, device=DEVICE),
    )
    torch.cuda.synchronize()

    for i, lid in enumerate(range(pool.start_layer, pool.start_layer + pool.layer_num)):
        kb = pool.get_key_buffer(lid)
        vb = pool.get_value_buffer(lid)
        assert torch.equal(kb[d], saved_k[i]), (
            f"layer {lid}: k_buffer[d] != relocated k_buffer[s] — move_kv_cache "
            f"is not byte-exact, so a KV migrate_slot would corrupt the slot"
        )
        assert torch.equal(vb[d], saved_v[i]), f"layer {lid}: v_buffer[d] mismatch"
        # src is left intact by the copy (the actuator caps/unmaps it after).
        assert torch.equal(kb[s], saved_k[i]), f"layer {lid}: src k clobbered"
    print("  PASS  A  move_kv_cache relocates slot s→d byte-exact across all "
          f"{pool.layer_num} layers (k+v)")


def test_B_req_to_token_rewrite_propagates_through_index_fill():
    """B (PARTIAL — data-driven fill, NOT a captured-graph replay): rewriting
    req_to_token[req,pos] s→d and RE-running create_flashinfer_kv_indices_
    triton reads the NEW slot d. This proves the index fill is data-driven —
    but it runs the fill directly, NOT through a captured+replayed CUDA graph,
    so it does not by itself prove the captured decode graph re-runs the fill
    on replay (that rests on the code-read of init_forward_metadata_replay_
    cuda_graph → indices_updater_decode.update → call_begin_forward). The
    rigorous capture+replay equivalence test is owed (#291)."""
    from sglang.srt.layers.attention.flashinfer_backend import (
        create_flashinfer_kv_indices_triton,
    )
    bs = 1
    seq_len = 5
    max_ctx = 16
    # req_to_token[req_pool_idx, pos] = kv_slot
    req_to_token = torch.zeros((bs, max_ctx), dtype=torch.int32, device=DEVICE)
    base = 100
    req_to_token[0, :seq_len] = torch.arange(base, base + seq_len,
                                             dtype=torch.int32, device=DEVICE)
    s, pos = base + 2, 2  # the slot we'll migrate (3rd token)
    d = 200

    def _fill():
        req_pool_indices = torch.tensor([0], dtype=torch.int64, device=DEVICE)
        paged_kernel_lens = torch.tensor([seq_len], dtype=torch.int64, device=DEVICE)
        kv_indptr = torch.zeros(bs + 1, dtype=torch.int64, device=DEVICE)
        kv_indptr[1:] = torch.cumsum(paged_kernel_lens, dim=0)
        kv_indices = torch.empty(seq_len, dtype=torch.int32, device=DEVICE)
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token, req_pool_indices, paged_kernel_lens,
            kv_indptr, None, kv_indices, req_to_token.shape[1],
        )
        torch.cuda.synchronize()
        return kv_indices

    before = _fill()
    assert int(before[pos].item()) == s, f"pre-rewrite slot should be {s}, got {before[pos]}"
    # The migrate's pointer rewrite (the KV analog of rewrite_ssm_state_indices).
    req_to_token[0, pos] = d
    after = _fill()
    assert int(after[pos].item()) == d, (
        f"req_to_token rewrite did NOT propagate: index-fill still reads {after[pos]} "
        f"not {d}. If true, captured-graph replay would read the stale slot and "
        f"live-KV migration would be unsafe."
    )
    assert s not in after.tolist(), "stale slot s must be gone from the refreshed indices"
    print("  PASS  B  req_to_token rewrite propagates through the (direct) "
          "flashinfer index fill — data-driven; captured-graph replay owed (#291)")


def main() -> int:
    tests = [
        test_A_move_kv_cache_relocates_slot_byte_exact,
        test_B_req_to_token_rewrite_propagates_through_index_fill,
    ]
    print(f"\n#271 KV-migrate feasibility spike (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc()
    verdict = (
        "data primitive OK + index-fill data-driven; captured-graph replay "
        "equivalence still owed (#291)"
        if passed == len(tests) else "CHECK FAILURES"
    )
    print(f"\n#271 spike: {passed}/{len(tests)} passed ({verdict})")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
