"""#271 step 3 — SchedulerStage0Handler.rewrite_kv_token_indices.

After `migrate_slot` relocates a KV token-slot's bytes src→dst, the owning
in-flight request's pointer must move too, or the next decode reads the
stale slot. For KV that pointer is `req_to_token_pool.req_to_token[
req_pool_idx, pos]` (the table the attention backend re-reads into
kv_indices every decode replay — see the spike's part B). This is the KV
analog of `rewrite_ssm_state_indices` (mamba's scalar per-req pointer).

A live KV slot is held at exactly one (req, pos). The rewrite is bounded to
the req's live length so a stale value lingering in the row's unused tail
can't be a false match. CPU-only (req_to_token is a plain int tensor).
"""
from __future__ import annotations

import sys
import types

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
import torch


def _handler(req_to_token, reqs):
    from sglang.srt.budgeter.scheduler_stage0_handler import SchedulerStage0Handler
    sched = types.SimpleNamespace(
        req_to_token_pool=types.SimpleNamespace(req_to_token=req_to_token),
        running_batch=types.SimpleNamespace(reqs=reqs),
    )
    return SchedulerStage0Handler(sched, kv_actuator=None, mamba_actuator=None)


def test_A_rewrites_the_owning_position():
    """src=101 is held by req(pool_idx=2) at pos 1 → rewrite to 200; other
    positions and other reqs untouched."""
    r2t = torch.zeros((4, 16), dtype=torch.int32)
    r2t[2, :3] = torch.tensor([100, 101, 102], dtype=torch.int32)
    r2t[1, :2] = torch.tensor([50, 51], dtype=torch.int32)
    reqs = [
        types.SimpleNamespace(req_pool_idx=1, seqlen=2, rid="r1"),
        types.SimpleNamespace(req_pool_idx=2, seqlen=3, rid="r2"),
    ]
    h = _handler(r2t, reqs)
    h.rewrite_kv_token_indices(101, 200)
    assert int(r2t[2, 1]) == 200, "owning position must be rewritten to dst"
    assert int(r2t[2, 0]) == 100 and int(r2t[2, 2]) == 102, "siblings intact"
    assert int(r2t[1, 0]) == 50 and int(r2t[1, 1]) == 51, "other req intact"
    print("  PASS  A  rewrite_kv_token_indices repoints the owning (req,pos)")


def test_B_stale_tail_is_not_a_false_match():
    """A freed slot id lingering in the row's UNUSED tail (beyond seqlen)
    must NOT be matched — only the live region [0, seqlen) is rewritten."""
    r2t = torch.zeros((4, 16), dtype=torch.int32)
    r2t[2, :2] = torch.tensor([100, 101], dtype=torch.int32)  # live: pos 0,1
    r2t[2, 5] = 300  # stale leftover in the unused tail
    reqs = [types.SimpleNamespace(req_pool_idx=2, seqlen=2, rid="r2")]
    h = _handler(r2t, reqs)
    raised = False
    try:
        h.rewrite_kv_token_indices(300, 999)  # 300 is only in the stale tail
    except RuntimeError:
        raised = True
    assert raised, "stale-tail-only slot must be treated as not-held (no live owner)"
    assert int(r2t[2, 5]) == 300, "the stale tail entry must NOT be rewritten"
    print("  PASS  B  stale tail (beyond seqlen) is not a false match")


def test_C_no_owner_raises():
    """No running req holds the slot → fail loud (Migration targets LIVE)."""
    r2t = torch.zeros((4, 16), dtype=torch.int32)
    r2t[2, :2] = torch.tensor([100, 101], dtype=torch.int32)
    reqs = [types.SimpleNamespace(req_pool_idx=2, seqlen=2, rid="r2")]
    h = _handler(r2t, reqs)
    raised = False
    try:
        h.rewrite_kv_token_indices(999, 200)
    except RuntimeError:
        raised = True
    assert raised, "no live owner of the migrated KV slot must raise"
    print("  PASS  C  no running owner → raises (LIVE-only contract)")


def test_D_stage0_kv_migration_end_to_end():
    """#271 steps 1-3 wired: XPoolActuator._run_stage0 with a kv_to_mamba
    plan carrying a migration must (a) relocate the slot via the KV
    actuator's migrate_slot (allocator byte-move + free/cap swap) AND
    (b) rewrite the owning req's req_to_token pointer — pool-agnostically,
    no NotImplementedError. CPU: a stub kvcache records the byte-move."""
    from sglang.srt.arena.kv_actuator import KVArenaActuator
    from sglang.srt.arena.xpool_actuator import XPoolActuator
    from sglang.srt.arena.fire_plan import FirePlan
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator

    class _StubKV:
        page_size = 1
        def can_move_kv_cache(self):
            return True
        def __init__(self):
            self.move_calls = []
        def get_kv_size_bytes(self):
            return 0
        def move_kv_cache(self, tgt, src):
            self.move_calls.append((tgt.tolist(), src.tolist()))

    kv = _StubKV()
    alloc = TokenToKVPoolAllocator(
        size=32, dtype=torch.float16, device="cpu", kvcache=kv, need_sort=False,
    )
    src, dst = 7, 20
    alloc.free_pages = alloc.free_pages[alloc.free_pages != src]  # src live
    # KVArenaActuator needs .pool + .allocator; only .allocator.migrate_slot
    # is exercised by the migration path.
    kv_act = KVArenaActuator.__new__(KVArenaActuator)
    kv_act.pool = kv
    kv_act.allocator = alloc

    # req_to_token holds src at (req_pool_idx=3, pos=2).
    r2t = torch.zeros((8, 16), dtype=torch.int32)
    r2t[3, :3] = torch.tensor([5, 6, src], dtype=torch.int32)
    reqs = [types.SimpleNamespace(req_pool_idx=3, seqlen=3, rid="r3")]
    handler = _handler(r2t, reqs)

    act = XPoolActuator.__new__(XPoolActuator)
    act.stage0_handler = handler
    plan = FirePlan(
        direction="kv_to_mamba", pages_to_unmap=[src],
        pages_to_map_dst=1, plan_seq=77, migrations=((src, dst),),
    )
    act._run_stage0(plan, kv_act)

    assert kv.move_calls == [([dst], [src])], (
        f"Stage-0 must relocate the KV slot via migrate_slot→move_kv_cache; "
        f"got {kv.move_calls}"
    )
    assert dst not in alloc.free_pages.tolist() and src in alloc.free_pages.tolist(), (
        "migrate_slot must swap dst→live / src→free"
    )
    assert int(r2t[3, 2]) == dst, "owning req's req_to_token pointer must move src→dst"
    print("  PASS  D  _run_stage0 KV migration: migrate_slot + req_to_token "
          "rewrite, pool-agnostic (no NotImplementedError)")


def test_E_missing_seqlen_fails_loud():
    """M1 (audit): rewrite_kv_token_indices must access Req.seqlen directly
    and fail loud — NOT fall back to searching the full row (the old
    getattr(...,None) fallback would reintroduce the stale-tail false match
    if seqlen were ever absent). Here a req lacks seqlen and a stale dup of
    the live slot sits in the tail; the rewrite must raise, not silently
    rewrite the stale tail position."""
    r2t = torch.zeros((4, 16), dtype=torch.int32)
    r2t[2, :2] = torch.tensor([100, 101], dtype=torch.int32)  # live
    r2t[2, 9] = 101  # stale dup in the unused tail
    req = types.SimpleNamespace(req_pool_idx=2, rid="r2")  # NO seqlen attr
    h = _handler(r2t, [req])
    raised = False
    try:
        h.rewrite_kv_token_indices(101, 200)
    except AttributeError:
        raised = True
    assert raised, (
        "a req missing seqlen must fail loud (no silent full-row fallback "
        "that could match the stale tail)"
    )
    print("  PASS  E  missing Req.seqlen fails loud (no full-row fallback)")


def test_F_stage0_migration_brackets_with_cuda_sync():
    """C1 (audit): _run_stage0 must bracket the migration byte-move with
    cuda.synchronize — before (drain in-flight readers of src) and after
    (relocated bytes + pointer rewrite visible before the next replay). The
    KV all-layer move races the live decode without it. Spy on synchronize;
    assert >= 2 calls for a 1-migration plan. (The full race repro is the
    GPU captured-graph test #291; this pins the contract.)"""
    from sglang.srt.arena.kv_actuator import KVArenaActuator
    from sglang.srt.arena.xpool_actuator import XPoolActuator
    from sglang.srt.arena.fire_plan import FirePlan
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator

    class _StubKV:
        page_size = 1
        def can_move_kv_cache(self):
            return True
        def get_kv_size_bytes(self):
            return 0
        def move_kv_cache(self, tgt, src):
            pass

    alloc = TokenToKVPoolAllocator(
        size=32, dtype=torch.float16, device="cpu",
        kvcache=_StubKV(), need_sort=False,
    )
    src, dst = 7, 20
    alloc.free_pages = alloc.free_pages[alloc.free_pages != src]
    kv_act = KVArenaActuator.__new__(KVArenaActuator)
    kv_act.pool, kv_act.allocator = _StubKV(), alloc
    r2t = torch.zeros((8, 16), dtype=torch.int32)
    r2t[3, :3] = torch.tensor([5, 6, src], dtype=torch.int32)
    reqs = [types.SimpleNamespace(req_pool_idx=3, seqlen=3, rid="r3")]
    handler = _handler(r2t, reqs)
    act = XPoolActuator.__new__(XPoolActuator)
    act.stage0_handler = handler
    plan = FirePlan(
        direction="kv_to_mamba", pages_to_unmap=[src],
        pages_to_map_dst=1, plan_seq=88, migrations=((src, dst),),
    )

    calls = []
    orig_sync = torch.cuda.synchronize
    orig_avail = torch.cuda.is_available
    torch.cuda.synchronize = lambda *a, **k: calls.append(1)
    torch.cuda.is_available = lambda: True
    try:
        act._run_stage0(plan, kv_act)
    finally:
        torch.cuda.synchronize = orig_sync
        torch.cuda.is_available = orig_avail
    assert len(calls) >= 2, (
        f"Stage-0 must cuda.synchronize before AND after the migration "
        f"byte-move; got {len(calls)} sync call(s)"
    )
    print("  PASS  F  _run_stage0 brackets the KV migration with cuda.sync")


def main() -> int:
    tests = [
        test_A_rewrites_the_owning_position,
        test_B_stale_tail_is_not_a_false_match,
        test_C_no_owner_raises,
        test_D_stage0_kv_migration_end_to_end,
        test_E_missing_seqlen_fails_loud,
        test_F_stage0_migration_brackets_with_cuda_sync,
    ]
    print(f"\n#271 step 3 KV pointer-rewrite tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {str(e)[:160]}")
            traceback.print_exc()
    print(f"\n#271 step 3: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
