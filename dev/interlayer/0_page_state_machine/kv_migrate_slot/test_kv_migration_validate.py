"""#271 HIGH-2 / M2 (audit) — _run_stage0 must validate-then-apply.

A migration mutates TWO places: the slot's bytes (`migrate_slot` →
`move_kv_cache` + free/cap swap) and the owning req's pointer
(`rewrite_kv_token_indices`). The bug: `_run_stage0` looped
migrate→rewrite per move, mutating as it went. If a LATER move's rewrite
raised (its src has no live owner) AFTER `migrate_slot` already freed that
src and moved its bytes, the owning req would read a freed slot and `dst`
would be live with no owner — and the raise is swallowed by `agent.tick`'s
`except Exception`, so the corruption silently survives. Earlier moves are
also left applied with no rollback.

Fix: validate EVERY source has a live owner (pure read) BEFORE relocating
any bytes, so a bad plan fails with ZERO mutations. This test builds a
2-move plan whose 2nd src has no running owner; pre-fix the 1st (and 2nd)
slot's bytes are moved before the raise (RED), post-fix nothing moves.
"""
from __future__ import annotations

import sys
import types

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
import torch


def _stub_kv():
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
    return _StubKV()


def test_bad_owner_aborts_with_zero_mutation():
    from sglang.srt.arena.kv_actuator import KVArenaActuator
    from sglang.srt.arena.xpool_actuator import XPoolActuator
    from sglang.srt.arena.fire_plan import FirePlan
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    from sglang.srt.budgeter.scheduler_stage0_handler import SchedulerStage0Handler

    kv = _stub_kv()
    alloc = TokenToKVPoolAllocator(
        size=32, dtype=torch.float16, device="cpu", kvcache=kv, need_sort=False,
    )
    good_src, dst0 = 7, 20   # good_src HAS a live owner
    bad_src, dst1 = 8, 21    # bad_src has NO live owner (the poison move)
    for s in (good_src, bad_src):
        alloc.free_pages = alloc.free_pages[alloc.free_pages != s]  # make live
    free_before = set(alloc.free_pages.tolist())

    kv_act = KVArenaActuator.__new__(KVArenaActuator)
    kv_act.pool, kv_act.allocator = kv, alloc

    # Only good_src is held by a running req; bad_src is owned by nobody.
    r2t = torch.zeros((8, 16), dtype=torch.int32)
    r2t[3, :2] = torch.tensor([5, good_src], dtype=torch.int32)
    reqs = [types.SimpleNamespace(req_pool_idx=3, seqlen=2, rid="r3")]
    sched = types.SimpleNamespace(
        req_to_token_pool=types.SimpleNamespace(req_to_token=r2t),
        running_batch=types.SimpleNamespace(reqs=reqs),
    )
    handler = SchedulerStage0Handler(sched, kv_actuator=None, mamba_actuator=None)
    act = XPoolActuator.__new__(XPoolActuator)
    act.stage0_handler = handler
    plan = FirePlan(
        direction="kv_to_mamba", pages_to_unmap=[good_src, bad_src],
        pages_to_map_dst=2, plan_seq=99,
        migrations=((good_src, dst0), (bad_src, dst1)),
    )

    raised = False
    try:
        act._run_stage0(plan, kv_act)
    except RuntimeError:
        raised = True
    assert raised, "a migration src with no live owner must abort the plan"
    # The load-bearing assertion: validate-then-apply means NO bytes moved.
    assert kv.move_calls == [], (
        f"validate-then-apply must relocate ZERO slots when the plan is "
        f"invalid; got move_kv_cache calls {kv.move_calls}"
    )
    assert set(alloc.free_pages.tolist()) == free_before, (
        "allocator free set must be unchanged after an aborted migration plan"
    )
    assert int(r2t[3, 1]) == good_src, "the good req's pointer must be untouched"
    print("  PASS  validate-then-apply: bad-owner plan aborts with zero mutation")


def main() -> int:
    tests = [test_bad_owner_aborts_with_zero_mutation]
    print(f"\n#271 HIGH-2 validate-then-apply tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {str(e)[:160]}")
            traceback.print_exc()
    print(f"\n#271 HIGH-2: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
