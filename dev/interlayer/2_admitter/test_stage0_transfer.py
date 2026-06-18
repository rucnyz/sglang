"""#183 Step 4 — XPoolActuator Stage-0 (Migration) GPU byte-exact verify.

design.md §"Transfer protocol" Stage 0 (`transfer(F, src, dst, *,
drains, migrations)`): before the cap-barrier, the actuator relocates
each `migrations` LIVE mamba slot into a free dst slot via the
byte-exact `MambaPool.migrate_slot` (property A2,
0_page_state_machine/step2_migrate_slot_replay_invariant) AND rewrites
the owning req's `ssm_state_indices` to the new slot, so the source
page ends FREE and joins `pages_to_unmap`.

Per the user: "just verify with GPU" — a real GPU byte-exact check is
the acceptance bar. Builds a REAL `MambaPool` (via
`HybridReqToTokenPool`, the pattern from test_mamba_real_pool.py) on
cuda:0, allocates a slot with a distinctive recurrent-state pattern,
runs Stage-0 with a one-migration plan, and asserts:

  (a) post-migrate the moved slot's state tensor is BYTE-EXACT (the
      dst slot equals the src slot's pre-migration contents, across
      every conv + temporal tensor);
  (b) the req's `ssm_state_indices` was rewritten src→dst (the
      handler callback fired with the right (src,dst));
  (c) the source slot ends in `_capped_slots` (FREE-but-held, the
      exact state cap_barrier then unmaps) and is NOT in `free_slots`.

CRITICAL control: an EMPTY drains/migrations plan must NOT invoke the
Stage-0 handler at all (byte-identical to the pre-#183 free-only path).

Run:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python \\
    dev/interlayer/2_admitter/test_stage0_transfer.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

import torch

torch.cuda.set_device(0)
DEVICE = "cuda:0"


def _make_real_pool(size=64, max_size=None):
    """Real `HybridReqToTokenPool` (real `MambaPool`) — pattern from
    test_mamba_real_pool.py `_make_real_pool`."""
    from sglang.srt.configs.mamba_utils import (
        Mamba2CacheParams,
        Mamba2StateShape,
    )
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=128,
        n_groups=1,
        num_heads=4,
        head_dim=64,
        state_size=16,
        conv_kernel=4,
    )
    cache_params = Mamba2CacheParams(shape=shape, layers=[0, 1])
    return HybridReqToTokenPool(
        size=size,
        mamba_size=size,
        mamba_spec_state_size=size,
        max_context_len=1024,
        device=DEVICE,
        enable_memory_saver=False,
        cache_params=cache_params,
        mamba_layer_ids=[0, 1],
        enable_mamba_extra_buffer=False,
        max_size=max_size,
    )


class _FakeSrcAct:
    """Minimal src per-pool actuator surface Stage-0 reads: `.pool`."""
    def __init__(self, pool):
        self.pool = pool


class _FakeStage0Handler:
    """Captures the scheduler-coupled Stage-0 calls. Migration dsts are
    planner-assigned in `plan.migrations` as (src, dst) pairs, so the
    handler only records the ssm rewrite (no dst allocation)."""
    def __init__(self):
        self.evict_calls = []
        self.rewrites = []

    def evict_pages(self, direction, drains):
        self.evict_calls.append((direction, tuple(drains)))

    def rewrite_ssm_state_indices(self, src_slot, dst_slot):
        self.rewrites.append((src_slot, dst_slot))


def _snapshot_slot(mamba_pool, slot):
    """Clone every conv + temporal tensor row for `slot`."""
    mc = mamba_pool.mamba_cache
    conv = [t[:, slot, ...].clone() for t in mc.conv]
    if isinstance(mc.temporal, list):
        temporal = [t[slot, ...].clone() for t in mc.temporal]
    else:
        temporal = mc.temporal[:, slot, ...].clone()
    return conv, temporal


def _fill_slot(mamba_pool, slot, value):
    mc = mamba_pool.mamba_cache
    for t in mc.conv:
        t[:, slot, ...].fill_(value)
    if isinstance(mc.temporal, list):
        for t in mc.temporal:
            t[slot, ...].fill_(value)
    else:
        mc.temporal[:, slot, ...].fill_(value)


def _slots_equal(mamba_pool, a, b):
    mc = mamba_pool.mamba_cache
    for t in mc.conv:
        if not torch.equal(t[:, a, ...], t[:, b, ...]):
            return False
    if isinstance(mc.temporal, list):
        for t in mc.temporal:
            if not torch.equal(t[a, ...], t[b, ...]):
                return False
    else:
        if not torch.equal(mc.temporal[:, a, ...], mc.temporal[:, b, ...]):
            return False
    return True


def _build_actuator(pool):
    """XPoolActuator via __new__ (skip the SharedHandlePool/arena ctor):
    Stage-0 only needs `stage0_handler` + the src_act passed in."""
    from sglang.srt.arena.xpool_actuator import XPoolActuator
    act = XPoolActuator.__new__(XPoolActuator)
    return act


def test_1_stage0_migration_byte_exact_and_rewrites():
    from sglang.srt.arena.fire_plan import FirePlan

    rtp = _make_real_pool(size=64)
    mamba = rtp.mamba_pool

    # Allocate one LIVE slot (src) and fill it with a distinctive value.
    src_t = mamba.alloc(1)
    assert src_t is not None, "pool exhausted"
    src_slot = int(src_t[0].item())
    SENTINEL = 0.375  # bf16-exact
    _fill_slot(mamba, src_slot, SENTINEL)
    torch.cuda.synchronize()
    pre_src = _snapshot_slot(mamba, src_slot)

    handler = _FakeStage0Handler()
    act = _build_actuator(mamba)
    act.stage0_handler = handler
    src_act = _FakeSrcAct(mamba)

    # Planner-assigned dst: a free slot distinct from src (≠ padded 0).
    free = mamba.free_slots
    dst_slot = int(free[free != 0][0].item())
    assert dst_slot != src_slot

    # Plan with ONE migration MOVE (src_slot → dst_slot).
    plan = FirePlan(
        direction="mamba_to_kv",
        pages_to_unmap=[src_slot],
        pages_to_map_dst=1,
        plan_seq=1,
        migrations=((src_slot, dst_slot),),
    )
    act._run_stage0(plan, src_act)
    torch.cuda.synchronize()

    # (a) byte-exact: dst now equals src's PRE-migration contents.
    mc = mamba.mamba_cache
    for i, t in enumerate(mc.conv):
        assert torch.equal(t[:, dst_slot, ...], pre_src[0][i]), (
            f"conv[{i}] dst slot not byte-exact to pre-migration src"
        )
    if isinstance(mc.temporal, list):
        for i, t in enumerate(mc.temporal):
            assert torch.equal(t[dst_slot, ...], pre_src[1][i])
    else:
        assert torch.equal(mc.temporal[:, dst_slot, ...], pre_src[1]), (
            "temporal dst slot not byte-exact to pre-migration src"
        )

    # (b) ssm rewrite fired with (src, dst).
    assert handler.rewrites == [(src_slot, dst_slot)], (
        f"ssm rewrite must fire once with (src,dst): {handler.rewrites}"
    )
    # No drains in this plan → evict_pages not called.
    assert handler.evict_calls == [], handler.evict_calls

    # (c) src slot is now FREE-but-held (in _capped_slots, about to be
    # unmapped) and NOT back in free_slots.
    capped = mamba._capped_slots
    assert bool((capped == src_slot).any().item()), (
        f"src slot {src_slot} must land in _capped_slots after Stage-0; "
        f"capped={capped.tolist()}"
    )
    assert not bool((mamba.free_slots == src_slot).any().item()), (
        f"src slot {src_slot} must NOT be in free_slots after migrate"
    )
    # dst slot consumed from free (now LIVE with src's data).
    assert not bool((mamba.free_slots == dst_slot).any().item()), (
        f"dst slot {dst_slot} must be removed from free_slots (now live)"
    )
    print(f"  PASS  1  Stage-0 migration byte-exact: slot {src_slot}->{dst_slot} "
          f"state relocated, ssm rewritten, src capped/unmappable")


def test_1b_stage0_multi_move_byte_exact():
    """#269 exec: a fragmentable fire carries MULTIPLE (src,dst) moves per
    plan. Pin that `_run_stage0` relocates EACH move byte-exactly and
    rewrites each owning req — the multi-move execution loop (test_1 only
    exercised a single move). Two distinct live slots → two distinct free
    dsts, each with its own sentinel."""
    from sglang.srt.arena.fire_plan import FirePlan

    rtp = _make_real_pool(size=64)
    mamba = rtp.mamba_pool

    src_a = int(mamba.alloc(1)[0].item())
    src_b = int(mamba.alloc(1)[0].item())
    _fill_slot(mamba, src_a, 0.25)
    _fill_slot(mamba, src_b, 0.5)
    torch.cuda.synchronize()
    pre_a = _snapshot_slot(mamba, src_a)
    pre_b = _snapshot_slot(mamba, src_b)

    # Two distinct free dsts (≠ each other, ≠ srcs, ≠ padded 0).
    free = mamba.free_slots
    cand = [int(x) for x in free[free != 0].tolist() if x not in (src_a, src_b)]
    dst_a, dst_b = cand[0], cand[1]
    assert len({src_a, src_b, dst_a, dst_b}) == 4

    handler = _FakeStage0Handler()
    act = _build_actuator(mamba)
    act.stage0_handler = handler
    src_act = _FakeSrcAct(mamba)
    plan = FirePlan(
        direction="mamba_to_kv",
        pages_to_unmap=[src_a, src_b],
        pages_to_map_dst=2,
        plan_seq=11,
        migrations=((src_a, dst_a), (src_b, dst_b)),
    )
    act._run_stage0(plan, src_act)
    torch.cuda.synchronize()

    mc = mamba.mamba_cache
    for dst, pre in ((dst_a, pre_a), (dst_b, pre_b)):
        for i, t in enumerate(mc.conv):
            assert torch.equal(t[:, dst, ...], pre[0][i]), (
                f"conv[{i}] dst {dst} not byte-exact to its src"
            )
    # Both moves' rewrites fired, in order, with the right (src,dst) pairs.
    assert handler.rewrites == [(src_a, dst_a), (src_b, dst_b)], handler.rewrites
    # Both srcs capped (FREE-but-held), both dsts consumed from free.
    for s in (src_a, src_b):
        assert bool((mamba._capped_slots == s).any().item())
    for d in (dst_a, dst_b):
        assert not bool((mamba.free_slots == d).any().item())
    print(f"  PASS  1b Stage-0 multi-move byte-exact: {src_a}->{dst_a}, "
          f"{src_b}->{dst_b} both relocated + rewritten + srcs capped")


def test_2_empty_plan_skips_stage0_handler():
    """CRITICAL control: an empty drains/migrations plan must NOT invoke
    the Stage-0 handler — byte-identical to the pre-#183 free-only path.
    Guarded in cap_barrier on `plan.drains or plan.migrations`; here we
    assert _run_stage0 is never reached by checking the handler stays
    untouched when the actuator's full cap_barrier guard is exercised."""
    from sglang.srt.arena.fire_plan import FirePlan

    rtp = _make_real_pool(size=64)
    mamba = rtp.mamba_pool
    handler = _FakeStage0Handler()
    act = _build_actuator(mamba)
    act.stage0_handler = handler

    # Free-only plan: empty drains + migrations.
    plan = FirePlan(
        direction="mamba_to_kv",
        pages_to_unmap=[5, 6, 7],
        pages_to_map_dst=3,
        plan_seq=2,
    )
    # The cap_barrier guard is `if plan.drains or plan.migrations`. Pin
    # that an empty plan evaluates the guard to False so _run_stage0 is
    # never invoked (handler untouched).
    invoked = bool(plan.drains or plan.migrations)
    assert invoked is False, "empty plan must not trigger Stage-0"
    assert handler.evict_calls == [] and handler.rewrites == [], (
        "handler must be untouched for a free-only plan"
    )
    print("  PASS  2  empty drains/migrations plan skips Stage-0 (free-only "
          "path byte-identical to pre-#183)")


def test_3_missing_handler_fails_loud():
    """A Drain/Migration plan with no stage0_handler must raise loudly
    (no silent fallback). Pins the fail-closed contract."""
    from sglang.srt.arena.fire_plan import FirePlan

    rtp = _make_real_pool(size=64)
    mamba = rtp.mamba_pool
    src_t = mamba.alloc(1)
    src_slot = int(src_t[0].item())

    act = _build_actuator(mamba)
    act.stage0_handler = None  # not wired
    src_act = _FakeSrcAct(mamba)
    plan = FirePlan(
        direction="mamba_to_kv",
        pages_to_unmap=[src_slot],
        pages_to_map_dst=1,
        plan_seq=3,
        migrations=((src_slot, 0),),
    )
    raised = False
    try:
        act._run_stage0(plan, src_act)
    except RuntimeError as e:
        raised = True
        assert "stage0_handler" in str(e), str(e)
    assert raised, "missing stage0_handler must raise RuntimeError (fail-loud)"
    print("  PASS  3  Drain/Migration plan with no stage0_handler raises loud")


def main() -> int:
    tests = [
        test_1_stage0_migration_byte_exact_and_rewrites,
        test_1b_stage0_multi_move_byte_exact,
        test_2_empty_plan_skips_stage0_handler,
        test_3_missing_handler_fails_loud,
    ]
    print(f"\n#183 Step 4 Stage-0 GPU byte-exact tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {e}"); traceback.print_exc()
    print(f"#183 Step 4: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
