"""Dverify — chunk→slot translation correctness (verify-gap-1).

Locks in the e5f6d34421 cap-barrier off-by-one fix at:
  - python/sglang/srt/arena/kv_actuator.py
  - python/sglang/srt/arena/mamba_actuator.py

Original bug: `expand_pages_to_token_slots` returned
[p*tps+1, (p+1)*tps+1) — wrong. Chunk p physically backs slot range
[p*tps, (p+1)*tps); the +1 form (a) missed slot p*tps (leaked into
engine free list → kernel writes to unmapped VA → IMA), and
(b) wastefully held slot (p+1)*tps (which belongs to chunk p+1).

Post-#226: chunk 0 is no longer a valid argument — it carries
padded slot 0 (design.md §"Per-unit sizes") and unmapping it
would corrupt the padded-output target. `expand_pages_to_token_slots
([0])` now raises ValueError; `page_is_fully_free(0, ...)` always
returns False. Tests 1 and 4 were rewritten to assert the new
fail-loud contract; the `_compute_fully_free_pages` upstream filter
(#226) prevents page 0 from ever being a planner candidate.

Currently only byte_transfer e2e catches the chunk-N regression
(via cudaErrorIllegalAddress 6s post-fire); if byte_transfer's
workload drifts (no fires happen, or fires don't pick chunks with
id > 0), the bug slips. This unit pins it directly.
"""
from __future__ import annotations

import sys
import types

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

import torch


# Build a minimal stub MHATokenToKVPool with `_kv_arena.tokens_per_chunk`
# so KVArenaActuator's __init__ doesn't crash.
class _StubArena:
    def __init__(self, tps, max_chunks=64):
        self.tokens_per_chunk = tps
        # MambaArenaActuator (Phase 7 dyn_admission_cap) reads this to
        # cap admission growth at arena VA upper bound — stub must
        # match the real arena's API.
        self.max_chunks_per_pool = max_chunks


class _StubKVPool:
    """Minimal stand-in for MHATokenToKVPool — just enough for
    KVArenaActuator to read `pool._kv_arena.tokens_per_chunk` and
    `pool.size` / `pool.page_size`."""
    def __init__(self, size, tps):
        self.size = size
        self.page_size = 1
        self._kv_arena = _StubArena(tps)

    def set_capacity_tokens(self, n_tokens):
        pass


class _StubMambaPool:
    """Minimal MambaPool stand-in."""
    def __init__(self, size, tps):
        import torch
        self.size = size
        self.live_size = size  # MambaArenaActuator reads pool.live_size at init
        self._mamba_temporal_arena = _StubArena(tps)
        self._capped_slots = torch.empty(0, dtype=torch.int64, device="cpu")
        # _MambaCapAllocator reads pool.free_slots.device — give a real tensor.
        self.free_slots = torch.arange(1, size + 1, dtype=torch.int64, device="cpu")

    def set_capacity_slots(self, n_slots):
        pass


class _StubAlloc:
    def __init__(self, size):
        self.size = size
        self.device = "cpu"


def _build_kv_actuator(size=1024, tps=64):
    from sglang.srt.arena.kv_actuator import KVArenaActuator
    pool = _StubKVPool(size=size, tps=tps)
    alloc = _StubAlloc(size=size)
    return KVArenaActuator(pool=pool, allocator=alloc), tps


def _build_mamba_actuator(size=512, tps=8):
    from sglang.srt.arena.mamba_actuator import MambaArenaActuator
    pool = _StubMambaPool(size=size, tps=tps)
    return MambaArenaActuator(pool=pool), tps


# ---------- sub-tests ----------

def test_1_kv_chunk_0_rejected_loudly():
    """Post-#226: chunk 0 carries padded slot 0 (design.md §"Per-unit
    sizes"); unmapping it would corrupt the padded-output target.
    `expand_pages_to_token_slots` must raise ValueError if page 0 is
    in the list — the upstream `_compute_fully_free_pages` filter
    prevents it from being a planner candidate, so any page 0
    reaching expand is a structural break.

    Pre-#226 the function returned `[1, tps)` for page 0 — defensible
    in isolation but the actuator would still unmap chunk 0 (the
    pages_to_unmap list contained the page id, not the slot list).
    The new contract eliminates that gap by failing loudly.
    """
    act, _ = _build_kv_actuator()
    try:
        act.expand_pages_to_token_slots([0])
    except ValueError as e:
        msg = str(e)
        assert "page 0" in msg and "padded slot 0" in msg, (
            f"raise must cite page 0 + padded slot 0 diagnostic; got: {msg}"
        )
    else:
        raise AssertionError(
            "expand_pages_to_token_slots([0]) must raise ValueError "
            "post-#226; got silent return. Without the raise, an "
            "upstream filter regression could silently route chunk 0 "
            "to cuMemUnmap → cudaErrorIllegalAddress on next padded write."
        )


def test_2_kv_chunk_n_is_half_open_at_correct_bounds():
    """For chunk N > 0, expansion is [N*tps, (N+1)*tps) — half-open
    at the right bounds. Pre-fix returned [N*tps+1, (N+1)*tps+1) which
    LEAKED slot N*tps (head of chunk N stays in engine free list →
    next alloc hands it out → kernel writes to about-to-be-unmapped
    VA → cudaErrorIllegalAddress)."""
    act, tps = _build_kv_actuator()
    for n in (1, 5, 15):
        slots = act.expand_pages_to_token_slots([n])
        expected = list(range(n * tps, (n + 1) * tps))
        assert slots == expected, (
            f"chunk {n} expand mismatch:\n  got      "
            f"[{slots[0]}, ..., {slots[-1]}] (len {len(slots)})\n"
            f"  expected [{expected[0]}, ..., {expected[-1]}] "
            f"(len {len(expected)})\n"
            f"pre-fix returned [{n*tps+1}, ..., {(n+1)*tps}] — leaks "
            f"slot {n*tps} (head of chunk {n}) and wastes slot "
            f"{(n+1)*tps} (head of chunk {n+1})."
        )


def test_3_kv_page_is_fully_free_uses_same_bounds():
    """`page_is_fully_free` must check the same slot range as expand;
    otherwise FirePlanner sees a page as 'fully free' but cap_barrier
    targets different slots."""
    act, tps = _build_kv_actuator()
    # Build a free_token_set that contains exactly [N*tps, (N+1)*tps)
    # for N=3. page_is_fully_free(3, ...) MUST return True.
    n = 3
    free = set(range(n * tps, (n + 1) * tps))
    assert act.page_is_fully_free(n, free), (
        f"page_is_fully_free({n}, range({n*tps}, {(n+1)*tps})) returned "
        f"False — bounds mismatch with expand_pages_to_token_slots. "
        f"Pre-fix used [{n*tps+1}, {(n+1)*tps+1}); slot {n*tps} would "
        f"be marked missing from free even though chunk n is fully free."
    )
    # Negative: if we drop slot N*tps from the set, page is NOT fully
    # free anymore — must return False.
    free_minus_head = free - {n * tps}
    assert not act.page_is_fully_free(n, free_minus_head), (
        f"page_is_fully_free({n}) returned True even though slot "
        f"{n*tps} (head of chunk) is missing from free_token_set. "
        f"Bounds are wrong (probably starting at {n*tps+1})."
    )


def test_4_mamba_chunk_0_rejected_loudly_chunk_n_matches_kv():
    """Same fail-loud contract as KV applies to MambaArenaActuator
    for chunk 0 (#226). For chunks N > 0, the off-by-one fix is
    identical to the KV side.

    Why mamba matters specifically: with tps == 1 the pre-#226 code
    silently returned [] for page 0 (via `start = max(1, 0)` →
    `range(1, 1)`). The actuator would then proceed to unmap chunk
    0 WITHOUT capping any slot — strictly worse than the KV path
    which at least capped [1, tps).
    """
    act, tps = _build_mamba_actuator()
    # Chunk 0 — must raise.
    try:
        act.expand_pages_to_token_slots([0])
    except ValueError as e:
        assert "page 0" in str(e), (
            f"mamba diagnostic missing 'page 0': {e}"
        )
    else:
        raise AssertionError(
            "MambaArenaActuator.expand_pages_to_token_slots([0]) must "
            "raise — silent [] return on tps=1 was the worst-case "
            "manifestation of the original bug (#226)."
        )

    # Chunk 7 — half-open bounds unchanged from KV pattern.
    slots7 = act.expand_pages_to_token_slots([7])
    assert slots7 == list(range(7 * tps, 8 * tps)), (
        f"mamba chunk 7 expand mismatch: got {slots7}"
    )
    # page_is_fully_free behavior.
    free = set(range(7 * tps, 8 * tps))
    assert act.page_is_fully_free(7, free)
    assert not act.page_is_fully_free(7, free - {7 * tps})
    # And page 0 always False (#226), regardless of free_set.
    assert not act.page_is_fully_free(0, set(range(tps))), (
        "page_is_fully_free(0, ...) must be False unconditionally (#226)"
    )


def test_5_multi_page_expansion_concatenates_correctly():
    """expand_pages_to_token_slots([a, b]) should be expand([a]) +
    expand([b]) — no boundary cross-contamination between chunks."""
    act, tps = _build_kv_actuator()
    multi = act.expand_pages_to_token_slots([3, 5])
    expected = list(range(3 * tps, 4 * tps)) + list(range(5 * tps, 6 * tps))
    assert multi == expected, (
        f"multi-page expand mismatch: got len {len(multi)}, "
        f"expected len {len(expected)}. Slots could overlap or skip."
    )


# ---------- runner ----------

def main():
    tests = [
        ("1 KV chunk 0 rejected loudly (#226)",
         test_1_kv_chunk_0_rejected_loudly),
        ("2 KV chunk N is half-open [N*tps, (N+1)*tps)",
         test_2_kv_chunk_n_is_half_open_at_correct_bounds),
        ("3 KV page_is_fully_free uses same bounds as expand",
         test_3_kv_page_is_fully_free_uses_same_bounds),
        ("4 Mamba chunk 0 rejected loudly + chunk N matches KV (#226)",
         test_4_mamba_chunk_0_rejected_loudly_chunk_n_matches_kv),
        ("5 multi-page expansion concatenates correctly",
         test_5_multi_page_expansion_concatenates_correctly),
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:
            print(f"  FAIL  {name}")
            import traceback
            traceback.print_exc()
    print(f"\nDverify chunk-slot: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
