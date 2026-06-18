"""#272 — mamba arena chunk-alignment validation.

`MambaPool.__init__` sizes the VMM arena from `tokens_per_chunk =
chunk_bytes // per_token_bytes`. The old code clamped this with
`max(1, ...)`, which (a) returned 1 when the per-token SSM state was LARGER
than one chunk (a config the arena cannot satisfy) and (b) floored a
non-dividing state — then sized the pool with that wrong value, so the
mismatch only surfaced later as a `MultiTensorArena` RuntimeError mid-boot.

`_arena_tokens_per_chunk` replaces the clamp: it enforces the SAME
`chunk_bytes % per_token_bytes == 0` invariant the arena enforces, at
pool-config time, with an actionable error (which multiple to set
`SGLANG_ARENA_CHUNK_BYTES` to). These tests pin both the happy path and the
fail-loud path.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

from sglang.srt.mem_cache.memory_pool import _arena_tokens_per_chunk  # noqa: E402

MiB = 1024 * 1024


def test_1_exact_divisor_returns_floor():
    # 2 MiB chunk, 2048-byte per-token state → 1024 slots/chunk, exact.
    assert _arena_tokens_per_chunk(2 * MiB, 2048) == 1024
    # per_token == chunk → exactly 1 slot/chunk.
    assert _arena_tokens_per_chunk(2 * MiB, 2 * MiB) == 1
    print("  PASS  1  exact divisor → chunk_bytes // per_token (1024, 1)")


def test_2_non_dividing_state_raises_with_suggestion():
    # 2 MiB chunk, 3000-byte state: 2097152 % 3000 = 152 ≠ 0.
    try:
        _arena_tokens_per_chunk(2 * MiB, 3000)
    except ValueError as e:
        msg = str(e)
        assert "multiple of" in msg and "3000" in msg, msg
        # suggested value is the smallest multiple of per_token ≥ chunk_bytes
        assert "2100000" in msg, f"expected rounded-up suggestion 2100000 in: {msg}"
        print("  PASS  2  non-dividing state → ValueError with multiple-of "
              "suggestion (2100000)")
        return
    raise AssertionError("non-dividing per-token state must raise (it floored "
                         "silently before #272)")


def test_3_state_larger_than_chunk_raises_not_clamp():
    # The #272 headline: per-token state (3 MiB) bigger than the chunk (2 MiB).
    # Old code returned max(1, 0) = 1 (wrong) and crashed later in the arena.
    try:
        _arena_tokens_per_chunk(2 * MiB, 3 * MiB)
    except ValueError as e:
        msg = str(e)
        # round-up suggestion is exactly one per-token state (3 MiB).
        assert str(3 * MiB) in msg, f"expected suggestion {3 * MiB} in: {msg}"
        print("  PASS  3  per-token state > chunk → ValueError (not the old "
              "silent max(1, 0)=1 clamp)")
        return
    raise AssertionError("per-token state larger than a chunk must raise, not "
                         "clamp to 1 slot/chunk (#272)")


def test_4_nonpositive_per_token_raises():
    for bad in (0, -1):
        try:
            _arena_tokens_per_chunk(2 * MiB, bad)
        except ValueError:
            continue
        raise AssertionError(f"per_token_bytes={bad} must raise")
    print("  PASS  4  non-positive per-token bytes → ValueError")


def test_5_matches_arena_constraint():
    # The helper accepts exactly the chunk sizes MultiTensorArena would accept
    # (chunk_bytes % per_token == 0) and rejects the rest — same invariant,
    # checked earlier. Spot-check a sweep.
    per_token = 4096
    for k in (1, 2, 7, 512, 1024):
        assert _arena_tokens_per_chunk(k * per_token, per_token) == k
    for bad in (per_token + 1, 3 * per_token - 1):
        try:
            _arena_tokens_per_chunk(bad, per_token)
        except ValueError:
            continue
        raise AssertionError(f"chunk_bytes={bad} (not a multiple of {per_token}) "
                             "must raise")
    print("  PASS  5  accepts exactly the arena-valid chunk sizes (multiples), "
          "rejects the rest")


if __name__ == "__main__":
    test_1_exact_divisor_returns_floor()
    test_2_non_dividing_state_raises_with_suggestion()
    test_3_state_larger_than_chunk_raises_not_clamp()
    test_4_nonpositive_per_token_raises()
    test_5_matches_arena_constraint()
    print("ALL PASS (#272 chunk-alignment)")
