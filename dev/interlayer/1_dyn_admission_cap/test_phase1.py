"""Phase 1 — ReqTokenVAArena unit tests.

Acceptance gate per `plan.md` Phase 1:
  1. construct + initial state
  2. grow + write
  3. grow again + preserve previous data
  4. data_ptr stable across grow (the WHOLE point)
  5. shrink preserves data in kept range
  6. cleanup unmaps + frees VA without faulting

Pure-Python, runs without sglang scheduler / pools. Uses cuda directly.
Run: .venv/bin/python dev/interlayer/1_dyn_admission_cap/test_phase1.py
"""
import sys

import torch

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
from sglang.srt.arena.req_token_arena import ReqTokenVAArena  # noqa: E402

DEVICE = 0
torch.cuda.set_device(DEVICE)

# 8 rows × 256 KiB per row = 2 MiB total. With chunk_bytes=2 MiB, that
# fits in 1 chunk → granularity is the whole tensor for these tests.
# To test multi-chunk grow, use chunk_bytes=1 MiB (still multiple of
# CUDA recommended granularity of 2 MiB? NO — see note below).
#
# NOTE: CUDA recommended granularity on H200 is 2 MiB. ChunkArena
# asserts chunk_bytes % granularity == 0, so we must use chunk_bytes
# that is a multiple of 2 MiB. For tests, use chunk_bytes=2 MiB and
# row_bytes that lets multiple chunks be useful: rows of 1 MiB each.
ROW_BYTES = 1 * 1024 * 1024  # 1 MiB per row
N_ROWS_MAX = 8               # 8 rows total = 8 MiB
CHUNK_BYTES = 2 * 1024 * 1024  # 2 MiB
ROWS_PER_CHUNK = CHUNK_BYTES // ROW_BYTES  # 2
N_CHUNKS_MAX = N_ROWS_MAX // ROWS_PER_CHUNK  # 4
MAX_BYTES = N_ROWS_MAX * ROW_BYTES

DTYPE = torch.int32
ELEMS_PER_ROW = ROW_BYTES // 4  # 262144 int32 per row
SHAPE = (N_ROWS_MAX, ELEMS_PER_ROW)


def make_arena():
    return ReqTokenVAArena(
        max_bytes=MAX_BYTES,
        device_id=DEVICE,
        chunk_bytes=CHUNK_BYTES,
    )


def test_1_construct():
    """Construct + initial state. data_ptr nonzero, mapped_bytes=0.

    NOTE: as_tensor requires at least 1 chunk mapped — PyTorch's
    `at::from_blob` runs `cudaPointerGetAttributes` which fails on an
    unmapped device VA. This is expected and matches the
    ReqToTokenPool usage pattern (we always boot with `init_size > 0`
    rows mapped).
    """
    a = make_arena()
    try:
        # Pre-construct state: VA reserved, no chunks mapped, no tensor yet.
        assert a.data_ptr() != 0, "data_ptr must be nonzero post-construction"
        assert a.mapped_bytes == 0, f"expected 0 mapped initially, got {a.mapped_bytes}"
        assert a.mapped_chunks == 0

        # Map 1 chunk so as_tensor can construct over the now-backed VA.
        a.set_mapped_bytes(CHUNK_BYTES)
        t = a.as_tensor(dtype=DTYPE, shape=SHAPE)
        assert a.data_ptr() == t.data_ptr(), \
            f"tensor data_ptr {t.data_ptr():#x} != arena va_base {a.data_ptr():#x}"
        assert t.shape == SHAPE
        assert t.dtype == DTYPE
        assert t.device.index == DEVICE
        print("  PASS  1  construct + initial state (data_ptr stable, "
              "as_tensor needs ≥1 chunk mapped)")
    finally:
        a.cleanup()


def test_2_grow_and_write():
    """Map 2 rows (1 chunk), write, read back exact bytes."""
    a = make_arena()
    try:
        a.set_mapped_bytes(2 * ROW_BYTES)  # 2 rows
        assert a.mapped_chunks == 1
        assert a.mapped_bytes == CHUNK_BYTES

        t = a.as_tensor(dtype=DTYPE, shape=SHAPE)
        # Write known pattern to rows 0 and 1
        pat = torch.arange(2 * ELEMS_PER_ROW, dtype=DTYPE, device=f"cuda:{DEVICE}")
        pat = pat.view(2, ELEMS_PER_ROW)
        t[:2] = pat
        torch.cuda.synchronize()
        out = t[:2].cpu()
        assert torch.equal(out, pat.cpu()), "rows 0:2 write/read mismatch"
        print("  PASS  2  grow + write/read 2 rows")
    finally:
        a.cleanup()


def test_3_grow_more_preserve_old():
    """Map 2 → 4 rows, verify rows 0:2 data preserved + rows 2:4 writable."""
    a = make_arena()
    try:
        a.set_mapped_bytes(2 * ROW_BYTES)
        t = a.as_tensor(dtype=DTYPE, shape=SHAPE)
        # Write pattern A to rows 0:2
        pat_a = torch.full((2, ELEMS_PER_ROW), 111, dtype=DTYPE, device=f"cuda:{DEVICE}")
        t[:2] = pat_a
        torch.cuda.synchronize()

        # Grow to 4 rows
        a.set_mapped_bytes(4 * ROW_BYTES)
        assert a.mapped_chunks == 2

        # Re-fetch tensor (same data_ptr expected, but be defensive)
        t2 = a.as_tensor(dtype=DTYPE, shape=SHAPE)
        assert torch.equal(t2[:2].cpu(), pat_a.cpu()), \
            "rows 0:2 data lost across grow"

        # Write rows 2:4 with pattern B
        pat_b = torch.full((2, ELEMS_PER_ROW), 222, dtype=DTYPE, device=f"cuda:{DEVICE}")
        t2[2:4] = pat_b
        torch.cuda.synchronize()
        assert torch.equal(t2[2:4].cpu(), pat_b.cpu()), \
            "rows 2:4 write/read mismatch"
        # rows 0:2 still good
        assert torch.equal(t2[:2].cpu(), pat_a.cpu())
        print("  PASS  3  grow more + preserve old data + new writable")
    finally:
        a.cleanup()


def test_4_data_ptr_stable_across_grow():
    """The WHOLE POINT: data_ptr never changes."""
    a = make_arena()
    try:
        a.set_mapped_bytes(2 * ROW_BYTES)
        t0 = a.as_tensor(dtype=DTYPE, shape=SHAPE)
        ptr0 = t0.data_ptr()
        arena_va0 = a.data_ptr()

        # Grow multiple times
        for n_rows in (4, 6, 8, 4, 2):
            a.set_mapped_bytes(n_rows * ROW_BYTES)
            t = a.as_tensor(dtype=DTYPE, shape=SHAPE)
            assert t.data_ptr() == ptr0, \
                f"data_ptr changed after grow→{n_rows}: " \
                f"{ptr0:#x} → {t.data_ptr():#x}"
            assert a.data_ptr() == arena_va0, \
                f"arena.data_ptr() changed: " \
                f"{arena_va0:#x} → {a.data_ptr():#x}"
        print("  PASS  4  data_ptr stable across multiple grow + shrink cycles")
    finally:
        a.cleanup()


def test_5_shrink_preserve_kept_range():
    """Map 4 rows, shrink to 2, verify rows 0:2 still intact."""
    a = make_arena()
    try:
        a.set_mapped_bytes(4 * ROW_BYTES)
        t = a.as_tensor(dtype=DTYPE, shape=SHAPE)
        # Write distinct patterns
        pat_a = torch.full((2, ELEMS_PER_ROW), 333, dtype=DTYPE, device=f"cuda:{DEVICE}")
        pat_b = torch.full((2, ELEMS_PER_ROW), 444, dtype=DTYPE, device=f"cuda:{DEVICE}")
        t[:2] = pat_a
        t[2:4] = pat_b
        torch.cuda.synchronize()

        # Shrink to 2 rows (drops rows 2:4)
        a.set_mapped_bytes(2 * ROW_BYTES)
        assert a.mapped_chunks == 1

        t2 = a.as_tensor(dtype=DTYPE, shape=SHAPE)
        assert torch.equal(t2[:2].cpu(), pat_a.cpu()), \
            "rows 0:2 data lost after shrink"
        # rows 2:4 are unmapped now — don't read them (would fault).

        # Re-grow + verify rows 2:4 are zeroed or at least writable
        a.set_mapped_bytes(4 * ROW_BYTES)
        t3 = a.as_tensor(dtype=DTYPE, shape=SHAPE)
        pat_c = torch.full((2, ELEMS_PER_ROW), 555, dtype=DTYPE, device=f"cuda:{DEVICE}")
        t3[2:4] = pat_c
        torch.cuda.synchronize()
        assert torch.equal(t3[2:4].cpu(), pat_c.cpu())
        # rows 0:2 still pat_a (never unmapped)
        assert torch.equal(t3[:2].cpu(), pat_a.cpu())
        print("  PASS  5  shrink preserves kept range; re-grow reusable")
    finally:
        a.cleanup()


def test_6_cleanup_no_fault():
    """cleanup() unmaps + frees VA; subsequent calls don't fault."""
    a = make_arena()
    a.set_mapped_bytes(4 * ROW_BYTES)
    t = a.as_tensor(dtype=DTYPE, shape=SHAPE)
    _ = t  # held during cleanup — tensor's no-op deleter means cleanup of
           # arena is what actually unmaps; tensor outliving arena is fine
           # IFF nobody reads from it afterwards.
    a.cleanup()
    # Construct + cleanup another instance to verify no global state corruption
    b = make_arena()
    b.set_mapped_bytes(2 * ROW_BYTES)
    b.cleanup()
    print("  PASS  6  cleanup unmaps + frees VA; second arena clean")


def main():
    tests = [test_1_construct, test_2_grow_and_write,
             test_3_grow_more_preserve_old, test_4_data_ptr_stable_across_grow,
             test_5_shrink_preserve_kept_range, test_6_cleanup_no_fault]
    print(f"\nReqTokenVAArena unit tests (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print(f"\nPhase 1: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
