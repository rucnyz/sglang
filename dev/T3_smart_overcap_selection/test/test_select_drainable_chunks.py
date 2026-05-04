"""T3 unit test for _select_drainable_chunks: given a known free-mask,
verify the helper returns the expected chunk indices (highest-first,
all-pages-free).
"""

import sys
import torch
sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

from sglang.srt.arena.cross_pool_actuator import _select_drainable_chunks


class _FakeAllocator:
    """Minimal stand-in: only needs `size` and `free_page_mask()`."""
    def __init__(self, size: int, free_indices, device: str):
        self.size = size
        self.device = device
        self._free = torch.tensor(free_indices, dtype=torch.int64, device=device)

    def free_page_mask(self) -> torch.Tensor:
        m = torch.zeros(self.size + 1, dtype=torch.bool, device=self.device)
        m[self._free] = True
        return m


class _FakeSrcAct:
    def __init__(self, allocator):
        self.allocator = allocator


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tpc = 4   # tokens_per_chunk

    # Case A: pages 1..40 all free → 10 chunks all drainable.
    a = _FakeAllocator(40, list(range(1, 41)), device)
    chunks = _select_drainable_chunks(_FakeSrcAct(a), n_chunks=3, tokens_per_chunk=tpc)
    assert chunks == [9, 8, 7], f"all-free expected highest 3 chunks: got {chunks}"
    print(f"[case A] all-free → top3 chunks: {chunks}")

    # Case B: pages 1..32 + 37,38 free; pages 33,34,35,36,39,40 in use.
    # Chunks (4 pages each, 0-indexed): chunk0=[1..4], chunk1=[5..8], ..., chunk9=[37..40]
    # chunk8 = pages [33..36] → not all free
    # chunk9 = pages [37..40] → 37,38 free, 39,40 not free → not all free
    # chunks 0-7 fully free (pages 1..32 → 8 full chunks).
    a2 = _FakeAllocator(40, list(range(1, 33)) + [37, 38], device)
    chunks2 = _select_drainable_chunks(_FakeSrcAct(a2), n_chunks=5, tokens_per_chunk=tpc)
    assert chunks2 == [7, 6, 5, 4, 3], \
        f"chunks 0..7 fully free, top5: got {chunks2}"
    print(f"[case B] partial-free at tail → top5 fully-free chunks: {chunks2}")

    # Case C: nothing free → empty result.
    a3 = _FakeAllocator(40, [], device)
    chunks3 = _select_drainable_chunks(_FakeSrcAct(a3), n_chunks=3, tokens_per_chunk=tpc)
    assert chunks3 == [], f"nothing free expected empty: got {chunks3}"
    print(f"[case C] nothing free → {chunks3}")

    # Case D: only chunk 5 fully free; ask for 3 → return only [5].
    a4 = _FakeAllocator(40, [21, 22, 23, 24], device)  # chunk 5 = pages 21..24
    chunks4 = _select_drainable_chunks(_FakeSrcAct(a4), n_chunks=3, tokens_per_chunk=tpc)
    assert chunks4 == [5], f"only chunk5 free, asked 3 got {chunks4}"
    print(f"[case D] sparse free → {chunks4} (under-supplied is OK)")

    # Case E: src_act with no allocator attribute → empty (graceful).
    class NoAllocSrc: pass
    chunks5 = _select_drainable_chunks(NoAllocSrc(), n_chunks=3, tokens_per_chunk=tpc)
    assert chunks5 == [], f"no allocator → empty: got {chunks5}"
    print(f"[case E] no-allocator src_act → {chunks5}")

    print("\nT3 _select_drainable_chunks unit test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
