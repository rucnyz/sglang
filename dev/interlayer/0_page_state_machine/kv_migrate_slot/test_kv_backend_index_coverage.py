"""#294 part (b) — per-backend KV-index-fill coverage guard (CPU).

The captured-graph replay safety property (a between-replay KV migration is
picked up because the decode index fill re-derives from req_to_token every
replay) was proven empirically for the index-fill KERNELS:
  - flashinfer: `create_flashinfer_kv_indices_triton`  (#291,
    test_kv_captured_replay.py)
  - FA3:        `normal_decode_set_metadata`           (#294b,
    test_kv_captured_replay_fa3.py)

The TRITON and AITER backends do not have their own fill kernel — they reuse
the SAME `create_flashinfer_kv_indices_triton` as flashinfer (verified by
identity below), so #291's kernel-level capture+replay proof transfers to
them unchanged. This guard makes that "covered by #291" claim SELF-CHECKING:
if any backend ever swaps in a different fill kernel, the identity assert
fails loudly here and a new empirical test (like the FA3 one) is owed before
that backend can be trusted for live KV migration.

(aiter is AMD/ROCm — its CUDA-graph machinery is not runnable on an NVIDIA
box, but the kernel-identity check is hardware-independent.)
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")


def test_backends_reuse_the_proven_index_kernel():
    from sglang.srt.layers.attention.utils import (
        create_flashinfer_kv_indices_triton as CANON,
    )

    import sglang.srt.layers.attention.flashinfer_backend as fb
    assert fb.create_flashinfer_kv_indices_triton is CANON, (
        "flashinfer backend no longer uses the kernel #291 captured — "
        "re-run/extend test_kv_captured_replay.py for the new fill"
    )

    import sglang.srt.layers.attention.triton_backend as tb
    assert tb.create_flashinfer_kv_indices_triton is CANON, (
        "triton backend no longer reuses create_flashinfer_kv_indices_triton — "
        "#291's proof no longer transfers; add an empirical capture+replay test "
        "for the triton fill"
    )
    print("  PASS  flashinfer + triton reuse the #291-proven index kernel")

    # aiter is AMD; importable here only if the aiter lib is present. When it
    # is, assert the same kernel identity (hardware-independent); else note it.
    try:
        import sglang.srt.layers.attention.aiter_backend as ab
    except Exception as e:  # pragma: no cover - depends on aiter availability
        print(f"  SKIP  aiter backend not importable here ({type(e).__name__}) "
              f"— AMD/ROCm; kernel-identity unchecked (code-read: reuses CANON)")
        return
    assert ab.create_flashinfer_kv_indices_triton is CANON, (
        "aiter backend no longer reuses create_flashinfer_kv_indices_triton — "
        "add an empirical test for its fill before trusting KV migration on AMD"
    )
    print("  PASS  aiter also reuses the #291-proven index kernel (AMD runtime "
          "untested on this box)")


def test_fa3_fill_symbol_exists():
    """FA3's distinct fill (covered empirically by test_kv_captured_replay_fa3)
    must still be the symbol that test exercises."""
    from sglang.srt.layers.attention.flashattention_backend import (
        normal_decode_set_metadata,
    )
    assert callable(normal_decode_set_metadata), (
        "FA3 normal_decode_set_metadata missing/renamed — update "
        "test_kv_captured_replay_fa3.py"
    )
    print("  PASS  FA3 normal_decode_set_metadata present (empirical: #294b)")


def main() -> int:
    tests = [
        test_backends_reuse_the_proven_index_kernel,
        test_fa3_fill_symbol_exists,
    ]
    print(f"\n#294(b) per-backend index-fill coverage guard (n={len(tests)}):")
    passed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {str(e)[:160]}")
            traceback.print_exc()
    print(f"\n#294(b) coverage guard: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
