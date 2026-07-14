"""HiMA must fail fast at boot when the tree cache is not a MambaRadixCache.

Reproduces the KimiLinearForCausalLM crash: sglang disables radix caching for
models whose MambaRadixCache support is not yet implemented
(support_mamba_cache=False), so the tree cache is a ChunkCache. HiMA's Budgeter
snapshot then reads MambaRadixCache-only eviction counters
(`_cumulative_evicted_mamba_slots`) and the Admitter reads `owner_provider`
from the arena, neither of which exists on a ChunkCache -> the scheduler
crashed mid-run with an obscure AttributeError / "owner_provider not wired".
The fix is a boot-time guard (`require_mamba_radix_cache_for_hima`) that raises
a clear RuntimeError instead. isinstance covers the HiMambaRadixCache subclass.
"""

import pytest

from sglang.srt.managers.scheduler import require_mamba_radix_cache_for_hima
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache


def test_chunk_cache_is_rejected():
    # ChunkCache = radix disabled (KimiLinear). __new__ skips the heavy
    # pool-backed __init__; the guard only inspects the type.
    chunk = ChunkCache.__new__(ChunkCache)
    with pytest.raises(RuntimeError, match="requires a MambaRadixCache"):
        require_mamba_radix_cache_for_hima(chunk, disable_radix_cache=True)


def test_mamba_radix_cache_is_accepted():
    mamba = MambaRadixCache.__new__(MambaRadixCache)
    # No raise -> HiMA can attach its Budgeter/Admitter to this cache.
    require_mamba_radix_cache_for_hima(mamba, disable_radix_cache=False)


def test_himamba_subclass_is_accepted():
    # HiMambaRadixCache (lpb eviction) is a MambaRadixCache subclass; the
    # isinstance guard (not exact-type) must accept it.
    from sglang.srt.mem_cache.hi_mamba_radix_cache import HiMambaRadixCache

    hi = HiMambaRadixCache.__new__(HiMambaRadixCache)
    require_mamba_radix_cache_for_hima(hi, disable_radix_cache=False)


def test_error_names_the_actual_cache_type():
    chunk = ChunkCache.__new__(ChunkCache)
    with pytest.raises(RuntimeError) as ei:
        require_mamba_radix_cache_for_hima(chunk, disable_radix_cache=True)
    msg = str(ei.value)
    assert "ChunkCache" in msg
    assert "disable_radix_cache=True" in msg


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
