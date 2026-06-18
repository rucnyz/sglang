"""Base allocator class for the token-to-KV pool allocators.

Carries HiMA's additions to the base class on top of upstream: _capped_pages,
_capped_lo, _set_capped_pages, live_size, _kv_grow_hook, _alloc_lock (all
needed for cross-pool cap management by the arena).
"""

from __future__ import annotations

import abc
import threading
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache

# Sentinel for `_capped_lo` when `_capped_pages` is empty: a value no page-id
# can reach. A fixed constant (not `size + 1`) so `__init__` need not read
# `self.size` — subclasses (SWA) expose `size` as a property whose backing is
# set only after `super().__init__()` returns.
_CAPPED_LO_EMPTY = 1 << 62


class BaseTokenToKVPoolAllocator(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
        max_size: Optional[int] = None,
    ):
        # Base ctor for the non-arena pools (Paged/SWA/HiSparse): they
        # pass `max_size is None`, so ceiling == live cap == size and
        # `_capped_pages` stays empty for the pool's lifetime (no cross-fire
        # grow/shrink). The dynamic-cap arena pool is the sibling
        # `TokenToKVPoolAllocator`, which bypasses this ctor and holds all
        # capped state in its own `CappedFreeList`.
        if max_size is None:
            max_size = size
        if max_size < size:
            raise ValueError(f"max_size={max_size} must be >= size={size}")
        self.max_size = max_size
        # Subclasses may override `size` as a read-only @property (e.g.
        # SWA returns min(_size_full, _size_swa)); in that case the
        # subclass owns the accessor and the base must not write to it.
        if not isinstance(getattr(type(self), "size", None), property):
            self.size = max_size  # page-id ceiling (= page space)
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache
        self.need_sort = need_sort

        self.free_pages = None
        self.release_pages = None
        # On-demand KV grow: an optional synchronous m2k grow callback the
        # BudgetAgent installs (grow KV from idle mamba). allocation.py's
        # alloc-fail path calls it before declaring OOM so the arena's live KV
        # cap grows to meet demand instead of crashing at the
        # budgeter-fire-paced cap. None when no budgeter / not arena-backed;
        # set unconditionally so the alloc-fail reader needs no fallback.
        self._kv_grow_hook = None
        self.is_not_in_free_group = True
        self.free_group = []
        # Held-out pages, always present so readers need no fallback. Empty
        # for the lifetime of a non-arena pool (no cross-fire cap). `live_size`
        # is `self.size - self._capped_pages.numel()`. `_capped_lo` mirrors
        # `min(_capped_pages)` as a plain Python int. `_set_capped_pages` is the
        # SINGLE writer and keeps `_capped_lo` in lockstep; empty → sentinel
        # `_CAPPED_LO_EMPTY`. Must not read `self.size` here — SWA's `size`
        # property backing isn't set until after `super().__init__()`.
        self._set_capped_pages(torch.empty(0, dtype=torch.int64, device=device))
        self._cap = size  # live admission cap (≤ max_size)
        # Protects free_pages / release_pages / _capped_pages against
        # concurrent mutation; held by alloc()/free() and inherited by every
        # subclass. Acquire is ~100 ns, negligible against alloc's per-call cost.
        self._alloc_lock = threading.Lock()

    @property
    def size_full(self):
        return self.size

    def debug_print(self) -> str:
        return ""

    def available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def get_kvcache(self):
        return self._kvcache

    @property
    def live_size(self) -> int:
        """Currently-active page capacity, read by sglang's pool-leak checker
        as `total`. For these non-arena pools `_capped_pages` is always empty,
        so this is just `size`; the arena pool overrides it to subtract its
        capped set. Defined here so the leak checker has a uniform accessor.
        """
        return self.size - int(self._capped_pages.numel())

    def _set_capped_pages(self, capped: torch.Tensor) -> None:
        """Single writer of `_capped_pages`, keeping `_capped_lo` (the min
        capped page-id, as a plain Python int) in lockstep. Empty → sentinel
        `_CAPPED_LO_EMPTY`. On these non-arena pools it is called once at boot
        with an empty tensor and the capped set stays empty for the pool's life.
        """
        self._capped_pages = capped
        self._capped_lo = int(capped.min()) if capped.numel() > 0 else _CAPPED_LO_EMPTY

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    def merge_and_sort_free(self):
        with self._alloc_lock:
            self._merge_and_sort_free_unlocked()

    def _merge_and_sort_free_unlocked(self):
        """Internal: caller must hold _alloc_lock. Used by alloc() which
        already acquired the lock — Python's threading.Lock is non-
        reentrant, so we can't recurse into the public method."""
        if len(self.release_pages) > 0:
            self.free_pages = torch.cat((self.free_pages, self.release_pages))
            self.free_pages, _ = torch.sort(self.free_pages)
            self.release_pages = torch.empty(
                (0,), dtype=self.release_pages.dtype, device=self.device
            )

    def get_cpu_copy(self, indices, mamba_indices=None):
        # FIXME: reuse the get_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        # FIXME: reuse the load_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def alloc_extend(self, *args, **kwargs):
        raise NotImplementedError("alloc_extend is only for paged allocator")

    def alloc_decode(self, *args, **kwargs):
        raise NotImplementedError("alloc_decode is only for paged allocator")

    def resize(self, config) -> None:
        self.size = config.max_total_num_tokens
        if self.page_size > 1:
            self.num_pages = config.max_total_num_tokens // self.page_size
        self.clear()

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def alloc(self, need_size: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def free(self, free_index: torch.Tensor):
        raise NotImplementedError()

    def free_segment(self, free_index: torch.Tensor, *, start_pos: int):
        """Free ``kv_row[start_pos : start_pos + n]`` of one request (or a
        page-aligned copy); subclasses may use ``start_pos`` to skip the
        data-dependent dedup. Default: plain free()."""
        self.free(free_index)

    def free_segments(self, segments):
        """Free disjoint ascending ``(free_index, start_pos)`` segments of one
        request's kv row; a boundary page shared by consecutive segments is
        emitted once (the later segment's head is trimmed)."""
        ps = self.page_size
        prev_end = None
        for free_index, start_pos in segments:
            n = free_index.numel()
            if n == 0:
                continue
            seg_end = start_pos + n
            if prev_end is not None and start_pos // ps == (prev_end - 1) // ps:
                boundary = (start_pos // ps + 1) * ps
                free_index = free_index[boundary - start_pos :]
                start_pos = boundary
            prev_end = seg_end
            self.free_segment(free_index, start_pos=start_pos)
