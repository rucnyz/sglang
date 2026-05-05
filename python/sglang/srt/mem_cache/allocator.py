from __future__ import annotations

"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Page-aligned memory pool.
"""

import abc
import logging
from typing import TYPE_CHECKING

import torch

logger = logging.getLogger(__name__)
import triton
import triton.language as tl

from sglang.srt.utils import get_bool_env_var, get_num_new_pages, next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


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
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache
        self.need_sort = need_sort

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []

    def debug_print(self) -> str:
        return ""

    def available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def get_kvcache(self):
        return self._kvcache

    def restore_state(self, state):
        self.free_pages, self.release_pages = state

    def backup_state(self):
        return (self.free_pages, self.release_pages)

    @property
    def live_size(self) -> int:
        """Currently-active page capacity (= size unless capped by budgeter)."""
        cap = getattr(self, "_cap", None)
        return cap if cap is not None else self.size

    def free_page_mask(self) -> torch.Tensor:
        """T3 (paper §3.2.2): return a boolean tensor of length `self.size + 1`
        with True at indices currently free (in `free_pages`, not held by
        any in-flight request, not in `release_pages`). Index 0 is the
        reserved-sentinel slot and is always False.

        Used by the actuator to pick which slots to unmap when shrinking
        the pool — replaces the assumption that the highest-VA slots are
        free with an explicit query of allocator state.

        Note: page indices in this allocator run 1..size (slot 0 is a
        dummy sentinel for padded-output writes); the mask matches that
        convention.
        """
        mask = torch.zeros(self.size + 1, dtype=torch.bool, device=self.device)
        if self.free_pages is not None and self.free_pages.numel() > 0:
            mask[self.free_pages] = True
        # release_pages also count as free for drain purposes — they
        # are released-but-not-yet-merged, no live req holds them.
        if self.release_pages is not None and self.release_pages.numel() > 0:
            mask[self.release_pages] = True
        return mask

    def mark_pages_capped(self, page_indices: torch.Tensor) -> int:
        """T3 (paper §3.2.2): hold the given page indices out of the
        free-list. Used after `shrink_explicit` unmaps non-tail chunks
        so the allocator stops handing those pages out.

        Removes `page_indices` from `free_pages` and `release_pages`
        (whichever list each occupies), and appends them to
        `_capped_pages`. Returns the number of pages actually moved
        (may be < len(page_indices) if some indices weren't currently
        in either free list).

        Pages re-enter the free list via the symmetric `unmark_pages_capped`
        when the actuator grows the pool back.
        """
        if page_indices is None or page_indices.numel() == 0:
            return 0
        # Build sets for fast membership.
        target = page_indices.to(self.device).to(torch.int64)
        # Drop matching ids out of free_pages.
        moved = 0
        if self.free_pages is not None and self.free_pages.numel() > 0:
            mask = torch.isin(self.free_pages, target)
            held = self.free_pages[mask]
            self.free_pages = self.free_pages[~mask]
            moved += int(held.numel())
            existing = getattr(self, "_capped_pages", None)
            if existing is None or existing.numel() == 0:
                self._capped_pages = held
            else:
                self._capped_pages = torch.cat([existing, held])
        # Drop matching ids out of release_pages.
        if self.release_pages is not None and self.release_pages.numel() > 0:
            mask = torch.isin(self.release_pages, target)
            held = self.release_pages[mask]
            self.release_pages = self.release_pages[~mask]
            moved += int(held.numel())
            existing = getattr(self, "_capped_pages", None)
            if existing is None or existing.numel() == 0:
                self._capped_pages = held
            else:
                self._capped_pages = torch.cat([existing, held])
        return moved

    def unmark_pages_capped(self, page_indices: torch.Tensor) -> int:
        """Reverse of `mark_pages_capped`: move given page ids from
        `_capped_pages` back to `free_pages` (or `release_pages` under
        need_sort=True so a subsequent merge sorts them).
        """
        if page_indices is None or page_indices.numel() == 0:
            return 0
        existing = getattr(self, "_capped_pages", None)
        if existing is None or existing.numel() == 0:
            return 0
        target = page_indices.to(self.device).to(torch.int64)
        mask = torch.isin(existing, target)
        held = existing[mask]
        self._capped_pages = existing[~mask]
        if held.numel() > 0:
            if self.need_sort:
                if self.release_pages is None or self.release_pages.numel() == 0:
                    self.release_pages = held
                else:
                    self.release_pages = torch.cat([self.release_pages, held])
            else:
                if self.free_pages is None or self.free_pages.numel() == 0:
                    self.free_pages = held
                else:
                    self.free_pages = torch.cat([self.free_pages, held])
        return int(held.numel())

    def select_drain_pages(self, n: int, prefer: str = "high") -> torch.Tensor:
        """T3 (paper §3.2.2): return up to `n` page indices that are
        currently free, suitable for cuMemUnmap.

        `prefer="high"`: pick the highest-indexed free pages — under T2's
        placement bias (live blocks at low indices), the high tail is
        most likely contiguously free, letting the actuator batch the
        unmap into a single VA range.

        `prefer="low"`: opposite preference; useful only for tests.

        Returns a 1-D int64 tensor with length min(n, available_free).
        Caller is responsible for verifying drain readiness for each
        returned index (via in-flight slot inspector); this function
        only checks allocator-side state.
        """
        if n <= 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        candidates = []
        if self.free_pages is not None and self.free_pages.numel() > 0:
            candidates.append(self.free_pages)
        if self.release_pages is not None and self.release_pages.numel() > 0:
            candidates.append(self.release_pages)
        if not candidates:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        all_free = torch.cat(candidates)
        sorted_free, _ = torch.sort(all_free)
        if prefer == "high":
            return sorted_free[-n:].to(torch.int64) if sorted_free.numel() > n \
                else sorted_free.to(torch.int64)
        elif prefer == "low":
            return sorted_free[:n].to(torch.int64)
        else:
            raise ValueError(f"unknown prefer={prefer!r}")

    def set_capacity_pages(self, n_pages: int) -> None:
        """Restrict the allocator to only hand out pages with id <= n_pages.

        Maintains a `_capped_pages` tensor of held-out indices so that a
        later grow can restore them. Initial cap = self.size (full
        capacity, _capped_pages empty).

        Does NOT verify that no in-flight allocation references id >
        n_pages — the caller (typically the budgeter) is responsible
        for that.
        """
        cap = getattr(self, "_cap", None)
        if cap is None:
            cap = self.size
        if n_pages == cap:
            return
        logger.info(
            "Allocator.set_capacity_pages: %d -> %d (size=%d, free=%d, capped=%d)",
            cap, n_pages, self.size,
            len(self.free_pages),
            getattr(self, "_capped_pages", torch.empty(0)).numel(),
        )
        if n_pages < cap:
            # Shrink: move free pages with id > n_pages out to _capped.
            mask = self.free_pages > n_pages
            held_now = self.free_pages[mask]
            self.free_pages = self.free_pages[~mask]
            existing = getattr(self, "_capped_pages", None)
            if existing is None or existing.numel() == 0:
                self._capped_pages = held_now
            else:
                self._capped_pages = torch.cat([existing, held_now])
        else:
            # Grow: move _capped pages with id <= n_pages back to free.
            held = getattr(self, "_capped_pages", None)
            if held is None or held.numel() == 0:
                pass
            else:
                mask = held <= n_pages
                move = held[mask]
                self.free_pages = torch.cat([self.free_pages, move])
                self._capped_pages = held[~mask]
        self._cap = n_pages

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    def merge_and_sort_free(self):
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

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def alloc(self, need_size: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def free(self, free_index: torch.Tensor):
        raise NotImplementedError()


class TokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """An allocator managing the indices to kv cache data."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        super().__init__(size, 1, dtype, device, kvcache, need_sort)
        self.clear()

    def clear(self):
        # Respect the budgeter's capacity cap if set. Without this,
        # /flush_cache (or scheduler.flush_cache) would reinstate ALL
        # pages 1..size into free_pages, including ones whose underlying
        # chunks have been unmapped by the cross-pool actuator. Subsequent
        # leak check trips because available > live.
        cap = getattr(self, "_cap", None)
        upper = cap if cap is not None else self.size
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, upper + 1, dtype=torch.int64, device=self.device
        )
        # Pages above the cap (id in (upper, self.size]) belong in
        # _capped_pages so a subsequent grow can restore them.
        if cap is not None and cap < self.size:
            self._capped_pages = torch.arange(
                cap + 1, self.size + 1, dtype=torch.int64, device=self.device
            )
        else:
            self._capped_pages = torch.empty((0,), dtype=torch.int64, device=self.device)
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)

    def available_size(self):
        # To avoid minor "len(free_pages) * 1" overhead
        return len(self.free_pages) + len(self.release_pages)

    def alloc(self, need_size: int):
        if self.need_sort and need_size > len(self.free_pages):
            self.merge_and_sort_free()

        if need_size > len(self.free_pages):
            return None

        select_index = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        # Paper §sec:design-l2 drain protocol: a freed page whose id is above
        # the current live cap belongs in _capped_pages, not back in
        # free_pages. The actuator may be in the middle of draining +
        # unmapping the tail; if we put the id back in free_pages, the
        # next alloc would hand it out and the next decode would touch
        # an unmapped chunk → cudaErrorIllegalAddress. Mirrors the cap-
        # aware free path on MambaPool.
        cap = getattr(self, "_cap", None)
        if cap is not None and self.is_not_in_free_group:
            mask_above = free_index > cap
            if mask_above.any():
                held_now = free_index[mask_above]
                kept = free_index[~mask_above]
                existing = getattr(self, "_capped_pages", None)
                if existing is None or existing.numel() == 0:
                    self._capped_pages = held_now
                else:
                    self._capped_pages = torch.cat([existing, held_now])
                if kept.numel() > 0:
                    if self.need_sort:
                        self.release_pages = torch.cat((self.release_pages, kept))
                    else:
                        self.free_pages = torch.cat((self.free_pages, kept))
                return

        if self.is_not_in_free_group:
            if self.need_sort:
                self.release_pages = torch.cat((self.release_pages, free_index))
            else:
                self.free_pages = torch.cat((self.free_pages, free_index))
        else:
            self.free_group.append(free_index)

    def get_cpu_copy(self, indices, mamba_indices=None):
        return self._kvcache.get_cpu_copy(indices, mamba_indices=mamba_indices)

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        return self._kvcache.load_cpu_copy(
            kv_cache_cpu, indices, mamba_indices=mamba_indices
        )


def alloc_extend_naive(
    prefix_lens,
    seq_lens,
    last_loc,
    free_pages,
    out_indices,
    page_size,
    device,
):
    extend_lens = seq_lens - prefix_lens
    end_pos = torch.cumsum(extend_lens, 0)
    start_pos = end_pos - extend_lens
    num_new_pages = (seq_lens + page_size - 1) // page_size - (
        prefix_lens + page_size - 1
    ) // page_size
    num_full_new_pages = (seq_lens) // page_size - (
        prefix_lens + page_size - 1
    ) // page_size
    need_page = num_new_pages - num_full_new_pages
    end_new_pages = torch.cumsum(num_new_pages, 0)
    start_new_pages = end_new_pages - num_new_pages
    pos_in_page = torch.arange(page_size, device=device, dtype=torch.int32)
    for i in range(len(prefix_lens)):
        num1 = (
            min(
                seq_lens[i],
                (prefix_lens[i] + page_size - 1) // page_size * page_size,
            )
            - prefix_lens[i]
        )
        if num1:
            out_indices[start_pos[i] : start_pos[i] + num1] = (
                last_loc[i] + 1 + pos_in_page[:num1].view(-1)
            )

        if prefix_lens[i] + num1 == seq_lens[i]:
            continue

        num2 = (
            seq_lens[i] // page_size - (prefix_lens[i] + page_size - 1) // page_size
        ) * page_size
        if num2:
            pages = (
                free_pages[start_new_pages[i] : end_new_pages[i] - need_page[i]]
                * page_size
            )
            out_indices[start_pos[i] + num1 : start_pos[i] + num1 + num2] = (
                pages.view(-1, 1) + pos_in_page.view(1, -1)
            ).view(-1)

        if prefix_lens[i] + num1 + num2 == seq_lens[i]:
            continue

        num3 = seq_lens[i] - seq_lens[i] // page_size * page_size
        if num3:
            out_indices[end_pos[i] - num3 : end_pos[i]] = (
                free_pages[end_new_pages[i] - 1] * page_size + pos_in_page[:num3]
            ).view(-1)


@triton.jit
def alloc_extend_kernel(
    pre_lens_ptr,
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)
    extend_lens = seq_lens - pre_lens

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = tl.load(pre_lens_ptr + pid)
    extend_len = seq_len - pre_len

    sum_extend_lens = tl.sum(extend_lens)
    output_start_loc = sum_extend_lens - extend_len

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    # Part 1: fill the old partial page
    last_loc = tl.load(last_loc_ptr + pid)
    num_part1 = (
        min(seq_len, (pre_len + page_size - 1) // page_size * page_size) - pre_len
    )
    offset_one_page = tl.arange(0, page_size)
    tl.store(
        out_indices + output_start_loc + offset_one_page,
        last_loc + 1 + offset_one_page,
        mask=offset_one_page < num_part1,
    )
    if pre_len + num_part1 == seq_len:
        return

    # Part 2: fill the new full pages using a dynamic blocked loop.
    # The loop bound is derived from num_part2 (runtime value), so Triton
    # generates a real loop instead of unrolling — no constexpr dependency
    # on extend size and only one kernel compilation.
    num_part2 = (
        seq_len // page_size * page_size
        - (pre_len + page_size - 1) // page_size * page_size
    )
    BLOCK_EXTEND: tl.constexpr = 4096
    num_blocks = (num_part2 + BLOCK_EXTEND - 1) // BLOCK_EXTEND
    for block_id in range(num_blocks):
        offset_in_block = tl.arange(0, BLOCK_EXTEND)
        offset = block_id * BLOCK_EXTEND + offset_in_block
        mask = offset < num_part2
        page_start = tl.load(
            free_page_ptr + new_page_start_loc + offset // page_size,
            mask=mask,
        )
        tl.store(
            out_indices + output_start_loc + num_part1 + offset,
            page_start * page_size + offset % page_size,
            mask=mask,
        )
    if pre_len + num_part1 + num_part2 == seq_len:
        return

    # Part 3: fill the new partial page
    num_part3 = seq_len - seq_len // page_size * page_size
    start_loc = tl.load(
        free_page_ptr + new_page_start_loc + num_page_start_loc_self - 1
    )
    tl.store(
        out_indices + output_start_loc + num_part1 + num_part2 + offset_one_page,
        start_loc * page_size + offset_one_page,
        mask=offset_one_page < num_part3,
    )


@triton.jit
def alloc_decode_kernel(
    seq_lens_ptr,
    last_loc_ptr,
    free_page_ptr,
    out_indices,
    bs_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.where(load_offset <= pid, seq_lens - 1, seq_lens)

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = seq_len - 1

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
        pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    if num_page_start_loc_self == 0:
        last_loc = tl.load(last_loc_ptr + pid)
        tl.store(out_indices + pid, last_loc + 1)
    else:
        page = tl.load(free_page_ptr + new_page_start_loc)
        tl.store(out_indices + pid, page * page_size)


class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """
    An allocator managing the indices to kv cache data.

    This class has the same interface as `TokenToKVPoolAllocator` but the output
    of one request is always page-aligned.

    TODO: fuse last_loc into the kernel.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)
        self.num_pages = size // page_size
        self.debug_mode = get_bool_env_var("SGLANG_DEBUG_MEMORY_POOL")
        self.clear()

    def alloc(self, need_size: int):
        # page-aligned allocation, returning contiguous indices of pages
        if self.debug_mode:
            assert (
                need_size % self.page_size == 0
            ), "The allocation size should be page-aligned"

        num_pages = need_size // self.page_size
        if self.need_sort and num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_pages > len(self.free_pages):
            return None

        out_pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]

        out_indices = (
            out_pages[:, None] * self.page_size
            + torch.arange(self.page_size, device=self.device)
        ).reshape(-1)

        return out_indices

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )

        bs = len(prefix_lens)
        if self.need_sort and extend_num_tokens // self.page_size + bs + 1 > len(
            self.free_pages
        ):
            self.merge_and_sort_free()

        out_indices = torch.empty(
            (extend_num_tokens,), dtype=torch.int64, device=self.device
        )

        alloc_extend_kernel[(bs,)](
            prefix_lens,
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            next_power_of_2(bs),
            self.page_size,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            prefix_lens=prefix_lens_cpu,
        )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )

        bs = len(seq_lens)
        if self.need_sort and bs > len(self.free_pages):
            self.merge_and_sort_free()

        out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
        alloc_decode_kernel[(bs,)](
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            next_power_of_2(bs),
            self.page_size,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            decode=True,
        )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            free_page_indices = torch.unique(free_index // self.page_size)
            if self.need_sort:
                self.release_pages = torch.cat((free_page_indices, self.release_pages))
            else:
                self.free_pages = torch.cat((free_page_indices, self.free_pages))
        else:
            self.free_group.append(free_index)

        if self.debug_mode:
            assert len(torch.unique(self.free_pages)) == len(self.free_pages)

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.num_pages + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)

    def get_cpu_copy(self, indices, mamba_indices=None):
        return self._kvcache.get_cpu_copy(indices, mamba_indices=mamba_indices)

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        return self._kvcache.load_cpu_copy(
            kv_cache_cpu, indices, mamba_indices=mamba_indices
        )
