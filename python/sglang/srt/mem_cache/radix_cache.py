from __future__ import annotations

from sglang.srt.mem_cache.cache_init_params import CacheInitParams

"""
Copyright 2023-2024 SGLang Team
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
The radix tree data structure for managing the KV cache.
"""

import collections
import hashlib
import heapq
import logging
import os
import sys
import time
from array import array
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.events import KVCacheEventMixin
from sglang.srt.mem_cache.utils import get_eviction_strategy, split_node_hash_value

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class RadixKey:
    """is_bigram=True: token_ids holds raw tokens (N+1 for N bigrams); slices share one boundary token."""

    __slots__ = ("token_ids", "extra_key", "is_bigram", "limit")

    def __init__(
        self,
        token_ids: array[int],
        extra_key: Optional[str] = None,
        is_bigram: bool = False,
        limit: Optional[int] = None,
    ):
        # token ids sequence (raw ints in both modes)
        self.token_ids = token_ids
        # extra key (e.g. lora_id, cache_salt)
        self.extra_key = extra_key
        # bigram view over token_ids: length = max(0, len(token_ids) - 1)
        self.is_bigram = is_bigram
        # Optional cap on raw tokens: behave as if token_ids were sliced to
        # token_ids[:limit], without the O(n) copy. None = use all tokens.
        self.limit = limit

    def _raw_len(self) -> int:
        n = len(self.token_ids)
        if self.limit is not None and self.limit < n:
            return self.limit
        return n

    def raw_token_ids(self) -> array:
        """token_ids honoring `limit` (copies only when capped)."""
        n = self._raw_len()
        t = self.token_ids
        return t if n == len(t) else t[:n]

    def __len__(self) -> int:
        n = self._raw_len()
        if self.is_bigram:
            return n - 1 if n > 0 else 0
        return n

    # TODO(Jialin): vectorize with numpy without PyLong boxing
    def __iter__(self) -> Iterator:
        t = self.token_ids
        n = self._raw_len()
        if self.is_bigram:
            for i in range(n - 1 if n > 0 else 0):
                yield (t[i], t[i + 1])
        elif n == len(t):
            yield from t
        else:
            for i in range(n):
                yield t[i]

    def __getitem__(self, idx: Union[int, slice]) -> RadixKey:
        # Normalize int -> 1-element slice so the rest handles one shape.
        if isinstance(idx, int):
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError(f"RadixKey index out of range: {idx}")
            idx = slice(idx, idx + 1)
        start, stop, step = idx.indices(len(self))
        if step != 1:
            raise ValueError("RadixKey slice step must be 1")

        if self.is_bigram:
            # bigrams [start, stop) span raw tokens [start, stop + 1);
            # empty slice -> empty raw tokens (not a dangling boundary token).
            raw = self.token_ids[start : stop + 1] if stop > start else array("q")
            return RadixKey(raw, self.extra_key, is_bigram=True)
        return RadixKey(self.token_ids[start:stop], self.extra_key)

    def __repr__(self) -> str:
        preview = self.token_ids[:10]
        return f"RadixKey(extra_key={self.extra_key!r}, token_ids={preview}{'...' if len(self.token_ids) > 10 else ''}, is_bigram={self.is_bigram})"

    def page_aligned(self, page_size: int) -> RadixKey:
        if page_size == 1:
            return self
        aligned_len = len(self) // page_size * page_size
        return self[:aligned_len]

    def maybe_to_bigram_view(
        self,
        is_eagle: bool,
        value: Optional[torch.Tensor] = None,
    ) -> Tuple[RadixKey, Optional[torch.Tensor]]:
        # O(1): flip the bigram flag instead of materializing a tuple list.
        # value is paired with raw tokens and gets truncated to the bigram count.
        if is_eagle and not self.is_bigram:
            self.is_bigram = True
            if value is not None:
                value = value[: len(self)]
        return self, value

    def _check_compatible(self, other: RadixKey) -> None:
        if self.extra_key != other.extra_key:
            raise ValueError(
                f"RadixKey operations require matching extra_key, but got "
                f"{self.extra_key=} != {other.extra_key=}"
            )

    def match(self, other: RadixKey, page_size: int = 1) -> int:
        """Logical-unit prefix length shared with ``other``. Result is rounded down to ``page_size``."""
        self._check_compatible(other)
        t0, t1 = self.token_ids, other.token_ids
        assert type(t0) is type(t1), (type(t0), type(t1))
        n = min(len(t0), len(t1))

        # Exponential search for the first diverging token: gallop in doubling
        # windows (one C-level slice compare each), then binary-search the window
        # holding the divergence -- no per-token Python loop on long shared prefixes.
        matched_tokens = n
        lo = 0
        step = 1
        while lo < n:
            hi = lo + step if lo + step < n else n
            if t0[lo:hi] != t1[lo:hi]:
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if t0[lo:mid] == t1[lo:mid]:
                        lo = mid
                    else:
                        hi = mid
                matched_tokens = lo
                break
            lo = hi
            step *= 2

        if self.is_bigram:
            matched = max(0, min(matched_tokens - 1, len(self), len(other)))
            return (matched // page_size) * page_size if page_size > 1 else matched

        matched_tokens = min(matched_tokens, len(self), len(other))
        if page_size == 1:
            return matched_tokens
        return (matched_tokens // page_size) * page_size

    def child_key(self, page_size: int = 1):
        """Hashable dict-key for the first ``page_size`` logical units, namespaced by ``extra_key``."""
        t = self.token_ids
        if self.is_bigram:
            if page_size == 1:
                plain = (t[0], t[1])
            else:
                plain = tuple((t[j], t[j + 1]) for j in range(page_size))
        else:
            plain = t[0] if page_size == 1 else tuple(t[:page_size])
        return plain if self.extra_key is None else (self.extra_key, plain)

    def hash_page(self, start: int, end: int, prior_hash: Optional[str] = None) -> str:
        """SHA256 for logical units [start, end); bigram mode feeds overlapping (t_i, t_{i+1}) byte pairs."""
        hasher = hashlib.sha256()
        if prior_hash:
            hasher.update(bytes.fromhex(prior_hash))
        t = self.token_ids
        if self.is_bigram:
            for j in range(start, end):
                hasher.update(t[j].to_bytes(4, byteorder="little", signed=False))
                hasher.update(t[j + 1].to_bytes(4, byteorder="little", signed=False))
        else:
            for j in range(start, end):
                hasher.update(t[j].to_bytes(4, byteorder="little", signed=False))
        return hasher.hexdigest()


class TreeNode:

    counter = 0
    # LPB eviction policy config (paper §sec:design-l1 eq:lpb-lru,
    # design.md §"Shared cost model" `c^evict_i`). Mirrors the
    # mamba-side knobs in `mamba_radix_cache.TreeNode`.
    lpb_window_s = float(os.environ.get("SGLANG_LPB_WINDOW_S", "60.0"))
    lpb_hit_deque_maxlen = int(
        os.environ.get("SGLANG_LPB_HIT_DEQUE_MAXLEN", "4096")
    )

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()

        self.hit_count = 0
        # indicating the node is locked to protect from eviction
        # incremented when the node is referenced by a storage operation
        self.host_ref_counter = 0
        # store the host indices of KV cache
        self.host_value: Optional[torch.Tensor] = None
        self.write_through_pending_id: Optional[int] = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None
        # priority for priority-aware eviction
        self.priority = priority

        # LPB eviction signal — windowed hit timestamps. Only
        # populated when `eviction_policy == "lpb"`; otherwise
        # unused. `record_hit` appends; `hits_in_window` lazy-prunes.
        self._hit_times: collections.deque = collections.deque(
            maxlen=TreeNode.lpb_hit_deque_maxlen
        )

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    # ---- LPB eviction signal (paper §sec:design-l1 eq:lpb-lru) ----

    def record_hit(self) -> None:
        """Append a hit timestamp + bump cumulative `hit_count`.
        Called from `RadixCache._match_prefix_helper` on every node
        visited during a successful prefix match (when policy=lpb)."""
        self._hit_times.append(time.monotonic())
        self.hit_count += 1

    def hits_in_window(self) -> int:
        """Sliding-window hit count over `TreeNode.lpb_window_s`
        seconds. Lazy-prunes old entries on read."""
        if not self._hit_times:
            return 0
        cutoff = time.monotonic() - TreeNode.lpb_window_s
        while self._hit_times and self._hit_times[0] < cutoff:
            self._hit_times.popleft()
        return len(self._hit_times)

    def lpb_priority(self) -> float:
        """LPB loss-per-byte: `ℓ(b) = n_b · c_kv(s_b) / B_b`.
        Higher loss = worse to evict; eviction picks the LOWEST.
        Matches design.md §"Shared cost model" `c^evict_i` formula
        and the mamba-side counterpart in `mamba_radix_cache`.

        Factor decomposition (KV-only — this is the plain RadixCache):
          - `n_b` = `self.hits_in_window()` — sliding-window hit count.
          - `c_kv(s_b)` = recompute cost to re-prefill this block
            (`CostCurves.c_kv_ms(len(self.key))` — block-local token
            count). Quadratic in `s_b` per the paper.
          - `B_b` = `value.numel()` — bytes the eviction would free.
            For ratio ordering across nodes the int64-per-page
            constant cancels, so we leave numel as-is.

        Returns `+inf` for hit-but-zero-byte (guards div/0) and `0`
        for never-hit zero-byte (degenerate; safe to evict first).

        No memoization: a prior implementation cached the priority
        on the node, but the cache went stale silently when
        `hits_in_window` pruned an expired entry without a
        `record_hit` in between (cache invalidation only fired on
        record_hit). Recompute on every call — cost is one cost-curve
        lookup + a divide, dominated by the deque walk in
        `hits_in_window` even in the cached case.
        """
        size_bytes = 0
        if self.value is not None:
            size_bytes = int(self.value.numel())
        n_hits = self.hits_in_window()
        if size_bytes == 0:
            return float("inf") if n_hits > 0 else 0.0
        from sglang.srt.budgeter.cost_model import get_cost_curves
        s_b = len(self.key) if self.key is not None else 0
        c_kv_ms = get_cost_curves().c_kv_ms(s_b)
        return n_hits * c_kv_ms / size_bytes

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def protect_host(self):
        """Protect the host value from eviction."""
        self.host_ref_counter += 1

    def release_host(self):
        """Release the host value, allowing it to be evicted."""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    def get_prefix_hash_values(self, node: TreeNode) -> List[str]:
        if node is None or node.hash_value is None:
            return []

        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: TreeNode):
        return self.last_access_time < other.last_access_time


class RadixCache(KVCacheEventMixin, BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        # Recovery-length EWMAs (paper sec:design-formalism-offline): written by
        # record_recovery_len_{kv,rec,retract} on eviction/retraction; the planner
        # c_sigma(L) and pressure adapter read them via the Budgeter snapshot.
        self._slow_recovery_len_kv_ewma = 0.0
        self._slow_recovery_len_rec_ewma = 0.0
        self._slow_recovery_len_retract_ewma = 0.0
        # Cumulative KV tokens reclaimed by admission-pressure eviction
        # (check_decode_mem); read via the Budgeter snapshot as the
        # deferred re-prefill cost the admission gate consults.
        self._admission_cumulative_evicted_tokens = 0
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.is_eagle = params.is_eagle
        self.disable_finished_insert = params.disable_finished_insert
        self.eviction_policy = params.eviction_policy.lower()

        self.kv_event_queue = []

        if params.enable_metrics:
            self.init_metrics_collector()

        if self.token_to_kv_pool_allocator:
            dev = self.token_to_kv_pool_allocator.device
            if isinstance(dev, (str, torch.device)):
                self.device = torch.device(dev)
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.eviction_strategy = get_eviction_strategy(self.eviction_policy)

        self.evictable_leaves = set()
        self.reset()

    @classmethod
    def create_simulated(
        self,
        disable: bool = False,
        mock_allocator: Optional[Any] = None,
        page_size: int = 1,
        enable_kv_cache_events: bool = False,
    ) -> RadixCache:
        """Init a radix cache without memory pools for simulation purpose."""
        params = CacheInitParams(
            disable=disable,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=page_size,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        return RadixCache(params)

    ##### Public API #####

    def reset(self):
        # Initialize root with minimum priority so any real priority overrides it
        self.root_node = TreeNode(priority=-sys.maxsize)
        self.root_node.key = RadixKey(token_ids=array("q"), extra_key=None)
        self.root_node.value = []
        self.root_node.host_value = []
        self.root_node.lock_ref = 1
        self.root_node.hash_value = []
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.evictable_leaves.clear()
        self._empty_match_result = MatchResult(
            device_indices=torch.empty(
                (0,),
                dtype=torch.int64,
                device=self.device,
            ),
            last_device_node=self.root_node,
            last_host_node=self.root_node,
            best_match_node=self.root_node,
        )
        self._record_all_cleared_event()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find the longest cached prefix of ``key`` in the radix tree.

        The logical namespace for prefix matching is determined by both the
        token id sequence and the optional ``extra_key`` carried by ``RadixKey``.
        Entries that share identical leading token ids but have *different*
        ``extra_key`` values are intentionally kept disjoint and never share
        prefix nodes. This is useful to:

        * Isolate KV cache lines for different LoRA / adapter IDs.
        * Separate requests that intentionally should not share state (e.g.,
          different sampling salt, cache version, or retrieval augmentation
          context) by supplying a distinct ``extra_key``.

        Args:
            params (MatchPrefixParams): Parameters containing the lookup key
                with a list of token ids and an optional ``extra_key`` namespace tag.
                If ``page_size > 1`` the length is internally truncated to a multiple
                of ``page_size`` before matching. Passing an empty key returns an
                empty result with the root as the last node.

        Returns:
            MatchResult: ``device_indices`` is a 1-D ``torch.int64`` tensor of
            the concatenated KV cache indices corresponding to the longest
            cached prefix (may be length 0).
            ``last_device_node`` and ``last_host_node`` (currently the same) are the tree node objects
            representing the terminal node of the matched prefix. This method
            may mutate internal structure by splitting an existing node if the
            match ends inside a stored segment.

        Internal updates:
            * Refreshes access metadata (timestamps) used by the
                configured eviction strategy.
            * If the lookup ends inside a stored segment the node is split once
                to expose a precise boundary; this structural refinement improves
                subsequent match efficiency and does not duplicate data.
        """
        key = params.key
        key, _ = key.maybe_to_bigram_view(self.is_eagle)

        if self.disable or len(key) == 0:
            return self._empty_match_result

        key = key.page_aligned(self.page_size)

        if len(key) == 0:
            return self._empty_match_result

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = self._empty_match_result.device_indices
        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
            best_match_node=last_node,
        )

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        priority = params.priority
        chunked = params.chunked

        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            value = value[: len(key)]
        else:
            # Debug/test fallback: use token ids themselves as values.
            value = torch.tensor(key.token_ids[: len(key)], dtype=torch.int64)

        prefix_len = self._insert_helper(self.root_node, key, value, priority, chunked)
        return InsertResult(prefix_len=prefix_len)

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        """Cache request when it finishes."""
        # In deterministic mode, disable finished request insertion to radix cache
        if self.disable_finished_insert:
            is_insert = False

        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        key_len = len(radix_key)
        values = kv_indices[:key_len].to(dtype=torch.int64, copy=True)

        # Radix Cache takes one ref in memory pool
        if is_insert:
            priority = getattr(req, "priority", 0) or 0
            result = self.insert(
                InsertParams(key=radix_key, value=values, priority=priority)
            )
            # Free the duplicates that were already in the tree
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : result.prefix_len]
            )
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : key_len]
            )

        # free the unaligned tail
        self.token_to_kv_pool_allocator.free(kv_indices[key_len:])

        # Remove req slot release the cache lock
        if req.last_node is not None:
            self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        token_ids = req.get_fill_ids()
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        values = kv_indices[: len(radix_key)].to(dtype=torch.int64, copy=True)

        # Radix Cache takes one ref in memory pool
        result = self.insert(
            InsertParams(
                key=radix_key,
                value=values,
                chunked=chunked,
                priority=getattr(req, "priority", 0) or 0,
            )
        )
        new_prefix_len = result.prefix_len

        self.token_to_kv_pool_allocator.free(
            kv_indices[req.cache_protected_len : new_prefix_len]
        )

        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )
        assert len(new_indices) == len(
            radix_key
        ), f"{len(new_indices)=}, {len(radix_key)=}"

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        # The cache_protected_len is not always equal to len(req.prefix_indices)
        # since for page_size > 1, the partial part is added to req.prefix_indices, but that part of kv indices is not added to the tree.
        # It should be freed in the next cache_unfinished_req and final cache_finished_req to avoid memory leak.
        # So we introduce this `cache_protected_len` field to make sure the partial part can be freed correctly.
        req.cache_protected_len = len(new_indices)

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        # - page_size != 1: there is a partial page at the end, keep the full kv_indices
        # - eagle case: bigram keys will only cache len - 1 kv indices
        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices

        req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper()

    def _iter_evict_victims(self, num_tokens: int):
        """Single source of truth for eviction victim selection
        (design.md §"Why exact c^evict"). Pure-read generator yielding
        the exact `TreeNode`s — in order — that would be popped to free
        `num_tokens` tokens under the active `eviction_strategy`.

        BOTH `evict()` (which then frees them) and
        `predict_evict_cost_us` (which sums their recompute cost)
        consume this generator, so the predicted set is byte-identical
        to the evicted set *by construction* — they cannot drift.

        Does NOT mutate tree / pool state. Parent-promotion is
        simulated via a per-parent child countdown (`effective_children`)
        that mirrors `_delete_leaf` emptying `parent.children`: after
        the last evictable child of an unlocked internal node is
        yielded, the parent is pushed into the heap exactly as the real
        eviction would promote it.

        IMPORTANT: callers that mutate (evict()) must materialise the
        full sequence FIRST (`list(...)`) before freeing — the
        generator reads live `len(parent.children)` to seed the
        countdown, so interleaving it with `_delete_leaf` would seed
        from already-mutated counts.

        Skip contract (intentional, safer than the pre-refactor inline
        loop): a popped node with `value is None` (already evicted) or
        a zero-length `value` is skipped — it frees no tokens and is
        left in the tree. The old inline `evict()` loop would have
        crashed on `value is None` (`len(None)`) and pruned a
        zero-length node; neither arises for a healthy evictable leaf.
        """
        if num_tokens <= 0:
            return
        leaves = list(self.evictable_leaves)
        if not leaves:
            return
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node)
            for node in leaves
        ]
        heapq.heapify(eviction_heap)
        effective_children: dict[int, int] = {}
        num_evicted = 0
        while eviction_heap and num_evicted < num_tokens:
            _priority, x = heapq.heappop(eviction_heap)
            if x.value is None:
                continue
            L_evicted = len(x.value)
            if L_evicted == 0:
                continue
            yield x
            num_evicted += L_evicted

            parent = x.parent
            if parent is None or parent is self.root_node:
                continue
            if parent.lock_ref != 0:
                continue
            key = id(parent)
            if key not in effective_children:
                effective_children[key] = len(parent.children)
            effective_children[key] -= 1
            if effective_children[key] == 0:
                heapq.heappush(
                    eviction_heap,
                    (self.eviction_strategy.get_priority(parent), parent),
                )

    def predict_evict_cost_us(self, num_tokens: int, pool: str = "kv") -> float:
        """Exact c^evict_i(X) predictor (design.md §"Shared cost model"
        "Why exact c^evict"). Sums `Σ n_b · c_kv(s_b)` over the
        exact set of blocks `evict()` would pick to free `num_tokens`
        tokens — obtained from the shared `_iter_evict_victims`
        generator, so the priced set IS the evicted set. Returns µs.

        n_b semantics (design.md §"LPB and the Admitter"):
          - LPB: `node.hits_in_window()` (path-counted by LPB anyway).
          - non-LPB: `n_b ≡ 1` (sglang doesn't path-count under LRU /
            LFU / FIFO / etc., so the predictor falls back to a
            uniform per-block hit count).

        Returns `+inf` when the cache cannot satisfy `num_tokens`
        (evictable supply too small) so the Admitter treats own-evict
        as infeasible — fail-closed.

        Caller is expected to hold the allocator's `_alloc_lock`
        across prediction → cap-barrier → execute (the Admitter wraps
        `Admitter.decide_for_req`). This method does NOT acquire the
        lock itself; called without the lock, the walk can race a
        concurrent `evict()` and produce stale costs.
        """
        if pool != "kv":
            # Plain RadixCache is a KV-only tree — it has no mamba
            # side. `CostModel.c_evict_us("mamba", ...)` should never
            # route here (it only does on hybrid models whose tree is
            # a MambaRadixCache). Crash loudly on misrouting.
            raise ValueError(
                f"RadixCache.predict_evict_cost_us: unsupported pool "
                f"{pool!r} (KV-only cache supports pool='kv' only)"
            )
        if num_tokens <= 0:
            return 0.0
        if self.disable:
            return float("inf")

        from sglang.srt.budgeter.cost_model import get_cost_curves
        curves = get_cost_curves()
        lpb_active = isinstance(self.eviction_strategy, LPBStrategy)

        num_evicted = 0
        total_cost_ms = 0.0
        for x in self._iter_evict_victims(num_tokens):
            s_b = len(x.key) if x.key is not None else 0
            n_b = x.hits_in_window() if lpb_active else 1
            total_cost_ms += n_b * curves.c_kv_ms(s_b)
            num_evicted += len(x.value)

        if num_evicted < num_tokens:
            return float("inf")
        return total_cost_ms * 1000.0

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        start_time = time.perf_counter()
        from sglang.srt.mem_cache.common import record_recovery_len_kv

        # Materialise the full victim list BEFORE freeing — the shared
        # generator reads live `len(parent.children)` to simulate
        # promotion; interleaving with `_delete_leaf` below would feed
        # it mutated counts (see `_iter_evict_victims` docstring).
        victims = list(self._iter_evict_victims(params.num_tokens))
        num_evicted = 0
        for x in victims:
            L_evicted = len(x.value)
            self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += L_evicted
            record_recovery_len_kv(self, L_evicted)
            self._delete_leaf(x)
            self._record_remove_event(x)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            node = node.parent
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return DecLockRefResult(delta=delta)

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        return self.protected_size_

    def all_values_flatten(self):
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    ##### Internal Helper Functions #####

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        access_time = time.monotonic()
        node.last_access_time = access_time
        # LPB needs per-node hit counting (paper §sec:design-l1
        # eq:lpb-lru `n_b`); skip the deque append in other modes
        # to keep LRU/LFU/SLRU paths zero-overhead.
        lpb_active = isinstance(self.eviction_strategy, LPBStrategy)

        child_key = key.child_key(self.page_size)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = access_time
            if lpb_active:
                child.record_hit()
            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = key.child_key(self.page_size)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # new_node -> child
        # New node inherits child's priority (represents shared prefix)
        new_node = TreeNode(priority=child.priority)
        new_node.hit_count = child.hit_count
        # LPB sliding-window hit signal lives on the shared-prefix
        # segment, which `new_node` now represents — every match that
        # credited `child` passed through this prefix. Move the deque
        # to `new_node`; the divergent tail (`child`) starts fresh
        # since it is now distinct, narrower content (paper
        # §sec:design-l1 eq:lpb-lru `n_b` is per-block).
        new_node._hit_times = child._hit_times
        child._hit_times = collections.deque(
            maxlen=TreeNode.lpb_hit_deque_maxlen
        )
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].clone()
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        new_node.parent.children[key.child_key(self.page_size)] = new_node

        # Split hash_value if it was already computed, otherwise leave as None
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        return new_node

    def _inc_hit_count(self, node: TreeNode, chunked: bool = False):
        # Skip the hit count update for chunked requests to avoid self-referencing
        # inflation where a chunked request increments hit_count on nodes it created
        # in previous chunks.
        if chunked:
            return
        node.hit_count += 1

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        priority: int = 0,
        chunked: bool = False,
    ):
        # Convert None priority to 0
        if priority is None:
            priority = 0
        access_time = time.monotonic()
        node.last_access_time = access_time
        # Update priority along the path (take max to propagate higher priority)
        node.priority = max(node.priority, priority)
        if len(key) == 0:
            return 0

        child_key = key.child_key(self.page_size)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = access_time
            prefix_len = node.key.match(key, page_size=self.page_size)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                new_node.priority = max(new_node.priority, priority)
                self._inc_hit_count(new_node, chunked)
                node = new_node
            else:
                node.priority = max(node.priority, priority)
                self._inc_hit_count(node, chunked)
            if len(key):
                child_key = key.child_key(self.page_size)

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            self._inc_hit_count(new_node, chunked)
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)
            # Hash will be computed lazily during event emission
            self._record_store_event(new_node)
        return total_prefix_length

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key.token_ids[:10],
                f"r={current_node.lock_ref}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == child.key.child_key(
                    self.page_size
                ), f"{key=}, {child.key.child_key(self.page_size)=}"

    def _delete_leaf(self, node):
        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.evictable_size_ -= len(node.key)
        if node in self.evictable_leaves:
            self.evictable_leaves.remove(node)
        self._update_leaf_status(node.parent)

    def _update_leaf_status(self, node: TreeNode):
        if node.evicted or node.lock_ref > 0:
            if node in self.evictable_leaves:
                self.evictable_leaves.remove(node)
            return

        for child in node.children.values():
            if not child.evicted:
                if node in self.evictable_leaves:
                    self.evictable_leaves.remove(node)
                return

        if node not in self.evictable_leaves:
            self.evictable_leaves.add(node)

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size


if __name__ == "__main__":
    tree = RadixCache.create_simulated()

    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [1, 2, 3]))))
    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [1, 2, 3]))))
    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [1, 2, 4, 5]))))
    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [1, 2, 4, 5, 6, 7]))))
    tree.insert(InsertParams(key=RadixKey(token_ids=array("q", [8, 9, 10, 11, 12]))))
    tree.pretty_print()

    print(
        tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=array("q", [1, 2, 3, 13, 14])))
        )
    )
