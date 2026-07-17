from __future__ import annotations

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
The radix tree data structure for managing the hybrid (full and Mamba) KV cache.
"""

import collections
import heapq
import os
import time
from array import array
from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from numpy import float64

from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
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
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.multi_ended_allocator import (
    UnifiedMambaTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.utils import split_node_hash_value
from sglang.srt.runtime_context import get_server_args

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

import logging

from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

# Debug-only invariant checks in the Mamba slot-donation path call tensor.item(),
# which forces a per-request cudaStreamSynchronize on the scheduler thread. Under
# load this can serialize/stall the scheduler. Gate them off by default; set
# SGLANG_MAMBA_DEBUG_ASSERTS=1 to re-enable for debugging.
_MAMBA_DEBUG_ASSERTS = os.environ.get("SGLANG_MAMBA_DEBUG_ASSERTS", "0") == "1"


class TreeNode:

    counter = 0
    last_access_time_counter_float = float64(1.0)
    # Paper §sec:design-l1 (eq:lpb-lru): sliding-window in seconds for
    # hits-per-byte eviction signal. Configurable via SGLANG_LPB_WINDOW_S;
    # default 60s matches the paper's narrative.
    lpb_window_s = float(os.environ.get("SGLANG_LPB_WINDOW_S", "60.0"))
    # Cap on per-node _hit_times deque to keep memory bounded under
    # long-running deployments with very hot nodes. With maxlen, the
    # deque becomes circular and silently drops the oldest entries on
    # append past the cap; hits_in_window() saturates at the cap value.
    # That's fine for LPB ordering: super-hot blocks still all score
    # above warm blocks, and ties break on last_access_time. Override
    # with SGLANG_LPB_HIT_DEQUE_MAXLEN; default 4096 is generous for
    # 60s windows at typical agent QPS.
    lpb_hit_deque_maxlen = int(
        os.environ.get("SGLANG_LPB_HIT_DEQUE_MAXLEN", "4096")
    )
    # Per-mamba-slot byte cost (B_b) used in eviction_priority()'s
    # denominator. MambaRadixCache.__init__ unconditionally overwrites
    # this with the actual pool byte/slot ratio
    # (`mamba_pool.mamba_cache.mem_usage_bytes() // (max_size + 1)`,
    # dividing by the physical allocated slot count per design.md
    # "Allocator padding") so LPB's KV-vs-mamba weighting reflects
    # reality, and fails loudly if the pool can't supply it. This 1024
    # class default applies only to test fixtures that build the cache
    # via `__new__` and bypass `__init__`.
    lpb_bytes_per_mamba_slot = 1024

    def __init__(self, id: Optional[int] = None):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.mamba_value: Optional[torch.Tensor] = None
        self.mamba_host_value: Optional[torch.Tensor] = None
        # invariant: for any node, if mamba_lock_ref is locked, full_lock_ref must be locked;
        # if full_lock_ref is locked, mamba_lock_ref doesn't need to be locked. So,
        # full_lock_ref is always >= mamba_lock_ref.
        # for full_lock, once it is locked, its parent must be locked as well
        # for mamba_lock, it only need lock node itself
        self.full_lock_ref = 0
        self.mamba_lock_ref = 0
        # last access time is only used for sanity check. LRU is maintained by the lru list.
        # `last_access_time` tracks the full LRU (whole matched path is reused as prefix);
        # `mamba_last_access_time` tracks the mamba LRU, which only touches the single state
        # actually consumed per access, so the two orders diverge and need separate stamps.
        self.last_access_time = get_last_access_time()
        self.mamba_last_access_time = self.last_access_time

        # Paper §sec:design-l1 (eq:lpb-lru): hits-per-byte eviction.
        # `hit_count` is the cumulative count (defined by upstream but
        # never incremented). `_hit_times` is the windowed deque of
        # timestamps used for the actual signal — `hits_in_window()`
        # lazily prunes old entries.
        self.hit_count = 0
        self._hit_times: collections.deque = collections.deque(
            maxlen=TreeNode.lpb_hit_deque_maxlen
        )
        self.host_ref_counter = 0
        self.host_mamba_ref_counter = 0
        # store the host indices of KV cache
        self.host_value = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None

        # for lru list, invariant:
        # 1. prev has greater last_access_time
        # 2. next has smaller last_access_time
        self.prev = None
        self.next = None
        self.mamba_prev = None
        self.mamba_next = None
        self.host_mamba_prev = None
        self.host_mamba_next = None

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def mamba_evicted(self):
        return self.mamba_value is None

    @property
    def backuped(self):
        return self.host_value is not None

    @property
    def mamba_backuped(self):
        return self.mamba_host_value is not None

    def protect_host(self):
        """Protect the host KV value from eviction."""
        self.host_ref_counter += 1

    def release_host(self):
        """Release the host KV value, allowing it to be evicted."""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def protect_host_mamba(self):
        """Protect the host mamba value from eviction."""
        self.host_mamba_ref_counter += 1

    def release_host_mamba(self):
        """Release the host mamba value, allowing it to be evicted."""
        if self.host_mamba_ref_counter > 0:
            self.host_mamba_ref_counter -= 1
        else:
            raise RuntimeError("Host mamba reference counter is already zero.")

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

    # ---- LPB eviction signal (paper §sec:design-l1, eq:lpb-lru) ------

    def record_hit(self) -> None:
        """Append a hit timestamp to the windowed deque.

        Called from `_match_prefix_helper` for every node visited during
        a successful prefix match. The deque is pruned lazily in
        `hits_in_window()`; we intentionally don't prune on insertion
        since the window cutoff floats with wall time.
        """
        self._hit_times.append(time.monotonic())
        self.hit_count += 1

    def hits_in_window(self) -> int:
        """Number of recorded hits within `TreeNode.lpb_window_s`."""
        if not self._hit_times:
            return 0
        cutoff = time.monotonic() - TreeNode.lpb_window_s
        # Lazy left-prune.
        while self._hit_times and self._hit_times[0] < cutoff:
            self._hit_times.popleft()
        return len(self._hit_times)

    def eviction_priority(self) -> float:
        """LPB loss-per-byte: `ℓ(b) = n_b · c_i(s_b) / B_b`.

        Higher loss = worse to evict; eviction picks the LOWEST.
        Matches design.md §"Shared cost model" `c^evict_i` formula and
        paper §sec:design-l1 eq:lpb-lru.

        Factor decomposition:
          - `n_b` = `self.hits_in_window()` — sliding-window hit
            count (see `record_hit` / `hits_in_window`).
          - `c_i(s_b)` = recompute cost to regenerate this block.
            `s_b = len(self.key)` (block-local token count). For a
            hybrid (KV + mamba) node, `c_i` is the SUM of `c_kv_ms(s_b)`
            and `c_m_ms(s_b)` — both costs would be paid on miss. For
            a KV-only or mamba-only node, only the matching curve
            contributes. Curves come from `get_cost_curves()` (boot-
            probed / built-in default).
          - `B_b` = bytes the eviction would actually free:
              * KV bytes (≈ `value.numel()`; the int64-per-page
                constant cancels for ratio ordering).
              * mamba snapshot bytes (`mamba_value.numel() ×
                lpb_bytes_per_mamba_slot`; the latter is the real
                per-slot byte cost from the mamba pool — its
                `mem_usage_bytes()` divided by the physical allocated
                slot count `max_size+1` (design.md "Padded slot 0"),
                set unconditionally in `MambaRadixCache.__init__`. The
                1024 class default applies only to `__new__` test
                fixtures that bypass `__init__`).

        Returns `+inf` for nodes that hit but free zero bytes
        (shouldn't happen but guards against div-by-zero) and `0` for
        never-hit zero-byte nodes (degenerate; safe to evict first).

        No memoization: a prior implementation cached the priority,
        but the cache went stale silently when `hits_in_window`
        pruned an expired entry without a `record_hit` in between
        (cache invalidation only fired on record_hit). Recompute on
        every call.
        """
        n_hits = self.hits_in_window()
        size_bytes = 0
        if self.value is not None:
            size_bytes += int(self.value.numel())
        if self.mamba_value is not None:
            size_bytes += (
                int(self.mamba_value.numel())
                * TreeNode.lpb_bytes_per_mamba_slot
            )
        if size_bytes == 0:
            return float("inf") if n_hits > 0 else 0.0

        # `c_i(s_b)` — recompute cost. Block-local token count is
        # the sgl `RadixKey` length.
        from sglang.srt.budgeter.cost_model import get_cost_curves
        curves = get_cost_curves()
        s_b = len(self.key) if self.key is not None else 0
        c_i_ms = 0.0
        if self.value is not None:
            c_i_ms += curves.c_kv_ms(s_b)
        if self.mamba_value is not None:
            c_i_ms += curves.c_m_ms(s_b)

        return n_hits * c_i_ms / size_bytes


def get_last_access_time() -> float64:
    ret = TreeNode.last_access_time_counter_float
    TreeNode.last_access_time_counter_float += 1.0
    return ret


class LRUList:
    def __init__(self, mamba: bool = False):
        self.mamba = mamba
        if self.mamba:
            self.prv = "mamba_prev"
            self.nxt = "mamba_next"
            self.lock_ref = "mamba_lock_ref"
            self.time_attr = "mamba_last_access_time"
        else:
            self.prv = "prev"
            self.nxt = "next"
            self.lock_ref = "full_lock_ref"
            self.time_attr = "last_access_time"
        # Initialize dummy head and tail nodes
        self.head = TreeNode()  # Most recently used side
        self.tail = TreeNode()  # Least recently used side
        setattr(self.head, self.nxt, self.tail)  # self.head.next = self.tail
        setattr(self.tail, self.prv, self.head)  # self.tail.prev = self.head
        self.cache = {}

    def _add_node(self, node):
        """Helper to add node right after head (most recently used)"""
        self._add_node_after(self.head, node)

    def _add_node_after(self, old_node, new_node):
        """Helper to add node right after old_node"""
        setattr(new_node, self.prv, old_node)  # new_node.prev = old_node
        setattr(
            new_node, self.nxt, getattr(old_node, self.nxt)
        )  # new_node.next = old_node.next
        setattr(
            getattr(old_node, self.nxt), self.prv, new_node
        )  # old_node.next.prev = new_node
        setattr(old_node, self.nxt, new_node)  # old_node.next = new_node

    def _remove_node(self, node):
        """Helper to remove node from linked list"""
        setattr(
            getattr(node, self.prv), self.nxt, getattr(node, self.nxt)
        )  # node.prev.next = node.next
        setattr(
            getattr(node, self.nxt), self.prv, getattr(node, self.prv)
        )  # node.next.prev = node.prev
        # Clear self pointers to break reference cycles among evicted nodes.
        setattr(node, self.prv, None)
        setattr(node, self.nxt, None)

    def _get_lru(self) -> Optional[TreeNode]:
        """
        Get the least recently used node
        """
        if len(self.cache) == 0:
            return None
        return getattr(self.tail, self.prv)

    def reset_node_mru(self, node):
        """
        Move a (existing) node to most recently used position
        """
        assert node.id in self.cache, f"Resetting node {node.id=} not in lru list"
        assert (
            not self.mamba or node.mamba_value is not None
        ), f"Resetting mamba tombstone node in mamba lru list: {node.id=}"
        if self.mamba:
            node.mamba_last_access_time = get_last_access_time()
        self._remove_node(node)
        self._add_node(node)

    def reset_node_and_parents_mru(self, node, root_node):
        """
        Move an (existing) node and its parents to most recently used position. Child node is
        more recently used than parent node.
        """
        prev_node = self.head
        while node != root_node:
            if not self.mamba or node.mamba_value is not None:
                assert (
                    node.id in self.cache
                ), f"Resetting node {node.id=} not in lru list when resetting node and parents mru"
                self._remove_node(node)
                self._add_node_after(prev_node, node)
                prev_node = node
            node = node.parent

    def insert_mru(self, node):
        """
        Insert a (new) node as most recently used
        """
        assert (
            not self.mamba or node.mamba_value is not None
        ), f"Inserting mamba tombstone node in mamba lru list: {node.id=}"
        assert (
            node.id not in self.cache
        ), f"Inserting node {node.id=} already in lru list, existing node: {self.cache[node.id].id=}"
        if self.mamba:
            node.mamba_last_access_time = get_last_access_time()
        self.cache[node.id] = node
        self._add_node(node)

    def remove_node(self, node: TreeNode):
        """
        Remove node from lru list
        """
        assert node.id in self.cache, f"Removing node {node.id=} not in lru list"
        assert (
            not self.mamba or node.mamba_value is not None
        ), f"Removing mamba tombstone node from mamba lru list: {node.id=}"
        del self.cache[node.id]
        self._remove_node(node)

    def get_lru_no_lock(self) -> Optional[TreeNode]:
        """
        Get the least recently used node that is not locked
        """
        return self.get_prev_no_lock(self.tail, check_id=False)

    def get_leaf_lru_no_lock(self) -> Optional[TreeNode]:
        """
        Get the least recently used leaf node that is not locked
        """
        return self.get_prev_leaf_no_lock(self.tail, check_id=False)

    def get_prev_no_lock(
        self, node: TreeNode, check_id: bool = True
    ) -> Optional[TreeNode]:
        """
        Get the previous (i.e. more recently used) node that is not locked
        """
        if check_id:
            assert (
                node.id in self.cache
            ), f"Getting prev of node {node.id=} not in lru list"
        x = getattr(node, self.prv)  # x = node.prev
        while getattr(x, self.lock_ref) > 0:
            x = getattr(x, self.prv)  # x = x.prev
        # if x is the head, it means there is no node in the lru list without lock
        if x == self.head:
            return None
        return x

    def get_prev_leaf_no_lock(self, node: TreeNode, check_id: bool = True):
        """
        Get the previous (i.e. more recently used) leaf node that is not locked
        """
        if check_id:
            assert (
                node.id in self.cache
            ), f"Getting prev of node {node.id=} not in lru list"
        x = getattr(node, self.prv)  # x = node.prev
        while getattr(x, self.lock_ref) > 0 or len(x.children) > 0:
            x = getattr(x, self.prv)  # x = x.prev
        # if x is the head, it means there is no leaf node in the lru list without lock
        if x == self.head:
            return None
        return x

    def in_list(self, node: Optional[TreeNode]):
        """
        Check if the node is in the lru list
        """
        if not node:
            return False
        return node.id in self.cache

    def pretty_print(self, tree_cache: Optional[MambaRadixCache] = None):
        """
        Pretty print the lru list
        """
        msg = f"{self.mamba=} LRU list: "
        x_lru = self._get_lru()
        while x_lru is not None and x_lru.id in self.cache:
            msg += f"[{x_lru.id}] {getattr(x_lru, self.time_attr):f} -> "
            x_lru = getattr(x_lru, self.prv)
        print(msg)

        if not tree_cache:
            return
        msg = f"{self.mamba=} Nodes (sorted by {self.time_attr}): "
        if self.mamba:
            nodes = tree_cache._collect_nontombstone_nodes()
        else:
            nodes = tree_cache._collect_all_nodes()
        nodes.sort(key=lambda n: getattr(n, self.time_attr))
        for x in nodes:
            msg += f"[{x.id}] {getattr(x, self.time_attr):f} -> "
        print(msg)

    # Note: this is expensive, only use for debug
    def sanity_check_evictable_size(self):
        """
        Check the evictable size (i.e. the size of the nodes that are not locked)
        """
        node = self.get_lru_no_lock()
        evictable_size = 0
        while self.in_list(node):
            evictable_size += (
                len(node.value) if not self.mamba else len(node.mamba_value)
            )
            node = self.get_prev_no_lock(node)
        return evictable_size

    # Note: this is expensive, only use for debug or idle check
    def sanity_check(self, tree_cache: MambaRadixCache):
        """
        Check the lru list is valid by rebuilding it from the tree, sorting by this list's
        access-time stamp, and checking the order matches the linked list.
        """
        try:
            if self.mamba:
                nodes = tree_cache._collect_nontombstone_nodes()
            else:
                nodes = tree_cache._collect_all_nodes()
            total_nodes = len(nodes)
            total_lru = len(self.cache)
            # rebuild expected order from this list's own access-time stamp (full and mamba
            # lists have independent recency, so they use different stamps)
            nodes.sort(key=lambda n: getattr(n, self.time_attr))
            # the root node is not in the lru list
            assert len(nodes) == (
                total_lru + (0 if self.mamba else 1)
            ), f"len(nodes): {len(nodes)}, total_lru: {total_lru}"

            x_lru = self._get_lru()
            for x in nodes:
                if x == tree_cache.root_node:
                    # root node is not in the lru list
                    continue
                assert (
                    x_lru is not None and x_lru.id in self.cache
                ), f"Incorrect LRU list, x_lru is None or not in cache: {x_lru=}, {x.id=}"

                assert (
                    x == x_lru
                ), f"Incorrect LRU list, {self.mamba=}, x: {x.id=} != x_lru: {x_lru.id=}, {getattr(x, self.time_attr)=}, {getattr(x_lru, self.time_attr)=}"
                assert (
                    x_lru.full_lock_ref == 0
                ), f"x_lru should not be locked when idle, {x_lru.full_lock_ref=}, {x_lru.id=}"
                assert (
                    x_lru.mamba_lock_ref == 0
                ), f"x_lru should not be locked when idle, {x_lru.mamba_lock_ref=}, {x_lru.id=}"
                x_lru = getattr(x, self.prv)

            if self.mamba:
                evictable_size = tree_cache.mamba_evictable_size()
                lru_list_evictable_size = self.sanity_check_evictable_size()
            else:
                evictable_size = tree_cache.full_evictable_size()
                lru_list_evictable_size = self.sanity_check_evictable_size()

            assert (
                evictable_size == lru_list_evictable_size
            ), f"{self.mamba=}, total nodes: {total_nodes}, total lru: {total_lru}, evictable size: {evictable_size} != lru list evictable size: {lru_list_evictable_size}"
        except Exception as e:
            if get_parallel().tp_rank == 0:
                msg = f"Mamba Radix tree sanity check failed, ping @yizhang2077: {e}"
                logger.error(msg)
                tree_cache.pretty_print()
                tree_cache.full_lru_list.pretty_print(tree_cache)
                tree_cache.mamba_lru_list.pretty_print(tree_cache)
                raise Exception(msg)


class MambaRadixCache(KVCacheEventMixin, BasePrefixCache):
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
        # Cumulative cache-eviction counters: the Budgeter's grow-side
        # eviction-rate signals (symmetric to the reuse-aware drain cost).
        # A pool actively shedding (hot) entries should be GROWN. KV's
        # admission-path tally misses cache eviction on a cache-bound
        # workload, so these count the actual radix-cache evictions;
        # BudgetAgent deltas them per tick.
        self._cumulative_evicted_mamba_slots = 0
        self._cumulative_evicted_kv_tokens = 0
        # LPB LOSS (reuse-weighted recompute cost, us) shed per pool — the
        # accurate cross-pool eviction-cost signal. Raw slot/token COUNT above
        # over-values low-reuse evictions (n_b=0 -> ~0 loss), which drove the
        # swarm k2m/m2k oscillation. BudgetAgent deltas these per tick.
        self._cumulative_evicted_kv_lpb_loss = 0.0
        self._cumulative_evicted_mamba_lpb_loss = 0.0
        assert (
            isinstance(params.token_to_kv_pool_allocator, TokenToKVPoolAllocator)
            or isinstance(
                params.token_to_kv_pool_allocator, PagedTokenToKVPoolAllocator
            )
            or isinstance(
                params.token_to_kv_pool_allocator, UnifiedMambaTokenToKVPoolAllocator
            )
        )
        self.req_to_token_pool: HybridReqToTokenPool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.mamba_cache_chunk_size = get_server_args().mamba_cache_chunk_size

        # Optional cross-pool grow hook: a callable `(n_slots) -> bool`
        # the Budgeter injects once its actuator chain is built. When a caching
        # fork can't get a mamba slot and `evict_mamba` finds no unlocked cold
        # cache to reclaim, this synchronously grows mamba from KV (k2m) and
        # the fork is retried — instead of asserting "Can not alloc mamba
        # cache". None on stock sglang / Budgeter off, so the
        # fork-failure path stays the original evict→assert (fail-loud).
        self._mamba_grow_hook = None

        self.page_size = params.page_size
        self.disable = params.disable
        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.enable_mamba_extra_buffer = params.enable_mamba_extra_buffer
        self.enable_mamba_extra_buffer_lazy = params.enable_mamba_extra_buffer_lazy
        self.kv_event_queue = []
        # LPB vs LRU eviction is a boot-time config, same as plain
        # `RadixCache` (paper §sec:design-l1). `evict_full` /
        # `evict_mamba` consult `_should_use_lpb()` rather than an env
        # var so a server launched with `--eviction-policy lpb` gets
        # LPB on hybrid models too.
        self.eviction_policy = params.eviction_policy

        if not self.enable_mamba_extra_buffer:
            assert (
                self.page_size == 1
            ), f"Page size must be 1 for MambaRadixCache v1, got {self.page_size}"

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if params.enable_metrics:
            self.init_metrics_collector()

        # Set the real per-mamba-slot byte cost on TreeNode so
        # eviction_priority() weights mamba vs KV by ground-truth
        # bytes (B_b) instead of an estimate. (Paper §sec:design-l1 —
        # accurate hits-per-byte denominator.) Access the pool APIs
        # directly: a missing/zero byte denominator must fail loudly
        # rather than silently shipping a placeholder that corrupts the
        # LPB ordering undetectably.
        mp = self.req_to_token_pool.mamba_pool
        assert mp.size > 0, (
            f"mamba_pool.size must be positive to compute B_b, got {mp.size}"
        )
        assert mp.max_size >= mp.size, (
            f"mamba_pool.max_size must be >= size to compute B_b, got "
            f"max_size={mp.max_size}, size={mp.size}"
        )
        # Per-slot physical byte cost for LPB's loss-per-byte ratio.
        # State.bytes_per_slot() normalizes each conv/temporal tensor by
        # its own slot-dim size, so the result is exact for the stacked,
        # per-layer, and arena (VA-inflated leading dim) layouts alike;
        # speculative draft caches are excluded. See
        # dev/interlayer/4_e2e/cc_zero_downside/test_mamba_bytes_per_slot.py.
        bytes_per_slot = mp.mamba_cache.bytes_per_slot()
        assert bytes_per_slot > 0, (
            f"mamba per-slot byte denominator (B_b) must be positive, got "
            f"{bytes_per_slot} from mem_usage_bytes={mp.mamba_cache.mem_usage_bytes()} "
            f"/ (max_size+1)={mp.max_size + 1}"
        )
        TreeNode.lpb_bytes_per_mamba_slot = bytes_per_slot
        # Use print so it shows up regardless of sglang's configured log
        # level — this is a one-time init line and useful for verifying
        # LPB's denominator uses ground-truth bytes.
        print(
            f"[LPB] bytes_per_mamba_slot = {bytes_per_slot} "
            f"(State.bytes_per_slot(): conv+temporal, slot-dim normalized; "
            f"max_size={mp.max_size}, size={mp.size})",
            flush=True,
        )

        self.reset()

    ##### Public API #####

    def supports_mamba(self) -> bool:
        return True

    def reset(self) -> None:
        self.root_node = TreeNode()
        self.root_node.key = RadixKey(array("q"), None)
        self.root_node.value = []
        self.root_node.hash_value = []
        self.root_node.full_lock_ref = 1
        self.root_node.mamba_lock_ref = 1
        self.full_evictable_size_ = 0
        self.mamba_evictable_size_ = 0
        self.full_protected_size_ = 0
        self.mamba_protected_size_ = 0
        self._cumulative_evicted_mamba_slots = 0
        self._cumulative_evicted_kv_tokens = 0
        self._cumulative_evicted_kv_lpb_loss = 0.0
        self._cumulative_evicted_mamba_lpb_loss = 0.0
        # LRU lists are used to maintain the order of eviction of the nodes in the tree
        self.full_lru_list = LRUList(mamba=False)
        self.mamba_lru_list = LRUList(mamba=True)
        self._record_all_cleared_event()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find the matching prefix from the radix tree.
        Args:
            params: MatchPrefixParams containing key and optional Mamba-specific parameters.
        Returns:
            A tuple of a tensor of matching prefix token IDs and
            the last node that contains the prefix values. Note that
            this API can modify the internal state of the Radix tree.
            The last node create a new child if the prefix is shorter
            than the last node's value.
        """
        key = self._match_pre_processor(params)
        if key is None:
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
                best_match_node=self.root_node,
            )

        value, last_node, best_value_len = self._match_prefix_helper(key)
        return self._match_post_processor(params, value, last_node, best_value_len)

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0, mamba_exist=False)

        key = params.key
        value = params.value
        mamba_value = params.mamba_value
        prev_prefix_len = params.prev_prefix_len

        if value is None:
            value = torch.tensor([x for x in key.raw_token_ids()], dtype=torch.int64)
        prefix_len, mamba_exist = self._insert_helper(
            self.root_node,
            key,
            value,
            mamba_value,
            params.chunked,
            prev_prefix_len,
        )
        return InsertResult(prefix_len=prefix_len, mamba_exist=mamba_exist)

    def cache_finished_req(
        self, req: Req, is_insert: bool = True, *, kv_len_to_handle: int
    ) -> None:
        """Cache request when it finishes."""
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_len_to_handle
            ]
            self.token_to_kv_pool_allocator.free_segment(kv_indices, start_pos=0)
            self.req_to_token_pool.free_mamba_cache(req)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_len_to_handle]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :kv_len_to_handle
        ]

        if is_insert:
            if self.enable_mamba_extra_buffer:
                cache_len = req.mamba_last_track_seqlen
            else:
                cache_len = len(token_ids)
                # ReplaySSM (no_buffer): `temporal[slot]` lags the live state by
                # the slot's unflushed ring depth (`write_pos`), so cap the
                # donate to the last flush boundary (where temporal is current)
                # and reset the cursor, keeping the donated checkpoint consistent
                # with its key length. page_size is asserted == 1, so no realign.
                write_pos_buf = self.req_to_token_pool.mamba_pool.replayssm_write_pos
                if write_pos_buf is not None:
                    cache_len -= int(write_pos_buf[req.mamba_pool_idx].item())
                    write_pos_buf[req.mamba_pool_idx] = 0
            if cache_len is None:
                cache_len = 0
            if cache_len != len(token_ids):
                cache_end_idx = max(cache_len, req.cache_protected_len)
                self.token_to_kv_pool_allocator.free_segment(
                    kv_indices[cache_end_idx:], start_pos=cache_end_idx
                )
                token_ids = token_ids[:cache_len]
                kv_indices = kv_indices[:cache_len]

            if self.page_size != 1:
                page_aligned_len = len(kv_indices) // self.page_size * self.page_size
                page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
                    dtype=torch.int64, copy=True
                )
            else:
                page_aligned_len = len(kv_indices)
                page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)

            assert (
                cache_len == page_aligned_len
            ), f"It is required {cache_len=}, {page_aligned_len=}, {kv_len_to_handle=}, {len(req.origin_input_ids)=}, {len(req.output_ids)=} ping @yizhang2077 if you see this"

            # Radix Cache takes one ref in memory pool
            # insert the token_ids and kv_indices into the radix tree
            if self.enable_mamba_extra_buffer:
                mamba_ping_pong_track_buffer_to_keep = (
                    self.req_to_token_pool.get_mamba_ping_pong_keep_idx(req)
                )
                src_active = req.mamba_ping_pong_track_buffer[
                    mamba_ping_pong_track_buffer_to_keep
                ].unsqueeze(-1)
                if _MAMBA_DEBUG_ASSERTS:
                    # .item() forces a cudaStreamSynchronize; only pay it when debugging.
                    assert src_active.item() != -1, (
                        f"Cached mamba slot is -1: keep_idx={mamba_ping_pong_track_buffer_to_keep}, "
                        f"buf={req.mamba_ping_pong_track_buffer.tolist()}, "
                        f"next_track_idx={req.mamba_next_track_idx}, "
                        f"last_track_seqlen={req.mamba_last_track_seqlen}, "
                        f"rid={req.rid}"
                    )
                if self.int8_ckpt_pool is not None:
                    mamba_value = self._commit_int8_checkpoint(src_active)
                    # quantized -> no ping-pong slot needs keeping
                    mamba_ping_pong_track_buffer_to_keep = None
                else:
                    mamba_value = src_active.clone()
            else:
                if self.int8_ckpt_pool is not None:
                    mamba_value = self._commit_int8_checkpoint(
                        req.mamba_pool_idx.unsqueeze(-1)
                    )
                else:
                    mamba_value = req.mamba_pool_idx.unsqueeze(-1).clone()
                mamba_ping_pong_track_buffer_to_keep = None

            result = self.insert(
                InsertParams(
                    key=RadixKey(token_ids[:page_aligned_len], req.extra_key),
                    value=page_aligned_kv_indices,
                    mamba_value=mamba_value,
                    prev_prefix_len=req.cache_protected_len,
                )
            )
            mamba_exist = result.mamba_exist
            if mamba_exist and self.int8_ckpt_pool is not None:
                # state already cached -> the int8 slot we just allocated is a duplicate
                self.int8_ckpt_pool.free(mamba_value)
        else:
            self.token_to_kv_pool_allocator.free_segment(
                kv_indices[req.cache_protected_len :],
                start_pos=req.cache_protected_len,
            )
            mamba_exist = True

        if mamba_exist:
            mamba_ping_pong_track_buffer_to_keep = None

        # With int8 checkpoints the radix owns an int8 slot (not the request's active
        # slot), so the active mamba slot must always be returned to the active pool.
        free_mamba_cache = (
            True
            if (self.enable_mamba_extra_buffer or self.int8_ckpt_pool is not None)
            else mamba_exist
        )

        if free_mamba_cache:
            self.req_to_token_pool.free_mamba_cache(
                req,
                mamba_ping_pong_track_buffer_to_keep=mamba_ping_pong_track_buffer_to_keep,
            )

        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req, chunked=False) -> None:
        """Cache request when it is unfinished."""

        def _skip_cache_unfinished_req(req: Req) -> None:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : req.extend_range.end
            ]

            # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
            req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
            return

        token_ids = req.get_fill_ids()
        cache_len = (
            req.mamba_last_track_seqlen
            if self.enable_mamba_extra_buffer
            else len(token_ids)
        )
        if self.disable or cache_len is None:
            return _skip_cache_unfinished_req(req)

        kv_indices_orig = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]
        # kv_indices is the kv indices to be cached
        kv_indices = kv_indices_orig[:cache_len]
        if self.page_size != 1:
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].to(
                dtype=torch.int64, copy=True
            )
        else:
            page_aligned_len = len(kv_indices)
            page_aligned_kv_indices = kv_indices.to(dtype=torch.int64, copy=True)

        assert page_aligned_len == len(
            kv_indices
        ), f"page_aligned_len != len(kv_indices), {page_aligned_len=}, {len(kv_indices)=}, {cache_len=}, {self.page_size=}, {self.mamba_cache_chunk_size=}"

        page_aligned_token_ids = token_ids[:page_aligned_len]

        # Donate the mamba index to the radix cache instead of copying.
        # This avoids a data copy that would race with the forward stream.
        if self.int8_ckpt_pool is not None:
            # int8 path: quantize the to-be-cached active state into an int8 slot
            # (strategy-agnostic donate hook).
            if self.enable_mamba_extra_buffer:
                new_slot = self._alloc_mamba_slot()
                src_active = self.req_to_token_pool.donate_mamba_ping_pong_slot(
                    req, new_slot
                )
                mamba_value_donated = self._commit_int8_checkpoint(src_active)
                self.req_to_token_pool.mamba_allocator.free(src_active)
            else:
                mamba_value_donated = self._commit_int8_checkpoint(
                    req.mamba_pool_idx.view(-1)
                )
        elif self.enable_mamba_extra_buffer:
            new_slot = self._alloc_mamba_slot()
            if new_slot is None:
                return _skip_cache_unfinished_req(req)
            mamba_value_donated = self.req_to_token_pool.donate_mamba_ping_pong_slot(
                req, new_slot
            )
        else:
            mamba_value_donated = self._alloc_mamba_slot()
            if mamba_value_donated is None:
                return _skip_cache_unfinished_req(req)
            # mamba_pool is a pure PHYSICAL store; translate both slot ids
            # virtual->physical (identity for the non-unified memory pool) before the copy.
            translate = self.req_to_token_pool.translate_mamba_indices
            self.req_to_token_pool.mamba_pool.copy_from(
                translate(req.mamba_pool_idx.unsqueeze(0)),
                translate(mamba_value_donated),
            )

        result = self.insert(
            InsertParams(
                key=RadixKey(page_aligned_token_ids, req.extra_key),
                value=page_aligned_kv_indices,
                mamba_value=mamba_value_donated,
                prev_prefix_len=req.cache_protected_len,
                chunked=chunked,
            )
        )
        new_prefix_len, mamba_exist = result.prefix_len, result.mamba_exist
        if mamba_exist:
            self._free_mamba_value(mamba_value_donated)

        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(
            MatchPrefixParams(key=RadixKey(page_aligned_token_ids, req.extra_key))
        )
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )

        if not mamba_exist:
            assert torch.equal(new_last_node.mamba_value, mamba_value_donated)

        assert (
            req.cache_protected_len <= len(new_indices) + self.page_size - 1
        ), f"{req.cache_protected_len=}, {len(new_indices)=}, {len(page_aligned_token_ids)=}, {mamba_exist=}"
        assert new_prefix_len <= len(
            new_indices
        ), f"{new_prefix_len=}, {len(new_indices)=}"

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        # NOTE: this is needed for both page_size == 1 and page_size > 1
        req.prefix_indices = torch.cat(
            [new_indices, kv_indices_orig[len(new_indices) :]]
        )
        req.cache_protected_len = len(new_indices)
        req.mamba_last_track_seqlen = None
        req.last_node = new_last_node

    def pretty_print(self) -> None:
        self._print_helper(self.root_node, 0)
        total_size, total_mamba_size = self._total_size_helper()
        print(f"#full_tokens: {total_size}, #mamba_num: {total_mamba_size}")

    def total_size(self) -> Tuple[int, int]:
        return self._total_size_helper()

    def _evict_leaf_node(
        self, x: TreeNode, is_evict_mamba: bool
    ) -> Tuple[int, int, TreeNode, TreeNode]:
        assert (
            x.full_lock_ref == 0 and x.mamba_lock_ref == 0
        ), f"evict leaf node invalid with {x.id=} {x.full_lock_ref=} {x.mamba_lock_ref=}"

        assert x.mamba_value is not None, f"leaf node mamba value is not None, {x.id=}"
        # 1. a leaf node, free full tokens and mamba. Record both pools'
        # \\bar L_i (paper §sec:design-formalism-offline): the per-segment
        # token count is the re-prefill length on KV side and the
        # chunked-scan distance to rebuild on the recurrent side.
        self._record_remove_event(x)
        from sglang.srt.mem_cache.common import (
            record_recovery_len_kv,
            record_recovery_len_rec,
        )

        record_recovery_len_kv(self, len(x.value))
        record_recovery_len_rec(self, len(x.value))
        # Tree values are page-aligned copies of a kv row: page-exact segment.
        self.token_to_kv_pool_allocator.free_segment(x.value, start_pos=0)
        full_num_evicted = len(x.value)
        self._free_mamba_value(x.mamba_value)
        mamba_num_evicted = len(x.mamba_value)

        # 2. get the next node, update the lru lists
        if is_evict_mamba:
            x_next = self.mamba_lru_list.get_prev_no_lock(x)
        else:
            x_next = self.full_lru_list.get_prev_leaf_no_lock(x)
        self.full_lru_list.remove_node(x)
        self.mamba_lru_list.remove_node(x)

        # 3. delete the leaf node
        self._delete_leaf(x)

        # 4. Iteratively delete tombstone leaves to maintain invariant that leaf nodes are not tombstone
        x, leaf_full_num_evicted = self._iteratively_delete_tombstone_leaf(x)
        full_num_evicted += leaf_full_num_evicted
        return full_num_evicted, mamba_num_evicted, x, x_next

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        full_num_evicted = 0
        mamba_num_evicted = 0

        if params.num_tokens > 0:
            full_num_evicted = self.evict_full(params.num_tokens)
        if params.mamba_num > 0:
            mamba_num_evicted = self.evict_mamba(params.mamba_num)

        return EvictResult(
            num_tokens_evicted=full_num_evicted, mamba_num_evicted=mamba_num_evicted
        )

    def _should_use_lpb(self) -> bool:
        """True iff this cache evicts by LPB (loss-per-byte) rather
        than recency-LRU. Single source of truth for both
        `evict_full` and `evict_mamba`, mirroring plain `RadixCache`'s
        `isinstance(self.eviction_strategy, LPBStrategy)` gate: driven
        by the `--eviction-policy` boot flag, NOT an env var, so the
        hybrid path matches the KV-only path."""
        return self.eviction_policy == "lpb"

    def _lpb_build_eviction_heap(self) -> list:
        """Paper §sec:design-l1 (eq:lpb-lru): build a min-heap of all
        currently-evictable mamba nodes ordered by hits-per-byte
        (lowest first). One-shot O(n) heapify; subsequent picks are
        O(log n) heappop instead of re-scanning the LRU list each time.
        Stale entries (nodes whose mamba_value / lock state changes
        after the heap was built) are filtered at pop time by
        `_lpb_pop_eviction_victim`.

        Tie-break by `last_access_time` ascending is preserved by the
        tuple key — same semantics as the prior O(n) scanner.
        """
        h = []
        for node in self.mamba_lru_list.cache.values():
            if node.mamba_lock_ref > 0 or node.mamba_value is None:
                continue
            h.append((
                node.eviction_priority(),
                node.last_access_time,
                node.id,  # final tiebreak for deterministic heap ordering
                node,
            ))
        heapq.heapify(h)
        return h

    def _lpb_pop_eviction_victim(self, heap: list) -> Optional[TreeNode]:
        """Pop the next valid eviction victim from the LPB heap.
        Skips entries whose underlying TreeNode is now locked, has had
        its mamba_value freed, or has left the mamba LRU list (e.g.
        evicted by a prior iteration). Returns None if the heap is
        exhausted."""
        while heap:
            _, _, _, node = heapq.heappop(heap)
            if node.mamba_lock_ref > 0 or node.mamba_value is None:
                continue
            if not self.mamba_lru_list.in_list(node):
                continue
            return node
        return None

    def _iter_mamba_victims(self):
        """Single source of truth for mamba victim ORDER (design.md
        §"Grow benefit and drain cost are both reuse-aware"). Pure-read
        lazy generator yielding the evictable mamba nodes, in the order
        eviction would pick them, under the active `--eviction-policy`.

        BOTH `evict_mamba` (which then frees them) and
        `_plan_mamba_eviction` / `predict_evict_cost_us(pool="mamba")`
        (which classify + price them) consume this, so the priced set
        is byte-identical to the evicted set by construction, mirroring
        the KV side's `_plan_full_eviction`.

        Order per policy:
          * LRU: `mamba_lru_list` tail (oldest) forward via the prev
            chain.
          * LPB: Phase 1 yields the contiguous tail run of cold
            (`hit_count == 0`) nodes first (LPB priority 0/size = 0, so
            they always sort first), walked O(1)/victim. The Phase-2
            heap over the remaining hit-bearing nodes is built LAZILY,
            only once the consumer pulls past the cold run, so a drain
            satisfied entirely by cold nodes never pays the O(n)
            heapify.

        Consumer contract: a node's successor pointer is captured
        BEFORE the node is yielded, so a consumer (`evict_mamba`) may
        free the just-yielded node before pulling the next without
        invalidating the walk. The captured successor is re-validated
        against `mamba_lru_list` membership before it is yielded, so a
        cascade that deletes it (a freed leaf sweeping tombstone
        parents) terminates the cold-tail walk exactly as the live walk
        would.
        """
        # Mamba eviction always uses LRU ordering (recency), even when
        # KV uses LPB. LPB's cost-per-byte normalization hurts mamba
        # because mamba's fixed bytes dilute long-prefix entries' value.
        # LRU's recency heuristic is better for mamba: active sessions
        # are recent (kept), finished sessions are old (evicted).
        node = self.mamba_lru_list.get_lru_no_lock()
        while node is not None:
            nxt = self.mamba_lru_list.get_prev_no_lock(node)
            yield node
            if nxt is None or not self.mamba_lru_list.in_list(nxt):
                return
            node = nxt

    def evict_mamba(self, mamba_num: int) -> int:
        """Evict mamba states. Returns the number of mamba states evicted.

        Consumes the shared `_iter_mamba_victims` generator (single
        source of truth, also consumed by `_plan_mamba_eviction` /
        `predict_evict_cost_us`), so the evicted set is byte-identical
        to the priced set, mirroring the KV-side `evict_full` /
        `_plan_full_eviction` pairing. For each yielded victim this
        frees either the mamba snapshot alone (internal node, leaving a
        KV tombstone) or KV + mamba (leaf, plus the tombstone-leaf
        cascade), until `mamba_num` slots are freed.

        Two policies, gated by `--eviction-policy lpb` (default lru):
          * Recency-LRU: oldest mamba node first.
          * LPB (hits-per-byte): cold (`hit_count == 0`) tail run first,
            then a heap over hit-bearing nodes — see
            `_iter_mamba_victims`.
        """
        if self.disable or mamba_num <= 0:
            return 0

        from sglang.srt.budgeter.cost_model import get_cost_curves, has_cost_curves

        # The per-victim LPB-loss telemetry below is consumed only by the
        # Budgeter. On a base server (hybrid model, --radix-eviction-policy lru,
        # no SGLANG_CSIGMA_*) there is no Budgeter and no calibrated cost model,
        # so skip the pricing rather than let get_cost_curves() fail-close and
        # crash the eviction path.
        track_loss = has_cost_curves()
        curves = get_cost_curves() if track_loss else None
        use_lpb = self._should_use_lpb()
        mamba_num_evicted = 0
        lpb_loss_us = 0.0
        for x in self._iter_mamba_victims():
            if mamba_num_evicted >= mamba_num:
                break
            assert x.mamba_value is not None, f"node has no mamba value, {x.id=}"
            assert (
                len(x.mamba_value) == 1
            ), f"node has abnormal mamba length, {x.id=}, {len(x.mamba_value)=}"
            assert x != self.root_node, f"root node is not evictable, {x.id=}"
            assert x.mamba_lock_ref == 0, f"node is in use by mamba kv indices, {x.id=}"

            # LPB loss of dropping this recurrent snapshot: n_b * (c_kv + c_m),
            # the full-prefix re-prefill needed to rebuild the mamba state
            # (a snapshot cannot be recovered from kept KV alone). n_b=0 =>
            # ~0 loss. Priced BEFORE eviction; matches predict_evict_cost_us.
            if track_loss:
                n_b = x.hits_in_window() if use_lpb else 1
                s_b = len(x.key) if x.key is not None else 0
                lpb_loss_us += n_b * (curves.c_kv_us(s_b) + curves.c_m_us(s_b))

            if len(x.children) > 0:
                from sglang.srt.mem_cache.common import record_recovery_len_rec

                record_recovery_len_rec(self, len(x.value))
                # 1. an internal node, free mamba tokens.
                self._free_mamba_value(x.mamba_value)
                mamba_num_evicted += len(x.mamba_value)
                self.mamba_lru_list.remove_node(x)
                self._tombstone_internal_node(x)
            else:
                _, mamba_evicted_delta, _, _ = self._evict_leaf_node(x, True)
                mamba_num_evicted += mamba_evicted_delta

        # Cumulative mamba-slot eviction count — the Budgeter's grow-side
        # eviction-rate signal (symmetric to the reuse-aware drain cost): a
        # pool actively shedding (hot) snapshots should be GROWN. KV's
        # admission-path tally is admission-only and misses cache eviction on
        # a mamba-bound workload; this counts the actual mamba cache
        # evictions. BudgetAgent deltas it per tick.
        self._cumulative_evicted_mamba_slots += mamba_num_evicted
        self._cumulative_evicted_mamba_lpb_loss += lpb_loss_us
        return mamba_num_evicted

    def _lpb_build_full_eviction_heap(self) -> list:
        """Heap of evictable LEAF nodes in full_lru_list, ordered by LPB
        priority (lowest first). Parallel to _lpb_build_eviction_heap
        but for full (KV) eviction. Only leaves are eligible because
        evict_full evicts leaves and walks parent-becomes-leaf
        chains; LPB just re-orders the leaf-selection step.
        """
        h = []
        for node in self.full_lru_list.cache.values():
            if node.full_lock_ref > 0:
                continue
            if len(node.children) > 0:
                continue  # not a leaf
            h.append((
                node.eviction_priority(),
                node.last_access_time,
                node.id,
                node,
            ))
        heapq.heapify(h)
        return h

    def _lpb_pop_full_eviction_victim(self, heap: list) -> Optional[TreeNode]:
        """Pop next valid leaf-eviction victim. Skips stale entries
        (locked, off-list, or no-longer-a-leaf — the last can happen
        if eviction created new children via tombstoning, which is
        not currently the case for evict_full but defensive)."""
        while heap:
            _, _, _, node = heapq.heappop(heap)
            if node.full_lock_ref > 0:
                continue
            if not self.full_lru_list.in_list(node):
                continue
            if len(node.children) > 0:
                continue
            return node
        return None

    def _plan_full_eviction(self, full_num_tokens: int):
        """Single source of truth for full (KV) eviction (design.md
        §"Why exact c^evict"). Pure-read; returns
        `(victims, swept_tombstones)`:

          * `victims` — the real leaves / promoted-parents (in order)
            that `evict_full` passes to `_evict_leaf_node` to free
            `full_num_tokens` KV tokens. Each carries a mamba snapshot;
            evicting it loses KV + mamba.
          * `swept_tombstones` — the tombstone internal nodes
            (`mamba_value is None`, left by a prior `evict_mamba`) that
            `_evict_leaf_node` sweeps as a side effect via
            `_iteratively_delete_tombstone_leaf` when their last child
            is freed. They free KV only.

        BOTH `evict_full` (frees `victims`; the sweeps happen inside
        `_evict_leaf_node`) and `predict_evict_cost_us(pool="kv")` (sums
        recompute over `victims` + `swept_tombstones`) consume this — so
        the priced set is byte-identical to the evicted set, and the
        stop point matches (`evict_full` counts swept tokens via the
        `_evict_leaf_node` delta, so the planner counts them too).

        Victim order mirrors `evict_full`'s sort key: LRU = lowest
        `last_access_time` first; LPB = lowest `eviction_priority()`.
        Parent-promotion + tombstone cascade are simulated via
        `effective_children`. Pure-read: `evict_full` consumes the
        returned lists BEFORE mutating. Skip contract: `value is None`
        / zero-length leaves are skipped.
        """
        victims: list = []
        swept_tombstones: list = []
        if full_num_tokens <= 0:
            return victims, swept_tombstones
        use_lpb = self._should_use_lpb()

        def _leaf_key(node):
            if use_lpb:
                return (node.eviction_priority(), node.last_access_time, node.id)
            return (node.last_access_time, node.id)

        heap = []
        for node in self.full_lru_list.cache.values():
            if node.full_lock_ref > 0:
                continue
            if len(node.children) > 0:
                continue
            heap.append((_leaf_key(node), node))
        heapq.heapify(heap)

        effective_children: dict[int, int] = {}
        num_evicted = 0
        while heap and num_evicted < full_num_tokens:
            _key, x = heapq.heappop(heap)
            if x.value is None:
                continue
            L_evicted = len(x.value)
            if L_evicted == 0:
                continue
            victims.append(x)
            num_evicted += L_evicted

            # Parent-promotion + tombstone-sweep cascade, mirroring
            # `_evict_leaf_node` step 4 (`_iteratively_delete_tombstone_
            # leaf`): when a node's last evictable child is gone, walk
            # up. A REAL parent (`mamba_value is not None`) becomes an
            # evictable leaf — promote it into the heap. A TOMBSTONE
            # parent is swept by `_evict_leaf_node`, not a victim — so
            # we record it (frees KV → counts toward the demand, same
            # as `evict_full`'s delta) and cascade to ITS parent so a
            # real grandparent still promotes at the right moment.
            node = x
            while True:
                parent = node.parent
                if parent is None or parent is self.root_node:
                    break
                if parent.full_lock_ref != 0:
                    break
                key = id(parent)
                if key not in effective_children:
                    effective_children[key] = len(parent.children)
                effective_children[key] -= 1
                if effective_children[key] != 0:
                    break
                if parent.mamba_value is not None:
                    heapq.heappush(heap, (_leaf_key(parent), parent))
                    break
                # tombstone parent → swept internally; its KV frees too.
                swept_tombstones.append(parent)
                if parent.value is not None:
                    num_evicted += len(parent.value)
                node = parent

        return victims, swept_tombstones

    def _plan_mamba_eviction(self, mamba_num: int):
        """Pure-read victim plan for mamba-side eviction (design.md
        §"Grow benefit and drain cost are both reuse-aware, not
        active-only"), parallel to `_plan_full_eviction` for KV. Returns
        `(leaf_victims, internal_victims, swept_tombstones)`:

        Sole consumer is `predict_evict_cost_us(pool="mamba")`, which
        prices the exact set `evict_mamba` would free. Victim ORDER
        comes from the shared `_iter_mamba_victims` generator that
        `evict_mamba` also consumes, so the priced set cannot drift from
        the evicted set (single source of truth, mirroring the KV-side
        `_plan_full_eviction`). This method adds only the pure-read
        classification (internal-tombstone vs leaf vs swept-cascade)
        that pricing needs.

          * `internal_victims` — nodes that still have children when
            evicted; `evict_mamba` frees only their mamba snapshot and
            leaves a KV tombstone (`_tombstone_internal_node`). Cost
            `c_m(s_b)`; frees 1 mamba slot; the node stays in the tree.
          * `leaf_victims` — nodes with no remaining children;
            `evict_mamba` frees BOTH KV and mamba via `_evict_leaf_node`
            and deletes the node. Cost `c_kv(s_b) + c_m(s_b)`; frees 1
            mamba slot; triggers the tombstone-leaf cascade.
          * `swept_tombstones` — KV tombstones (mamba already gone) the
            leaf cascade (`_iteratively_delete_tombstone_leaf`) deletes
            when their last child is freed. Cost `c_kv(s_b)`; frees 0
            mamba slots (KV recompute only).

        The consumer sums `Σ n_b · c_i(s_b)` over these three sets, so
        the priced set is byte-identical to the evicted set.

        Victim order comes from `_iter_mamba_victims` (the same source
        `evict_mamba` consumes). Effective-children bookkeeping mirrors
        `_plan_full_eviction`: a leaf eviction (or a tombstone sweep)
        decrements its parent's child count, so a parent that loses its
        last child is classified as a leaf / swept exactly as the live
        walk would. Internal tombstoning does NOT delete the node, so it
        does not decrement its parent. Pure-read: consumes the generator
        without mutating; safe to call before `evict_mamba`.
        """
        leaf_victims: list = []
        internal_victims: list = []
        swept_tombstones: list = []
        if mamba_num <= 0:
            return leaf_victims, internal_victims, swept_tombstones

        # Shared victim ORDER (pure-read). The generator already yields
        # only unlocked, mamba-bearing, in-list nodes.
        order = list(self._iter_mamba_victims())

        # ---- Simulate the walk with effective-children bookkeeping ----
        eff_children: dict[int, int] = {}

        def _eff(node) -> int:
            k = id(node)
            if k not in eff_children:
                eff_children[k] = len(node.children)
            return eff_children[k]

        tombstoned_now: set = set()  # nodes we virtually tombstoned (mamba freed)
        deleted: set = set()         # nodes removed from the tree (leaf / swept)

        def _sweep_cascade(start_leaf) -> None:
            # Mirror `_iteratively_delete_tombstone_leaf`: walk up while
            # the parent is a childless KV tombstone (mamba already gone,
            # either an original tombstone or one we tombstoned above).
            node = start_leaf
            while True:
                parent = node.parent
                if parent is None or parent is self.root_node:
                    break
                if parent.full_lock_ref > 0:
                    break
                eff_children[id(parent)] = _eff(parent) - 1
                if eff_children[id(parent)] != 0:
                    break
                is_tombstone = (
                    parent.mamba_value is None or id(parent) in tombstoned_now
                )
                if not is_tombstone:
                    break
                swept_tombstones.append(parent)
                deleted.add(id(parent))
                node = parent

        collected = 0
        for x in order:
            if collected >= mamba_num:
                break
            if id(x) in deleted or id(x) in tombstoned_now:
                continue
            if _eff(x) > 0:
                # internal node → tombstone (frees mamba only); stays in tree.
                internal_victims.append(x)
                tombstoned_now.add(id(x))
                collected += 1  # len(mamba_value) == 1
            else:
                # leaf node → frees KV + mamba, deleted + cascade.
                leaf_victims.append(x)
                deleted.add(id(x))
                collected += 1
                _sweep_cascade(x)

        return leaf_victims, internal_victims, swept_tombstones

    def evict_full(self, full_num_tokens: int) -> int:
        """Evict full KV cache. Returns the number of tokens evicted.

        Victim selection is the shared `_plan_full_eviction` (single
        source of truth, also consumed by `predict_evict_cost_us`);
        this method consumes the plan — computed pure-read BEFORE any
        mutation — then frees each victim via `_evict_leaf_node` (which
        sweeps the planned tombstones internally).

        Two policies, gated by `--eviction-policy lpb` (default lru):
          * Recency-LRU baseline: least-recently-accessed leaf first.
          * LPB (hits-per-byte): lowest `eviction_priority()` first.
        """
        if self.disable or full_num_tokens <= 0:
            return 0

        from sglang.srt.budgeter.cost_model import get_cost_curves, has_cost_curves

        # Skip the Budgeter-only LPB-loss pricing when no cost model is
        # calibrated (base LRU serving); see the twin note in evict_mamba.
        track_loss = has_cost_curves()
        curves = get_cost_curves() if track_loss else None
        use_lpb = self._should_use_lpb()
        victims, _swept = self._plan_full_eviction(full_num_tokens)
        full_num_evicted = 0
        lpb_loss_us = 0.0
        for x in victims:
            assert (
                x != self.root_node
            ), f"root node should not exist in full lru list, {x.id=}"
            # LPB loss BEFORE the node is freed: n_b * c_recompute(s_b), the
            # reuse-weighted re-prefill cost if this block is re-requested.
            # n_b=0 for never-reused cache => ~0 loss (the accurate signal that
            # a low-reuse pool is cheap to shrink; see predict_evict_cost_us).
            if track_loss:
                n_b = x.hits_in_window() if use_lpb else 1
                s_b = len(x.key) if x.key is not None else 0
                lpb_loss_us += n_b * (
                    curves.c_kv_us(s_b)
                    + (curves.c_m_us(s_b) if x.mamba_value is not None else 0.0)
                )
            full_num_evicted_delta, _, _, _ = self._evict_leaf_node(x, False)
            full_num_evicted += full_num_evicted_delta

        # Cumulative KV-token cache-eviction count — the symmetric grow-side
        # signal for nb_m2k (grow KV when KV sheds hot prefixes). See the
        # mamba counterpart in `evict_mamba`. BudgetAgent deltas it per tick.
        self._cumulative_evicted_kv_tokens += full_num_evicted
        self._cumulative_evicted_kv_lpb_loss += lpb_loss_us
        return full_num_evicted

    def predict_evict_cost_us(self, num_tokens: int, pool: str = "kv") -> float:
        """Exact c^evict_i(X) predictor for the hybrid cache (design.md
        §"Shared cost model" "Why exact c^evict"). Sums
        `Σ n_b · c_i(s_b)` over the exact blocks the matching pool's
        eviction would pick to free `num_tokens`, via the same
        victim-selection plan eviction uses — so the priced set IS the
        evicted set.

        `pool="kv"`: mirrors `evict_full`. Each picked full-tree leaf
        carries both a KV value and a mamba snapshot; evicting it loses
        both, so its recompute is `c_kv(s_b) + c_m(s_b)` — the same
        `c_i` decomposition `eviction_priority()` uses for the LPB sort
        key. Tombstone internal nodes swept as a side effect free KV
        only, so they contribute `c_kv(s_b)` (no `c_m` — mamba already
        gone). `n_b` is `hits_in_window()` under LPB, else 1.

        `pool="mamba"`: cross_evict / Budgeter m2k-drain (src=mamba).
        `num_tokens` is the number of mamba SLOTS to free (each
        evictable node holds one). Prices the `_plan_mamba_eviction`
        mirror of `evict_mamba`: an internal victim loses only its mamba
        snapshot (`c_m`), a leaf victim loses both KV and mamba
        (`c_kv + c_m`), and a tombstone swept by the leaf cascade frees
        KV only (`c_kv`). `n_b` is `hits_in_window()` under LPB (so a
        hot cache is expensive to drain and a cold cache is ~free —
        Phase-1 cold nodes carry `n_b = 0`), else 1. This is the
        reuse-aware drain cost the Budgeter consumes via
        `snapshot["mamba_drain_cost_us"]` (design.md §"Grow benefit and
        drain cost are both reuse-aware, not active-only").

        Returns `+inf` when the cache cannot satisfy `num_tokens`
        (fail-closed). Does NOT acquire `_alloc_lock`; the caller (the
        Admitter decision) already holds it.
        """
        if pool not in ("kv", "mamba"):
            raise ValueError(
                f"MambaRadixCache.predict_evict_cost_us: unknown pool "
                f"{pool!r} (expected 'kv' or 'mamba')"
            )
        if num_tokens <= 0:
            return 0.0
        if self.disable:
            return float("inf")

        from sglang.srt.budgeter.cost_model import get_cost_curves
        curves = get_cost_curves()
        use_lpb = self._should_use_lpb()

        if pool == "mamba":
            leaf_v, internal_v, swept = self._plan_mamba_eviction(num_tokens)
            mamba_freed = len(leaf_v) + len(internal_v)
            if mamba_freed < num_tokens:
                return float("inf")
            total_cost_ms = 0.0
            # An internal victim whose KV is ALSO freed this pass (its leaf
            # cascade tombstones it into `swept`) has its c_kv counted there,
            # so it contributes c_m here and c_kv via `swept` — together the
            # whole-prefix total. But an internal victim whose KV STAYS (live
            # descendants keep it) is never swept, so it would be priced c_m
            # alone — which collapses to ~0 under κ_M = 0 and lets the m2k
            # drain over-harvest a hot snapshot. Recovering that
            # snapshot still needs a full prefix re-prefill (attention and
            # recurrent layers interleave; the kept KV cannot stand in for the
            # rest of the forward), so price it by the whole-prefix total
            # c_kv + c_m, matching a leaf victim and eviction_priority().
            swept_ids = {id(t) for t in swept}
            for x in internal_v:
                s_b = len(x.key) if x.key is not None else 0
                n_b = x.hits_in_window() if use_lpb else 1
                c_i_ms = curves.c_m_ms(s_b)
                if id(x) not in swept_ids:        # KV stays → full re-prefill
                    c_i_ms += curves.c_kv_ms(s_b)
                total_cost_ms += n_b * c_i_ms
            for x in leaf_v:
                s_b = len(x.key) if x.key is not None else 0
                n_b = x.hits_in_window() if use_lpb else 1
                total_cost_ms += n_b * (curves.c_kv_ms(s_b) + curves.c_m_ms(s_b))
            for t in swept:
                s_b = len(t.key) if t.key is not None else 0
                n_b = t.hits_in_window() if use_lpb else 1
                total_cost_ms += n_b * curves.c_kv_ms(s_b)   # KV only
            return total_cost_ms * 1000.0

        victims, swept_tombstones = self._plan_full_eviction(num_tokens)
        num_evicted = 0
        total_cost_ms = 0.0
        for x in victims:
            s_b = len(x.key) if x.key is not None else 0
            c_i_ms = curves.c_kv_ms(s_b)          # KV value present
            if x.mamba_value is not None:
                c_i_ms += curves.c_m_ms(s_b)       # + mamba snapshot
            n_b = x.hits_in_window() if use_lpb else 1
            total_cost_ms += n_b * c_i_ms
            num_evicted += len(x.value)
        for t in swept_tombstones:
            # tombstone frees KV only (mamba already gone) → c_kv.
            s_b = len(t.key) if t.key is not None else 0
            n_b = t.hits_in_window() if use_lpb else 1
            total_cost_ms += n_b * curves.c_kv_ms(s_b)
            if t.value is not None:
                num_evicted += len(t.value)

        if num_evicted < num_tokens:
            return float("inf")
        return total_cost_ms * 1000.0

    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        """
        Increment the lock reference count for the node.
        It locks the full_lock_ref for nodes between the [last node, root), exclusive.
        It locks the mamba_lock_ref for current node if its mamba_value exists.
        """
        if self.disable:
            return IncLockRefResult()

        # protect mamba value in current node if it exists
        if node.mamba_value is not None:
            if node.mamba_lock_ref == 0:
                self.mamba_evictable_size_ -= len(node.mamba_value)
                self.mamba_protected_size_ += len(node.mamba_value)
            node.mamba_lock_ref += 1

        while node != self.root_node:
            # lock full from node to root
            assert (
                node.full_lock_ref >= 0
            ), f"inc_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 0:
                self.full_evictable_size_ -= len(node.value)
                self.full_protected_size_ += len(node.value)
            node.full_lock_ref += 1
            node = node.parent
        return IncLockRefResult()

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        """
        Decrement the lock reference count for the node.
        It unlocks the full_lock_ref for nodes between the [last node, root), exclusive.
        It unlocks the mamba_lock_ref for current node if its mamba_value exists.
        """
        if self.disable:
            return DecLockRefResult()

        if node.mamba_value is not None:
            assert (
                node.mamba_lock_ref > 0
            ), f"dec_lock_ref on node with {node.mamba_lock_ref=}, {node.id=}"
            if node.mamba_lock_ref == 1:
                self.mamba_evictable_size_ += len(node.mamba_value)
                self.mamba_protected_size_ -= len(node.mamba_value)
            node.mamba_lock_ref -= 1

        while node != self.root_node:
            assert (
                node.full_lock_ref > 0
            ), f"dec_lock_ref on node with {node.full_lock_ref=}, {node.id=}"
            if node.full_lock_ref == 1:
                self.full_evictable_size_ += len(node.value)
                self.full_protected_size_ -= len(node.value)
            node.full_lock_ref -= 1
            node = node.parent

        return DecLockRefResult()

    def sanity_check(self):
        if self.disable:
            return
        self.full_lru_list.sanity_check(self)
        self.mamba_lru_list.sanity_check(self)

    def evictable_size(self) -> Tuple[int, int]:
        # Note: use full_evictable_size() and mamba_evictable_size() instead.
        raise NotImplementedError

    def full_evictable_size(self) -> int:
        return self.full_evictable_size_

    def mamba_evictable_size(self) -> int:
        return self.mamba_evictable_size_

    def protected_size(self) -> Tuple[int, int]:
        # Note: use full_protected_size() and mamba_protected_size() instead.
        raise NotImplementedError

    def full_protected_size(self) -> int:
        # protected size refers to the size of the full cache that is locked
        return self.full_protected_size_

    def mamba_protected_size(self) -> int:
        # protected size refers to the size of the mamba cache that is locked
        return self.mamba_protected_size_

    def all_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values) if len(values) > 0 else torch.tensor([])

    def all_mamba_values_flatten(self) -> torch.Tensor:
        values = []

        def _dfs_helper(node: TreeNode):
            if node.mamba_value is not None:
                values.append(node.mamba_value)
            for _, child in node.children.items():
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values) if len(values) > 0 else torch.tensor([])

    def available_and_evictable_str(self) -> str:
        full_available_size = self.token_to_kv_pool_allocator.available_size()
        full_evictable_size = self.full_evictable_size()
        return (
            f"Available full tokens: {full_available_size + full_evictable_size} ({full_available_size=} + {full_evictable_size=})\n"
            f"Full LRU list evictable size: {self.full_lru_list.sanity_check_evictable_size()}\n"
        )

    ##### Internal Helper Functions #####

    def _alloc_mamba_slot(self) -> Optional[torch.Tensor]:
        pool = self.req_to_token_pool.mamba_allocator
        slot = self.req_to_token_pool.mamba_allocator.alloc(1)
        if slot is not None:
            return slot
        self.evict(EvictParams(num_tokens=0, mamba_num=1))
        slot = self.req_to_token_pool.mamba_allocator.alloc(1)
        if slot is not None:
            return slot
        if self._mamba_grow_hook is not None and self._mamba_grow_hook(1):
            slot = self.req_to_token_pool.mamba_allocator.alloc(1)
        return slot

    @property
    def int8_ckpt_pool(self):
        """The int8 checkpoint pool, or None when --enable-int8-mamba-checkpoint is off.
        When enabled, radix-cached mamba states live HERE (int8), not in the active
        bf16 pool -> ~2x cached-prefix capacity at fixed memory."""
        return getattr(self.req_to_token_pool, "mamba_ckpt_pool", None)

    def _alloc_int8_ckpt_slot(self) -> torch.Tensor:
        """Allocate one int8 checkpoint slot, evicting cached states if the pool is full."""
        slot = self.int8_ckpt_pool.alloc(1)
        if slot is None:
            self.evict(EvictParams(num_tokens=0, mamba_num=1))
            slot = self.int8_ckpt_pool.alloc(1)
            assert slot is not None, "Can not alloc int8 mamba checkpoint slot"
        return slot

    def _commit_int8_checkpoint(self, active_slots: torch.Tensor) -> torch.Tensor:
        """Quantize the active-pool state at ``active_slots`` into a fresh int8
        checkpoint slot and return that slot. Strategy-agnostic donate hook: both
        no_buffer (copy_from) and extra_buffer (ping-pong) converge here. The caller
        frees ``active_slots`` separately."""
        ckpt_slot = self._alloc_int8_ckpt_slot()
        self.int8_ckpt_pool.store_from_active(
            self.req_to_token_pool.mamba_pool, active_slots, ckpt_slot
        )
        return ckpt_slot

    def _free_mamba_value(self, mamba_value: torch.Tensor) -> None:
        """Free a node's mamba_value to the right allocator (int8 ckpt pool or the
        active mamba allocator)."""
        if self.int8_ckpt_pool is not None:
            self.int8_ckpt_pool.free(mamba_value)
        else:
            self.req_to_token_pool.mamba_allocator.free(mamba_value)

    def _match_prefix_helper(
        self, key: RadixKey
    ) -> Tuple[List[torch.Tensor], TreeNode, int]:
        """
        Mamba prefix matching helper. It factors in the sliding window size such that
        the matched node is guaranteed to either 1. connected to root without mamba tombstone,
        or 2. the number of matching tokens from the matched node to the last mamba tombstone
        node is greater than or equal to the sliding window size.
        """
        node = self.root_node
        child_key = key.child_key(self.page_size)

        value: List[torch.Tensor] = []
        best_value_len = 0
        best_last_node = node
        # Paper §sec:design-l1: under LPB, record a hit on every visited
        # node so eviction_priority() reflects recent value to the
        # workload (without this, hit_count never increments). Skip the
        # deque append in LRU/other modes to keep them zero-overhead,
        # mirroring `RadixCache._match_prefix_helper`.
        lpb_active = self._should_use_lpb()
        hit_path: List[TreeNode] = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            # update best_value_len and best_last_node if needed
            if node.mamba_value is not None:
                best_value_len = len(value)
                best_last_node = node

            prefix_len = child.key.match(key, page_size=self.page_size)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                if lpb_active:
                    hit_path.append(new_node)
                break
            else:
                value.append(child.value)
                node = child
                if lpb_active:
                    hit_path.append(child)
                key = key[prefix_len:]

                if len(key):
                    child_key = key.child_key(self.page_size)
        # handle best_value_len and best_last_node, for the case that last node is fully matched
        if node.mamba_value is not None:
            best_value_len = len(value)
            best_last_node = node

        # Record hits for every node we visited in the matching walk.
        # Only do this under LPB and when there was at least some prefix
        # match — zero-length matches don't credit any node.
        if lpb_active and best_value_len > 0:
            for n in hit_path:
                n.record_hit()

        return value, best_last_node, best_value_len

    def _match_pre_processor(self, params: MatchPrefixParams) -> Optional[RadixKey]:
        """Preprocess the key before matching."""
        key = params.key

        if self.disable or len(key) == 0:
            return None

        return key

    def _match_post_processor(
        self,
        params: MatchPrefixParams,
        value: List[torch.Tensor],
        last_node: TreeNode,
        best_value_len: int,
    ) -> MatchResult:
        """Post-process the matched result."""
        cow_mamba = params.cow_mamba
        req = params.req

        # Full KV of the whole matched path is reused as prefix, so refresh the entire
        # chain (nodes closer to root end up least recently used, evicted first).
        node_update = last_node
        self.full_lru_list.reset_node_and_parents_mru(node_update, self.root_node)
        # Mamba only consumes last_node's state (cf. inc_lock_ref, which locks just this
        # node's mamba_value). Refreshing ancestors would keep a whole session's states
        # adjacent in the mamba LRU and evict cold sessions wholesale; touch only the used
        # state so older leaves survive.
        if last_node is not self.root_node and last_node.mamba_value is not None:
            self.mamba_lru_list.reset_node_mru(last_node)

        # This last_access_time is for sanity check, can be deleted after validation in production
        cur_time = get_last_access_time()
        while node_update:
            node_update.last_access_time = cur_time
            cur_time -= (
                0.00001  # assuming less than 100000 nodes in a branch of the tree
            )
            node_update = node_update.parent

        # Calculate the branching point. It is defined as the last aligned position that
        # does not have a mamba value.
        if len(value) > best_value_len:
            chunk_aligned_seqlen = (
                sum(len(v) for v in value) // self.mamba_cache_chunk_size
            ) * self.mamba_cache_chunk_size
            mamba_branching_seqlen = (
                chunk_aligned_seqlen if chunk_aligned_seqlen > 0 else None
            )
        else:
            mamba_branching_seqlen = None

        # Defer COW to forward stream: record source index, allocate destination
        if cow_mamba and last_node.mamba_value is not None:
            if req.mamba_pool_idx is None:
                dst_index = self._cow_mamba_slot_or_none(last_node)
                if dst_index is None:
                    return self._no_mamba_match_result()
                req.mamba_pool_idx = dst_index[0]
            req.mamba_cow_src_index = last_node.mamba_value
            req.mamba_needs_clear = False

        value = value[:best_value_len]
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
            best_match_node=last_node,
            mamba_branching_seqlen=mamba_branching_seqlen,
        )

    def _cow_mamba_slot_or_none(self, last_node: TreeNode) -> Optional[torch.Tensor]:
        """Acquire one mamba slot to copy `last_node.mamba_value` into, or None.

        Order, mirroring `_fork_mamba_with_recovery`: alloc → (None) lock
        `last_node` + evict an unlocked cold-cache slot + alloc → (None) grow
        mamba from KV via `_mamba_grow_hook` + alloc → (None). The caller
        degrades to a mamba cache miss on None; this never asserts.
        """
        pool = self.req_to_token_pool.mamba_pool
        dst_index = self.req_to_token_pool.mamba_allocator.alloc(1)
        if dst_index is not None:
            return dst_index
        # Protect last_node so the evict cannot reclaim the slot we are copying.
        self.inc_lock_ref(last_node)
        self.evict(EvictParams(num_tokens=0, mamba_num=1))
        dst_index = self.req_to_token_pool.mamba_allocator.alloc(1)
        if dst_index is None and self._mamba_grow_hook is not None:
            if self._mamba_grow_hook(1):
                dst_index = self.req_to_token_pool.mamba_allocator.alloc(1)
        self.dec_lock_ref(last_node)
        return dst_index

    def _no_mamba_match_result(self) -> MatchResult:
        """The empty / root match: no reusable KV or mamba prefix for this
        request. Same shape as `match_prefix` on an empty key, so the request
        re-prefills from scratch. Used when a COW copy cannot be allocated."""
        return MatchResult(
            device_indices=torch.empty((0,), dtype=torch.int64, device=self.device),
            last_device_node=self.root_node,
            last_host_node=self.root_node,
            best_match_node=self.root_node,
        )

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int) -> TreeNode:
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {key[split_len:].child_key(self.page_size): child}
        new_node.parent = child.parent
        new_node.mamba_value = None  # mamba cache can not be split
        new_node.full_lock_ref = child.full_lock_ref
        new_node.mamba_lock_ref = 0
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].clone()
        # LPB sliding-window hit signal belongs to the shared-prefix
        # segment (`new_node`); the divergent tail (`child`) starts
        # fresh. Both `hit_count` (Phase-1 tail walk in `evict_mamba`)
        # and `_hit_times` (`eviction_priority` denominator) carry over
        # (paper §sec:design-l1 eq:lpb-lru `n_b` is per-block).
        new_node.hit_count = child.hit_count
        new_node._hit_times = child._hit_times
        child.hit_count = 0
        child._hit_times = collections.deque(
            maxlen=TreeNode.lpb_hit_deque_maxlen
        )

        # child time should be later than the new parent's time in the full LRU
        child.last_access_time = get_last_access_time()

        # A split does not change the set of live mamba states (child keeps its value,
        # new_node is a mamba tombstone), so the mamba LRU is left untouched — only the
        # full LRU reorders around the new intermediate node.
        self.full_lru_list.remove_node(child)
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        new_node.parent.children[key.child_key(self.page_size)] = new_node
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        # insert the new node and child into the full lru list, insert
        # parent first so that parent is after child in the lru list
        self.full_lru_list.insert_mru(new_node)
        self.full_lru_list.insert_mru(child)
        return new_node

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value,
        mamba_value,
        chunked: bool = False,
        prev_prefix_len: int = 0,
    ) -> Tuple[int, bool]:
        # Refresh the full LRU from root to leaf (the whole path is reused as prefix).
        # The mamba states of these existing nodes were not recomputed this insert, so
        # the mamba LRU is left untouched here; only genuinely new mamba states (the new
        # leaf / a revived tombstone below) are inserted.
        assert mamba_value is not None, "Mamba value should not be None here."
        node.last_access_time = get_last_access_time()
        if node != self.root_node:
            self.full_lru_list.reset_node_mru(node)
        if len(key) == 0:
            return 0, True

        child_key = key.child_key(self.page_size)

        total_prefix_length = 0
        # Track the deepest-snapshot depth during traversal so
        # that the value we return mirrors what _match_prefix_helper would
        # return (it stops updating best_value_len at the deepest mamba_value
        # node). Required for the cache_unfinished_req invariant
        # `insert.prefix_len <= len(match_prefix.device_indices)` to hold
        # when the path goes through tombstone-internal-nodes past the
        # deepest snapshot.
        deepest_snapshot_depth = (
            total_prefix_length if node.mamba_value is not None else 0
        )
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = get_last_access_time()
            self.full_lru_list.reset_node_mru(node)
            prefix_len = node.key.match(key, page_size=self.page_size)

            if prev_prefix_len < total_prefix_length + prefix_len:
                start = max(0, prev_prefix_len - total_prefix_length)
                # value sits at offset total_prefix_length of the kv row; match()
                # rounds prefix_len to page multiples, so frees never share a page.
                self.token_to_kv_pool_allocator.free_segment(
                    value[start:prefix_len],
                    start_pos=total_prefix_length + start,
                )

            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            # After potential split, `node` is the matched-up-to-here node.
            # Update deepest_snapshot_depth if this node carries a snapshot.
            if node.mamba_value is not None:
                deepest_snapshot_depth = total_prefix_length

            if len(key):
                child_key = key.child_key(self.page_size)

        # A freshly-inserted leaf always carries its mamba snapshot, so we never
        # create a tombstone leaf here. Tombstones arise only when mamba eviction
        # later drops a snapshot from an existing internal node (the `elif` below
        # and _tombstone_internal_node). Snapshot-bearing leaves preserve two
        # invariants: _evict_leaf_node asserts `mamba_value is not None`, and
        # match_prefix's `best_value_len` (which only advances at snapshot nodes)
        # stays consistent with the inserted depth.
        mamba_value_exist = False
        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            new_node.mamba_value = mamba_value
            self.full_lru_list.insert_mru(new_node)
            self.mamba_lru_list.insert_mru(new_node)
            self.mamba_evictable_size_ += len(mamba_value)
            node.children[child_key] = new_node
            self.full_evictable_size_ += len(value)
            self._record_store_event(new_node)
        elif node.mamba_value is None:  # add for mamba tombstone (or stay tombstone)
            if mamba_value is not None:
                node.mamba_value = mamba_value
                self.mamba_lru_list.insert_mru(node)
                self.mamba_evictable_size_ += len(mamba_value)
                deepest_snapshot_depth = total_prefix_length
            self.full_lru_list.reset_node_mru(node)
            node.last_access_time = get_last_access_time()
        else:  # mamba value already exists
            mamba_value_exist = True
            self.full_lru_list.reset_node_mru(node)
            node.last_access_time = get_last_access_time()
            deepest_snapshot_depth = total_prefix_length

        # match_prefix only advances `best_value_len` at snapshot-bearing nodes,
        # so a matched path can descend past the deepest snapshot through
        # tombstone-internal nodes that mamba eviction left behind. Return the
        # deepest-snapshot depth in that case so insert.prefix_len <=
        # len(match_prefix.device_indices) holds in cache_unfinished_req.
        if deepest_snapshot_depth < total_prefix_length:
            return deepest_snapshot_depth, mamba_value_exist
        return total_prefix_length, mamba_value_exist

    def _iteratively_delete_tombstone_leaf(
        self, node: TreeNode
    ) -> Tuple[TreeNode, int]:
        full_num_evicted = 0
        while node.parent.mamba_value is None and len(node.parent.children) == 0:
            # root node is not evictable
            if node.parent == self.root_node:
                break
            # if locked, means node is in use, skip
            if node.parent.full_lock_ref > 0:
                break
            assert (
                node.parent.mamba_lock_ref == 0
            ), f"tombstone mamba_lock_ref should always be 0, {node.parent.full_lock_ref=}, {node.parent.mamba_lock_ref=}, {node.parent.id=}"
            # delete tombstone node evicts full tokens
            self._record_remove_event(node.parent)
            self.token_to_kv_pool_allocator.free_segment(node.parent.value, start_pos=0)
            full_num_evicted += len(node.parent.value)
            self.full_lru_list.remove_node(node.parent)
            self._delete_tombstone_leaf(node.parent)
            node = node.parent

        return node, full_num_evicted

    def _delete_leaf(self, node: TreeNode) -> None:
        assert (
            node.mamba_value is not None
        ), f"Invariant violated: leaf node is a tombstone, {node.id=}"
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.full_evictable_size_ -= len(node.key)
        self.mamba_evictable_size_ -= len(node.mamba_value)

    def _tombstone_internal_node(self, node: TreeNode) -> None:
        assert len(node.children) != 0, f"Cannot tombstone a leaf node, {node.id=}"
        self.mamba_evictable_size_ -= len(node.mamba_value)
        node.mamba_value = None

    def _delete_tombstone_leaf(self, node: TreeNode) -> None:
        assert (
            node.mamba_value is None
        ), f"Deleting a unexpected non-tombstone leaf node, {node.id=}"
        assert len(node.children) == 0, f"leaf node has children, {node.id=}"
        key = node.key.child_key(self.page_size)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.full_evictable_size_ -= len(node.key)

    def _collect_nontombstone_nodes(self) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if cur_node.mamba_value is not None:
                ret_list.append(cur_node)
            stack.extend(cur_node.children.values())

        return ret_list

    def _collect_all_nodes(self) -> List[TreeNode]:
        ret_list = []
        stack = [self.root_node]
        while stack:
            cur_node = stack.pop()
            ret_list.append(cur_node)
            stack.extend(cur_node.children.values())
        return ret_list

    def _print_helper(self, node: TreeNode, indent: int) -> None:
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                f"[{current_node.id}]",
                len(current_node.key),
                f"fr={current_node.full_lock_ref}",
                f"mr={current_node.mamba_lock_ref}",
                f"fll={self.full_lru_list.in_list(current_node)}",
                f"mll={self.mamba_lru_list.in_list(current_node)}",
                f"mv={current_node.mamba_value}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == child.key.child_key(
                    self.page_size
                ), f"{key=}, {child.key.child_key(self.page_size)=}"

    def _total_size_helper(self) -> Tuple[int, int]:
        total_size = 0
        total_mamba_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            if current_node.mamba_value is not None:
                total_mamba_size += len(current_node.mamba_value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size, total_mamba_size
