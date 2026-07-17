"""
ChunkArena: shared-VA multi-pool allocator over CUDA VMM (paper §sec:design-l2).
  - One contiguous virtual-address range, divided into uniform-size chunks.
  - Each pool gets a contiguous *sub-range* of the arena (its own VA window).
  - A central pool of `cuMemCreate`-d physical handles is the actuator's
    only resource: handles can be unmapped from one pool's window and
    re-mapped into another, transferring bytes from one pool to another
    without disturbing the pools' tensor pointers.

This module exposes the data structure and the actuator (transfer_chunks).
Per-pool eviction policy and Layer 2's Lagrange planner sit on top.

Note on terminology:
  - "Pool" here is a Layer-2 pool (paged-KV / mamba / LoRA / prefix).
  - "Slot" is an index into a pool's VA sub-range, in chunk units.
  - "Handle" is a physical memory allocation (cuMemCreate output).
"""

import ctypes
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


CUDA = ctypes.CDLL("libcuda.so")

# Fault-injection hook for D1 race-detection test. When set, the
# pre-unmap sync is skipped — exposes the cuMemUnmap-vs-kernel race
# that the test then catches via cudaErrorIllegalAddress in a
# subprocess. NEVER set this in production.

CU_SUCCESS = 0
CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CUmemAllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _CUmemAllocFlags),
    ]


class _CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _CUmemLocation), ("flags", ctypes.c_int)]


_HANDLE = ctypes.c_ulonglong
_DPTR = ctypes.c_ulonglong


def _setup_argtypes():
    CUDA.cuInit.argtypes = [ctypes.c_uint]
    CUDA.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    CUDA.cuMemGetAllocationGranularity.argtypes = [
        ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_int
    ]
    CUDA.cuMemAddressReserve.argtypes = [
        ctypes.POINTER(_DPTR), ctypes.c_size_t, ctypes.c_size_t, _DPTR, ctypes.c_ulonglong
    ]
    CUDA.cuMemAddressFree.argtypes = [_DPTR, ctypes.c_size_t]
    CUDA.cuMemCreate.argtypes = [
        ctypes.POINTER(_HANDLE), ctypes.c_size_t, ctypes.c_void_p, ctypes.c_ulonglong
    ]
    CUDA.cuMemRelease.argtypes = [_HANDLE]
    CUDA.cuMemMap.argtypes = [_DPTR, ctypes.c_size_t, ctypes.c_size_t, _HANDLE, ctypes.c_ulonglong]
    CUDA.cuMemUnmap.argtypes = [_DPTR, ctypes.c_size_t]
    CUDA.cuMemSetAccess.argtypes = [_DPTR, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    CUDA.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    CUDA.cuCtxSynchronize.argtypes = []


_setup_argtypes()


def _check(rc: int, what: str) -> None:
    if rc != CU_SUCCESS:
        msg = ctypes.c_char_p()
        CUDA.cuGetErrorString(rc, ctypes.byref(msg))
        raise RuntimeError(f"{what} failed: {rc} {msg.value.decode() if msg.value else ''}")


class SharedHandlePool:
    """Owns a pool of cuMemCreate'd physical handles that may be shared
    between multiple ChunkArenas on the same device.

    Cross-arena (KV ↔ mamba) transfer needs both arenas to share a single
    bag of handles. `ChunkArena.__init__(external_handle_pool=...)`
    references this object instead of creating its own handle list.

    The pool can be created empty and grown incrementally: each arena
    calls `grow(n_handles_needed)` at its own init time, so the engine
    doesn't need to pre-compute a total handle budget across pools.

    **Thread-safety contract** (Phase 4 Admitter, audit_test_depth_phase4
    Concern 1): `grow()` and `cleanup()` are init-time / shutdown-time only.
    The free-handle list (`self.free`) is mutated at runtime by
    `ChunkArena.grow / shrink / shrink_explicit`, which are themselves
    invoked exclusively from `XPoolActuator._execute_async_locked()` —
    guarded by `XPoolActuator._fire_inflight`. Calling `SharedHandlePool.grow()`
    after the actuator has started would race with the actuator's fires.
    A latch (`self._frozen`) below makes that race fail loudly instead
    of silently corrupting state.
    """

    def __init__(
        self,
        device_id: int,
        chunk_size: int,
        n_handles: int = 0,
    ) -> None:
        self.device_id = device_id
        self.chunk_size = chunk_size

        prop = _CUmemAllocationProp()
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
        prop.requestedHandleTypes = 0
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device_id
        self._prop = prop  # kept for sanity-equality checks

        self.handles: List[int] = []
        # Free handles: indices into `self.handles` that are currently unmapped
        # *anywhere*. Both arenas pop/push on this list.
        self.free: List[int] = []

        # C-side arena_multi64.so pool indices reserved by participating
        # MultiTensorArenas. Each one calls `allocate_subpool_range(n)` to
        # claim a disjoint range; the pool tracks the watermark so two
        # arenas in the same process don't collide.
        self._next_subpool_idx: int = 0

        # Latches to True the first time freeze() is called (by XPoolActuator
        # at its own init). Any subsequent grow() raises — this is what
        # makes the thread-safety contract above noisy instead of silent.
        self._frozen: bool = False

        if n_handles > 0:
            self.grow(n_handles)

    def freeze(self) -> None:
        """Called by XPoolActuator after both arenas are wired. After this
        point any grow() call is a programming bug — fires from the actuator
        race the grow on `self.free` without protection.
        """
        self._frozen = True

    def grow(self, n_more: int) -> None:
        """Create `n_more` cuMemCreate'd handles, append to the pool."""
        if n_more <= 0:
            return
        if self._frozen:
            raise RuntimeError(
                "SharedHandlePool.grow() called after freeze(); fires from "
                "XPoolActuator would race the free-handle list. If you "
                "genuinely need to grow the shared pool at runtime, wrap "
                "the grow() in XPoolActuator._fire_inflight first."
            )
        for _ in range(n_more):
            h = _HANDLE(0)
            _check(CUDA.cuMemCreate(
                ctypes.byref(h), self.chunk_size, ctypes.byref(self._prop), 0),
                "cuMemCreate (SharedHandlePool.grow)")
            idx = len(self.handles)
            self.handles.append(h.value)
            self.free.append(idx)

    def allocate_subpool_range(self, n: int) -> int:
        """Reserve the next `n` C-side pool indices and return the start.

        The C-side arena_multi64.so allocator has 64 fixed pool slots
        (numbered 0..63). When two MultiTensorArenas live in one process
        they must use disjoint sub-ranges; this helper hands them out
        sequentially.
        """
        if n <= 0:
            raise ValueError(f"allocate_subpool_range n={n} must be > 0")
        offset = self._next_subpool_idx
        self._next_subpool_idx += n
        return offset

    def free_count(self) -> int:
        return len(self.free)

    def total_count(self) -> int:
        return len(self.handles)

    def cleanup(self) -> None:
        for h in self.handles:
            CUDA.cuMemRelease(h)
        self.handles.clear()
        self.free.clear()
        self._next_subpool_idx = 0


@dataclass
class _PoolState:
    name: str
    va_base: int       # start of this pool's VA sub-range
    n_slots: int       # capacity in chunks
    # mapped[i] is the handle-index currently bound at slot i, or None.
    mapped: List[Optional[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.mapped = [None] * self.n_slots

    def mapped_count(self) -> int:
        return sum(1 for h in self.mapped if h is not None)

    def first_free_slot(self) -> Optional[int]:
        for i, h in enumerate(self.mapped):
            if h is None:
                return i
        return None

    def last_mapped_slot(self) -> Optional[int]:
        for i in range(self.n_slots - 1, -1, -1):
            if self.mapped[i] is not None:
                return i
        return None


class ChunkArena:
    """Shared-VA multi-pool allocator over CUDA Virtual Memory Management.

    Reserves one contiguous VA range of size `chunk_size * sum(pool_capacities)`,
    creates `n_handles` physical handles, and lets a budgeter move handles
    between per-pool VA sub-ranges via `transfer_chunks`.

    Tensor pointers each pool exposes (= the VA base of its sub-range)
    are stable for the lifetime of the arena.
    """

    def __init__(
        self,
        device_id: int,
        chunk_size: int,
        n_handles: int,
        pool_capacities: List[Tuple[str, int]],
        external_handle_pool: Optional["SharedHandlePool"] = None,
    ) -> None:
        """
        Args:
            device_id: CUDA device for cuMemCreate.
            chunk_size: bytes per chunk; must be a multiple of recommended granularity.
            n_handles: total physical handles available across all pools.
                Ignored when `external_handle_pool` is provided.
            pool_capacities: list of (pool_name, max_chunks_in_pool).
                The sum of max_chunks may exceed n_handles (over-provisioned VA).
            external_handle_pool: if provided, this arena's handle list and
                free-list are aliased to the shared pool. Two ChunkArenas
                with the same external pool can transfer handles between
                each other via `cross_arena_transfer`.
        """
        self.device_id = device_id
        self.chunk_size = chunk_size

        # Build allocation prop for handle creation.
        self._prop = _CUmemAllocationProp()
        self._prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
        self._prop.requestedHandleTypes = 0
        self._prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        self._prop.location.id = device_id

        # Verify chunk_size is a multiple of recommended granularity.
        g = ctypes.c_size_t()
        _check(CUDA.cuMemGetAllocationGranularity(
            ctypes.byref(g), ctypes.byref(self._prop), CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
            "cuMemGetAllocationGranularity")
        if chunk_size % g.value != 0:
            raise ValueError(
                f"chunk_size {chunk_size} not a multiple of recommended granularity {g.value}"
            )

        # Reserve one big VA range.
        total_chunks = sum(cap for _, cap in pool_capacities)
        self.total_va_size = chunk_size * total_chunks
        ptr = _DPTR(0)
        _check(CUDA.cuMemAddressReserve(
            ctypes.byref(ptr), self.total_va_size, 0, 0, 0), "cuMemAddressReserve")
        self.va_base = ptr.value

        # Handle ownership: either we create our own (single-pool mode)
        # or alias an external SharedHandlePool (cross-pool mode, the
        # path the inter-pool actuator uses).
        if external_handle_pool is None:
            self._owned_handles: List[int] = []
            for _ in range(n_handles):
                h = _HANDLE(0)
                _check(CUDA.cuMemCreate(
                    ctypes.byref(h), chunk_size, ctypes.byref(self._prop), 0), "cuMemCreate")
                self._owned_handles.append(h.value)
            self._handles: List[int] = self._owned_handles
            self._free_handles: List[int] = list(range(n_handles))
            self._external_pool: Optional["SharedHandlePool"] = None
        else:
            if external_handle_pool.chunk_size != chunk_size:
                raise ValueError(
                    f"external pool chunk_size {external_handle_pool.chunk_size} "
                    f"!= arena chunk_size {chunk_size}"
                )
            if external_handle_pool.device_id != device_id:
                raise ValueError(
                    f"external pool device {external_handle_pool.device_id} "
                    f"!= arena device {device_id}"
                )
            # Ensure the shared pool has enough free handles for this
            # arena's full grow path (init plus any future planner-driven
            # growth, depending on caller). If it doesn't, grow it. If it
            # already does (e.g., pre-sized at construction or shared
            # with a peer arena that has spare handles), no-op.
            shortfall = n_handles - external_handle_pool.free_count()
            if shortfall > 0:
                external_handle_pool.grow(shortfall)
            self._owned_handles = []
            self._handles = external_handle_pool.handles
            self._free_handles = external_handle_pool.free
            self._external_pool = external_handle_pool

        # Carve VA sub-ranges per pool.
        self.pools: dict[str, _PoolState] = {}
        offset = 0
        for name, cap in pool_capacities:
            self.pools[name] = _PoolState(
                name=name, va_base=self.va_base + offset * chunk_size, n_slots=cap)
            offset += cap

        # Access descriptor (re-used).
        self._desc = _CUmemAccessDesc()
        self._desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        self._desc.location.id = device_id
        self._desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE

    # -- helpers --------------------------------------------------------

    def _map_handle_to_slot(self, pool: _PoolState, slot: int, handle_idx: int) -> None:
        """Map one handle. Does NOT call cuMemSetAccess (caller batches it)."""
        va = pool.va_base + slot * self.chunk_size
        _check(CUDA.cuMemMap(va, self.chunk_size, 0, self._handles[handle_idx], 0),
               f"cuMemMap pool={pool.name} slot={slot}")
        pool.mapped[slot] = handle_idx

    def _unmap_slot(self, pool: _PoolState, slot: int) -> int:
        if pool.mapped[slot] is None:
            raise RuntimeError(f"pool={pool.name} slot={slot} not mapped")
        handle_idx = pool.mapped[slot]
        va = pool.va_base + slot * self.chunk_size
        _check(CUDA.cuMemUnmap(va, self.chunk_size),
               f"cuMemUnmap pool={pool.name} slot={slot}")
        pool.mapped[slot] = None
        return handle_idx

    @staticmethod
    def _runs(sorted_slots):
        """Yield (start, end) inclusive pairs for maximal contiguous runs
        in a sorted unique list of ints."""
        if not sorted_slots:
            return
        run_start = sorted_slots[0]
        prev = run_start
        for s in sorted_slots[1:]:
            if s == prev + 1:
                prev = s
                continue
            yield (run_start, prev)
            run_start = s
            prev = s
        yield (run_start, prev)

    def _unmap_slots_batched(self, pool: _PoolState, slots) -> list:
        """Unmap a set of slots, batching contiguous runs into single
        cuMemUnmap calls. Returns the freed handle indices in slot order.

        `slots` may include out-of-range and already-None slots; those are
        skipped (same forgiving semantic as ``shrink_explicit``).

        Safety contract — NO defensive synchronize before unmap. The
        caller's layer-0 invariant ("fire_planner only picks free pages
        that no in-flight req's state_indices references") is the sole
        safety mechanism. A1 evidence
        (dev/interlayer/0_page_state_machine/step1_stream_isolated_unmap)
        shows raw cuMemUnmap concurrent with captured Triton-graph
        replays of index-gated kernels is safe given that invariant.
        If layer-0 is violated, the next captured-graph replay touching
        the unmapped VA will fault with cudaErrorIllegalAddress —
        fail-fast diagnosis is the chosen design point per design.md
        §"Transfer protocol" Stage 3.
        """
        valid = sorted({int(s) for s in slots
                        if 0 <= int(s) < pool.n_slots
                        and pool.mapped[int(s)] is not None})
        if not valid:
            return []
        freed = [pool.mapped[s] for s in valid]
        for run_start, run_end in self._runs(valid):
            run_len = run_end - run_start + 1
            va = pool.va_base + run_start * self.chunk_size
            _check(CUDA.cuMemUnmap(va, run_len * self.chunk_size),
                   f"cuMemUnmap pool={pool.name} slots=[{run_start},{run_end}]")
            for s in range(run_start, run_end + 1):
                pool.mapped[s] = None
        return freed

    def _setaccess_slots_batched(self, pool: _PoolState, slots) -> None:
        """cuMemSetAccess on a set of just-mapped slots, batching contiguous
        runs. Caller must guarantee every slot is currently mapped."""
        if not slots:
            return
        sorted_slots = sorted({int(s) for s in slots})
        for run_start, run_end in self._runs(sorted_slots):
            run_len = run_end - run_start + 1
            va = pool.va_base + run_start * self.chunk_size
            _check(CUDA.cuMemSetAccess(va, run_len * self.chunk_size,
                                        ctypes.byref(self._desc), 1),
                   f"cuMemSetAccess pool={pool.name} slots=[{run_start},{run_end}]")

    # -- API ------------------------------------------------------------

    def pool_va_base(self, pool_name: str) -> int:
        """Start address of the pool's VA sub-range (stable forever)."""
        return self.pools[pool_name].va_base

    def pool_va_size(self, pool_name: str) -> int:
        """Total VA reservation for the pool (= max-chunks * chunk_size)."""
        return self.pools[pool_name].n_slots * self.chunk_size

    def pool_mapped_chunks(self, pool_name: str) -> int:
        """How many chunks are physically backed in this pool right now."""
        return self.pools[pool_name].mapped_count()

    def pool_mapped_bytes(self, pool_name: str) -> int:
        return self.pool_mapped_chunks(pool_name) * self.chunk_size

    def free_handle_count(self) -> int:
        return len(self._free_handles)

    def grow(self, pool_name: str, n: int) -> list[int]:
        """Map n free handles into the pool's first n free slots.

        cuMemMap is per-handle (no batch API for arbitrary handles),
        but the trailing cuMemSetAccess is batched across contiguous
        runs of newly-mapped slots — collapsing N syscalls to ~1 in the
        common tail-fill case.

        Returns: list of slot IDs actually mapped, in `first_free_slot`
        order (low-id first). May be shorter than `n` if free handles
        or free slots are exhausted. Callers that only need the count
        wrap in `len()`.

        Returning the IDs (not just count) is the foundation for the
        dst cap-bump rewrite: xpool_actuator passes these IDs straight
        to `dst_alloc.unmark_pages_capped` / `dst_pool.unmark_slots`,
        eliminating the prior helper (`unmark_lowest_capped_after_grow`)
        that re-derived them by sorting `_capped_pages` (correct by
        coincidence — chunk_arena maps at lowest unmapped position
        which matches the lowest capped IDs in steady state — but
        fragile coupling we don't want long-term).
        """
        pool = self.pools[pool_name]
        newly_mapped_slots = []
        for _ in range(n):
            if not self._free_handles:
                break
            slot = pool.first_free_slot()
            if slot is None:
                break
            handle_idx = self._free_handles.pop()
            self._map_handle_to_slot(pool, slot, handle_idx)
            newly_mapped_slots.append(slot)
        # Batched access set across contiguous runs of the just-mapped slots
        self._setaccess_slots_batched(pool, newly_mapped_slots)
        return newly_mapped_slots

    def shrink(self, pool_name: str, n: int, evict_policy: str = "tail") -> int:
        """Unmap n chunks from this pool. Returns number actually unmapped.

        Tail-evict by default. Implemented via batched cuMemUnmap — the
        contiguous tail collapses to a single syscall.

        evict_policy:
            "tail" — pop from the highest-indexed mapped slot first.
        """
        if evict_policy != "tail":
            raise ValueError(f"unknown evict_policy {evict_policy!r}")
        if n <= 0:
            return 0
        pool = self.pools[pool_name]
        # Collect up to n tail slots that are mapped
        slots_to_unmap = []
        for i in range(pool.n_slots - 1, -1, -1):
            if pool.mapped[i] is not None:
                slots_to_unmap.append(i)
                if len(slots_to_unmap) == n:
                    break
        freed = self._unmap_slots_batched(pool, slots_to_unmap)
        self._free_handles.extend(freed)
        return len(freed)

    def shrink_explicit(self, pool_name: str, slot_indices) -> int:
        """Unmap a caller-specified list of slot indices (paper §3.2.2).

        Unlike ``shrink()``, the caller supplies which slots to unmap: the
        fire planner builds them via the owner-provider cost-order walk into
        ``plan.pages_to_unmap``, and the actuator passes that list here.
        cuMemUnmap is batched across contiguous runs of the supplied list.

        Returns: number actually unmapped. Slots that aren't currently
        mapped (None in pool.mapped), or out-of-range, are skipped silently.

        slot_indices: any iterable of int (e.g., list, torch.Tensor).
        """
        pool = self.pools[pool_name]
        # Accept torch.Tensor by converting to a Python list of ints.
        if hasattr(slot_indices, "tolist"):
            slots = slot_indices.tolist()
        else:
            slots = list(slot_indices)
        freed = self._unmap_slots_batched(pool, slots)
        self._free_handles.extend(freed)
        return len(freed)

    def transfer_chunks(self, from_pool: str, to_pool: str, n: int) -> int:
        """Move n physical handles from `from_pool` to `to_pool`.

        Implemented as: shrink(from, n) which frees n handles into the
        free-handle list, then grow(to, n) which pulls them back. The
        two-step structure keeps the actuator policy-agnostic about which
        chunks of `from_pool` are evictable; in production, shrink calls
        the spec's per-pool eviction protocol.

        Returns: number of chunks actually transferred.
        """
        if from_pool == to_pool:
            raise ValueError("from_pool == to_pool")
        unmapped = self.shrink(from_pool, n)
        # Grow may transfer fewer if the destination has too few free slots.
        # In that case, the unmapped-but-not-remapped handles stay in
        # _free_handles and can be retried later.
        # grow() returns list[int] of mapped slot IDs; this helper
        # only reports the count, so wrap in len().
        granted = len(self.grow(to_pool, unmapped))
        return granted

    def cleanup(self) -> None:
        """Tear down the arena: unmap everything, release handles, free VA.

        Owned handles are released here. External (SharedHandlePool) handles
        are NOT released — that's the pool's responsibility.

        No defensive sync: caller (process exit or test teardown) is
        responsible for ensuring no GPU work is in flight against this
        arena's VA. Same fail-fast contract as ``_unmap_slots_batched``.
        """
        for pool in self.pools.values():
            for slot, handle_idx in enumerate(pool.mapped):
                if handle_idx is not None:
                    va = pool.va_base + slot * self.chunk_size
                    rc = CUDA.cuMemUnmap(va, self.chunk_size)
                    if rc != CU_SUCCESS:
                        # Best-effort cleanup; print but do not raise.
                        print(f"warning: cuMemUnmap failed during cleanup: rc={rc}")
        if self._external_pool is None:
            for h in self._owned_handles:
                CUDA.cuMemRelease(h)
        CUDA.cuMemAddressFree(self.va_base, self.total_va_size)


def cross_arena_transfer(
    from_arena: "ChunkArena",
    from_pool: str,
    to_arena: "ChunkArena",
    to_pool: str,
    n: int,
) -> int:
    """Move n physical handles from `from_arena[from_pool]` to `to_arena[to_pool]`.

    Both arenas MUST share the same `SharedHandlePool` (i.e. they were
    constructed with the same `external_handle_pool=`); otherwise the
    handles would not be reachable from `to_arena`'s grow path. This is
    the cross-arena equivalent of `transfer_chunks`, used for KV ↔ mamba
    physical-byte movement (paper §sec:design-l2).

    Returns: number of chunks actually transferred.
    """
    if from_arena is to_arena:
        raise ValueError(
            "from_arena is to_arena — use ChunkArena.transfer_chunks for "
            "intra-arena transfers."
        )
    if from_arena._external_pool is None or to_arena._external_pool is None:
        raise ValueError(
            "cross_arena_transfer requires both arenas to use a "
            "SharedHandlePool (external_handle_pool=...)."
        )
    if from_arena._external_pool is not to_arena._external_pool:
        raise ValueError(
            "cross_arena_transfer requires the two arenas to share the "
            "SAME SharedHandlePool instance, not just equivalent ones."
        )
    unmapped = from_arena.shrink(from_pool, n)
    # grow() returns list[int]; this helper reports only count.
    granted = len(to_arena.grow(to_pool, unmapped))
    return granted
