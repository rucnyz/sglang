"""
Phase 2e.3 — ArenaSpec: a generic StateSpec backed by a ChunkArena pool.

This is the bridge from the StateSpec protocol (paper §4.1) down to the
arena actuator (paper §4.4). It holds:
  - a reference to a ChunkArena and the name of one pool inside it;
  - a callable `pressure_signal: () -> float` that returns the pool's
    current marginal-utility estimate (the V_sigma' signal).

Concrete pool implementations (LoRA, paged-KV, mamba, prefix) are this
class plus a pool-specific pressure callback. The pool-specific
"draining" logic that has to run before a shrink (e.g., paged-KV
must release blocks above the new cap as their owners complete) is
parameterised via the optional `before_shrink` and `after_grow` hooks.
"""

from __future__ import annotations
from typing import Callable, Optional

from chunk_arena import ChunkArena
from state_spec import ResizeError, StateSpec


class ArenaSpec(StateSpec):
    def __init__(
        self,
        arena: ChunkArena,
        pool_name: str,
        min_chunks: int,
        marginal_value_fn: Callable[[], float],
        value_at_fn: Optional[Callable[[int], float]] = None,
        before_shrink: Optional[Callable[[int], None]] = None,
        after_grow: Optional[Callable[[int], None]] = None,
        resize_cost_fn: Optional[Callable[[int], float]] = None,
    ) -> None:
        self._arena = arena
        self._pool_name = pool_name
        self._min_chunks = min_chunks
        self._marginal_value_fn = marginal_value_fn
        self._value_at_fn = value_at_fn
        self._before_shrink = before_shrink
        self._after_grow = after_grow
        self._resize_cost_fn = resize_cost_fn

        max_chunks = arena.pools[pool_name].n_slots
        if min_chunks > max_chunks:
            raise ValueError(
                f"min_chunks {min_chunks} > pool max {max_chunks} for {pool_name}"
            )

    # -- StateSpec reads ------------------------------------------------

    @property
    def name(self) -> str:
        return self._pool_name

    def allocated_bytes(self) -> int:
        return self._arena.pool_mapped_bytes(self._pool_name)

    def min_bytes(self) -> int:
        return self._min_chunks * self._arena.chunk_size

    def max_bytes(self) -> int:
        return self._arena.pools[self._pool_name].n_slots * self._arena.chunk_size

    def marginal_value(self) -> float:
        return self._marginal_value_fn()

    def value_at(self, m: int) -> float:
        if self._value_at_fn is not None:
            return self._value_at_fn(m)
        return super().value_at(m)

    def resize_cost(self, m: int) -> float:
        if self._resize_cost_fn is not None:
            return self._resize_cost_fn(m)
        return super().resize_cost(m)

    # -- StateSpec write ------------------------------------------------

    def resize(self, m: int) -> None:
        m = max(self.min_bytes(), min(self.max_bytes(), m))
        # Round to chunk granularity (round-to-nearest, biased down).
        chunk = self._arena.chunk_size
        target_chunks = m // chunk
        target_chunks = max(self._min_chunks, target_chunks)
        current_chunks = self._arena.pool_mapped_chunks(self._pool_name)
        if target_chunks == current_chunks:
            return
        if target_chunks < current_chunks:
            n = current_chunks - target_chunks
            if self._before_shrink is not None:
                self._before_shrink(n)
            unmapped = self._arena.shrink(self._pool_name, n)
            if unmapped < n:
                raise ResizeError(
                    f"{self._pool_name}: tried to shrink by {n}, only {unmapped} succeeded"
                )
        else:
            n = target_chunks - current_chunks
            granted = self._arena.grow(self._pool_name, n)
            if granted < n:
                raise ResizeError(
                    f"{self._pool_name}: tried to grow by {n}, only {granted} succeeded"
                )
            if self._after_grow is not None:
                self._after_grow(n)
