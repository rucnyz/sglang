"""
Phase 2e.3 — StateSpec interface (paper §4.1).

Each pool exposes its budgeter-facing view through this protocol. The
planner (paper §4.3) and actuator (paper §4.4) interact with pools
exclusively through these methods; concrete details (block tables,
adapter weights, KV layout) stay inside each pool's implementation.

Five required reads:
  - allocated_bytes(): bytes currently backed by physical memory.
  - min_bytes(): floor below which the pool refuses to operate.
  - max_bytes(): ceiling (this pool's static VA reservation).
  - marginal_value(): V_sigma'(m_sigma) at the current allocation.
  - value_at(m): approximate V_sigma at allocation m, for `what-if` queries.

One required write:
  - resize(m): drive the pool to allocation `m` bytes, returning when the
    arena's bytes have actually moved. Should be idempotent w.r.t. the
    current state and respect min_bytes / max_bytes / chunk granularity.

One optional helper:
  - resize_cost(m): expected wall-clock to resize to `m`. The planner uses
    this to gate hysteresis (if cost > expected benefit, skip).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ResizeRequest:
    """A planner decision asking a spec to resize itself.

    The actuator translates one ResizeRequest per spec per control tick
    into ChunkArena.transfer_chunks calls, taking from shrinking specs
    and giving to growing specs.
    """
    spec_name: str
    target_bytes: int


class ResizeError(Exception):
    """Spec refused or could not complete a resize."""


class StateSpec(ABC):
    """Abstract base; every pool exposes one of these to the planner.

    The planner queries reads on each tick, computes new targets via
    Lagrange equalization, and issues resize() calls.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    # Reads ------------------------------------------------------------

    @abstractmethod
    def allocated_bytes(self) -> int:
        """Currently-backed physical bytes."""

    @abstractmethod
    def min_bytes(self) -> int:
        """Refuse to shrink below this."""

    @abstractmethod
    def max_bytes(self) -> int:
        """Refuse to grow above this (= the spec's arena VA reservation)."""

    @abstractmethod
    def marginal_value(self) -> float:
        """V'(m) at current allocation. Higher = should grow.

        Units: arbitrary, but consistent across all specs in the engine
        (so Lagrange equalization is meaningful). The convention is
        "throughput recovery per byte per second."
        """

    def value_at(self, m: int) -> float:
        """Approximate V(m). Default: integrate marginal_value as constant.

        Specs with curve estimators (e.g., LoRA's miss-rate curve) should
        override this with a real interpolation.
        """
        return self.marginal_value() * (m - self.allocated_bytes())

    def resize_cost(self, m: int) -> float:
        """Expected wall-clock cost of a resize, in seconds.

        Default: zero (used by the simplest specs whose resize is just a
        bookkeeping update). Pools that must drain in-flight work
        (paged-KV waiting for chunked-prefill checkpoints, etc.) should
        override.
        """
        return 0.0

    # Writes -----------------------------------------------------------

    @abstractmethod
    def resize(self, m: int) -> None:
        """Move the pool to `m` allocated bytes.

        Implementations should be idempotent (resize to current size = no-op)
        and clamp `m` to [min_bytes, max_bytes]. They should raise
        ResizeError if they cannot reach `m` (e.g., shrinking would evict
        a pinned resource).
        """
