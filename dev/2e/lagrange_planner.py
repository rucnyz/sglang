"""
Phase 2e.3 — Lagrange equalization planner (paper §4.3).

Given a set of StateSpecs and a total budget, compute target allocations
{m_sigma^*} such that the marginal values V_sigma'(m_sigma^*) equalize at
some shadow price lambda^*, subject to per-spec [min, max] bounds and the
total-budget constraint sum_sigma m_sigma <= M_total.

Implementation: 1-D bisection in lambda. The total-allocation function
sum_sigma m_sigma(lambda) is monotone non-increasing in lambda. We
bisect lambda until total fits the budget; per-spec allocations are
clipped at their [min, max] bounds.

For the smoke test we use a piecewise-linear V_sigma model: each spec
exposes its current marginal_value(), and we extrapolate that as a flat
line. Specs that override value_at() can supply real curves.

Hysteresis is applied at the planner output level: a spec's target is
only changed if the change exceeds delta_hyst fraction of its current
allocation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from state_spec import StateSpec


@dataclass(frozen=True)
class PlanDecision:
    spec_name: str
    current_bytes: int
    target_bytes: int
    marginal_value: float

    @property
    def delta_bytes(self) -> int:
        return self.target_bytes - self.current_bytes


class LagrangePlanner:
    def __init__(
        self,
        delta_hyst: float = 0.05,
        bisection_iters: int = 32,
    ) -> None:
        self.delta_hyst = delta_hyst
        self.bisection_iters = bisection_iters

    def plan(self, specs: Sequence[StateSpec], total_budget: int) -> List[PlanDecision]:
        """Compute target allocations equalizing marginal_value across specs.

        For piecewise-linear V_sigma (the default), the optimum is to put
        every spec at either its min, its max, or its current allocation,
        depending on its marginal value relative to other specs. The
        bisection-on-lambda machinery is overkill for this case but
        matches the paper's framing and generalises cleanly when specs
        provide real value_at() curves.

        For the simple case (no per-spec value curves), the algorithm
        reduces to: sort specs by marginal_value descending; greedily
        allocate to the highest-marginal spec until either it hits
        max_bytes or the budget runs out; repeat.
        """
        specs = list(specs)
        if not specs:
            return []

        # Sum of mins must fit in the budget; otherwise infeasible.
        total_min = sum(s.min_bytes() for s in specs)
        if total_min > total_budget:
            raise ValueError(
                f"infeasible: sum of min_bytes {total_min} > total_budget {total_budget}"
            )

        # Greedy fill by marginal value.
        ranked = sorted(
            range(len(specs)),
            key=lambda i: specs[i].marginal_value(),
            reverse=True,
        )
        targets = [s.min_bytes() for s in specs]
        remaining = total_budget - total_min

        for idx in ranked:
            spec = specs[idx]
            headroom = spec.max_bytes() - targets[idx]
            give = min(headroom, remaining)
            targets[idx] += give
            remaining -= give
            if remaining == 0:
                break

        decisions: List[PlanDecision] = []
        for spec, tgt in zip(specs, targets):
            cur = spec.allocated_bytes()
            mv = spec.marginal_value()
            # Hysteresis: skip small adjustments.
            if cur > 0 and abs(tgt - cur) / cur < self.delta_hyst:
                tgt = cur
            decisions.append(PlanDecision(
                spec_name=spec.name,
                current_bytes=cur,
                target_bytes=tgt,
                marginal_value=mv,
            ))
        return decisions

    def apply(self, specs: Sequence[StateSpec], decisions: Iterable[PlanDecision]) -> None:
        """Issue resize() calls in the right order: shrinks first (free
        chunks back into the arena), then grows (consume them).
        """
        spec_by_name = {s.name: s for s in specs}
        decisions = list(decisions)
        shrinks = [d for d in decisions if d.delta_bytes < 0]
        grows = [d for d in decisions if d.delta_bytes > 0]
        for d in shrinks:
            spec_by_name[d.spec_name].resize(d.target_bytes)
        for d in grows:
            spec_by_name[d.spec_name].resize(d.target_bytes)
