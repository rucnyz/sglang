"""Joint memory budgeter (Phase 2).

In-process agent that runs inside the SGLang scheduler. Each scheduler tick
it (a) snapshots per-pool pressure from SchedulerStats, (b) optionally fits
$V_\\sigma(m_\\sigma)$ slopes per pool, (c) optionally calls actuators to
shift logical pool ownership.

Activation gates:
  SGLANG_BUDGETER=1            -> instantiate the agent (read-only, JSONL logging)
  SGLANG_BUDGETER_ACTUATE=1    -> additionally allow actuation (evict / capacity changes)
  SGLANG_BUDGETER_LOG=path     -> JSONL log path (default: /tmp/sglang_budgeter.jsonl)
  SGLANG_BUDGETER_TICK_S=1.0   -> tick interval in seconds

Phase plan: see dev/2.md.
"""

from sglang.srt.budgeter.agent import BudgetAgent

# T6 (paper §3.2.4): process-singleton accessor so the admission-time
# alloc_token_slots hook can find the budgeter without touching
# tree_cache's API surface. Set in BudgetAgent.__init__ when enabled.
_BUDGET_AGENT_SINGLETON = None


def get_budget_agent():
    """Return the process-wide BudgetAgent if instantiated, else None."""
    return _BUDGET_AGENT_SINGLETON


def _set_budget_agent_singleton(agent):
    global _BUDGET_AGENT_SINGLETON
    _BUDGET_AGENT_SINGLETON = agent


__all__ = ["BudgetAgent", "get_budget_agent"]
