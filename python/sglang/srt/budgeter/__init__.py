"""Joint memory budgeter.

In-process agent that runs inside the SGLang scheduler. Each scheduler tick
it snapshots per-pool pressure from SchedulerStats and optionally drives
cross-pool VMM transfers via the planner / actuator path.

Activation gates:
  SGLANG_BUDGETER=1            -> instantiate the agent (snapshot + JSONL logging)
  SGLANG_BUDGETER_LOG=path     -> JSONL log path (default: /tmp/sglang_budgeter.jsonl)
  SGLANG_BUDGETER_TICK_S=1.0   -> tick interval in seconds
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
