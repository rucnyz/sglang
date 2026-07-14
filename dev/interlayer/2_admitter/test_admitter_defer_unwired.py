"""Admitter.decide_for_req must DEFER (return None), not crash, when the
BudgetAgent has not yet wired owner_provider. Reproduces the swarm race:
400 requests co-arrive the instant the server is ready, before the Budgeter's
first tick builds the actuator chain -> decide_for_req was raising
'owner_provider not wired' and crashing the scheduler (seen on Ling-2.6-flash
tp=4 swarm). The fix returns None (normal admission) until the chain is wired.
"""
from sglang.srt.budgeter.admitter import Admitter
from sglang.srt.budgeter.cost_model import get_cost_model


def test_defers_when_owner_provider_unwired():
    adm = Admitter(cost_model=get_cost_model())
    assert adm.owner_provider is None
    # No tokens_per_page override + unwired chain -> defer, do NOT raise.
    assert adm.decide_for_req(req=object(), scheduler=object()) is None


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v"]))
