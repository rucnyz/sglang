"""
Policy implementations for KV-cache scheduling baselines.

Each policy is a specialization of the per-unit value rule from paper Section 7:
    V_u(tau)        = p_hat_u * [R(u,0) - R(u,tau)]  -  h_tau(used_tau) * b_u / lambda_u
    Vt_u(old->new)  = V_u(new) - M_eff(u, old->new)
    a_t*            = argmax_a sum_{(u,tau) in a} Vt_u(...)   s.t. capacity

See paper Section 8 (Baselines Subsumed) for which restrictions each baseline imposes.
"""
from .base import Policy, ReuseUnit, SchedulerState, Action
from .lru import LRUPolicy
from .thunder_agent import ThunderAgentPolicy
from .infercept import InferCeptPolicy
from .continuum import ContinuumPolicy
from .kvflow import KVFlowPolicy
from .ours_greedy import OursGreedyPolicy

__all__ = [
    "Policy",
    "ReuseUnit",
    "SchedulerState",
    "Action",
    "LRUPolicy",
    "ThunderAgentPolicy",
    "InferCeptPolicy",
    "ContinuumPolicy",
    "KVFlowPolicy",
    "OursGreedyPolicy",
]
