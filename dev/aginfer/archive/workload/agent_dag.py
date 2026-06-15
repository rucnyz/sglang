"""
Agent DAG model (paper Section 2.4).

Online-revealed DAG: at decision time t, only V^{<=t} and E^{<=t} are observed.
Future nodes are not enumerable. Future predictors are statistical (see typed
lambda_u estimators in paper Section 7).

This file defines the *data model* + *event stream* the workload driver and
policies share. It is independent of any specific serving engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class NodeKind(str, Enum):
    LLM = "llm"      # V^LLM:  consume/produce KV
    TOOL = "tool"    # V^tool: opaque external call
    SUB = "sub"      # V^sub:  sub-agent dispatch marker


class SubMode(str, Enum):
    BLOCKING = "blocking"
    ASYNC = "async"


@dataclass
class Node:
    id: str
    kind: NodeKind
    # LLM-specific
    n_input_tokens: int = 0
    n_output_tokens: int = 0
    # Tool-specific
    tool_name: str = ""
    tool_duration_sec: float = 0.0
    # Sub-agent specific
    sub_mode: Optional[SubMode] = None
    child_session_id: Optional[str] = None
    join_point: Optional[str] = None


@dataclass
class Session:
    id: str
    nodes: List[Node]
    edges: List[tuple]                  # data dependencies (parent_id, child_id)
    cursor: int = 0                     # index into nodes; cursor advances
    pending_tools: List[str] = field(default_factory=list)
    parent_session: Optional[str] = None


class EventKind(str, Enum):
    SESSION_ARRIVAL = "session_arrival"
    LLM_PREFILL = "llm_prefill"
    LLM_DECODE = "llm_decode"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    SUB_DISPATCH_BLOCKING = "sub_dispatch_blocking"
    SUB_DISPATCH_ASYNC = "sub_dispatch_async"
    SUB_RETURN = "sub_return"
    SESSION_END = "session_end"
    MEMORY_PRESSURE = "memory_pressure"


@dataclass
class Event:
    t: float
    kind: EventKind
    session_id: str
    node_id: Optional[str] = None
    payload: Dict = field(default_factory=dict)


class Workload:
    """Holds the multi-session world. The driver materializes sessions over time
    and emits events the scheduler reacts to. Real implementations will plug
    this into an SGLang client that issues actual generate calls."""

    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self.event_log: List[Event] = []

    def add_session(self, sess: Session) -> None:
        self.sessions[sess.id] = sess
